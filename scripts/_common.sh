#!/usr/bin/env bash
# Shared helpers: run from the RLAIF root, load .env, use the project venv, log every stage.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; [ -f .env ] && . ./.env; set +a
export LITELLM_LOCAL_MODEL_COST_MAP="${LITELLM_LOCAL_MODEL_COST_MAP:-True}"
export TOKENIZERS_PARALLELISM=false
# Explicit import path: macOS file sync can flag the venv's editable .pth files "hidden", which Python skips.
export PYTHONPATH="$PWD/src:$PWD/third_party/mini-swe-agent/src${PYTHONPATH:+:$PYTHONPATH}"
PY=".venv/bin/python"
mkdir -p outputs/logs
# GPU layout from the GPU count and size: the last GPU serves vLLM (rollouts, agent runs), the others train.
#   4 x B200 (Lambda):            train 0-2, vLLM + A4B judge on 3
#   8 x B200 (CoreWeave HGX):     train 0-6, vLLM + A4B judge on 7
#   8 x H100/H200 (< 150 GB):     train 0-5, vLLM on 7, judge on 6 (the 52 GB judge does not fit next to the policy)
# The count is the machine's real one (0 on the Mac); a GPU step on a machine with too few GPUs stops in need_gpus.
NGPU="${OA_NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)}"; NGPU="${NGPU:-0}"
GPU_GB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | awk '{ print int($1 / 1024) }' || true)"
GPU_GB="${GPU_GB:-180}"
if [ "$NGPU" -ge 2 ]; then
  VLLM_GPU="${VLLM_GPU:-$((NGPU - 1))}"
  if [ -z "${JUDGE_GPU:-}" ]; then
    if [ "$GPU_GB" -lt 150 ] && [ "$NGPU" -ge 4 ]; then JUDGE_GPU=$((NGPU - 2)); else JUDGE_GPU="$VLLM_GPU"; fi
  fi
  LAST_TRAIN=$((NGPU - 2)); [ "$JUDGE_GPU" != "$VLLM_GPU" ] && LAST_TRAIN=$((NGPU - 3))
  TRAIN_GPUS="${TRAIN_GPUS:-$(seq 0 "$LAST_TRAIN" | paste -sd, -)}"
else
  VLLM_GPU="${VLLM_GPU:-}"; JUDGE_GPU="${JUDGE_GPU:-}"; TRAIN_GPUS="${TRAIN_GPUS:-}"
fi
NTRAIN="$(echo "$TRAIN_GPUS" | tr ',' '\n' | grep -c . || true)"
EVAL_TP="${EVAL_TP:-$(( NGPU < 1 ? 1 : (NGPU < 4 ? NGPU : 4) ))}"   # vLLM tensor parallel for batch evaluation
# the judge shares the policy's GPU (0.45 of it) unless it has its own
[ "$JUDGE_GPU" = "$VLLM_GPU" ] && JUDGE_GPU_UTIL="${JUDGE_GPU_UTIL:-0.45}" || JUDGE_GPU_UTIL="${JUDGE_GPU_UTIL:-0.85}"
export VLLM_GPU JUDGE_GPU TRAIN_GPUS NTRAIN
log() { echo "[$(date '+%H:%M:%S')] $*"; }
# GPU steps call this first: enough GPUs, and every index in the layout exists on this machine.
need_gpus() {
  local min="${OA_MIN_GPUS:-4}" i
  if [ "$NGPU" -lt "$min" ]; then
    log "this machine shows $NGPU GPU(s) (nvidia-smi -L); the pipeline needs at least $min (3 train + 1 for vLLM and the judge)."
    log "Use a 4-GPU (or 8-GPU) instance. OA_MIN_GPUS=2 allows a smaller test run (1 train + 1 vLLM; slow, may not fit)."
    exit 1
  fi
  if [ "$GPU_GB" -lt "${OA_MIN_GPU_GB:-75}" ]; then
    log "GPU memory is ${GPU_GB} GB; the A4B judge alone needs about 52 GB plus cache, so each GPU needs >= 80 GB"
    log "(B200, H200, H100 80GB, A100 80GB). Use such an instance."
    exit 1
  fi
  for i in $(echo "$TRAIN_GPUS,$VLLM_GPU,$JUDGE_GPU" | tr ',' ' '); do
    [ "$i" -ge 0 ] 2>/dev/null && [ "$i" -lt "$NGPU" ] || {
      log "GPU index '$i' does not exist here (GPUs 0-$((NGPU - 1))); check VLLM_GPU / JUDGE_GPU / TRAIN_GPUS in .env"; exit 1; }
  done
}
train_launch() { need_gpus; CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" .venv/bin/accelerate launch --num_processes "$NTRAIN" "$@"; }
budget_gate() { $PY -c "from oa_rlaif.budget import BudgetGuard; g=BudgetGuard(); g.check('$1'); print('budget ok:', g.summary())"; }
vllm_up() {  # $1 = model path or id; serves on $VLLM_GPU for TRL rollouts
  need_gpus
  local model="$1"
  CUDA_VISIBLE_DEVICES="$VLLM_GPU" nohup .venv/bin/trl vllm-serve --model "$model" --port "${VLLM_PORT:-8000}" \
      --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.85}" --max-model-len 8192 > outputs/logs/vllm_server.log 2>&1 &
  echo $! > outputs/vllm_server.pid
  for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:${VLLM_PORT:-8000}/health/" >/dev/null && { log "vLLM up ($model)"; return 0; }; sleep 5; done
  log "vLLM did not start; see outputs/logs/vllm_server.log"; return 1
}
vllm_down() { [ -f outputs/vllm_server.pid ] && kill "$(cat outputs/vllm_server.pid)" 2>/dev/null || true; rm -f outputs/vllm_server.pid; }
# A4B judge served locally for GRPO (decided 2026-10-10: API quotas and API latency would idle the GPUs).
# On B200 it shares $VLLM_GPU with the policy server: start the policy first (VLLM_GPU_UTIL=0.40), then this
# (0.45 of 180 GB). On smaller GPUs it gets its own GPU ($JUDGE_GPU, 0.85).
JUDGE_LOCAL_MODEL="${JUDGE_LOCAL_MODEL:-google/gemma-4-26B-A4B-it}"
judge_up() {
  need_gpus
  CUDA_VISIBLE_DEVICES="$JUDGE_GPU" nohup .venv/bin/vllm serve "$JUDGE_LOCAL_MODEL" --port "${JUDGE_PORT:-8001}" \
      --served-model-name a4b-judge --gpu-memory-utilization "$JUDGE_GPU_UTIL" --max-model-len 8192 \
      > outputs/logs/judge_server.log 2>&1 &
  echo $! > outputs/judge_server.pid
  for _ in $(seq 1 180); do curl -sf "http://127.0.0.1:${JUDGE_PORT:-8001}/health" >/dev/null && break; sleep 5; done
  curl -sf "http://127.0.0.1:${JUDGE_PORT:-8001}/health" >/dev/null || { log "judge did not start; see outputs/logs/judge_server.log"; return 1; }
  export JUDGE_API_BASE="http://127.0.0.1:${JUDGE_PORT:-8001}/v1" JUDGE_API_KEY=local JUDGE_MODEL=a4b-judge \
         JUDGE_RPM=0 JUDGE_MAX_TOKENS=512 JUDGE_USD_PER_MTOK_IN=0 JUDGE_USD_PER_MTOK_OUT=0 \
         JUDGE_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
  log "A4B judge up locally ($JUDGE_LOCAL_MODEL, thinking off)"
}
judge_down() { [ -f outputs/judge_server.pid ] && kill "$(cat outputs/judge_server.pid)" 2>/dev/null || true; rm -f outputs/judge_server.pid; }

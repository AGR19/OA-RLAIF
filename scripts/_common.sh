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
# OA_PLAN_TIME: one planning instant for every GPU process, so they all plan the same step count
train_launch() { need_gpus; OA_PLAN_TIME="$(date +%s)" CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" .venv/bin/accelerate launch --num_processes "$NTRAIN" "$@"; }
budget_gate() { $PY -c "from oa_rlaif.budget import BudgetGuard; g=BudgetGuard(); g.check('$1'); print('budget ok:', g.summary())"; }
# ---- vLLM servers ------------------------------------------------------------------------------------------------
# One helper starts every server (rollouts, judge, agent runs). A server runs in its own session (setsid), so stopping
# it stops its engine processes too. GPU scripts call own_servers first: it clears servers left by a crashed earlier
# run (any code version) and stops this script's servers on every exit. A server counts as up only if the process
# just started is alive AND the port serves the expected model: a leftover server on the same port is never reused.
kill_tree() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$c"; done; kill -TERM "$1" 2>/dev/null || true; }
stop_server() {  # $1 = vllm_server | judge_server
  local f="outputs/$1.pid" pid
  [ -f "$f" ] || return 0
  pid="$(cat "$f")"
  kill -TERM -- "-$pid" 2>/dev/null || kill_tree "$pid"
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$pid" 2>/dev/null || true
  rm -f "$f"
}
stop_all_servers() { stop_server judge_server; stop_server vllm_server; }
stop_leftovers() {  # vLLM servers, engines and trainers started from this folder by an earlier run that crashed
  local p cmd n=0
  for p in $(pgrep -f -i "vllm|oa_rlaif\.(train|eval)|accelerate" 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    [ "$(readlink "/proc/$p/cwd" 2>/dev/null)" = "$PWD" ] || continue
    cmd="$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | cut -c1-90)"
    log "stopping a leftover process from an earlier run: $p $cmd"; kill_tree "$p"; n=$((n + 1))
  done
  if [ "$n" -gt 0 ]; then sleep 10; for p in $(pgrep -f -i "vllm|oa_rlaif\.(train|eval)|accelerate" 2>/dev/null); do
    [ "$(readlink "/proc/$p/cwd" 2>/dev/null)" = "$PWD" ] && kill -KILL "$p" 2>/dev/null; done; fi
  return 0
}
own_servers() { stop_all_servers; stop_leftovers; trap stop_all_servers EXIT; }
gpu_mem_gb() { nvidia-smi --query-gpu="memory.$2" --format=csv,noheader,nounits -i "$1" 2>/dev/null | awk '{ print int($1 / 1024) }'; }
wait_for_memory() {  # $1 = GPU, $2 = share of its memory the server will take
  local need; need="$(awk -v t="$(gpu_mem_gb "$1" total)" -v u="$2" 'BEGIN { print int(t * u) }')"
  for _ in $(seq 1 60); do [ "$(gpu_mem_gb "$1" free)" -ge "$need" ] && return 0; sleep 2; done
  log "GPU $1 has $(gpu_mem_gb "$1" free) GB free; this server needs $need GB. What holds it:"; nvidia-smi || true
  return 1
}
serve() {  # serve NAME GPU SHARE PORT MODEL_ID LOG -- command...
  local name="$1" gpu="$2" share="$3" port="$4" expect="$5" logf="$6"; shift 7
  if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    log "port $port already answers: stopping that server first"; stop_server "$name"; stop_leftovers
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && {
      log "another program serves port $port (not started from this folder); stop it and run again"; return 1; }
  fi
  wait_for_memory "$gpu" "$share" || return 1
  CUDA_VISIBLE_DEVICES="$gpu" setsid nohup "$@" > "$logf" 2>&1 &
  local pid=$!
  echo "$pid" > "outputs/$name.pid"
  for _ in $(seq 1 180); do
    kill -0 "$pid" 2>/dev/null || { log "$name stopped during start-up; end of $logf:"; tail -n 20 "$logf"; return 1; }
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -qF "\"$expect\"" && { log "$name up on GPU $gpu ($expect)"; return 0; }
      log "port $port answers but does not serve $expect"; return 1
    fi
    sleep 5
  done
  log "$name did not start within 15 minutes; see $logf"; return 1
}
vllm_up() {  # $1 = model path or id; serves on $VLLM_GPU for TRL rollouts
  need_gpus
  serve vllm_server "$VLLM_GPU" "${VLLM_GPU_UTIL:-0.85}" "${VLLM_PORT:-8000}" "$1" outputs/logs/vllm_server.log -- \
    .venv/bin/trl vllm-serve --model "$1" --port "${VLLM_PORT:-8000}" \
    --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.85}" --max-model-len 8192
}
vllm_down() { stop_server vllm_server; }
agent_server_up() {  # $1 = model, $2 = log: the model behind the mini-SWE-agent runs, as oa-rlaif-e4b
  need_gpus
  serve vllm_server "$VLLM_GPU" 0.85 8000 oa-rlaif-e4b "$2" -- .venv/bin/vllm serve "$1" --port 8000 \
    --served-model-name oa-rlaif-e4b --gpu-memory-utilization 0.85 --max-model-len 8192
}
# A4B judge served locally for GRPO (decided 2026-10-10: API quotas and API latency would idle the GPUs).
# On B200 it shares $VLLM_GPU with the policy server: start the policy first (VLLM_GPU_UTIL=0.40), then this
# (0.45 of 180 GB). On smaller GPUs it gets its own GPU ($JUDGE_GPU, 0.85).
JUDGE_LOCAL_MODEL="${JUDGE_LOCAL_MODEL:-google/gemma-4-26B-A4B-it}"
judge_up() {
  need_gpus
  serve judge_server "$JUDGE_GPU" "$JUDGE_GPU_UTIL" "${JUDGE_PORT:-8001}" a4b-judge outputs/logs/judge_server.log -- \
    .venv/bin/vllm serve "$JUDGE_LOCAL_MODEL" --port "${JUDGE_PORT:-8001}" --served-model-name a4b-judge \
    --gpu-memory-utilization "$JUDGE_GPU_UTIL" --max-model-len 8192 || return 1
  export JUDGE_API_BASE="http://127.0.0.1:${JUDGE_PORT:-8001}/v1" JUDGE_API_KEY=local JUDGE_MODEL=a4b-judge \
         JUDGE_RPM=0 JUDGE_MAX_TOKENS=512 JUDGE_USD_PER_MTOK_IN=0 JUDGE_USD_PER_MTOK_OUT=0 \
         JUDGE_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
  log "A4B judge up locally ($JUDGE_LOCAL_MODEL, thinking off)"
}
judge_down() { stop_server judge_server; }

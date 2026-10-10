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
log() { echo "[$(date '+%H:%M:%S')] $*"; }
budget_gate() { $PY -c "from oa_rlaif.budget import BudgetGuard; g=BudgetGuard(); g.check('$1'); print('budget ok:', g.summary())"; }
vllm_up() {  # $1 = model path or id; serves on GPU 3 for TRL rollouts
  local model="$1"
  CUDA_VISIBLE_DEVICES="${VLLM_GPU:-3}" nohup .venv/bin/trl vllm-serve --model "$model" --port "${VLLM_PORT:-8000}" \
      --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.85}" --max-model-len 8192 > outputs/logs/vllm_server.log 2>&1 &
  echo $! > outputs/vllm_server.pid
  for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:${VLLM_PORT:-8000}/health/" >/dev/null && { log "vLLM up ($model)"; return 0; }; sleep 5; done
  log "vLLM did not start; see outputs/logs/vllm_server.log"; return 1
}
vllm_down() { [ -f outputs/vllm_server.pid ] && kill "$(cat outputs/vllm_server.pid)" 2>/dev/null || true; rm -f outputs/vllm_server.pid; }
# A4B judge served locally for GRPO (decided 2026-10-10: API quotas and API latency would idle the GPUs).
# Shares GPU 3 with the policy server; start the policy first (VLLM_GPU_UTIL=0.40), then this (0.45 of 180 GB).
JUDGE_LOCAL_MODEL="${JUDGE_LOCAL_MODEL:-google/gemma-4-26B-A4B-it}"
judge_up() {
  CUDA_VISIBLE_DEVICES="${VLLM_GPU:-3}" nohup .venv/bin/vllm serve "$JUDGE_LOCAL_MODEL" --port "${JUDGE_PORT:-8001}" \
      --served-model-name a4b-judge --gpu-memory-utilization "${JUDGE_GPU_UTIL:-0.45}" --max-model-len 8192 \
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

#!/usr/bin/env bash
# GPU box: evaluate a merged model:  scripts/50_eval.sh sdft   |   scripts/50_eval.sh grpo --full
#   1. held-out + zero-shot decisions and closed-book knowledge (vLLM batch, tensor parallel $EVAL_TP)
#   2. with --full: the whole Anatomy alignment -> OAEI results row (P, R, F, Recall+, Coherent)
#   3. the same model as a mini-SWE-agent in the sandbox (Docker, or the user sandbox; 200 Anatomy episodes), comparable to the
#      base-model agent run in 30_baseline.sh
. "$(dirname "$0")/_common.sh"
need_gpus
TAG="$1"; shift || true
budget_gate "eval_$TAG"
$PY -m oa_rlaif.eval.run_eval --model "outputs/merged/$TAG" --tag "$TAG" --tp "$EVAL_TP" 2>&1 | tee "outputs/logs/eval_$TAG.log"
if [ "${1:-}" = "--full" ]; then
  RAW="${OA_RAW_DIR:-data/raw}"
  $PY -m oa_rlaif.eval.full_align --model "outputs/merged/$TAG" --tp "$EVAL_TP" --name anatomy \
     --source "$RAW/anatomy-dataset-2/mouse.owl" --target "$RAW/anatomy-dataset-2/human.owl" \
     --reference "$RAW/anatomy-dataset-2/reference.rdf"
fi
CUDA_VISIBLE_DEVICES="$VLLM_GPU" nohup .venv/bin/vllm serve "outputs/merged/$TAG" --port 8000 --served-model-name oa-rlaif-e4b \
  --max-model-len 8192 > "outputs/logs/vllm_agent_$TAG.log" 2>&1 &
echo $! > outputs/vllm_server.pid; trap vllm_down EXIT
for _ in $(seq 1 120); do curl -sf http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
curl -sf http://127.0.0.1:8000/health >/dev/null || { log "vLLM did not start for the agent run; see outputs/logs/vllm_agent_$TAG.log"; exit 1; }
$PY -m oa_rlaif.eval.agent_eval --episodes data/eval/heldout_anatomy.jsonl --n 200 --tag "$TAG" --env "${OA_AGENT_ENV:-docker}" \
  2>&1 | tee "outputs/logs/agent_eval_$TAG.log"

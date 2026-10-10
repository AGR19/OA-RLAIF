#!/usr/bin/env bash
# GPU box: base Gemma 4 E4B on all held-out sets (vLLM batch) + mini-SWE-agent episodes in the sandbox (OA_AGENT_ENV).
. "$(dirname "$0")/_common.sh"
budget_gate baseline
MODEL="$(.venv/bin/python -c "from oa_rlaif.train.common import load_config; print(load_config()['model']['id'])")"
$PY -m oa_rlaif.eval.run_eval --model "$MODEL" --tag base --tp "$EVAL_TP" 2>&1 | tee outputs/logs/eval_base.log
CUDA_VISIBLE_DEVICES="$VLLM_GPU" nohup .venv/bin/vllm serve "$MODEL" --port 8000 --served-model-name oa-rlaif-e4b --max-model-len 8192 > outputs/logs/vllm_agent.log 2>&1 &
echo $! > outputs/vllm_server.pid; trap vllm_down EXIT
until curl -sf http://127.0.0.1:8000/health >/dev/null; do sleep 5; done
$PY -m oa_rlaif.eval.agent_eval --episodes data/eval/heldout_anatomy.jsonl --n 200 --tag base --env "${OA_AGENT_ENV:-docker}"

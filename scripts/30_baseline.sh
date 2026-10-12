#!/usr/bin/env bash
# GPU box: base Gemma 4 E4B on all held-out sets (vLLM batch) + mini-SWE-agent episodes in the sandbox (OA_AGENT_ENV).
. "$(dirname "$0")/_common.sh"
need_gpus
own_servers
budget_gate baseline
MODEL="$(.venv/bin/python -c "from oa_rlaif.train.common import load_config; print(load_config()['model']['id'])")"
$PY -m oa_rlaif.eval.run_eval --model "$MODEL" --tag base --tp "$EVAL_TP" 2>&1 | tee outputs/logs/eval_base.log
agent_server_up "$MODEL" outputs/logs/vllm_agent.log
$PY -m oa_rlaif.eval.agent_eval --episodes data/eval/heldout_anatomy.jsonl --n 200 --tag base --env "${OA_AGENT_ENV:-docker}"

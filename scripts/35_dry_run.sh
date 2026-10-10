#!/usr/bin/env bash
# LAMBDA: 20 SDFT steps + 10 GRPO steps (no judge) to measure throughput; writes outputs/dry_run.json.
. "$(dirname "$0")/_common.sh"
budget_gate dry_run
vllm_up "$(.venv/bin/python -c "from oa_rlaif.train.common import load_config; print(load_config()['model']['id'])")"
trap vllm_down EXIT
CUDA_VISIBLE_DEVICES=0,1,2 .venv/bin/accelerate launch --num_processes 3 -m oa_rlaif.train.sdft --dry-run --max-steps 20 2>&1 | tee outputs/logs/dry_run_sdft.log
$PY -m oa_rlaif.progress status

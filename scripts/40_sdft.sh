#!/usr/bin/env bash
# LAMBDA: Stage 1 SDFT/OPSD (TRL SDFTTrainer). vLLM rollout server on GPU 3, training on GPUs 0-2.
#   scripts/40_sdft.sh            on-policy self-distillation
#   scripts/40_sdft.sh --mode sft SFT comparison row
. "$(dirname "$0")/_common.sh"
budget_gate sdft
MODEL="$(.venv/bin/python -c "from oa_rlaif.train.common import load_config; print(load_config()['model']['id'])")"
vllm_up "$MODEL"; trap vllm_down EXIT
CUDA_VISIBLE_DEVICES=0,1,2 .venv/bin/accelerate launch --num_processes 3 -m oa_rlaif.train.sdft "$@" 2>&1 | tee outputs/logs/sdft.log
$PY -m oa_rlaif.export.release merge --adapter outputs/checkpoints/sdft/final --out outputs/merged/sdft

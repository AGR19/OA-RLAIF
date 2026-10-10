#!/usr/bin/env bash
# GPU box: Stage 1 OPSD (TRL SDFTTrainer). vLLM rollout server on the last GPU, training on the others.
#   scripts/40_sdft.sh            on-policy self-distillation
#   scripts/40_sdft.sh --mode sft SFT comparison row
. "$(dirname "$0")/_common.sh"
budget_gate sdft
MODEL="$(.venv/bin/python -c "from oa_rlaif.train.common import load_config; print(load_config()['model']['id'])")"
vllm_up "$MODEL"; trap vllm_down EXIT
train_launch -m oa_rlaif.train.sdft "$@" 2>&1 | tee outputs/logs/sdft.log
MODE=sdft; case " $* " in *" --mode sft "*) MODE=sft ;; esac
$PY -m oa_rlaif.export.release merge --adapter "outputs/checkpoints/$MODE/final" --out "outputs/merged/$MODE"

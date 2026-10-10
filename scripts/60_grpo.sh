#!/usr/bin/env bash
# LAMBDA: Stage 2 GRPO from the merged SDFT model. R = 0.8*(0.5 gold + 0.5 DRAL-OA Eq.1) + 0.2*A4B judge/10.
# The last GPU hosts two vLLM servers: the policy (rollouts) and the A4B judge (local, thinking off). Pass --no-judge to skip it.
. "$(dirname "$0")/_common.sh"
budget_gate grpo
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.40}" vllm_up outputs/merged/sdft; trap 'judge_down; vllm_down' EXIT
case " $* " in *" --no-judge "*) ;; *) judge_up ;; esac
train_launch -m oa_rlaif.train.grpo --init outputs/merged/sdft "$@" 2>&1 | tee outputs/logs/grpo.log
$PY -m oa_rlaif.export.release merge --adapter outputs/checkpoints/grpo/final --base outputs/merged/sdft --out outputs/merged/grpo

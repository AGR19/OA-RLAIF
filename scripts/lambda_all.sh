#!/usr/bin/env bash
# ONE COMMAND on a fresh Lambda 4xB200 instance (Ubuntu). From the cloned repo:
#
#   git clone https://github.com/AGR19/OA-RLAIF.git && cd OA-RLAIF   # private repo: use a GitHub token as password
#   scripts/lambda_all.sh
#
# It asks for the Hugging Face token once (after installing), then runs every step without further input and shows a
# live dashboard: total progress bar, the current step with its own bar, elapsed time, ETA and GPU spend so far.
# Full output of each step: outputs/logs/stage_<step>.log
#
#   1 setup       packages, Gemma 4 E4B + 26B-A4B judge weights, Docker sandbox, llama.cpp
#   2 data        private HF datasets -> data/
#   3 check_env   4 GPUs, libraries, data, Docker agent sandbox (stops here if anything is missing)
#   4 gguf_check  converts the BASE model to GGUF first, so export problems show before training
#   5 dry_run     times SDFT and GRPO (with the judge) and plans the step counts to fit the budget
#   6 baseline    base E4B: held-out sets + mini-SWE-agent episodes
#   7 sdft        Stage 1 on-policy self-distillation, merged -> outputs/merged/sdft
#   8 eval_sdft   SDFT model: held-out sets + agent episodes
#   9 grpo        Stage 2 GRPO: DRAL-OA reward + local A4B judge (RLAIF), merged -> outputs/merged/grpo
#  10 eval_grpo   GRPO model: held-out, zero-shot, full Anatomy alignment (OAEI row), agent episodes
#  11 export      GGUF Q4_K_M + Ollama Modelfile (base + SDFT + GRPO in one file)
#  12 push        private HF: oa-rlaif-e4b-gguf, oa-rlaif-e4b-sdft, oa-rlaif-e4b-grpo, oa-rlaif-results
#  (optional)     RUN_SFT_ARM=1 adds the plain-SFT comparison model (~3 GPU h, ~$80)
#
# Resumable: finished steps are skipped on the next run (outputs/stages/<step>.done). After a failure the dashboard
# shows the end of that step's log; fix the cause and run the same command again.
# Budget: every billable step stops at $450 of $500.  WHEN IT SAYS DONE, TERMINATE THE INSTANCE (billing continues).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/stages outputs/logs
export OA_RAW_DIR="${OA_RAW_DIR:-data/raw}"
[ -f outputs/stages/run.start ] || date +%s > outputs/stages/run.start
DASH="python3 scripts/dashboard.py --clear"

stage() {
  local name="$1"; shift
  [ -f "outputs/stages/$name.done" ] && return 0
  rm -f "outputs/stages/$name.failed"
  date +%s > "outputs/stages/$name.start"
  "$@" > "outputs/logs/stage_$name.log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do $DASH || true; sleep "${DASH_INTERVAL:-10}"; done
  if wait "$pid"; then
    touch "outputs/stages/$name.done"
    $DASH || true
  else
    touch "outputs/stages/$name.failed"
    $DASH || true
    echo; echo "STEP '$name' FAILED. Last lines of outputs/logs/stage_$name.log:"; echo
    tail -n 40 "outputs/logs/stage_$name.log"
    echo; echo "Fix the cause, then run scripts/lambda_all.sh again (finished steps are skipped)."
    exit 1
  fi
}

stage setup scripts/lambda_setup.sh

# Hugging Face login, once (the token stays in your terminal and in ~/.cache/huggingface on this machine).
if ! .venv/bin/hf auth whoami >/dev/null 2>&1; then
  echo; echo "Hugging Face login needed (the account that owns the private datasets). Paste your token:"
  .venv/bin/hf auth login
fi
NS="$(.venv/bin/python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")"
sed -i '/^HF_NAMESPACE=/d; /^OA_RAW_DIR=/d' .env && printf 'HF_NAMESPACE=%s\nOA_RAW_DIR=%s\n' "$NS" "$OA_RAW_DIR" >> .env

stage data       env PYTHONPATH=src .venv/bin/python -m oa_rlaif.data.hf_push --pull
stage check_env  scripts/00_check_env.sh
stage gguf_check scripts/72_gguf_check.sh
stage dry_run    scripts/35_dry_run.sh
stage baseline   scripts/30_baseline.sh
stage sdft       scripts/40_sdft.sh
stage eval_sdft  scripts/50_eval.sh sdft
if [ "${RUN_SFT_ARM:-0}" = 1 ]; then
  stage sft_arm  scripts/40_sdft.sh --mode sft
  stage eval_sft scripts/50_eval.sh sft
fi
stage grpo       scripts/60_grpo.sh
stage eval_grpo  scripts/50_eval.sh grpo --full
stage export     scripts/70_export.sh
stage push       env PYTHONPATH=src .venv/bin/python -m oa_rlaif.data.hf_push --models

echo
echo "DONE. Your GGUF: https://huggingface.co/$NS/oa-rlaif-e4b-gguf (private)"
echo "Also: $NS/oa-rlaif-e4b-sdft and $NS/oa-rlaif-e4b-grpo (LoRA adapters), dataset $NS/oa-rlaif-results."
echo "NOW TERMINATE THE INSTANCE in the Lambda console. Billing continues until you do."

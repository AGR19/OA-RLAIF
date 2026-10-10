#!/usr/bin/env bash
# ONE COMMAND on a fresh Lambda 4xB200 instance. Run from the cloned repo:
#
#   git clone https://github.com/AGR19/OA-RLAIF.git && cd OA-RLAIF   # private: sign in with a GitHub token
#   scripts/lambda_all.sh            # first run installs, then asks you to run `.venv/bin/hf auth login`
#   scripts/lambda_all.sh            # (again) runs everything below and pushes the model (private)
#
# Stages (each one leaves outputs/stages/<name>.done; re-running skips finished stages, so after a failure
# fix the cause and run the same command again):
#   setup        uv venv + pinned wheels, Gemma 4 E4B + 26B-A4B judge weights, Docker sandbox, llama.cpp
#   data         pull the private HF datasets into data/ (episodes, SDFT/GRPO files, eval sets, DRAL layers)
#   check_env    4 GPUs at sm_100, libraries, data present, Docker sandbox agent test
#   gguf_check   convert the BASE model to GGUF first, so export problems surface before training
#   dry_run      20 SDFT + 10 GRPO steps; measures throughput (outputs/dry_run.json)
#   baseline     base E4B on all held-out sets
#   sdft         Stage 1 on-policy self-distillation, then merge -> outputs/merged/sdft
#   eval_sdft    held-out evaluation of the SDFT model
#   grpo         Stage 2 GRPO (DRAL-OA reward + local A4B judge), then merge -> outputs/merged/grpo
#   eval_grpo    held-out + zero-shot evaluation and the full Anatomy alignment (OAEI row)
#   export       merged GRPO model -> GGUF Q4_K_M + Ollama Modelfile
#   push         private HF: LoRA adapters, the GGUF, the evaluation results
#
# Every billable stage checks the budget first and stops at OA_BUDGET_STOP_USD ($450 of $500).
# Watch from a second terminal:  scripts/status.sh --live
# When it finishes: TERMINATE THE INSTANCE in the Lambda console. Billing continues until you do.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/stages outputs/logs

export OA_RAW_DIR="${OA_RAW_DIR:-data/raw}"

stage() {
  local name="$1"; shift
  if [ -f "outputs/stages/$name.done" ]; then echo "[skip] $name (done earlier)"; return 0; fi
  echo "[$(date '+%H:%M:%S')] >>> $name"
  "$@" 2>&1 | tee "outputs/logs/stage_$name.log"
  touch "outputs/stages/$name.done"
}
py() { .venv/bin/python "$@"; }

stage setup      scripts/lambda_setup.sh
# Hugging Face login (private datasets and the final push). Either `.venv/bin/hf auth login` or export HF_TOKEN.
.venv/bin/hf auth whoami >/dev/null 2>&1 || { echo "Not logged in to Hugging Face: run '.venv/bin/hf auth login' (or export HF_TOKEN=...) and re-run this script."; exit 1; }
NS="$(.venv/bin/python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")"
sed -i '/^HF_NAMESPACE=/d; /^OA_RAW_DIR=/d' .env && printf 'HF_NAMESPACE=%s\nOA_RAW_DIR=%s\n' "$NS" "$OA_RAW_DIR" >> .env
echo "Hugging Face namespace: $NS"
stage data       env PYTHONPATH=src .venv/bin/python -m oa_rlaif.data.hf_push --pull
stage check_env  scripts/00_check_env.sh
stage gguf_check scripts/72_gguf_check.sh
stage dry_run    scripts/35_dry_run.sh
stage baseline   scripts/30_baseline.sh
stage sdft       scripts/40_sdft.sh
stage eval_sdft  scripts/50_eval.sh sdft
stage grpo       scripts/60_grpo.sh
stage eval_grpo  scripts/50_eval.sh grpo --full
stage export     scripts/70_export.sh
stage push       env PYTHONPATH=src .venv/bin/python -m oa_rlaif.data.hf_push --models

echo
echo "DONE. Private HF repos under $NS: oa-rlaif-e4b-gguf (GGUF + Modelfile), oa-rlaif-e4b-sdft,"
echo "oa-rlaif-e4b-grpo (LoRA adapters), oa-rlaif-results (dataset). Spend: scripts/status.sh"
echo "NOW TERMINATE THE INSTANCE in the Lambda console."

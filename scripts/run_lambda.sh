#!/usr/bin/env bash
# LAMBDA: the whole run in one command, after setup and the dry run:
#   scripts/lambda_setup.sh && scripts/00_check_env.sh && scripts/35_dry_run.sh   (once; check outputs/dry_run.json)
#   scripts/run_lambda.sh                                                         (baseline -> ... -> GGUF)
# Every stage checks the budget first (stops at OA_BUDGET_STOP_USD) and the chain stops at the first failure.
# Watch from a second shell:  scripts/status.sh --live
# Nothing is pushed: after it finishes, review the results, then run  scripts/80_verify_and_push.sh --models
# (with your OK) and TERMINATE THE INSTANCE in the Lambda console: billing continues until you do.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -s outputs/dry_run.json ] || { echo "run scripts/35_dry_run.sh first (it sets the step counts)"; exit 1; }
run() { echo "[$(date '+%H:%M:%S')] >>> $*"; "$@"; }
run scripts/30_baseline.sh
run scripts/40_sdft.sh
run scripts/50_eval.sh sdft
run scripts/60_grpo.sh
run scripts/50_eval.sh grpo --full
run scripts/70_export.sh
echo "[$(date '+%H:%M:%S')] done. GGUF: outputs/release/oa-rlaif-e4b/ · results: outputs/eval/ · spend: scripts/status.sh"
echo "Next: review results, then (with OK) scripts/80_verify_and_push.sh --models, then TERMINATE the instance."

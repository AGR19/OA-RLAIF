#!/usr/bin/env bash
# ONE COMMAND on a Vast.ai GPU instance (4 x B300 planned): the Top-k evaluation (k = 1, 3, 5) of the five models
# (base, + SFT, + OPSD, + OPSD -> GRPO, All merged; thinking off) and All merged + mini-SWE-agent (thinking on) on every
# dataset: every source class with a partner in the reference (no held-out view), and the complete SNOMED CT -> ICD-11
# map of ICD's scope; within FULL_EVAL_BUDGET_USD (default $425, setup included; expected ~$363, ~15 h at $24/h).
# FULL_EVAL_SCOPE=all aligns every class (needed only for OAEI scores). Run it inside tmux.
#
#   full_eval/vast_full_eval.sh            setup, inputs, plan (asks once), run, report, upload; resumable
#   full_eval/vast_full_eval.sh --plan     only show the plan and what the money buys
#   FULL_EVAL_RERUN=base,sdft,grpo full_eval/vast_full_eval.sh   run only these methods again; the other methods'
#                                          files are copied from <you>/oa-rlaif-full-eval-results (read only) and
#                                          everything goes to <you>/oa-rlaif-fixed-full-eval-results
#
# Steps: 1 setup (price, packages, Python env, agent sandbox)  2 Hugging Face login  3 inputs from the private repos
#        4 the run (models rebuilt, episodes built, GPUs, scoring, report)  5 upload of every output to the
#        private dataset <you>/oa-rlaif-full-eval-results. Logs: outputs/full_eval/logs/. Run the same command again after any
#        stop: finished work is kept.
set -euo pipefail
cd "$(dirname "$0")/.."
export FULL_EVAL_BUDGET_USD="${FULL_EVAL_BUDGET_USD:-425}"
mkdir -p outputs/full_eval/logs
ngpu="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)"
gpu_gb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | awk '{ print int($1 / 1024) }' || true)"
if [ "${ngpu:-0}" -lt 1 ] || [ "${gpu_gb:-0}" -lt 40 ]; then
  echo "This machine shows ${ngpu:-0} GPU(s) with ${gpu_gb:-0} GB each; the evaluation needs GPUs with >= 40 GB"
  echo "(planned: 4 x B300). Nothing was installed."
  exit 1
fi
echo "[full-eval] ${ngpu} GPU(s), ${gpu_gb} GB each; cap \$${FULL_EVAL_BUDGET_USD} (machine time from its start, setup included)"
[ -f outputs/full_eval/.setup.done ] || { full_eval/setup.sh 2>&1 | tee -a outputs/full_eval/logs/setup.log; touch outputs/full_eval/.setup.done; }
set -a; . ./.env; set +a
export PYTHONPATH="$PWD/src:$PWD" TOKENIZERS_PARALLELISM=false LITELLM_LOCAL_MODEL_COST_MAP=True
if ! .venv/bin/hf auth whoami >/dev/null 2>&1; then
  echo; echo "Hugging Face login needed (the account with the private repos; a token with write access). Paste it:"
  .venv/bin/hf auth login
fi
[ -f outputs/full_eval/.inputs.done ] || { .venv/bin/python -m full_eval.pull_inputs 2>&1 | tee -a outputs/full_eval/logs/inputs.log; touch outputs/full_eval/.inputs.done; }
set -a; . ./.env; set +a
.venv/bin/python -m full_eval.run "$@"
[ "${1:-}" = "--plan" ] && exit 0
.venv/bin/python -m full_eval.push_results 2>&1 | tee -a outputs/full_eval/logs/push_results.log
NS="$(.venv/bin/python -c "from oa_rlaif.data.hf_push import namespace; print(namespace())")"
REPO="$(.venv/bin/python -c "from full_eval.push_results import target_repo; print(target_repo())")"
echo
echo "DONE. Report: outputs/full_eval/report.md; everything: https://huggingface.co/datasets/$NS/$REPO (private)"
echo "NOW DESTROY THE VAST.AI INSTANCE in the console (a stopped instance still bills for its disk)."

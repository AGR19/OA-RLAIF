#!/usr/bin/env bash
# ONE COMMAND on a Vast.ai GPU instance (4 x B200 planned): pplx-decider-v1.2 (a decision model) controls the
# All-merged Gemma 4 E4B agent through JevAgentHarness, against the All-merged rows of oa-rlaif-full-eval-results.
#   arm A  pplx-decider alone (every GPU)
#   arm B  JevAgentHarness + pplx-decider + All merged + mini-SWE-agent (route thinking on/off at p(hard) >= 0.75,
#          the NONE/invalid guard at 0.5; thresholds fixed before the run)
# Every class with a partner in the reference and the complete SNOMED CT -> ICD-11 map (299,961 per arm; no held-out
# view), Top-k evaluation (k = 1, 3, 5), within DM_BUDGET_USD (default $250, setup included). Run it inside tmux.
#
#   dm_harness/vast_dm_harness.sh           setup, inputs, plan (asks once), both arms, report, upload; resumable
#   dm_harness/vast_dm_harness.sh --plan    only show the plan and what the money buys
#   dm_harness/vast_dm_harness.sh --yes     no question
#   DM_ARMS=decider ...                     one arm only (decider | jah-agent)
#   FULL_EVAL_TASKS=icd__snomed-to-icd11,cyber__iotsec-uco ...   only these tasks
#
# Results: the private dataset <you>/oa-rlaif-dm-harness-results (oa-rlaif-full-eval-results is only read).
set -euo pipefail
cd "$(dirname "$0")/.."
export DM_BUDGET_USD="${DM_BUDGET_USD:-250}"
export FULL_EVAL_BUDGET_USD="$DM_BUDGET_USD" FULL_EVAL_SCOPE=reference   # the price check shows this cap
mkdir -p outputs/full_eval/logs
ngpu="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)"
gpu_gb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | awk '{ print int($1 / 1024) }' || true)"
if [ "${ngpu:-0}" -lt 2 ] || [ "${gpu_gb:-0}" -lt 80 ]; then
  echo "This machine shows ${ngpu:-0} GPU(s) with ${gpu_gb:-0} GB each; the decision model (52 GB of weights plus"
  echo "working memory) and the agent need at least 2 GPUs with >= 80 GB (planned: 4 x B200). Nothing was installed."
  exit 1
fi
echo "[dm] ${ngpu} GPU(s), ${gpu_gb} GB each; cap \$${DM_BUDGET_USD} (machine time from its start, setup included)"
[ -f outputs/full_eval/.setup.done ] || { full_eval/setup.sh 2>&1 | tee -a outputs/full_eval/logs/setup.log; touch outputs/full_eval/.setup.done; }
set -a; . ./.env; set +a
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH="$PWD/src:$PWD" TOKENIZERS_PARALLELISM=false LITELLM_LOCAL_MODEL_COST_MAP=True
# the JevAgentHarness proxy (vendored, MIT) next to the evaluation's packages (aiohttp and pyyaml are already there)
.venv/bin/python -c "import jevagentharness" 2>/dev/null || uv pip install --python .venv/bin/python --no-deps -e third_party/JevAgentHarness
if ! .venv/bin/hf auth whoami >/dev/null 2>&1; then
  echo; echo "Hugging Face login needed (the account with the private repos; a token with write access). Paste it:"
  .venv/bin/hf auth login
fi
[ -f outputs/full_eval/.inputs.done ] || { .venv/bin/python -m full_eval.pull_inputs 2>&1 | tee -a outputs/full_eval/logs/inputs.log; touch outputs/full_eval/.inputs.done; }
[ -f outputs/full_eval/.decider.done ] || { dm_harness/setup_decider.sh 2>&1 | tee -a outputs/full_eval/logs/decider_setup.log; touch outputs/full_eval/.decider.done; }
[ -f outputs/full_eval/.reference.done ] || { .venv/bin/python -m dm_harness.pull_reference 2>&1 | tee -a outputs/full_eval/logs/reference.log; touch outputs/full_eval/.reference.done; }
.venv/bin/python -m dm_harness.run "$@"
[ "${1:-}" = "--plan" ] && exit 0
.venv/bin/python -m dm_harness.push_results 2>&1 | tee -a outputs/full_eval/logs/push_results.log
NS="$(.venv/bin/python -c "from oa_rlaif.data.hf_push import namespace; print(namespace())")"
echo
echo "DONE. Report: outputs/full_eval/dm_report.md; everything: https://huggingface.co/datasets/$NS/oa-rlaif-dm-harness-results (private)"
echo "NOW DESTROY THE VAST.AI INSTANCE in the console (a stopped instance still bills for its disk)."

#!/usr/bin/env bash
# The one-command run, shared by scripts/lambda_all.sh and scripts/coreweave_all.sh:
#
#   scripts/run_all.sh lambda|coreweave|vast
#
# It asks for the Hugging Face token once (after installing), then runs every step without further input and shows a
# live dashboard: total progress bar, the current step with its own bar, elapsed time, ETA and GPU spend so far.
# Full output of each step: outputs/logs/stage_<step>.log
#
#   1 setup       packages, Gemma 4 E4B + 26B-A4B judge weights, agent sandbox (Docker or user), llama.cpp
#   2 data        private HF datasets -> data/
#   3 check_env   GPUs, shared memory, libraries, data, agent sandbox (stops here if anything is missing)
#   4 gguf_check  converts the BASE model to GGUF first, so export problems show before training
#   5 dry_run     times OPSD, SFT and GRPO (with the judge), checks the judge, plans steps that fit the budget
#   6 baseline    base E4B: held-out sets + mini-SWE-agent episodes
#   7 sdft        Stage 1 OPSD (teacher sees the gold answer), merged -> outputs/merged/sdft
#   8 eval_sdft   OPSD model: held-out sets + agent episodes
#   9 grpo        Stage 2 GRPO: DRAL-OA reward + local A4B judge (RLAIF), merged -> outputs/merged/grpo
#  10 eval_grpo   GRPO model: held-out, zero-shot, ICD-11<->SNOMED, full Anatomy alignment (OAEI row), agent
#  11 export      GGUF Q4_K_M + Ollama Modelfile (base + OPSD + GRPO in one file)
#  12 push        private HF: oa-rlaif-e4b-gguf, oa-rlaif-e4b-sdft, oa-rlaif-e4b-grpo (the GGUF is safe from here)
#  13 sft         compulsory SFT comparison: same rows and steps as OPSD, merged -> outputs/merged/sft
#  14 eval_sft    SFT model: held-out sets + agent episodes
#  15 report      results tables: base / SFT / OPSD / GRPO, ICD-11<->SNOMED, agents, OAEI row
#  16 results     private HF dataset oa-rlaif-results (tables, metrics, ICD-11<->SNOMED SSSOM, alignments)
#
# Resumable: finished steps are skipped on the next run (outputs/stages/<step>.done). After a failure the dashboard
# shows the end of that step's log; fix the cause and run the same command again.
# Budget (src/oa_rlaif/cost.py): the price of THIS machine (provider API, else asked and checked against its GPUs)
# times the machine's billed time since it started; every billable step stops at $450 of $500. The training stages
# get shares of the money left (OPSD 40%, GRPO 40%, SFT 20%), not time slots.
set -euo pipefail
cd "$(dirname "$0")/.."
PLATFORM="${1:?usage: scripts/run_all.sh lambda|coreweave|vast}"
case "$PLATFORM" in
  lambda)    SETUP="scripts/cloud_setup.sh lambda"; WHERE="Lambda"
             RELEASE="TERMINATE THE INSTANCE in the Lambda console. Billing continues until you do." ;;
  coreweave) SETUP="scripts/cloud_setup.sh coreweave"; WHERE="CoreWeave"
             RELEASE="RELEASE THE COREWEAVE NODE: delete the pod or instance, or scale its node pool to 0. Billing continues until you do." ;;
  vast)      SETUP="scripts/cloud_setup.sh vast"; WHERE="Vast.ai"; export OA_SANDBOX_FALLBACK="${OA_SANDBOX_FALLBACK:-user}"
             RELEASE="DESTROY THE VAST.AI INSTANCE in the console (a stopped instance still bills for its disk)." ;;
  *) echo "unknown platform '$PLATFORM' (lambda, coreweave or vast)"; exit 1 ;;
esac
export OA_PLATFORM="$WHERE"
mkdir -p outputs/stages outputs/logs
export OA_RAW_DIR="${OA_RAW_DIR:-data/raw}"
# GPU check first, before anything is installed or downloaded (a wrong instance fails in seconds, not after 70 GB)
ngpu="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)"
gpu_gb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | awk '{ print int($1 / 1024) }' || true)"
if [ "${ngpu:-0}" -lt "${OA_MIN_GPUS:-4}" ] || [ "${gpu_gb:-0}" -lt "${OA_MIN_GPU_GB:-75}" ]; then
  echo "This machine shows ${ngpu:-0} GPU(s) with ${gpu_gb:-0} GB each (nvidia-smi -L):"
  nvidia-smi -L 2>/dev/null || echo "  (nvidia-smi not found)"
  echo "The pipeline needs at least ${OA_MIN_GPUS:-4} GPUs (3 train + 1 for vLLM and the A4B judge) with >= 80 GB each"
  echo "(B200, H200, H100 80GB, A100 80GB): the A4B judge alone is about 52 GB. Rent such an instance."
  echo "Nothing was installed. (OA_MIN_GPUS=2 allows a smaller test run.)"
  exit 1
fi
envval() { grep -E "^$1=" .env 2>/dev/null | tail -n 1 | cut -d= -f2- | tr -d '"' || true; }

# python3 runs the price calculator and the dashboard before anything else is installed
if ! command -v python3 >/dev/null; then
  SUDO=""; [ "$(id -u)" -eq 0 ] || SUDO="sudo"
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 update -qq
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y -qq python3 >/dev/null
fi
scripts/price.sh "$PLATFORM"          # the price of THIS machine (provider API, else asked and checked); what it buys
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
    echo; echo "Fix the cause, then run the same command again (finished steps are skipped)."
    exit 1
  fi
}

stage setup $SETUP
# the HF login reads this from .env (written by the setup script)
HF_HOME_ENV="$(envval HF_HOME)"; [ -n "$HF_HOME_ENV" ] && export HF_HOME="$HF_HOME_ENV"
# A restart can stop the Docker daemon: make sure the sandbox works (seconds when it does)
scripts/sandbox_setup.sh --start

# Hugging Face login, once (the token stays in your terminal and in the HF cache on this machine).
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
stage grpo       scripts/60_grpo.sh
stage eval_grpo  scripts/50_eval.sh grpo --full
stage export     scripts/70_export.sh
stage push       env PYTHONPATH=src .venv/bin/python -m oa_rlaif.data.hf_push --models
# the GGUF is now safe on Hugging Face; the compulsory SFT comparison row follows
stage sft        scripts/40_sdft.sh --mode sft
stage eval_sft   scripts/50_eval.sh sft
stage report     bash -c 'A=data/raw/anatomy-dataset-2; PYTHONPATH=src .venv/bin/python -m oa_rlaif.eval.align --backend lexical \
                   --model lexical --source $A/mouse.owl --target $A/human.owl --reference $A/reference.rdf --format tsv \
                   --no-coherence --out outputs/align/anatomy_lexical_baseline && PYTHONPATH=src .venv/bin/python -m oa_rlaif.eval.report'
stage results    env PYTHONPATH=src .venv/bin/python -m oa_rlaif.data.hf_push --results

echo
echo "DONE on $WHERE. Your GGUF: https://huggingface.co/$NS/oa-rlaif-e4b-gguf (private)"
echo "Also: $NS/oa-rlaif-e4b-sdft and $NS/oa-rlaif-e4b-grpo (LoRA adapters), dataset $NS/oa-rlaif-results."
echo "Results table: outputs/eval/report.md (also in $NS/oa-rlaif-results)"; echo; cat outputs/eval/report.md
echo "NOW $RELEASE"

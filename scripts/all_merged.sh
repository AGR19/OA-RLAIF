#!/usr/bin/env bash
# Research variant, separate from the one-command run: "all merged" = base E4B + OPSD + GRPO (RLAIF) + SFT.
# The GRPO model (outputs/merged/grpo) already holds base + OPSD + GRPO. The SFT LoRA adapter, trained on the plain
# base, is added on top: its weight change is summed in (task arithmetic). The released GGUF does not change.
#
#   scripts/all_merged.sh             merge (once), then the same evaluations as for GRPO: held-out, zero-shot,
#                                     closed-book, ICD, mini-SWE-agent runs, whole-Anatomy alignment (OAEI row);
#                                     then its own comparison tables (outputs/all_merged_report/report.md: base, SFT,
#                                     OPSD, GRPO and All merged). The main report (outputs/eval/report.md) is unchanged.
#   scripts/all_merged.sh --no-full   without the whole-Anatomy alignment (saves about 25 minutes)
#   scripts/all_merged.sh --push      also upload everything all-merged to its OWN private repo
#                                     <you>/oa-rlaif-e4b-all-combined-model-safetensors (weights, evaluation, comparison report). Nothing
#                                     goes into the existing repos (oa-rlaif-heldout-eval-results, the GGUF, the adapters).
#
# Run it after the one-command run has printed DONE: it needs steps 9 (grpo) and 13 (sft), and it uses the same
# GPUs and port as the run's evaluations (its start-up clears servers left in this folder, so it must not overlap).
# About 1-1.5 h of machine time; the budget guard still stops at $450.
. "$(dirname "$0")/_common.sh"
PUSH=0; FULL=--full
for arg in "$@"; do
  case "$arg" in --push) PUSH=1 ;; --no-full) FULL="" ;; *) echo "unknown option: $arg"; exit 1 ;; esac
done
if pgrep -f "scripts/run_all.sh" >/dev/null; then
  log "the one-command run is still running; run this after it prints DONE (both need the GPUs and port 8000)"
  exit 1
fi
for done_marker in outputs/stages/grpo.done outputs/stages/sft.done; do
  [ -f "$done_marker" ] || { log "missing $done_marker: steps 9 (grpo) and 13 (sft) must have finished"; exit 1; }
done
budget_gate all_merged

if [ ! -f outputs/merged/all_merged/config.json ]; then
  # the SFT adapter must change something, or "all merged" would just be the GRPO model under another name
  $PY - <<'PY'
from safetensors.torch import load_file
w = load_file("outputs/checkpoints/sft/final/adapter_model.safetensors")
moved = [k for k, v in w.items() if "lora_B" in k and v.abs().sum() > 0]
if not moved:
    raise SystemExit("the SFT adapter has no trained weights (every lora_B is zero): all merged would equal GRPO")
print(f"SFT adapter: {len(moved)} trained LoRA tensors; all merged will differ from the GRPO model")
PY
  log "merging the SFT adapter onto the GRPO model (base + OPSD + GRPO) -> outputs/merged/all_merged"
  rm -rf outputs/merged/all_merged.partial
  $PY -m oa_rlaif.export.release merge --adapter outputs/checkpoints/sft/final --base outputs/merged/grpo \
    --out outputs/merged/all_merged.partial
  mv outputs/merged/all_merged.partial outputs/merged/all_merged   # only a finished merge gets the final name
else
  log "outputs/merged/all_merged exists; evaluating it"
fi

scripts/50_eval.sh all_merged $FULL
$PY -m oa_rlaif.eval.report --all-merged

if [ "$PUSH" = 1 ]; then
  $PY -m oa_rlaif.data.push_all_merged   # its own repo only; the existing results repo is not touched
fi
log "all merged: done. Model: outputs/merged/all_merged; comparison tables: outputs/all_merged_report/report.md"

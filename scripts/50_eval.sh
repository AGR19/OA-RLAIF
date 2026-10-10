#!/usr/bin/env bash
# LAMBDA: evaluate a merged model:  scripts/50_eval.sh sdft   |   scripts/50_eval.sh grpo --full
. "$(dirname "$0")/_common.sh"
TAG="$1"; shift || true
budget_gate "eval_$TAG"
$PY -m oa_rlaif.eval.run_eval --model "outputs/merged/$TAG" --tag "$TAG" --tp 4 2>&1 | tee "outputs/logs/eval_$TAG.log"
if [ "${1:-}" = "--full" ]; then
  RAW="${OA_RAW_DIR:-data/raw}"
  $PY -m oa_rlaif.eval.full_align --model "outputs/merged/$TAG" --tp 4 --name anatomy \
     --source "$RAW/anatomy-dataset-2/mouse.owl" --target "$RAW/anatomy-dataset-2/human.owl" \
     --reference "$RAW/anatomy-dataset-2/reference.rdf"
fi

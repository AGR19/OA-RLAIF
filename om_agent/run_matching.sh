#!/usr/bin/env bash
# Align mouse.owl -> human.owl with the OA-RLAIF all-merged model as a mini-SWE-agent, EVERY mouse class, locally.
# Everything stays in this folder: the model weights (model/), Ollama's model store (ollama-models/), the outputs
# (output/), logs (logs/) and caches (cache/). Ollama runs on its own port (11435), so no other Ollama is touched.
#
#   ./run_matching.sh                every class, scored against ontologies/reference.rdf
#   ./run_matching.sh --limit 20     the first 20 classes only (a quick test)
#   ./run_matching.sh --workers 1    one episode at a time (less memory)
# Stopping and running again continues where it stopped (output/decisions.jsonl).
set -euo pipefail
. "$(cd "$(dirname "$0")" && pwd)/setup_ollama.sh"   # this folder's Ollama + the model

# 3. the matcher: every mouse class, one mini-SWE-agent episode each
O="$HERE/ontologies"
# Reference alignment (optional): scoring only, Top-k evaluation (k = 1, 3, 5) (results.md, topk.svg);
# the agent never sees it.
# OA_REFERENCE=<file> sets it; otherwise it is asked for here (Enter = skip). Thinking on (--no-think: off).
REF="${OA_REFERENCE:-}"
if [ -z "$REF" ] && [ -t 0 ]; then
  read -r -p "Reference alignment (optional) [Enter = skip, e.g. $O/reference.rdf]: " REF
fi
REF_ARG=(); [ -n "$REF" ] && REF_ARG=("file://${REF/#\~/$HOME}")
"$RLAIF/.venv/bin/python" "$HERE/test_matching_ai.py" "file://$O/mouse.owl" "file://$O/human.owl" "${REF_ARG[@]}" \
  --out "$HERE/output/mouse-human" "$@" 2>&1 | tee -a "$HERE/logs/matching.log"
echo "results: $HERE/output/mouse-human (alignment.rdf, decisions.tsv, decisions.jsonl, summary.json, tasks/)"

#!/usr/bin/env bash
# LAMBDA: GGUF (Q4_K_M) + Ollama Modelfile from the merged GRPO model; mini-SWE-agent smoke run on Ollama.
. "$(dirname "$0")/_common.sh"
LLAMA_CPP_DIR=third_party/llama.cpp $PY -m oa_rlaif.export.release gguf --merged outputs/merged/grpo --out outputs/release/oa-rlaif-e4b
if command -v ollama >/dev/null; then
  (cd outputs/release/oa-rlaif-e4b && ollama create oa-rlaif-e4b -f Modelfile)
  $PY -m oa_rlaif.eval.agent_eval --episodes data/eval/heldout_anatomy.jsonl --n 20 --tag ollama_smoke \
     --model ollama/oa-rlaif-e4b --api-base http://127.0.0.1:11434 --env "${OA_AGENT_ENV:-docker}"
fi

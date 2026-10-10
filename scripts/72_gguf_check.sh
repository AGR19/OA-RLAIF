#!/usr/bin/env bash
# LAMBDA (before training): prove the GGUF export works for Gemma 4 E4B by converting the BASE model.
# Fails early (minutes, CPU) instead of after 12 h of training. Output is deleted afterwards.
. "$(dirname "$0")/_common.sh"
BASE="$($PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('google/gemma-4-E4B-it', local_files_only=True))")"
rm -rf outputs/gguf_check
LLAMA_CPP_DIR=third_party/llama.cpp $PY -m oa_rlaif.export.release gguf --merged "$BASE" --out outputs/gguf_check
ls -la outputs/gguf_check
[ -s outputs/gguf_check/oa-rlaif-e4b-Q4_K_M.gguf ] || { echo "GGUF check FAILED"; exit 1; }
echo "GGUF check passed ($(du -h outputs/gguf_check/oa-rlaif-e4b-Q4_K_M.gguf | cut -f1))"
rm -rf outputs/gguf_check

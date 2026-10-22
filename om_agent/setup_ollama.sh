#!/usr/bin/env bash
# Sourced by oa_agent.sh and run_matching.sh: this folder's Ollama server (127.0.0.1:11435, model store in
# ollama-models/) and the all-merged model, imported once from model/all_merged.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The OA-RLAIF repo: this folder's parent when it is om_agent/ in the repo, else OA_RLAIF_DIR, else the Mac copy.
if [ -z "${OA_RLAIF_DIR:-}" ]; then
  if [ -d "$HERE/../src/oa_rlaif" ]; then OA_RLAIF_DIR="$(cd "$HERE/.." && pwd)"; else OA_RLAIF_DIR="$HOME/Desktop/oa-sft-sm-1/RLAIF"; fi
fi
export OA_RLAIF_DIR
RLAIF="$OA_RLAIF_DIR"
export OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS="$HERE/ollama-models" OLLAMA_NUM_PARALLEL=2 \
       OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_KEEP_ALIVE=30m OLLAMA_API_BASE=http://127.0.0.1:11435/v1
mkdir -p "$HERE/logs" "$HERE/output" "$HERE/cache" "$OLLAMA_MODELS"
if ! curl -sf http://127.0.0.1:11435/api/version >/dev/null; then
  echo "starting Ollama on 127.0.0.1:11435 (models in $OLLAMA_MODELS)"
  nohup ollama serve > "$HERE/logs/ollama_server.log" 2>&1 &
  echo $! > "$HERE/logs/ollama_server.pid"
  for _ in $(seq 1 30); do curl -sf http://127.0.0.1:11435/api/version >/dev/null && break; sleep 1; done
fi
if ! ollama list | grep -q "^oa-rlaif-all-merged"; then
  [ -f "$HERE/model/all_merged/model.safetensors" ] || { echo "missing model/all_merged/model.safetensors"; exit 1; }
  [ -f "$HERE/cache/import/kv_shared.safetensors" ] || "$RLAIF/.venv/bin/python" "$HERE/fix_kv_shared.py"
  echo "importing the all-merged model into Ollama (once, a few minutes; ${OA_QUANTIZE:-int8})"
  (cd "$HERE" && ollama create oa-rlaif-all-merged --quantize "${OA_QUANTIZE:-int8}" -f Modelfile) 2>&1 | tail -3
fi

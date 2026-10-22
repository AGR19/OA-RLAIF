#!/usr/bin/env bash
# GGUF conversion on Hugging Face Jobs (not on this machine): the job downloads a safetensors model repo, converts it
# with llama.cpp b11206 (the first prebuilt image after b11205, the release the first GGUF used) to f16, quantizes it
# to Q4_K_M and uploads the GGUF and an Ollama Modelfile to the target repo. Runs inside the job container; launch it from the repo root with:
#
#   hf jobs run --flavor cpu-upgrade --timeout 3h --secrets HF_TOKEN --detach --name gguf-all-combined \
#     -e SRC=gonibeed/oa-rlaif-e4b-all-combined-model-safetensors \
#     -e DST=gonibeed/oa-rlaif-e4b-all-combined-model-gguf -e NAME=oa-rlaif-e4b-all-combined-Q4_K_M.gguf \
#     ghcr.io/ggml-org/llama.cpp:full-b11206 bash -c "$(cat scripts/hf_gguf_job.sh)"
#
# HF_TOKEN comes from the user's `hf auth login` (the token needs the "Jobs" permission); it is never printed.
# cpu-upgrade: 8 vCPU, 32 GB RAM, 50 GB disk (model 16 GB + f16 GGUF ~16 GB at the peak), $0.03/h.
set -euo pipefail
: "${SRC:?}" "${DST:?}" "${NAME:?}"
CONVERT=/app/convert_hf_to_gguf.py                    # the llama.cpp "full" image keeps its tools in /app
QUANTIZE="$(command -v llama-quantize || ls /app/llama-quantize 2>/dev/null || true)"
echo "[gguf] convert: $CONVERT; quantize: ${QUANTIZE:-missing}"
[ -f "$CONVERT" ] && [ -n "$QUANTIZE" ] || { ls /app; echo "[gguf] llama.cpp tools not found"; exit 1; }
python3 -c "import huggingface_hub" 2>/dev/null || python3 -m pip install -q huggingface_hub
python3 - <<PY
from huggingface_hub import snapshot_download
import json
from pathlib import Path
# the model files only (the repo also holds eval/ and report/)
snapshot_download("$SRC", local_dir="/tmp/model", allow_patterns=[
    "config.json", "generation_config.json", "processor_config.json", "preprocessor_config.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "chat_template.jinja",
    "model.safetensors", "model-*.safetensors", "model.safetensors.index.json"])
# transformers 5 (which saved the model) writes extra_special_tokens as a list; the image's transformers 4 reads a
# dict there and fails. In this working copy only, the list goes to additional_special_tokens, the transformers 4
# field for the same tokens; the vocabulary does not change.
path = Path("/tmp/model/tokenizer_config.json")
cfg = json.loads(path.read_text())
extra = cfg.get("extra_special_tokens")
if isinstance(extra, list):
    cfg["additional_special_tokens"] = list(dict.fromkeys((cfg.get("additional_special_tokens") or []) + extra))
    del cfg["extra_special_tokens"]
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    print(f"[gguf] tokenizer_config: {len(extra)} extra special tokens moved to additional_special_tokens")
PY
df -h /tmp | tail -n 1
python3 "$CONVERT" /tmp/model --outfile /tmp/f16.gguf --outtype f16
rm -rf /tmp/model
"$QUANTIZE" /tmp/f16.gguf "/tmp/$NAME" Q4_K_M
rm -f /tmp/f16.gguf
sha256sum "/tmp/$NAME"
printf 'FROM ./%s\nPARAMETER temperature 0\nPARAMETER num_ctx 16384\nPARAMETER stop "<turn|>"\nSYSTEM "You are OA-RLAIF, an ontology alignment agent."\n' "$NAME" > /tmp/Modelfile
python3 - <<PY
from huggingface_hub import HfApi
api = HfApi()
assert api.repo_info("$DST").private, "target repo is not private"
for local, remote in (("/tmp/$NAME", "$NAME"), ("/tmp/Modelfile", "Modelfile")):
    api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id="$DST",
                    commit_message="GGUF Q4_K_M from $SRC (llama.cpp b11206, Hugging Face Jobs)")
print("[gguf] uploaded to $DST")
PY

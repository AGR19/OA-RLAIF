#!/usr/bin/env bash
# GGUF conversion on Hugging Face Jobs (not on this machine): the job downloads a safetensors model repo, converts it
# with llama.cpp (the release the first GGUF used, b11205) to f16, quantizes it to Q4_K_M and uploads the GGUF and an
# Ollama Modelfile to the target repo. Runs inside the job container; launch it from the repo root with:
#
#   hf jobs run --flavor cpu-upgrade --timeout 3h --secrets HF_TOKEN --detach --name gguf-all-combined \
#     -e SRC=gonibeed/oa-rlaif-e4b-all-combined-model-safetensors \
#     -e DST=gonibeed/oa-rlaif-e4b-all-combined-model-gguf -e NAME=oa-rlaif-e4b-all-combined-Q4_K_M.gguf \
#     ghcr.io/ggml-org/llama.cpp:full-b11205 bash -c "$(cat scripts/hf_gguf_job.sh)"
#
# HF_TOKEN comes from the user's `hf auth login` (the token needs the "Jobs" permission); it is never printed.
# cpu-upgrade: 8 vCPU, 32 GB RAM, 50 GB disk (model 16 GB + f16 GGUF ~16 GB at the peak), $0.03/h.
set -euo pipefail
: "${SRC:?}" "${DST:?}" "${NAME:?}"
CONVERT="$(find / -name convert_hf_to_gguf.py -not -path '/proc/*' 2>/dev/null | head -n 1)"
QUANTIZE="$(command -v llama-quantize || find / -name llama-quantize -type f -not -path '/proc/*' 2>/dev/null | head -n 1)"
echo "[gguf] convert: $CONVERT; quantize: $QUANTIZE"
python3 -c "import huggingface_hub" 2>/dev/null || python3 -m pip install -q huggingface_hub
python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$SRC", local_dir="/tmp/model", allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer*"])
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
                    commit_message="GGUF Q4_K_M from $SRC (llama.cpp b11205, Hugging Face Jobs)")
print("[gguf] uploaded to $DST")
PY

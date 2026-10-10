#!/usr/bin/env bash
# One-time install on the GPU box. Downloads happen only here.
# Lambda (4xB200, Ubuntu, Docker preinstalled) runs it directly; CoreWeave runs it from scripts/coreweave_setup.sh
# after the price, system packages and Docker are in place (the price lines below then keep CoreWeave's value).
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv .venv --python 3.11
uv pip sync --python .venv/bin/python requirements-train.txt
[ -f .env ] || cp .env.example .env
grep -q '^OA_BILLABLE=' .env || echo 'OA_BILLABLE=1' >> .env          # GPU time counts against the budget here
grep -q '^OA_GPU_USD_PER_HOUR=' .env || echo 'OA_GPU_USD_PER_HOUR=27.16' >> .env
# Gemma 4 E4B weights (one-time; afterwards HF_HUB_OFFLINE=1 is set by the trainers)
OA_ALLOW_HUB_DOWNLOAD=1 .venv/bin/hf download google/gemma-4-E4B-it
# Gemma 4 26B-A4B (51.6 GB, not gated): the local GRPO judge served by vLLM on GPU 3 (scripts/60_grpo.sh)
OA_ALLOW_HUB_DOWNLOAD=1 .venv/bin/hf download "${JUDGE_LOCAL_MODEL:-google/gemma-4-26B-A4B-it}"
# Sandbox image for mini-SWE-agent (one-time base image pull; runs with --network none)
docker build --network none -t oa-rlaif-om:latest -f docker/Dockerfile.om .   # no step needs a network
# llama.cpp for GGUF export (pinned tag)
[ -d third_party/llama.cpp ] || git clone --depth 1 --branch "${LLAMA_CPP_TAG:-b11205}" https://github.com/ggml-org/llama.cpp third_party/llama.cpp
echo "setup done; next: scripts/00_check_env.sh"
# build llama-quantize (CPU is enough for quantisation)
[ -x third_party/llama.cpp/build/bin/llama-quantize ] || (cd third_party/llama.cpp && cmake -B build -DGGML_CUDA=OFF && cmake --build build -j --target llama-quantize)

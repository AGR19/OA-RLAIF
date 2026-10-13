#!/usr/bin/env bash
# Push everything of the SFT comparison arm to private Hugging Face repos. Separate from the one-command run.
#   scripts/push_sft.sh             after step 13 (sft); run it again after step 14 (eval_sft) to add the evaluation
#   scripts/push_sft.sh --dry-run   list what would be uploaded
# Repos: <you>/oa-rlaif-e4b-sft (adapter + eval + logs) and <you>/oa-rlaif-e4b-sft-merged (merged model, ~16 GB).
# Uses the Hugging Face login of this machine (the one-command run already did `hf auth login`).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src .venv/bin/python -m oa_rlaif.data.push_sft "$@"

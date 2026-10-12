#!/usr/bin/env bash
# The hourly price of THIS machine, recorded once in outputs/cost_ledger.json (see src/oa_rlaif/cost.py):
# read from the provider's API (Vast.ai; Lambda with LAMBDA_API_KEY in .env), else asked and checked against the GPUs.
#   scripts/price.sh lambda|coreweave|vast
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python3 -m oa_rlaif.cost price --platform "${1:-lambda}"

#!/usr/bin/env bash
# MAC -> LAMBDA: copy code + built data (no venv, no raw ontologies, no fast.bin needed on the GPU box).
#   scripts/sync_to_lambda.sh ubuntu@<lambda-ip> [~/OA-RLAIF]
set -euo pipefail
HOST="$1"; DEST="${2:-OA-RLAIF}"
cd "$(dirname "$0")/.."
rsync -avz --progress --exclude .venv --exclude outputs --exclude data/profiles --exclude '__pycache__' \
      --exclude '.env' ./ "$HOST:$DEST/"
echo "now on Lambda: cd $DEST && scripts/lambda_setup.sh && scripts/00_check_env.sh"

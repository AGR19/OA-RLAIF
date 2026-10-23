#!/usr/bin/env bash
# Once on the GPU box (dm_harness/vast_dm_harness.sh runs it): the decision model's weights and its own Python
# environment. perplexity-ai/pplx-decider-v1.2 (Apache-2.0, public, 52 GB BF16) goes to outputs/dm_models/; it needs
# Python 3.12 and its own pinned torch / transformers (model card: requirements.txt, and requirements-fast.txt for the
# fast path), so it gets .venv-decider next to the evaluation's .venv. pplx_decider is installed from the snapshot's
# source/. The decision model cannot run on vLLM (noncausal attention, a readout head); dm_harness.decider_server
# serves it behind POST /v1/systemone.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$PWD/.cache/uv" UV_LINK_MODE=copy
DIR="${DM_DECIDER_DIR:-outputs/dm_models/pplx-decider-v1.2}"
echo "[decider] weights -> $DIR"
.venv/bin/python - "$DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("perplexity-ai/pplx-decider-v1.2", local_dir=sys.argv[1], max_workers=16)
PY
[ -x .venv-decider/bin/python ] || uv venv .venv-decider --python 3.12
uv pip install --python .venv-decider/bin/python -r "$DIR/requirements.txt" -r "$DIR/requirements-fast.txt" \
  fastapi==0.136.3 uvicorn==0.54.0
uv pip install --python .venv-decider/bin/python --no-deps "$DIR/source"
.venv-decider/bin/python -c "import pplx_decider, fastapi, uvicorn; print('[decider] environment ok')"

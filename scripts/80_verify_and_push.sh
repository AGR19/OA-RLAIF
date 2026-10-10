#!/usr/bin/env bash
# Gates, then (only with flags, after the user's OK) push to PRIVATE repos.
#   --hf         datasets to private Hugging Face repos (needs `hf auth login` + HF_NAMESPACE)
#   --models     adapters to private Hugging Face model repos (on Lambda)
. "$(dirname "$0")/_common.sh"
log "tests"; $PY -m pytest -q
log "parse + reference verification"; $PY -m oa_rlaif.data.parse_verify
log "verify"; $PY -m oa_rlaif.data.verify
for arg in "$@"; do
  case "$arg" in
    --hf) $PY -m oa_rlaif.data.hf_push --datasets ;;
    --models) $PY -m oa_rlaif.data.hf_push --models ;;
  esac
done

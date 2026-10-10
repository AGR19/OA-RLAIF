#!/usr/bin/env bash
# MAC: knowledge questions from all passages via the A4B API (falls back to heuristic if no key), then re-render.
. "$(dirname "$0")/_common.sh"
MODE=api; [ -z "${JUDGE_API_KEY:-}" ] && { MODE=heuristic; log "JUDGE_API_KEY unset -> heuristic questions"; }
$PY -m oa_rlaif.data.datasource --mode "$MODE" | tee outputs/logs/datasource.log
$PY -m oa_rlaif.data.render > outputs/logs/render.log

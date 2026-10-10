#!/usr/bin/env bash
# MAC: coverage manifest -> full data build (episodes, facts, complex, baselines, passages, ICD) -> render.
. "$(dirname "$0")/_common.sh"
log "coverage"; $PY -m oa_rlaif.data.coverage
log "build ${*:-all}"; $PY -m oa_rlaif.data.build "$@" 2>&1 | tee outputs/logs/build.log
log "render"; $PY -m oa_rlaif.data.render | tee outputs/logs/render.log

#!/usr/bin/env bash
# Whole-pipeline progress bar, ETA and budget meter. --live: refreshing view of running stages and the Mac queue.
. "$(dirname "$0")/_common.sh"
if [ "${1:-}" = "--live" ]; then $PY -m oa_rlaif.progress live; else $PY -m oa_rlaif.progress status; fi

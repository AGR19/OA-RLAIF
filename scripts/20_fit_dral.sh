#!/usr/bin/env bash
# MAC: fit the DRAL-OA mapping layer per family group (train-split references only) -> data/dral/*.npz
. "$(dirname "$0")/_common.sh"
$PY -m oa_rlaif.reward.fit_all 2>&1 | tee outputs/logs/fit_dral.log

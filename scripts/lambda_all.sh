#!/usr/bin/env bash
# ONE COMMAND on a fresh Lambda 4xB200 instance (Ubuntu, Docker preinstalled). From the cloned repo:
#
#   git clone https://github.com/AGR19/OA-RLAIF.git && cd OA-RLAIF   # private repo: use a GitHub token as password
#   scripts/lambda_all.sh
#
# Runs the 16 stages listed in scripts/run_all.sh with a live dashboard; asks once for the Hugging Face token.
# Resumable: run it again after a failure. WHEN IT SAYS DONE, TERMINATE THE INSTANCE (billing continues).
# CoreWeave: use scripts/coreweave_all.sh instead (installs Docker, asks for the price, any GPU count).
exec "$(dirname "$0")/run_all.sh" lambda "$@"

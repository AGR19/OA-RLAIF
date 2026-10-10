#!/usr/bin/env bash
# ONE COMMAND on CoreWeave: an Ubuntu GPU machine (a CKS pod or DevPod, a Virtual Server, or a bare-metal node).
# Put the repo on the large persistent volume, then:
#
#   git clone https://github.com/AGR19/OA-RLAIF.git && cd OA-RLAIF   # private repo: use a GitHub token as password
#   scripts/coreweave_all.sh
#
# The same 16 stages as on Lambda (scripts/run_all.sh), plus what CoreWeave needs:
#   - asks once for the hourly price of this machine and shows what the $450 stop buys (scripts/price.sh);
#     the dry run then fits the training slots to the money left
#   - installs Docker for the agent sandbox when it is missing (scripts/sandbox_setup.sh); in a pod this needs
#     `privileged: true` or a Docker-in-Docker sidecar, and the script prints both fixes if neither is there
#   - checks /dev/shm and disk space, keeps the HF and uv caches next to the repo (scripts/cloud_setup.sh)
#   - uses every GPU: the last one serves vLLM and the judge, the others train (4 GPUs: train 0-2; 8 GPUs: train 0-6)
# Resumable: run it again after a failure or a pod restart. WHEN IT SAYS DONE, RELEASE THE NODE (billing continues).
exec "$(dirname "$0")/run_all.sh" coreweave "$@"

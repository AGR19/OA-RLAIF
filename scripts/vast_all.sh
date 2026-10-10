#!/usr/bin/env bash
# ONE COMMAND on Vast.ai: a GPU instance (a Docker container on a marketplace host, or a VM instance), Ubuntu image,
# >= 4 GPUs, >= 250 GB disk chosen at creation. From the repo (Vast opens tmux when you SSH in):
#
#   git clone https://github.com/AGR19/OA-RLAIF.git && cd OA-RLAIF   # private repo: use a GitHub token as password
#   scripts/vast_all.sh
#
# The same 16 stages as on Lambda (scripts/run_all.sh), plus what Vast.ai needs:
#   - asks once for the instance's $/hr and shows what the $450 stop buys (scripts/price.sh)
#   - the agent sandbox (scripts/sandbox_setup.sh): a normal Vast.ai instance is itself an unprivileged container
#     and cannot run Docker, so the agent runs as the unprivileged user oa-agent (no gold in its folder, repo closed,
#     network cut when the container allows it). On a VM instance Docker is installed and used instead.
#   - checks /dev/shm and disk space (scripts/cloud_setup.sh); uses every GPU (scripts/_common.sh)
# Resumable: run it again after a failure or a restart. WHEN IT SAYS DONE, DESTROY THE INSTANCE (a stopped instance
# still bills for its disk).
exec "$(dirname "$0")/run_all.sh" vast "$@"

#!/usr/bin/env bash
# Vast.ai GPU box, once (full_eval/vast_full_eval.sh runs it): the price of this machine, system packages,
# /dev/shm and disk checks, the pinned Python environment and the agent's user sandbox (the unprivileged user
# oa-agent: Vast.ai containers cannot run Docker). Unlike scripts/cloud_setup.sh: no Docker, no 26B judge weights and
# no llama.cpp (the full evaluation needs none of them).
set -euo pipefail
cd "$(dirname "$0")/.."
SUDO=""; [ "$(id -u)" -eq 0 ] || SUDO="sudo"
say() { echo "[setup] $*"; }
touch .env
pkgs=()
command -v python3 >/dev/null || pkgs+=(python3)
command -v git >/dev/null || pkgs+=(git)
command -v curl >/dev/null || pkgs+=(curl ca-certificates)
command -v tmux >/dev/null || pkgs+=(tmux)
command -v pkill >/dev/null || pkgs+=(procps)
if [ "${#pkgs[@]}" -gt 0 ]; then
  say "installing: ${pkgs[*]}"
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 update -qq
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y -qq "${pkgs[@]}" >/dev/null
fi
# the price first: from here on every minute of this machine counts against the cap
PYTHONPATH="$PWD/src:$PWD" python3 -m full_eval.price --platform "${OA_PLATFORM_NAME:-vast}"
shm_gb="$(df -Pk /dev/shm | awk 'NR == 2 { print int($2 / 1048576) }')"
if [ "${shm_gb:-0}" -lt 8 ]; then
  say "/dev/shm is ${shm_gb} GB; vLLM needs more. On Vast.ai choose another offer, or add --shm-size=64g to the"
  say "template's Docker options if your host allows it, then create the instance again."
  exit 1
fi
free_gb="$(df -Pk . | awk 'NR == 2 { print int($4 / 1048576) }')"
if [ "$free_gb" -lt 200 ]; then
  say "need about 200 GB free here (five models 80 GB, venv 15 GB, inputs 8 GB, episodes and outputs 30 GB); $free_gb GB free."
  say "On Vast.ai the disk size is fixed at creation: create the instance again with >= 250 GB."
  exit 1
fi
grep -q '^HF_HOME=' .env || echo "HF_HOME=$PWD/.cache/huggingface" >> .env
export HF_HOME="$(grep -E '^HF_HOME=' .env | tail -n 1 | cut -d= -f2-)" UV_CACHE_DIR="$PWD/.cache/uv" UV_LINK_MODE=copy
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -x .venv/bin/python ] || uv venv .venv --python 3.11
uv pip sync --python .venv/bin/python requirements-train.txt
# the mini-SWE-agent method runs every agent command as oa-agent in a task folder outside the repo (self-tested)
OA_AGENT_ENV=user scripts/sandbox_setup.sh --start
sed -i '/^OA_AGENT_ENV=/d' .env; echo "OA_AGENT_ENV=user" >> .env
say "done: /dev/shm ${shm_gb} GB, free disk ${free_gb} GB, HF cache $HF_HOME, agent sandbox: user oa-agent"

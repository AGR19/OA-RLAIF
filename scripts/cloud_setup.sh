#!/usr/bin/env bash
# GPU box (Lambda, CoreWeave, Vast.ai): one-time setup, stage 1 of lambda_all.sh / coreweave_all.sh / vast_all.sh.
# Nothing is assumed on the host: Docker, build tools and the Python requirements are all installed here.
# Safe to run again.
#   scripts/cloud_setup.sh lambda|coreweave|vast
#   1. the hourly price (scripts/price.sh; asks only if .env has none)
#   2. system packages the install needs (images differ: PyTorch images often lack cmake, slim ones git/curl)
#   3. shared memory and disk checks (Kubernetes pods default to a 64 MB /dev/shm, which crashes NCCL and vLLM)
#   4. the agent sandbox (scripts/sandbox_setup.sh): Docker where it can run; on Vast.ai container instances, which
#      cannot run Docker, the user sandbox (OA_SANDBOX_FALLBACK=user)
#   5. the install shared with Lambda (scripts/lambda_setup.sh): uv venv from the lock, E4B + A4B weights, llama.cpp
# Caches (Hugging Face weights, uv wheels) go under .cache/ in this folder, so they live on the same volume as the
# repo: put the repo on the large persistent volume and they survive a pod restart.
set -euo pipefail
cd "$(dirname "$0")/.."
PLATFORM="${1:-lambda}"
SUDO=""; [ "$(id -u)" -eq 0 ] || SUDO="sudo"
say() { echo "[setup] $*"; }
touch .env

scripts/price.sh "$PLATFORM"

# 2. system packages
pkgs=()
command -v git >/dev/null || pkgs+=(git)
command -v curl >/dev/null || pkgs+=(curl ca-certificates)
command -v cmake >/dev/null || pkgs+=(cmake)
command -v g++ >/dev/null || pkgs+=(build-essential)
command -v tmux >/dev/null || pkgs+=(tmux)
command -v pkill >/dev/null || pkgs+=(procps)
command -v python3 >/dev/null || pkgs+=(python3)
dpkg -s libssl-dev >/dev/null 2>&1 || pkgs+=(libssl-dev)                       # llama.cpp build
dpkg -s libcurl4-openssl-dev >/dev/null 2>&1 || pkgs+=(libcurl4-openssl-dev)   # llama.cpp build (older tags)
if [ "${#pkgs[@]}" -gt 0 ]; then
  say "installing: ${pkgs[*]}"
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 update -qq
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y -qq "${pkgs[@]}" >/dev/null
fi

# 3. shared memory and disk
shm_gb="$(df -Pk /dev/shm | awk 'NR == 2 { print int($2 / 1048576) }')"
if [ "${shm_gb:-0}" -lt 8 ]; then
  echo "[setup] /dev/shm is ${shm_gb} GB. Multi-GPU training (NCCL) and vLLM need much more."
  if [ "$PLATFORM" = vast ]; then
    echo "On Vast.ai, choose another offer, or add  --shm-size=64g  to the template's Docker options if your host"
    echo "allows it, then create the instance again."
    exit 1
  fi
  cat <<EOF
Kubernetes gives pods 64 MB by default. Add a memory volume at /dev/shm to the pod spec, recreate the pod, and run
the same command again:
    volumes:      [{name: dshm, emptyDir: {medium: Memory, sizeLimit: 128Gi}}]
    volumeMounts: [{name: dshm, mountPath: /dev/shm}]          # in the container that runs this repo
EOF
  exit 1
fi
if ! grep -q '^HF_HOME=' .env; then
  echo "HF_HOME=$PWD/.cache/huggingface" >> .env
fi
export HF_HOME="$(grep -E '^HF_HOME=' .env | tail -n 1 | cut -d= -f2- | tr -d '"')"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}" UV_LINK_MODE=copy
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"
free_gb="$(df -Pk . | awk 'NR == 2 { print int($4 / 1048576) }')"
say "/dev/shm ${shm_gb} GB; free disk here ${free_gb} GB; HF cache: $HF_HOME"
if [ "$free_gb" -lt 180 ]; then
  say "need about 180 GB free here (weights 68 GB, venv 15 GB, three merged models 48 GB, GGUF 21 GB, data 4 GB)."
  if [ "$PLATFORM" = vast ]; then say "On Vast.ai the disk size is fixed at creation: create the instance again with >= 250 GB."
  else say "Clone the repo onto the large volume (for example your PVC mount) and run again."; fi
  exit 1
fi
[ "$free_gb" -lt 250 ] && say "warning: under 250 GB free; the run fits, with little room to spare"

# 4. the agent sandbox + self-test
if [ "$PLATFORM" = vast ]; then export OA_SANDBOX_FALLBACK="${OA_SANDBOX_FALLBACK:-user}"; fi
scripts/sandbox_setup.sh

# 5. the shared install
scripts/lambda_setup.sh
say "$PLATFORM setup done"

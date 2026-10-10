#!/usr/bin/env bash
# The mini-SWE-agent sandbox on a cloud GPU box (CoreWeave, Vast.ai; Lambda has Docker preinstalled).
#
#   scripts/sandbox_setup.sh            Docker if it can run here, else (if allowed) the user sandbox; self-test
#   scripts/sandbox_setup.sh --start    only make sure it still works (run_all.sh calls this on every run)
#
# Writes OA_AGENT_ENV=docker|user to .env; the agent evaluations use it.
#
# Docker (preferred: no network, only the task folder visible). Detected cases:
#   A. Docker already works (an image with Docker, or a dind sidecar with DOCKER_HOST in .env): nothing to install.
#   B. Virtual Server / VM / node with systemd and root or sudo (CoreWeave Virtual Server, Vast.ai VM instance):
#      Docker Engine from Docker's apt repository, started with systemd.
#   C. A privileged container (CoreWeave pod with privileged: true): Engine packages, dockerd in the background
#      (vfs storage when the data root is on overlayfs; no bridge/iptables when they cannot be set up).
#   A container without CAP_SYS_ADMIN (a normal Vast.ai instance, a non-privileged pod) cannot run dockerd, so
#   nothing is installed for it.
# User sandbox (OA_SANDBOX_FALLBACK=user, set for Vast.ai): when Docker cannot run, every agent command runs as the
#   unprivileged user oa-agent with an empty environment, in a task folder outside the repo that holds only the
#   public episode; the repo is closed to other users; the network is cut with `unshare -n` when the container
#   allows it, else the results say "network open" (src/oa_rlaif/harness/sandbox.py).
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
SUDO=""; [ "$(id -u)" -eq 0 ] || SUDO="sudo"
mkdir -p outputs/logs
LOG="$PWD/outputs/logs/dockerd.log"
IMAGE=oa-rlaif-om:latest
FALLBACK="${OA_SANDBOX_FALLBACK:-none}"
say() { echo "[sandbox] $*"; }

setenv() { sed -i "/^$1=/d" .env 2>/dev/null || true; echo "$1=$2" >> .env; }
docker_ok() { docker info >/dev/null 2>&1; }
has_systemd() { [ "$(ps -p 1 -o comm= 2>/dev/null)" = "systemd" ]; }
cap_sys_admin() {   # bit 21 of the effective capabilities: needed to run dockerd (mounts, namespaces)
  local hex; hex="$(awk '/^CapEff/ { print $2 }' /proc/self/status 2>/dev/null || echo 0)"
  [ $(( (16#${hex:-0} >> 21) & 1 )) -eq 1 ]
}
apt_get() { $SUDO env DEBIAN_FRONTEND=noninteractive apt-get "$@"; }

add_docker_repo() {
  . /etc/os-release
  case "${ID:-}" in ubuntu|debian) ;; *) say "unsupported OS '${ID:-?}': install Docker Engine by hand, then rerun"; exit 1 ;; esac
  apt_get update -qq
  apt_get install -y -qq ca-certificates curl gnupg acl iptables procps >/dev/null
  $SUDO install -m 0755 -d /etc/apt/keyrings
  $SUDO curl -fsSL "https://download.docker.com/linux/$ID/gpg" -o /etc/apt/keyrings/docker.asc
  $SUDO chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/$ID ${UBUNTU_CODENAME:-$VERSION_CODENAME} stable" \
    | $SUDO tee /etc/apt/sources.list.d/docker.list >/dev/null
  apt_get update -qq
}

install_cli_only() {
  command -v docker >/dev/null && return 0
  say "installing the docker CLI (daemon: ${DOCKER_HOST})"
  add_docker_repo
  apt_get install -y -qq docker-ce-cli >/dev/null
}

install_engine() {
  command -v dockerd >/dev/null && return 0
  say "installing Docker Engine from download.docker.com"
  add_docker_repo
  apt_get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin >/dev/null
}

grant_socket() {
  [ -z "$SUDO" ] && return 0
  $SUDO usermod -aG docker "$USER" || true
  $SUDO setfacl -m "u:$USER:rw" /var/run/docker.sock || true
}

fs_type() { stat -f -c %T "$1" 2>/dev/null || echo unknown; }

start_dockerd() {
  local root="${OA_DOCKER_DATA_ROOT:-/var/lib/docker}"
  $SUDO mkdir -p "$root"
  local storage=overlay2
  case "$(fs_type "$root")" in overlay*) storage=vfs ;; esac
  for attempt in "default network" "no bridge"; do
    for driver in "$storage" vfs; do
      local args=(--data-root "$root" --storage-driver "$driver")
      [ "$attempt" = "no bridge" ] && args+=(--bridge=none --iptables=false --ip6tables=false)
      $SUDO pkill -x dockerd 2>/dev/null || true; sleep 1
      say "starting dockerd (${driver}, ${attempt}); log: outputs/logs/dockerd.log"
      $SUDO sh -c "nohup setsid dockerd ${args[*]} >> '$LOG' 2>&1 &"
      for _ in $(seq 1 30); do docker_ok && break; sleep 1; done
      if docker_ok && docker run --rm --network none hello-world >/dev/null 2>&1; then
        echo "${args[*]}" > outputs/dockerd.args
        return 0
      fi
      [ "$driver" = vfs ] && break
    done
  done
  $SUDO pkill -x dockerd 2>/dev/null || true
  return 1
}

restart_with_saved_args() {
  [ -f outputs/dockerd.args ] || return 1
  say "dockerd is not running; starting it again with the saved settings"
  $SUDO sh -c "nohup setsid dockerd $(cat outputs/dockerd.args) >> '$LOG' 2>&1 &"
  for _ in $(seq 1 30); do docker_ok && return 0; sleep 1; done
  return 1
}

# Try to get a working Docker daemon. Returns 1 when Docker cannot run on this machine.
docker_daemon() {
  docker_ok && return 0
  if [ -n "${DOCKER_HOST:-}" ]; then
    install_cli_only
    docker_ok && return 0
    say "DOCKER_HOST=${DOCKER_HOST} is set but the daemon does not answer. Is the dind sidecar running?"; exit 1
  fi
  if has_systemd; then
    install_engine
    say "systemd found (VM / Virtual Server / node): enabling the docker service"
    $SUDO systemctl enable --now docker
    grant_socket
    for _ in $(seq 1 20); do docker_ok && return 0; sleep 1; done
    say "the docker service did not start: sudo journalctl -u docker"; exit 1
  fi
  if ! cap_sys_admin; then
    say "this is a container without CAP_SYS_ADMIN: dockerd cannot run here (nothing installed)"
    return 1
  fi
  install_engine
  restart_with_saved_args && return 0
  start_dockerd && return 0
  return 1
}

docker_image_and_selftest() {
  docker build --network none -t "$IMAGE" -f docker/Dockerfile.om .
  local probe="$PWD/outputs/docker_selftest"; mkdir -p "$probe"; echo "bind mount ok" > "$probe/probe.txt"
  local out
  out="$(docker run --rm --network none -v "$probe:/task" "$IMAGE" sh -c 'cat /task/probe.txt && om --help >/dev/null && echo om ok')" || out=""
  if ! printf '%s' "$out" | grep -q "om ok"; then
    say "self-test FAILED: the sandbox could not read the bind-mounted episode folder or run om."
    say "With a dind sidecar, mount the workspace at the same path in both containers."
    exit 1
  fi
  rm -rf "$probe"
  say "docker sandbox self-test passed (no network, bind mount, om tool)"
}

# ---- user sandbox (no Docker) -------------------------------------------------------------------------------------
user_sandbox_setup() {
  [ "$(id -u)" -eq 0 ] || { say "the user sandbox needs root (Vast.ai instances run as root)"; exit 1; }
  command -v python3 >/dev/null || { apt_get update -qq; apt_get install -y -qq python3 >/dev/null; }
  command -v runuser >/dev/null || { apt_get update -qq; apt_get install -y -qq util-linux >/dev/null; }
  id oa-agent >/dev/null 2>&1 || useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin oa-agent
  # the same `om` tool as in the Docker image (stdlib only)
  install -d -m 0755 /opt/om/bin
  install -m 0644 src/oa_rlaif/harness/om_cli.py /opt/om/om_cli.py
  printf '#!/bin/sh\nexec python3 /opt/om/om_cli.py "$@"\n' > /opt/om/bin/om; chmod 0755 /opt/om/bin/om
  install -d -m 0711 /var/tmp/oa-sandbox
  chown oa-agent:oa-agent /var/tmp/oa-sandbox; chmod 0700 /var/tmp/oa-sandbox
  # close the repo (data, gold, outputs, caches) to every other user, the sandbox user included
  chmod o-rwx "$PWD"
}

user_sandbox_selftest() {
  local task=/var/tmp/oa-sandbox/selftest/e0 secret="$PWD/outputs/sandbox_probe.txt"
  echo "secret" > "$secret"
  mkdir -p "$task"
  printf '{"id": "e0", "pair_id": "t/p", "source_iri": "http://s#1", "source_card": "IRI: http://s#1", "candidates": [{"iri": "http://t#1", "card": "IRI: http://t#1"}], "relation": "="}\n' > "$task/episode.json"
  chown -R oa-agent:oa-agent /var/tmp/oa-sandbox/selftest
  local net=False; unshare -n true 2>/dev/null && net=True
  local run_om run_secret
  run_om="$(PYTHONPATH=src python3 -c "from oa_rlaif.harness.sandbox import wrap_command; print(wrap_command('om candidates http://s#1 --k 5 && whoami', '$task', $net))")"
  run_secret="$(PYTHONPATH=src python3 -c "from oa_rlaif.harness.sandbox import wrap_command; print(wrap_command('cat $secret', '$task', $net))")"
  local out_om out_secret
  out_om="$(sh -c "$run_om" 2>&1)" || true
  out_secret="$(sh -c "$run_secret" 2>&1)" || true
  rm -rf /var/tmp/oa-sandbox/selftest "$secret"
  if ! printf '%s' "$out_om" | grep -q "oa-agent" || ! printf '%s' "$out_om" | grep -q "http://t#1"; then
    say "user sandbox self-test FAILED (om as oa-agent): $out_om"; exit 1
  fi
  if printf '%s' "$out_secret" | grep -q "^secret"; then
    say "user sandbox self-test FAILED: oa-agent can read the repo"; exit 1
  fi
  [ "$net" = True ] && setenv OA_AGENT_NET none || setenv OA_AGENT_NET open
  say "user sandbox self-test passed: om runs as oa-agent, the repo is closed to it, network: $( [ "$net" = True ] && echo cut || echo 'open (unshare not allowed here)')"
}

# ---- main ---------------------------------------------------------------------------------------------------------
if [ "${1:-}" = "--start" ] && [ "${OA_AGENT_ENV:-}" = user ]; then
  id oa-agent >/dev/null 2>&1 && [ -x /opt/om/bin/om ] && { say "user sandbox ok"; exit 0; }
  user_sandbox_setup; user_sandbox_selftest; exit 0
fi

if docker_daemon; then
  say "docker ok: $(docker info --format '{{.ServerVersion}}, storage {{.Driver}}' 2>/dev/null)"
  setenv OA_AGENT_ENV docker
  [ "${1:-}" = "--start" ] && exit 0
  docker_image_and_selftest
  exit 0
fi

if [ "$FALLBACK" = user ]; then
  say "Docker cannot run here; using the user sandbox for the agent evaluations"
  user_sandbox_setup
  user_sandbox_selftest
  setenv OA_AGENT_ENV user
  exit 0
fi

cat <<'EOF'

[sandbox] Docker cannot run here (a container without the privilege to start dockerd; see outputs/logs/dockerd.log).
The mini-SWE-agent sandbox needs Docker. Choose one, then run the same command again:

  1. Privileged pod (simplest). In the pod spec:  securityContext: {privileged: true}

  2. Docker-in-Docker sidecar (no privilege for the main container). Add to the same pod:
         - name: dind
           image: docker:28-dind
           securityContext: {privileged: true}
           env: [{name: DOCKER_TLS_CERTDIR, value: ""}]
           volumeMounts: [{name: <your workspace volume>, mountPath: <the SAME path as in the main container>}]
     and put  DOCKER_HOST=tcp://localhost:2375  in .env (each agent episode bind-mounts its folder).

  3. A VM / Virtual Server, or an image with Docker preinstalled.

  4. Or accept the weaker no-Docker sandbox:  OA_SANDBOX_FALLBACK=user  (as on Vast.ai; see VAST_RUNBOOK.md).

EOF
exit 1

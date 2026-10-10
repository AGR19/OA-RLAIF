#!/usr/bin/env bash
# CoreWeave (or any Ubuntu/Debian GPU box): make Docker work for the mini-SWE-agent sandbox and build its image.
#
#   scripts/coreweave_docker.sh            install if needed, start the daemon, build oa-rlaif-om:latest, self-test
#   scripts/coreweave_docker.sh --start    only make sure the daemon runs (coreweave_all.sh calls this on every run)
#
# It detects which CoreWeave setup it is on:
#   A. Docker already works: an image with the `docker` feature tag, or a Docker-in-Docker sidecar with DOCKER_HOST
#      set in .env. Nothing is installed (only the docker CLI, if it is missing).
#   B. Virtual Server or bare-metal node (systemd, root or sudo): Docker Engine from Docker's apt repository,
#      started with systemd; your user gets access to the socket.
#   C. Kubernetes pod (CKS, DevPod, a SUNK container; no systemd): Docker Engine packages, then dockerd in the
#      background. A pod can run dockerd only when it is privileged; otherwise this stops and prints the fixes.
# The sandbox runs every container with --network none, so the daemon needs no bridge network. If iptables or the
# bridge cannot be set up (common in pods), dockerd starts without them. If overlay2 cannot work (the data root is on
# overlayfs), it falls back to the vfs storage driver (slower, fine for the 200 MB sandbox image).
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
SUDO=""; [ "$(id -u)" -eq 0 ] || SUDO="sudo"
mkdir -p outputs/logs
LOG="$PWD/outputs/logs/dockerd.log"
IMAGE=oa-rlaif-om:latest
say() { echo "[docker] $*"; }

docker_ok() { docker info >/dev/null 2>&1; }
has_systemd() { [ "$(ps -p 1 -o comm= 2>/dev/null)" = "systemd" ]; }
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

install_cli_only() {   # case A without a CLI: a sidecar daemon is reachable through DOCKER_HOST
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

grant_socket() {       # case B as a normal user: group membership applies at next login, the ACL applies now
  [ -z "$SUDO" ] && return 0
  $SUDO usermod -aG docker "$USER" || true
  $SUDO setfacl -m "u:$USER:rw" /var/run/docker.sock || true
}

fs_type() { stat -f -c %T "$1" 2>/dev/null || echo unknown; }

start_dockerd() {      # case C: no systemd, so run dockerd ourselves (pods need privileged: true)
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
  return 1
}

restart_with_saved_args() {
  [ -f outputs/dockerd.args ] || return 1
  say "dockerd is not running; starting it again with the saved settings"
  $SUDO sh -c "nohup setsid dockerd $(cat outputs/dockerd.args) >> '$LOG' 2>&1 &"
  for _ in $(seq 1 30); do docker_ok && return 0; sleep 1; done
  return 1
}

explain_pod_fix() {
  cat <<'EOF'

[docker] dockerd cannot run in this pod: it is not privileged (see outputs/logs/dockerd.log).
The mini-SWE-agent sandbox needs Docker. Choose one, then run the same command again:

  1. Privileged pod (simplest). In the pod spec of your CKS workload / DevPod:
         securityContext:
           privileged: true
     Recreate the pod; this script then starts dockerd inside it.

  2. Docker-in-Docker sidecar (no privilege for the main container). Add to the same pod:
         - name: dind
           image: docker:28-dind
           securityContext: {privileged: true}
           env: [{name: DOCKER_TLS_CERTDIR, value: ""}]
           volumeMounts: [{name: <your workspace volume>, mountPath: <the SAME path as in the main container>}]
     and put  DOCKER_HOST=tcp://localhost:2375  in .env. The workspace must be mounted at the same path in both
     containers, because each agent episode bind-mounts its folder (outputs/agent_runs/...) into the sandbox.

  3. A CoreWeave Virtual Server or node image with the `docker` feature tag (Docker Engine preinstalled).

EOF
}

ensure_daemon() {
  docker_ok && return 0
  if [ -n "${DOCKER_HOST:-}" ]; then
    install_cli_only
    docker_ok && return 0
    say "DOCKER_HOST=${DOCKER_HOST} is set but the daemon does not answer. Is the dind sidecar running?"; exit 1
  fi
  install_engine
  if has_systemd; then
    say "systemd found (Virtual Server / node): enabling the docker service"
    $SUDO systemctl enable --now docker
    grant_socket
    for _ in $(seq 1 20); do docker_ok && return 0; sleep 1; done
    say "the docker service did not start: sudo journalctl -u docker"; exit 1
  fi
  restart_with_saved_args && return 0
  start_dockerd && return 0
  explain_pod_fix; exit 1
}

ensure_daemon
say "daemon ok: $(docker info --format '{{.ServerVersion}}, storage {{.Driver}}, cgroup {{.CgroupVersion}}' 2>/dev/null)"
[ "${1:-}" = "--start" ] && exit 0

# Sandbox image. No step needs a network, so the build runs with none (works without a bridge too).
docker build --network none -t "$IMAGE" -f docker/Dockerfile.om .
# Self-test exactly as the harness runs it: no network, the episode folder bind-mounted at /task.
probe="$PWD/outputs/docker_selftest"; mkdir -p "$probe"; echo "bind mount ok" > "$probe/probe.txt"
out="$(docker run --rm --network none -v "$probe:/task" "$IMAGE" sh -c 'cat /task/probe.txt && om --help >/dev/null && echo om ok')" || out=""
if ! printf '%s' "$out" | grep -q "om ok"; then
  say "self-test FAILED: the sandbox could not read the bind-mounted episode folder or run om."
  say "With a dind sidecar, mount the workspace at the same path in both containers (see option 2 above)."
  exit 1
fi
rm -rf "$probe"
say "sandbox self-test passed (no network, bind mount, om tool)"

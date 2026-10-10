"""What the agent's sandbox may see, and a sandbox for machines that cannot run Docker (Vast.ai container instances).

public_episode(): the episode fields `om` needs. The task folder never holds the answer (gold, gold_all,
none_reason, hard_negatives, kind, provenance, split), because the agent can `cat` every file in it.

UserSandbox ("user" agent environment): each agent command runs as the unprivileged user `oa-agent`, with an empty
environment, in a task folder outside the repo that holds only the public episode. The repo (data, gold, outputs,
caches) is closed to other users by scripts/sandbox_setup.sh, so the agent cannot read it. The network is cut with
`unshare -n` when the machine allows it (it needs CAP_SYS_ADMIN, which most containers lack); otherwise the run is
labelled "network open". Docker (scripts/sandbox_setup.sh) stays the default and gives the stronger isolation.

Standard library only: scripts/sandbox_setup.sh imports it with the system Python, before the venv exists.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from functools import lru_cache
from pathlib import Path

PUBLIC_FIELDS = ("id", "pair_id", "source_iri", "source_card", "candidates", "relation", "source_profile",
                 "target_profile", "source_lang", "target_lang")
SANDBOX_USER = os.environ.get("OA_SANDBOX_USER", "oa-agent")
SANDBOX_ROOT = Path(os.environ.get("OA_SANDBOX_ROOT", "/var/tmp/oa-sandbox"))
OM_BIN = "/opt/om/bin"


def public_episode(episode: dict) -> dict:
    """The episode as the agent may see it: what `om` reads, nothing that reveals the answer."""
    return {k: episode[k] for k in PUBLIC_FIELDS if k in episode}


@lru_cache(maxsize=1)
def network_isolation() -> bool:
    """True when `unshare -n` works here, so agent commands can run without any network."""
    try:
        return subprocess.run(["unshare", "-n", "true"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def wrap_command(command: str, task_dir: Path | str, isolate_net: bool, user: str = SANDBOX_USER) -> str:
    """The shell command that runs `command` as the sandbox user, with an empty environment, in task_dir."""
    task_dir = str(task_dir)
    env = (f"env -i PATH={OM_BIN}:/usr/local/bin:/usr/bin:/bin HOME={shlex.quote(task_dir)} "
           f"OM_TASK={shlex.quote(task_dir + '/episode.json')} PAGER=cat LANG=C.UTF-8")
    run = f"cd {shlex.quote(task_dir)} && exec runuser -u {user} -- {env} bash -c {shlex.quote(command)}"
    run = f"bash -c {shlex.quote(run)}"
    return f"unshare -n {run}" if isolate_net else run


def prepare_task_dir(tag: str, episode_id: str) -> Path:
    """A fresh task folder owned by the sandbox user, outside the repo."""
    import pwd

    entry = pwd.getpwnam(SANDBOX_USER)
    task_dir = SANDBOX_ROOT / tag / episode_id
    task_dir.mkdir(parents=True, exist_ok=True)
    for path in (SANDBOX_ROOT / tag, task_dir):
        os.chown(path, entry.pw_uid, entry.pw_gid)
        path.chmod(0o700)
    return task_dir


def make_user_sandbox(task_dir: Path, timeout: int = 30):
    """mini-swe-agent environment: LocalEnvironment whose commands run through wrap_command()."""
    from minisweagent.environments.local import LocalEnvironment

    isolate_net = network_isolation()

    class UserSandboxEnvironment(LocalEnvironment):
        def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None):
            wrapped = dict(action, command=wrap_command(action.get("command", ""), task_dir, isolate_net))
            return super().execute(wrapped, cwd=str(task_dir), timeout=timeout)

    return UserSandboxEnvironment(cwd=str(task_dir), env={}, timeout=timeout)


def describe(kind: str) -> dict:
    """How the agent was isolated, for summary.json and the report."""
    if kind == "docker":
        return {"sandbox": "docker", "network": "none", "filesystem": "task folder only"}
    if kind == "user":
        return {"sandbox": f"user {SANDBOX_USER}", "network": "none" if network_isolation() else "open",
                "filesystem": "task folder; repo closed to other users"}
    return {"sandbox": "local (tests only)", "network": "open", "filesystem": "not isolated"}

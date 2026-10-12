"""The agent's task folder never holds the answer; the no-Docker sandbox command drops to an unprivileged user."""

import json

from oa_rlaif.harness.sandbox import PUBLIC_FIELDS, public_episode, wrap_command


def test_public_episode_hides_the_answer():
    episode = {"id": "e1", "pair_id": "a/b", "source_iri": "s", "source_card": "c", "candidates": [{"iri": "t"}],
               "relation": "=", "gold": ["t"], "gold_all": ["t"], "none_reason": "x", "hard_negatives": ["u"],
               "kind": "gold", "provenance": {"ref": "r"}, "split": "test", "source_profile": "p"}
    public = public_episode(episode)
    assert set(public) <= set(PUBLIC_FIELDS)
    for hidden in ("gold", "gold_all", "none_reason", "hard_negatives", "kind", "provenance", "split"):
        assert hidden not in public
    assert public["candidates"] == [{"iri": "t"}] and public["source_card"] == "c"


def test_wrap_command_runs_as_sandbox_user_with_empty_env():
    cmd = wrap_command("om candidates 'x y' --k 5", "/var/tmp/oa-sandbox/base/e1", isolate_net=False)
    assert "runuser -u oa-agent --" in cmd and "env -i" in cmd and "OM_TASK=/var/tmp/oa-sandbox/base/e1/episode.json" in cmd
    assert not cmd.startswith("unshare")
    assert wrap_command("true", "/t", isolate_net=True).startswith("unshare -n ")


def test_agent_task_folder_has_no_gold(tmp_path):
    import pytest

    pytest.importorskip("minisweagent")
    from minisweagent.models.test_models import DeterministicModel

    from oa_rlaif.eval.agent_eval import run
    from oa_rlaif.io import read_yaml
    from oa_rlaif.paths import ROOT

    episode = {"id": "h2", "pair_id": "t/p", "kind": "gold", "relation": "=", "source_iri": "http://s#1",
               "source_card": "IRI: http://s#1", "candidates": [{"iri": "http://t#7", "card": "IRI: http://t#7"}],
               "gold": ["http://t#7"], "gold_all": ["http://t#7"]}
    turn = {"role": "assistant", "content": "THOUGHT: x\n```mswea_bash_command\ncat episode.json\n```",
            "extra": {"actions": [{"command": "cat episode.json"}]}}
    done = {"role": "assistant", "content": "x", "extra": {"actions": [
        {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && om submit C1 --relation = --confidence 0.9"}]}}
    results = run([episode], DeterministicModel(outputs=[turn, done]), "local",
                  tmp_path, read_yaml(ROOT / "configs" / "mini_swe_agent.yaml"))
    written = json.loads((tmp_path / "h2" / "episode.json").read_text())
    assert "gold" not in written and "gold_all" not in written
    assert results[0]["correct"]  # scored against the gold kept in the harness
    assert "gold_all" not in (tmp_path / "h2" / "trajectory.json").read_text()


def test_only_the_sandbox_setup_calls_docker():
    """Machines without Docker (Vast.ai containers) use the user sandbox: no other setup or stage script may call
    docker (an unconditional `docker build` in lambda_setup.sh once stopped the Vast.ai setup)."""
    import re

    from oa_rlaif.paths import ROOT

    allowed = {"sandbox_setup.sh", "00_check_env.sh", "smoke_mac.sh"}  # check_env: only when OA_AGENT_ENV=docker
    offenders = []
    for path in sorted((ROOT / "scripts").glob("*.sh")):
        if path.name in allowed:
            continue
        for line in path.read_text().splitlines():
            code = line.split("#", 1)[0]
            if re.search(r"(^|[\s;&|(])docker\s", code):
                offenders.append(f"{path.name}: {line.strip()}")
    assert not offenders, offenders

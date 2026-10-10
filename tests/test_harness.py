"""End-to-end harness test: mini-SWE-agent (deterministic model, local env) runs `om` and submits."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("minisweagent")

from oa_rlaif.eval.agent_eval import parse_submission, run  # noqa: E402
from oa_rlaif.io import read_yaml  # noqa: E402
from oa_rlaif.paths import ROOT  # noqa: E402


def episode():
    return {"id": "h1", "pair_id": "test/pair", "kind": "gold", "relation": "=", "source_iri": "http://s#1",
            "source_card": "IRI: http://s#1\nname: spinal cord grey matter",
            "candidates": [{"iri": "http://t#9", "card": "IRI: http://t#9\nname: white matter"},
                           {"iri": "http://t#7", "card": "IRI: http://t#7\nname: gray matter of spinal cord"}],
            "gold": ["http://t#7"]}


def turn(command):
    return {"role": "assistant", "content": f"THOUGHT: x\n```mswea_bash_command\n{command}\n```",
            "extra": {"actions": [{"command": command}]}}


def test_agent_loop_submits_through_om(tmp_path):
    from minisweagent.models.test_models import DeterministicModel

    model = DeterministicModel(outputs=[
        turn("om show http://s#1"), turn("om candidates http://s#1 --k 20"),
        turn("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && om submit C2 --relation = --confidence 0.9")])
    cfg = read_yaml(ROOT / "configs" / "mini_swe_agent.yaml")
    results = run([episode()], model, "local", tmp_path, cfg)
    assert results[0]["exit"] == "Submitted"
    assert results[0]["target"] == "http://t#7" and results[0]["correct"]
    trajectory = json.loads((tmp_path / "h1" / "trajectory.json").read_text())
    assert any("gray matter of spinal cord" in str(m.get("content", "")) for m in trajectory)


def test_parse_submission():
    text = "source_iri\ttarget_iri\trelation\tconfidence\nhttp://s#1\tNONE\t=\t0.5000\n"
    assert parse_submission(text)["target"] == "NONE"
    assert parse_submission("garbage") is None


def _docker_image_ready() -> bool:
    import shutil
    import subprocess

    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "image", "inspect", "oa-rlaif-om:latest"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _docker_image_ready(), reason="docker image oa-rlaif-om:latest not built")
def test_agent_loop_in_docker_sandbox(tmp_path):
    """Same loop inside the real sandbox: `om` from the image, task mounted at /task, no network."""
    from minisweagent.models.test_models import DeterministicModel

    # the printed tokens (OPEN_42 / BLOCKED_42) do not occur in the command text, so the check reads the output only
    probe = "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)\" && echo OPEN_$((40+2)) || echo BLOCKED_$((40+2))"
    model = DeterministicModel(outputs=[
        turn(probe), turn("om profile"), turn("om candidates http://s#1 --k 20"),
        turn("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && om submit C2 --relation = --confidence 0.9")])
    cfg = read_yaml(ROOT / "configs" / "mini_swe_agent.yaml")
    results = run([episode()], model, "docker", tmp_path, cfg)
    assert results[0]["exit"] == "Submitted" and results[0]["correct"]
    trajectory = json.dumps(json.loads((tmp_path / "h1" / "trajectory.json").read_text()))
    assert "BLOCKED_42" in trajectory and "OPEN_42" not in trajectory.replace("BLOCKED_42", "")

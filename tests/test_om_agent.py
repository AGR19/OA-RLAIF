"""om_agent/: the mini-SWE-agent harness of the local console, also the full evaluation's agent method."""

from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_importing_the_matcher_finds_the_repo_and_changes_no_paths(monkeypatch):
    monkeypatch.delenv("OA_OUT_DIR", raising=False)
    monkeypatch.delenv("OA_DATA_DIR", raising=False)
    module = importlib.import_module("om_agent.test_matching_ai")
    assert module.RLAIF == ROOT  # om_agent/ inside the repo: the repo is its parent
    assert "OA_OUT_DIR" not in os.environ and "OA_DATA_DIR" not in os.environ  # only own_folders() sets them


def test_matcher_settings_for_vllm_and_ollama():
    from om_agent.test_matching_ai import Matcher

    vllm = Matcher("oa-rlaif-e4b", ROOT / "outputs" / "unused", 1, "local", think=True, backend="vllm",
                   api_base="http://127.0.0.1:8100/v1")
    kwargs = vllm.model_cfg["model_kwargs"]
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}, "skip_special_tokens": False}
    assert kwargs["api_base"] == "http://127.0.0.1:8100/v1" and kwargs["max_tokens"] >= 4096
    ollama = Matcher("oa-rlaif-all-merged", ROOT / "outputs" / "unused", 1, "local", think=False)
    assert ollama.model_cfg["model_kwargs"]["extra_body"] == {"reasoning_effort": "none"}
    assert ollama.agent_cfg["step_limit"] == 6


def test_wording_is_formal():
    name = re.compile("mehr" + "noosh", re.I)
    files = [p for d in ("om_agent", "full_eval", "src", "tests", "configs") for p in (ROOT / d).rglob("*")
             if p.is_file() and p.suffix in {".py", ".sh", ".md", ".yaml", ""} and "__pycache__" not in p.parts]
    assert not [str(p.relative_to(ROOT)) for p in files if name.search(p.read_text(errors="ignore"))]

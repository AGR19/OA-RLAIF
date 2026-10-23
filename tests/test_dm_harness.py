"""dm_harness/: the candidate question, the System-One server (lexical stand-in), the proxy configuration, the
NONE/invalid guard, the new methods' folders and labels, and that the first run's repo is only read."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dm_harness import questions  # noqa: E402
from full_eval import schedule, score  # noqa: E402


def episode(relation: str = "=") -> dict:
    names = ["Liver", "Heart", "Heart valve"]
    return {"id": "e1", "pair_id": "anatomy/mouse-human", "relation": relation, "source_iri": "http://m#1",
            "source_card": "IRI: http://m#1\nname: heart\nparents: organ", "source_profile": "", "target_profile": "",
            "candidates": [{"iri": f"http://h#{i}", "card": f"IRI: http://h#{i}\nname: {n}\nparents: organ"}
                           for i, n in enumerate(names, 1)],
            "ranking": ["http://h#1", "http://h#3", "http://h#2"]}


def test_candidate_question_uses_the_agents_option_names_and_none():
    ep = episode()
    req = questions.request(ep)
    q = req["questions"][questions.QUESTION_ID]
    assert q["type"] == "choice" and list(q["criteria"]) == ["C1", "C2", "C3", "NONE"]
    assert q["criteria"]["C2"] == "name: Heart | parents: organ"  # the card without its IRI line
    assert "name: heart" in req["state"] and "same concept" in q["instructions"]
    assert "classifies" in questions.request(episode("classified_under"))["questions"]["match"]["instructions"]


def test_answer_gives_the_pick_and_ranks_candidates_by_probability():
    ep = episode()
    out = questions.read(ep, {"answers": {"match": {"probabilities": {"C1": 0.1, "C2": 0.6, "C3": 0.2, "NONE": 0.1}}}})
    assert (out["pick"], out["best_candidate"]) == ("http://h#2", "C2")
    assert out["ranking"] == ["http://h#2", "http://h#3", "http://h#1"]
    none = questions.read(ep, {"answers": {"match": {"probabilities": {"C1": 0.1, "C2": 0.2, "C3": 0.1, "NONE": 0.6}}}})
    assert none["pick"] == questions.NONE and none["best_candidate"] == "C2" and none["p_best_candidate"] == 0.2


def test_arm_a_rows_are_scored_with_the_decision_models_order():
    ep = episode()
    answer = questions.read(ep, {"answers": {"match": {"probabilities": {"C1": 0.1, "C2": 0.6, "C3": 0.2, "NONE": 0.1}}}})
    rec = score.decide(ep, questions.reply_row(ep, answer, 0.1, {"input_tokens": 300}))
    assert (rec["decision"], rec["targets"]) == ("match", ["http://h#2"])
    assert score.ranked_answers(rec) == ["http://h#2", "http://h#3", "http://h#1"]  # not the retrieval order


def test_server_accepts_the_proxys_request_and_answers_in_its_format(monkeypatch):
    """The JevAgentHarness backend sends an extra 'model' key and reads answers[id].probabilities."""
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from jevagentharness.decision import JevBackend, Question

    from dm_harness import decider_server

    monkeypatch.setenv("DM_FAKE", "lexical")
    client = TestClient(decider_server.build_app(None))
    body = {"model": "pplx-decider-v1.2", **questions.request(episode())}
    assert decider_server.clean(body).keys() == {"state", "questions"}
    data = client.post("/v1/systemone", json=body).json()
    probs = data["answers"]["match"]["probabilities"]
    assert set(probs) == {"C1", "C2", "C3", "NONE"} and max(probs, key=probs.get) == "C2"  # 'heart' ~ 'Heart'
    route = Question(id="next_step", text="routine or hard?", options=["routine", "hard"])
    state = "TASK: ...\n[3] ACTION: om submit\n    RESULT: FormatError: no command found"
    answer = client.post("/v1/systemone", json={"state": state, "model": "m", "questions": {"next_step": route.wire()}})
    decision = JevBackend(provider="local")._parse(answer.json(), route, 0.01)
    assert decision.p["hard"] > 0.75 and abs(sum(decision.p.values()) - 1) < 1e-9
    assert client.get("/health").json()["fake"] is True


def test_proxy_config_loads_with_the_proxys_own_validator(tmp_path):
    from jevagentharness.config import load_config

    from dm_harness.jah_config import GUARD_TAU, ROUTING_TAU, write

    cfg = load_config(write(tmp_path / "c.yaml", tmp_path / "logs"))
    assert cfg.tiers["strong"].extra["chat_template_kwargs"] == {"enable_thinking": True}
    assert cfg.tiers["fast"].extra["chat_template_kwargs"] == {"enable_thinking": False}
    assert cfg.router.decision["provider"] == "local" and cfg.router.tau == ROUTING_TAU == 0.75 and GUARD_TAU == 0.5
    assert cfg.models["jah-strong"] == {"tier": "strong"} and cfg.port == 8787


def run_guard(monkeypatch, first: dict, check: dict, redo: dict | None = None) -> tuple[dict, list]:
    from dm_harness import harness
    from om_agent.test_matching_ai import Matcher

    calls = []

    def fake_run(self, ep, on_message=None, approve=None, model_name=None, model_kwargs=None, note=""):
        calls.append((model_name, (model_kwargs or {}).get("api_base"), note))
        base = {"decision": "match", "targets": [], "exit": "Submitted", "model_calls": 1, "seconds": 1.0,
                "prompt_tokens": 10, "completion_tokens": 5, "thinking_chars": 0, "turns": [{}]}
        return {**base, **(redo if model_name else first)}

    monkeypatch.setattr(Matcher, "run_episode", fake_run)
    monkeypatch.setattr(harness, "decide", lambda ep, url=None: check)
    m = harness.HarnessMatcher.__new__(harness.HarnessMatcher)
    m.proxy, m.decider_url, m.guard_tau = "http://p", "http://x", 0.5
    return m.run_episode(episode()), calls


def test_guard_leaves_matches_alone_and_keeps_an_unsure_none(monkeypatch):
    out, calls = run_guard(monkeypatch, {"decision": "match", "targets": ["http://h#2"]}, {})
    assert "guard" not in out and calls == [(None, "http://p/s/e1/v1", "")]
    low = {"best_candidate": "C2", "p_best_candidate": 0.3, "pick": "NONE", "p": 0.6}
    out, calls = run_guard(monkeypatch, {"decision": "NONE"}, low)
    assert out["decision"] == "NONE" and out["guard"]["outcome"] == "kept_none" and len(calls) == 1


def test_guard_redoes_a_likely_none_on_the_strong_tier_with_the_hint(monkeypatch):
    high = {"best_candidate": "C2", "p_best_candidate": 0.8, "pick": "http://h#2", "p": 0.8}
    out, calls = run_guard(monkeypatch, {"decision": "NONE"}, high, {"decision": "match", "targets": ["http://h#2"]})
    assert calls[1][0] == "jah-strong" and calls[1][1] == "http://p/s/e1-redo/v1" and "candidate C2 (Heart)" in calls[1][2]
    assert (out["decision"], out["guard"]["outcome"], out["model_calls"]) == ("match", "redo_match", 2)


def test_guard_falls_back_to_the_decision_model_when_the_redo_is_invalid(monkeypatch):
    check = {"best_candidate": "C3", "p_best_candidate": 0.4, "pick": "http://h#3", "p": 0.4}
    out, calls = run_guard(monkeypatch, {"decision": "no submission"}, check, {"decision": "no submission"})
    assert calls[1][2] == ""  # p below 0.5: the redo gets no hint (the proxy's fail-safe: an error -> strong)
    assert (out["decision"], out["targets"], out["guard"]["outcome"]) == ("match", ["http://h#3"], "decider_pick")


def test_new_methods_have_their_own_folders_and_labels():
    assert schedule.folder_name("decider") == "7-pplx-decider"
    assert schedule.folder_name("jah-agent") == "8-jah-all-merged-agent"
    assert schedule.label("decider") == "pplx-decider-v1.2 alone" and "JevAgentHarness" in schedule.label("jah-agent")
    assert schedule.method("jah-agent").agent and not schedule.method("decider").agent


def test_the_first_runs_repo_is_only_read():
    from dm_harness import pull_reference, push_results
    from full_eval.push_results import REPO as FIRST_RUN

    assert push_results.REPO == "oa-rlaif-dm-harness-results" != FIRST_RUN
    assert pull_reference.PATTERNS == ["results/*/5-all-merged/scores.json",
                                       "results/*/6-all-merged-agent-thinking/scores.json"]
    assert not any("5-all-merged" in p or "6-all-merged" in p for p in push_results.PATTERNS)


def test_plan_runs_only_the_selected_tasks_for_each_arm(monkeypatch, tmp_path):
    """FULL_EVAL_TASKS narrows the plan exactly as it narrows the episode builder (else a run waits for episodes that
    are never built)."""
    from dm_harness import run

    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("FULL_EVAL_FAKE", "lexical")
    monkeypatch.setenv("FULL_EVAL_TASKS", "dh,cyber__iotsec-uco")
    monkeypatch.delenv("DM_ARMS", raising=False)
    rows = run.plan(schedule.Budget(250.0))
    assert [(r["arm"], r["task"]) for r in rows] == [("decider", "dh"), ("decider", "cyber__iotsec-uco"),
                                                     ("jah-agent", "dh"), ("jah-agent", "cyber__iotsec-uco")]
    assert all(r["fits"] for r in rows)

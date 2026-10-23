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
    m.proxy, m.decider_url, m.guard_tau, m.route_model, m.suffix = "http://p", "http://x", 0.5, "jah-auto", ""
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
    assert schedule.folder_name("jah-fit-agent") == "9-jah-all-merged-agent-fitted"
    assert schedule.method("jah-fit-agent").agent and schedule.label("jah-fit-agent") != schedule.label("jah-agent")
    assert schedule.label("decider") == "pplx-decider-v1.2 alone" and "JevAgentHarness" in schedule.label("jah-agent")
    assert schedule.method("jah-agent").agent and not schedule.method("decider").agent


def test_the_first_runs_repo_is_only_read():
    from dm_harness import pull_reference, push_results
    from full_eval.push_results import REPO as FIRST_RUN

    assert push_results.REPO == "oa-rlaif-dm-harness-results" != FIRST_RUN
    assert pull_reference.PATTERNS == ["results/*/5-all-merged/scores.json",
                                       "results/*/6-all-merged-agent-thinking/scores.json"]
    assert not any("5-all-merged" in p or "6-all-merged" in p for p in push_results.PATTERNS)
    for arm in ("results/*/7-pplx-decider/*", "results/*/8-jah-all-merged-agent/*",
                "results/*/9-jah-all-merged-agent-fitted/*", "replies/jah-fit-agent/*", "calibration/*"):
        assert arm in push_results.PATTERNS  # every arm's results and arm C's thresholds are uploaded


def test_plan_runs_only_the_selected_tasks_for_each_arm(monkeypatch, tmp_path):
    """FULL_EVAL_TASKS narrows the plan exactly as it narrows the episode builder (else a run waits for episodes that
    are never built)."""
    from dm_harness import run

    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("FULL_EVAL_FAKE", "lexical")
    monkeypatch.setenv("FULL_EVAL_TASKS", "dh,cyber__iotsec-uco")
    monkeypatch.delenv("DM_ARMS", raising=False)
    monkeypatch.setenv("DM_CAL_N", "200")
    rows = run.plan(schedule.Budget(450.0))
    assert [(r["arm"], r["task"]) for r in rows] == [
        ("decider", "dh"), ("decider", "cyber__iotsec-uco"), ("jah-agent", "dh"), ("jah-agent", "cyber__iotsec-uco"),
        ("calibrate", "sample of 200"), ("jah-fit-agent", "dh"), ("jah-fit-agent", "cyber__iotsec-uco")]
    assert all(r["fits"] for r in rows)
    # arm C aligns every class of its tasks, the calibration sample included (not a held-out view)
    assert [r["classes"] for r in rows if r["arm"] == "jah-fit-agent"] == [r["classes"] for r in rows
                                                                           if r["arm"] == "jah-agent"]


def test_threshold_fit_follows_the_papers_rule():
    """Equal-mass bins of >= 20; tau = the lowest score from which every higher bin has a non-negative gain."""
    from dm_harness.calibrate import MIN_BIN, fit_threshold

    scores = [i / 100 for i in range(100)]  # 5 bins of 20
    gains = [-1.0] * 40 + [0.5] * 20 + [-0.1] * 20 + [1.0] * 20  # bin 4 is negative: tau starts at bin 5
    tau, bins = fit_threshold(scores, gains)
    assert len(bins) == 5 and all(b["n"] == MIN_BIN for b in bins) and tau == 0.8
    assert [b["acts"] for b in bins] == [False, False, False, False, True]
    assert fit_threshold(scores, [1.0] * 100)[0] == 0.0  # every bin pays: always act
    assert fit_threshold(scores, [1.0] * 80 + [-1.0] * 20)[0] is None  # the top bin loses: never act
    assert fit_threshold(scores[:19], [1.0] * 19) == (None, [])  # fewer than one bin: no threshold
    assert fit_threshold(scores, [-1e-5] * 100)[0] is None  # a cost of a fraction of a cent still counts
    tied = [0.2] * 70 + [0.5 + i / 100 for i in range(30)]  # a bin edge must not split equal scores
    tau, bins = fit_threshold(tied, [-1.0] * 70 + [1.0] * 30)
    assert tau == 0.5 and bins[0]["to"] == 0.2 and all(b["from"] > 0.2 for b in bins[1:])


def test_fit_writes_frozen_thresholds_from_the_shadow_runs(monkeypatch, tmp_path):
    from dm_harness import calibrate

    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    task = type("T", (), {"slug": "toy", "pair_ids": ["p"]})()
    monkeypatch.setattr(calibrate, "task_order", lambda ts: [task])
    monkeypatch.setattr(calibrate, "selected_tasks", lambda: [task])
    monkeypatch.setattr(calibrate.labels, "partners", lambda t: {"p": {f"s{i}": {"t"} for i in range(40)}})
    rows, hard = [], {}
    for i in range(40):  # hard classes (p 0.9): only thinking on is right; routine (p 0.1): both are right
        strong, fast = ["t"], (["t"] if i < 20 else ["x"])
        hard[f"e{i}"] = 0.1 if i < 20 else 0.9
        rows.append({"id": f"e{i}", "pair_id": "p", "source": f"s{i}",
                     "decider": {"answer": ["x"], "confidence": 0.3, "best_candidate": "C1", "p_best": 0.3},
                     "strong": {"decision": "match", "answer": strong}, "fast": {"decision": "match", "answer": fast}})
    monkeypatch.setattr(calibrate, "p_hard", lambda: hard)
    (calibrate.cal_dir() / "shadow.jsonl").write_text("\n".join(map(__import__("json").dumps, rows)))
    out = calibrate.fit(0.01)
    assert out["routing"]["tau"] == 0.9 and out["routing"]["decisions"] == 40  # thinking on only where it pays
    assert out["guard"]["tau"] is None and out["cascade"]["tau"] is None  # no redo; the decider is never right
    saved = __import__("json").loads((calibrate.cal_dir() / "thresholds.json").read_text())
    assert saved["routing"]["tau"] == 0.9 and len(saved["routing"]["bins"]) == 2
    assert len(__import__("json").loads((calibrate.cal_dir() / "sample_ids.json").read_text())) == 40


def test_arm_c_uses_the_fitted_thresholds_and_the_cascade(monkeypatch):
    from dm_harness import harness
    from om_agent.test_matching_ai import Matcher

    def th(r, g, c):
        return {"routing": {"tau": r}, "guard": {"tau": g}, "cascade": {"tau": c}}

    monkeypatch.setattr(Matcher, "__init__", lambda self, *a, **k: None)
    assert harness.FittedMatcher("o", 1, "local", th(None, 0.5, None)).route_model == "jah-fast"
    assert harness.FittedMatcher("o", 1, "local", th(0.0, 0.5, None)).route_model == "jah-strong"
    m = harness.FittedMatcher("o", 1, "local", th(0.62, None, 0.9))
    assert (m.route_model, m.guard_tau, m.suffix, m.cascade_tau) == ("jah-fit", None, "-fit", 0.9)
    calls = []
    monkeypatch.setattr(harness.HarnessMatcher, "run_episode", lambda self, ep, *a, **k: calls.append(ep["id"]) or
                        {"decision": "NONE", "targets": []})
    m.decider_rows = {"e1": {"id": "e1", "decision": "match", "targets": ["http://h#2"], "confidence": 0.95}}
    out = m.run_episode(episode())
    assert out["targets"] == ["http://h#2"] and out["cascade"] == {"p": 0.95, "tau": 0.9} and calls == []
    m.decider_rows["e1"]["confidence"] = 0.8  # below tau_c: the agent decides
    assert m.run_episode(episode())["decision"] == "NONE" and calls == ["e1"]


def test_proxy_config_carries_the_shadow_and_fitted_models(tmp_path):
    from jevagentharness.config import load_config

    from dm_harness.jah_config import write

    cfg = load_config(write(tmp_path / "c.yaml", tmp_path / "logs", fitted_tau=0.62))
    assert cfg.models["jah-fit"]["tau"] == 0.62 and cfg.models["jah-shadow"]["decisions"]["routing"]["mode"] == "shadow"
    assert "jah-fit" not in load_config(write(tmp_path / "d.yaml", tmp_path / "logs")).models


def test_a_cascade_answer_is_scored_as_arm_cs_row(monkeypatch):
    """Arm C's cascade returns arm A's reply row; the worker's row_of and full_eval.score must read it as the class's
    answer (with arm A's probability order for ranks 2-5) and count it as a cascade."""
    from dm_harness import harness
    from full_eval.agent_run import row_of
    from om_agent.test_matching_ai import Matcher

    monkeypatch.setattr(Matcher, "__init__", lambda self, *a, **k: None)
    ep = episode()
    answer = questions.read(ep, {"answers": {"match": {"probabilities": {"C1": 0.05, "C2": 0.9, "C3": 0.03,
                                                                         "NONE": 0.02}}}})
    m = harness.FittedMatcher("o", 1, "local", {"routing": {"tau": None}, "guard": {"tau": None},
                                                "cascade": {"tau": 0.8}})
    m.decider_rows = {"e1": questions.reply_row(ep, answer, 0.1, {"input_tokens": 300})}
    rec = score.decide(ep, row_of(ep, m.run_episode(ep)))
    assert (rec["decision"], rec["targets"], rec["stats"]["cascade"]) == ("match", ["http://h#2"], True)
    assert score.ranked_answers(rec) == ["http://h#2", "http://h#1", "http://h#3"]
    out = score.behaviour([rec], "jah-fit-agent")
    assert out["cascade"] == 1.0 and out["match"] == 1.0


def test_upload_goes_to_the_new_private_repo_with_every_arm(monkeypatch, tmp_path):
    """dm_harness.push_results with a stand-in HfApi: a private dataset repo of its own, the three arms' results and
    arm C's calibration, never the first run's rows."""
    import huggingface_hub

    from dm_harness import push_results
    from oa_rlaif.data import hf_push

    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    out = tmp_path / "full_eval"
    for f in ["results/dh/7-pplx-decider/scores.json", "results/dh/9-jah-all-merged-agent-fitted/scores.json",
              "calibration/thresholds.json", "results/dh/5-all-merged/scores.json", "dm_report.md"]:
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        (out / f).write_text("{}")
    calls = []

    class Api:
        def create_repo(self, repo, **kw):
            calls.append(("create", repo, kw))

        def repo_info(self, repo, **kw):
            return type("I", (), {"private": True})()

        def upload_large_folder(self, **kw):
            calls.append(("upload", kw["repo_id"], kw))

        def upload_file(self, **kw):
            calls.append(("card", kw["repo_id"], kw))

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(hf_push, "namespace", lambda: "ns")
    assert push_results.main([]) == 0
    (_, repo, create), (_, _, upload), (_, _, card) = calls
    assert repo == "ns/oa-rlaif-dm-harness-results" and create["private"] and create["repo_type"] == "dataset"
    assert upload["private"] and upload["allow_patterns"] == push_results.PATTERNS and upload["folder_path"] == str(out)
    assert b"9-jah-all-merged-agent-fitted" in card["path_or_fileobj"] and b"Three arms" in card["path_or_fileobj"]

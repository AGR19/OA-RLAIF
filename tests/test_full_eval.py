"""full_eval/: which classes run (FULL_EVAL_SCOPE: the reference classes by default, every class with 'all'), the
methods and their order, the scoring rules (the Top-k evaluation; OAEI scores with 'all'; no held-out view), the money
planner, thinking replies, the agent's replies and the report."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from full_eval import schedule, score, tasks  # noqa: E402
from full_eval.infer import LexicalEngine, reply_row  # noqa: E402
from oa_rlaif.eval.metrics import AlignmentEvaluator  # noqa: E402
from oa_rlaif.eval.thinking import split  # noqa: E402
from oa_rlaif.harness.prompts import decision_messages, parse_decision  # noqa: E402
from oa_rlaif.ontology.models import Entity, Label, OntologyProfile  # noqa: E402


def profile(*entities: Entity) -> OntologyProfile:
    return OntologyProfile("t", "t.owl", {e.iri: e for e in entities})


def entity(iri: str, label: str = "", declared: bool = True) -> Entity:
    e = Entity(iri, declared=declared)
    if label:
        e.labels.add(Label(label, "en", "label"))
    return e


def test_scope_leaves_out_vocabulary_placeholders_deprecated_and_obsolete():
    p = profile(entity("http://m.owl#MA_1", "heart"), entity("http://www.w3.org/2002/07/owl#Thing"),
                entity("http://www.geneontology.org/formats/oboInOwl#Synonym", "Synonym"),
                entity("http://m.owl#UNDEFINED_part_of"), entity("http://m.owl#MA_2", "old"),
                entity("http://m.owl#MA_3", "obsolete liver"), entity("http://m.owl#parent", "p", declared=False))
    assert tasks.scope(p, deprecated={"http://m.owl#MA_2"}) == ["http://m.owl#MA_1"]


def test_deprecated_scan_reads_rdfxml_entities_and_functional_syntax(tmp_path, monkeypatch):
    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("OA_RAW_DIR", str(tmp_path))
    rdf = tmp_path / "a.owl"
    rdf.write_text('<!DOCTYPE rdf:RDF [\n<!ENTITY obo "http://purl.obolibrary.org/obo/" >\n]>\n'
                   '<owl:Class rdf:about="&obo;DOID_1">\n<owl:deprecated rdf:datatype="x">true</owl:deprecated>\n</owl:Class>\n'
                   '<owl:Class rdf:about="&obo;DOID_2">\n<rdfs:label>kept</rdfs:label>\n</owl:Class>\n')
    assert tasks.deprecated_iris(rdf) == {"http://purl.obolibrary.org/obo/DOID_1"}
    ofn = tmp_path / "b.owl"
    ofn.write_text('Prefix(:=<http://snomed.info/id/>)\nAnnotationAssertion(owl:deprecated :123 "true"^^xsd:boolean)\n')
    assert tasks.deprecated_iris(ofn) == {"http://snomed.info/id/123"}


def test_unit_order_each_method_on_every_task_thinking_off_first():
    """Each thinking-off method on every task (small, SNOMED CT -> ICD-11, Bio-ML), so it loads once; then thinking on."""
    order = schedule.units(["grpo", "all_merged-agent", "grpo-think"],
                           ["anatomy__mouse-human", "dh", "icd__snomed-to-icd11", "bioml__NCIT-DOID"])
    tasks_in_order = ["anatomy__mouse-human", "dh", "icd__snomed-to-icd11", "bioml__NCIT-DOID"]
    assert order == [(m, t) for m in ("grpo", "all_merged-agent", "grpo-think") for t in tasks_in_order]


def test_default_methods_five_models_thinking_off_and_the_agent_thinking_on(monkeypatch):
    assert len(schedule.METHODS) == 12 and "all_merged-agent-think" in schedule.METHODS
    m = schedule.method("all_merged-agent-think")
    assert (m.model, m.think, m.agent, m.kind, m.base) == ("all_merged", True, True, "agent-think", "all_merged-agent")
    assert schedule.method("grpo").kind == "decide" and schedule.label("sft-think") == "+ SFT, thinking on"
    for name in ("FULL_EVAL_METHODS", "FULL_EVAL_MODELS", "FULL_EVAL_THINKING", "FULL_EVAL_BUDGET_USD"):
        monkeypatch.delenv(name, raising=False)
    assert schedule.selected_methods() == ["grpo", "sft", "all_merged", "sdft", "base", "all_merged-agent-think"]
    monkeypatch.setenv("FULL_EVAL_THINKING", "off")
    assert schedule.selected_methods() == schedule.MODELS
    monkeypatch.setenv("FULL_EVAL_METHODS", "all")
    assert schedule.selected_methods() == schedule.METHODS_OFF
    monkeypatch.delenv("FULL_EVAL_THINKING")
    assert schedule.selected_methods() == schedule.METHODS
    monkeypatch.setenv("FULL_EVAL_FAKE", "lexical")
    assert schedule.Budget().cap == 425.0


def test_estimates_count_the_classes_each_scope_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    monkeypatch.delenv("FULL_EVAL_SCOPE", raising=False)
    assert sum(schedule.ESTIMATES["reference"].values()) == 299_961
    assert schedule.decisions("anatomy__mouse-human") == (1_497, False)
    assert schedule.decisions("icd__snomed-to-icd11") == (141_182, False)  # the complete map in either scope
    monkeypatch.setenv("FULL_EVAL_SCOPE", "everything")
    with pytest.raises(SystemExit):
        tasks.eval_scope()


def test_planner_starts_only_units_that_fit_the_money(monkeypatch, tmp_path):
    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("FULL_EVAL_FAKE", "lexical")  # $1/h, no ledger
    sizes = {"a": 360_000, "b": 3_600_000, "c": 36_000}
    monkeypatch.setattr(schedule, "decisions", lambda slug: (sizes[slug], True))
    budget = schedule.Budget(cap=5.0)
    budget.reserve = 1.0
    speed = schedule.Speed(path=tmp_path / "speed.json")
    monkeypatch.setattr(speed, "rate", lambda kind="decide": (100.0, "test"))
    monkeypatch.setattr(speed, "load_seconds", lambda: 0.0)
    plan = schedule.project([("m", "a"), ("m", "b"), ("m", "c")], budget, speed, loaded="m")
    assert [p["fits"] for p in plan] == [True, False, True]  # $1 fits, $10 does not, $0.10 still fits after it
    assert plan[0]["usd"] == pytest.approx(1.0, abs=0.01)


def reference_pair(tmp_path, monkeypatch):
    """A pair whose reference has: m#1 = h#1, m#2 = h#2 (m#2 deprecated), h#3 = m#3 written target -> source, and a
    link to a class of neither ontology. m#4 and m#5 have no partner."""
    from oa_rlaif.data.families import PairSpec
    from oa_rlaif.ontology.alignment_io import write_oaei_rdf
    from oa_rlaif.ontology.models import Alignment

    source = profile(*(entity(f"http://m#{i}", f"m{i}") for i in range(1, 6)))
    target = profile(*(entity(f"http://h#{i}", f"h{i}") for i in range(1, 4)))
    ref = tmp_path / "reference.rdf"
    write_oaei_rdf(ref, [Alignment("http://m#1", "http://h#1", "=", 1.0, "ref"),
                         Alignment("http://m#2", "http://h#2", "=", 1.0, "ref"),
                         Alignment("http://h#3", "http://m#3", "=", 1.0, "ref"),
                         Alignment("http://m#5", "http://x#9", "=", 1.0, "ref")])
    spec = PairSpec("test/m-h", "anatomy", tmp_path / "m.owl", tmp_path / "h.owl", [ref])
    monkeypatch.setattr(score, "load_profile_any", lambda path: source if path.name == "m.owl" else target)
    monkeypatch.setattr(score, "zero_shot_pairs", lambda: set())
    return spec, source, target


def test_reference_scope_runs_exactly_the_classes_topk_scores(tmp_path, monkeypatch):
    """The builder (sources_to_align) and the scorer (PairScorer.partners) use one rule: the classes with a reference
    partner between the two ontologies. Deprecated reference classes are not run and count as misses."""
    from full_eval.build_episodes import sources_to_align

    spec, source, target = reference_pair(tmp_path, monkeypatch)
    task = tasks.Task("t", "anatomy", "Anatomy", [spec])
    partners = score.PairScorer(spec, oaei=False).partners()
    assert partners == {"http://m#1": {"http://h#1"}, "http://m#2": {"http://h#2"}, "http://m#3": {"http://h#3"}}
    monkeypatch.setenv("FULL_EVAL_SCOPE", "reference")
    iris, info = sources_to_align(task, spec, source, target, {"http://m#2"})
    assert iris == ["http://m#1", "http://m#3"]
    assert info == {"scope": "reference", "reference_classes": 3, "reference_classes_left_out": 1}
    monkeypatch.setenv("FULL_EVAL_SCOPE", "all")
    assert sources_to_align(task, spec, source, target, {"http://m#2"})[0] == [f"http://m#{i}" for i in (1, 3, 4, 5)]


def test_reference_scope_scores_topk_and_writes_the_alignment_without_oaei(tmp_path, monkeypatch):
    spec, _, _ = reference_pair(tmp_path, monkeypatch)
    task = tasks.Task("t", "anatomy", "Anatomy", [spec])

    def rec(src, target, ranking):
        return {"pair_id": spec.pair_id, "source_iri": src, "source_label": src, "decision": "match",
                "targets": [target], "target_labels": [target], "relation": "=", "confidence": 0.9,
                "candidates": ranking, "ranking": ranking}

    records = [rec("http://m#1", "http://h#1", ["http://h#1", "http://h#2"]),
               rec("http://m#3", "http://h#2", ["http://h#2", "http://h#3"])]
    monkeypatch.setenv("FULL_EVAL_SCOPE", "reference")
    out = score.score_pairs("grpo", task, {spec.pair_id: records}, tmp_path / "out")
    pair = out["pairs"][spec.pair_id]
    assert "oaei" not in pair and out["oaei"] == {}
    assert pair["topk"]["classes_scored"] == 3 and pair["reference_classes_not_aligned"] == 1  # m#2: deprecated
    assert pair["topk"]["k"][1]["Precision"] == pytest.approx(1 / 3) and pair["topk"]["k"][3]["Recall"] == pytest.approx(2 / 3)
    assert (tmp_path / "out" / "pairs" / "test__m-h" / "alignment.rdf").exists()  # one file per method and pair
    monkeypatch.setenv("FULL_EVAL_SCOPE", "all")
    assert "oaei" in score.score_pairs("grpo", task, {spec.pair_id: records}, tmp_path / "all")["pairs"][spec.pair_id]


def test_reference_scope_report_has_topk_tables_only(tmp_path, monkeypatch):
    from full_eval import report

    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("FULL_EVAL_SCOPE", "reference")
    for name in ("FULL_EVAL_METHODS", "FULL_EVAL_MODELS", "FULL_EVAL_THINKING"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "full_eval").mkdir()
    k = {str(k): {"Precision": 0.9, "Recall": 0.8, "F-measure": 0.85} for k in (1, 3, 5)}
    unit = {"anatomy__mouse-human": {"pairs": {"anatomy/mouse-human": {
        "zero_shot": False, "retrieval_ceiling": 0.97, "topk": {"classes_scored": 1497, "k": k}}},
        "behaviour": {"classes": 1497, "match": 0.95, "none": 0.04, "invalid": 0.01, "prompt_tokens_per_class": 900,
                      "completion_tokens_per_class": 40}, "runtime_s": 30}}
    monkeypatch.setattr(report, "collect", lambda: {"grpo": unit, "all_merged-agent-think": unit})
    built = report.build()
    md = (tmp_path / "full_eval" / "report.md").read_text()
    assert set(built["tables"]) == {"t1", "t2", "t3", "t4", "t5"} and built["scope"] == "reference"
    assert "F-Measure" not in md and "Recall+" not in md and "OAEI score" not in md  # the OAEI 2026 track is named
    assert "Top-k evaluation (k = 1, 3, 5)" in md and "All merged + mini-SWE-agent, thinking on" in md


def test_full_alignment_scores_every_reference_link():
    reference = {("s1", "t1"), ("s2", "t2"), ("s3", "t3")}
    predicted = {("s1", "t1"), ("s2", "t9"), ("s3", "t3"), ("s4", "t4")}  # s4 has no partner: a wrong match
    s = AlignmentEvaluator().evaluate(predicted, reference)
    assert (s["Precision"], s["Recall"], s["true_positive"], s["reference"]) == (0.5, 2 / 3, 2, 3)


def test_aggregate_gives_macro_and_micro():
    rows = [{"Precision": 1.0, "Recall": 0.5, "F-Measure": 2 / 3, "true_positive": 1, "Size": 1, "reference": 2,
             "Recall+": None},
            {"Precision": 0.5, "Recall": 1.0, "F-Measure": 2 / 3, "true_positive": 2, "Size": 4, "reference": 2,
             "Recall+": 0.5}]
    a = score.oaei_aggregate(rows)
    assert a["macro"]["F-Measure"] == pytest.approx(2 / 3)
    assert a["micro"]["Precision"] == pytest.approx(3 / 5) and a["micro"]["Recall"] == pytest.approx(3 / 4)
    assert a["macro_recall_plus"] == 0.5


def episode(*names: str) -> dict:
    return {"id": "e1", "pair_id": "anatomy/mouse-human", "relation": "=", "source_iri": "http://m#1",
            "source_card": "IRI: http://m#1\nname: heart", "source_profile": "", "target_profile": "",
            "candidates": [{"iri": f"http://h#{i}", "card": f"IRI: http://h#{i}\nname: {n}"} for i, n in enumerate(names, 1)]}


def test_decide_threshold_none_and_invalid_replies():
    ep = episode("Liver", "Heart")
    sure = "THOUGHT: x\n```mswea_bash_command\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && om submit C2 --relation = --confidence 0.9\n```"
    unsure = sure.replace("0.9", "0.4")
    none = sure.replace("C2", "NONE")
    r = score.decide(ep, {"reply": sure})
    assert (r["decision"], r["targets"], r["target_labels"]) == ("match", ["http://h#2"], ["Heart"])
    assert [a.pair for a in score.picks_of(r)] == [("http://m#1", "http://h#2")]
    assert score.picks_of(score.decide(ep, {"reply": unsure})) == []  # below the 0.5 threshold: no correspondence
    assert score.decide(ep, {"reply": none})["decision"] == "NONE"
    assert score.decide(ep, {"reply": "I think C2"})["decision"] == "invalid"


def test_agent_rows_are_scored_from_the_submission():
    ep = {**episode("Liver", "Heart", "Heart valve"), "ranking": ["http://h#3", "http://h#2", "http://h#1"]}
    row = {"decision": "match", "targets": ["http://h#2"], "relation": "=", "confidence": 0.9, "exit": "Submitted",
           "model_calls": 2, "extra_commands": 1, "format_errors": 0, "prompt_tokens": 9000, "completion_tokens": 80,
           "thinking_chars": 0, "answer_in_thinking": 0, "turns": [{"answer": "a", "thinking": "", "finish_reason": "stop"}],
           "seconds": 3.0, "thought": "THOUGHT: heart"}
    r = score.decide(ep, row)
    assert (r["decision"], r["targets"], r["target_labels"]) == ("match", ["http://h#2"], ["Heart"])
    assert score.ranked_answers(r) == ["http://h#2", "http://h#3", "http://h#1"]  # own choice, then retrieval order
    lost = score.decide(ep, {**row, "decision": "no submission", "targets": [], "exit": "LimitsExceeded"})
    assert (lost["decision"], lost["error"]) == ("invalid", "LimitsExceeded")
    b = score.behaviour([r, lost], "all_merged-agent")
    assert b["model_calls_per_class"] == 2 and b["invalid"] == 0.5 and b["exit_statuses"] == {"Submitted": 1, "LimitsExceeded": 1}


def test_thinking_reply_is_split_and_scored_from_its_answer():
    ep = episode("Liver", "Heart")
    answer = "THOUGHT: x\n```mswea_bash_command\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && om submit C2 --relation = --confidence 0.9\n```"
    row = reply_row(ep, f"<|channel>thought\nheart matches<channel|>{answer}", 100, 20, "stop", think=True)
    assert (row["reply"], row["thinking"], row["answer_in_thinking"]) == (answer, "heart matches", False)
    inside = reply_row(ep, f"<|channel>thought\n{answer}", 100, 20, "length", think=True)  # never closed
    assert inside["reply"] == answer and inside["answer_in_thinking"]
    r = score.decide(ep, inside)
    assert r["targets"] == ["http://h#2"] and r["stats"]["truncated"] and r["stats"]["answer_in_thinking"]
    assert split("", reasoning=answer)["answer"] == answer  # Ollama: the thinking comes in its own field


def test_lexical_rehearsal_engine_answers_in_the_trained_format():
    ep = episode("Liver", "Heart", "Heart valve")
    (reply, *_), = LexicalEngine().chat([decision_messages(ep)])
    d = parse_decision(reply, ep)
    assert d.valid and d.choice == "http://h#2"
    (raw, *_), = LexicalEngine(think=True).chat([decision_messages(ep)])
    assert parse_decision(reply_row(ep, raw, 0, 0, "stop", think=True)["reply"], ep).choice == "http://h#2"


def test_every_task_family_of_the_earlier_report_is_present():
    families = {t.column for t in tasks.tasks()}
    assert {"Anatomy", "Bio-ML", "MultiFarm", "DH", "DH-ML", "Biodiv", "ICD-10→11", "SNOMED→ICD-11"} <= families
    assert set(tasks.COLUMNS) >= {"Cybersecurity", "Zero-shot"}


def test_topk_follows_clf_py_precision_and_recall_at_k():
    """clf.py: only classes with a reference partner count; P@k = hits / min(k, partners), R@k = hits / partners."""
    partners = {"a": {"x"}, "b": {"y", "z"}, "c": {"w"}}
    rankings = {"a": ["x", "q"], "b": ["z", "q", "y"], "c": [], "d": ["q"]}  # c: NONE; d has no partner: never scored
    s = AlignmentEvaluator().topk(rankings, partners)
    assert s["method"] == "Top-k evaluation (k = 1, 3, 5)" and s["classes_scored"] == 3
    one, three = s["k"][1], s["k"][3]
    assert one["Precision"] == pytest.approx(2 / 3) and one["Recall"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert three["Precision"] == pytest.approx((1 + 1 + 0) / 3) and three["Recall"] == pytest.approx((1 + 1 + 0) / 3)
    assert one["F-measure"] == pytest.approx(2 * one["Precision"] * one["Recall"] / (one["Precision"] + one["Recall"]))


def test_report_cells_and_the_agent_difference():
    from full_eval import report

    def unit(f, top1):
        k = {str(k): {"Precision": top1, "Recall": top1, "F-measure": top1} for k in (1, 3, 5)}
        return {"anatomy__mouse-human": {"pairs": {"anatomy/mouse-human": {
            "zero_shot": False, "oaei": {"F-Measure": f}, "topk": {"classes_scored": 10, "k": k}}}}}

    results = {"all_merged": unit(0.80, 0.90), "all_merged-agent": unit(0.85, 0.92)}
    assert report.oaei_value(results["all_merged"], "Anatomy") == pytest.approx(80.0)
    assert report.oaei_value(results["all_merged"], "DH") is None  # not run: "–", never estimated
    rows = dict(report.table7(results))
    assert rows["OAEI F-measure, thinking off"][0] == pytest.approx(5.0)
    assert rows["Top-k F-measure@1, thinking off"][0] == pytest.approx(2.0)
    assert rows["OAEI F-measure, thinking on"][0] is None


def test_blank_labels_fall_back_to_the_iri_name():
    """IoTSec writes empty rdfs:label elements: cards and retrieval must use the IRI's local name instead."""
    from oa_rlaif.data.cards import names
    from oa_rlaif.retrieval.candidates import label_texts

    e = entity("http://iotsec#BlockCipher")
    e.labels.add(Label("", "en", "label"))
    assert names(e)[0] == "BlockCipher" and label_texts(e) == ["BlockCipher"]
    e.labels.add(Label("block cipher", "en", "label"))
    assert names(e)[0] == "block cipher" and label_texts(e) == ["block cipher"]


def test_biodiv2026_is_its_own_zero_shot_column_and_never_training_data():
    """OAEI 2026 Biodiversity and Ecology: 9 pairs in the full evaluation only (not in families.discover, which the
    training-data build reads), each source.xml -> target.xml, classes only, all zero-shot."""
    from oa_rlaif.data.families import discover

    specs = tasks.biodiv2026_specs()
    if not specs:
        pytest.skip("raw folder biodiv-2026/ not present")
    assert len(specs) == 9 and all(s.source.name == "source.xml" and s.target.name == "target.xml" for s in specs)
    assert not {s.pair_id for s in discover()} & {s.pair_id for s in specs}
    task = next(t for t in tasks.tasks() if t.slug == "biodiv2026")
    assert task.column == "Biodiv 2026" and task.classes_only
    assert {s.pair_id for s in specs} <= tasks.zero_shot_pairs()


def test_results_are_sorted_by_dataset_then_method(tmp_path, monkeypatch):
    """results/<task>/<method folder>/, method folders numbered in table order; thinking on adds -thinking."""
    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    names = [schedule.folder_name(t) for t in ("base", "sft", "sdft", "grpo", "all_merged", "all_merged-agent-think")]
    assert names == ["1-base", "2-sft", "3-opsd", "4-opsd-grpo", "5-all-merged", "6-all-merged-agent-thinking"]
    assert schedule.results_dir("grpo", "anatomy__mouse-human") == \
        tmp_path / "full_eval" / "results" / "anatomy__mouse-human" / "4-opsd-grpo"


def test_progress_line_has_a_bar_counts_speed_and_time_left():
    line = schedule.progress_text("icd/who-icd10-to-icd11", 5_300, 12_597, 12.4, "episodes")
    assert line.startswith("icd/who-icd10-to-icd11 [") and " 42.1% 5,300/12,597 episodes · 12.4/s · ETA 0:09:48" in line
    assert schedule.bar(0, 0) == "[" + "-" * 24 + "]   0.0%" and schedule.clock(None) == "?"


def test_replies_are_not_cut_at_256_tokens_and_cut_replies_are_counted(monkeypatch):
    """The 2026-10-22 run cut base, OPSD and GRPO replies at 256 new tokens before their `om submit`; the limit is now
    1,024 (FULL_EVAL_MAX_TOKENS) and every reply records its finish reason."""
    from full_eval import infer

    monkeypatch.delenv("FULL_EVAL_MAX_TOKENS", raising=False)
    import importlib

    assert importlib.reload(infer).MAX_NEW_TOKENS == 1024
    ep = episode("Liver", "Heart")
    cut = reply_row(ep, "THOUGHT: the source is a heart, and C2 ...", 900, 1024, "length", think=False)
    assert cut["finish_reason"] == "length" and "thinking" not in cut
    r = score.decide(ep, cut)
    assert r["decision"] == "invalid" and r["stats"]["truncated"]
    done = score.decide(ep, reply_row(ep, "THOUGHT: x\n```mswea_bash_command\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                                          " && om submit C2 --relation = --confidence 0.9\n```", 900, 60, "stop", think=False))
    assert score.behaviour([r, done], "grpo")["truncated"] == 0.5


def test_rerun_keeps_the_finished_results_of_the_other_methods(monkeypatch):
    from full_eval.pull_inputs import reuse_patterns

    for name in ("FULL_EVAL_METHODS", "FULL_EVAL_MODELS", "FULL_EVAL_THINKING"):
        monkeypatch.delenv(name, raising=False)
    assert reuse_patterns(["base", "sdft", "grpo"]) == [
        "results/*/2-sft/scores.json", "results/*/5-all-merged/scores.json",
        "results/*/6-all-merged-agent-thinking/scores.json"]
    with pytest.raises(SystemExit):
        reuse_patterns(["opsd"])  # the tag is sdft

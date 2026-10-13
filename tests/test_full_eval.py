"""full_eval/: what counts as a class to align, the OAEI and Bio-ML scoring rules, the money planner and the replies."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from full_eval import schedule, score, tasks  # noqa: E402
from full_eval.infer import LexicalEngine  # noqa: E402
from oa_rlaif.eval.metrics import AlignmentEvaluator  # noqa: E402
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


def test_unit_order_puts_grpo_and_the_small_tasks_first():
    order = schedule.units(["grpo", "sft"], ["anatomy__mouse-human", "dh", "icd__snomed-to-icd11", "bioml__NCIT-DOID"])
    assert order == [("grpo", "anatomy__mouse-human"), ("grpo", "dh"), ("grpo", "icd__snomed-to-icd11"),
                     ("sft", "anatomy__mouse-human"), ("sft", "dh"), ("grpo", "bioml__NCIT-DOID"),
                     ("sft", "icd__snomed-to-icd11"), ("sft", "bioml__NCIT-DOID")]


def test_planner_starts_only_units_that_fit_the_money(monkeypatch, tmp_path):
    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("FULL_EVAL_FAKE", "lexical")  # $1/h, no ledger
    sizes = {"a": 360_000, "b": 3_600_000, "c": 36_000}
    monkeypatch.setattr(schedule, "decisions", lambda slug: (sizes[slug], True))
    budget = schedule.Budget(cap=5.0)
    budget.reserve = 1.0
    speed = schedule.Speed(path=tmp_path / "speed.json")
    monkeypatch.setattr(speed, "rate", lambda: (100.0, "test"))
    monkeypatch.setattr(speed, "load_seconds", lambda: 0.0)
    plan = schedule.project([("m", "a"), ("m", "b"), ("m", "c")], budget, speed, loaded="m")
    assert [p["fits"] for p in plan] == [True, False, True]  # $1 fits, $10 does not, $0.10 still fits after it
    assert plan[0]["usd"] == pytest.approx(1.0, abs=0.01)


def test_heldout_view_sets_training_links_to_null():
    reference = {("s1", "t1"), ("s2", "t2"), ("s3", "t3")}
    predicted = {("s1", "t1"), ("s2", "t9"), ("s3", "t3")}
    flagged = {("s1", "t1")}  # a training link: neither right nor wrong
    s = AlignmentEvaluator().evaluate(predicted, reference, flagged=flagged)
    assert (s["Precision"], s["Recall"], s["true_positive"], s["reference"]) == (0.5, 0.5, 1, 2)


def test_aggregate_gives_macro_and_micro():
    rows = [{"Precision": 1.0, "Recall": 0.5, "F-Measure": 2 / 3, "true_positive": 1, "Size": 1, "predicted_flagged": 0,
             "reference": 2, "Recall+": None},
            {"Precision": 0.5, "Recall": 1.0, "F-Measure": 2 / 3, "true_positive": 2, "Size": 4, "predicted_flagged": 0,
             "reference": 2, "Recall+": 0.5}]
    a = score.aggregate(rows)
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
    r = score.decide(ep, sure)
    assert (r["decision"], r["targets"], r["target_labels"]) == ("match", ["http://h#2"], ["Heart"])
    assert [a.pair for a in score.picks_of(r)] == [("http://m#1", "http://h#2")]
    assert score.picks_of(score.decide(ep, unsure)) == []  # below the 0.5 threshold: no correspondence
    assert score.decide(ep, none)["decision"] == "NONE"
    assert score.decide(ep, "I think C2")["decision"] == "invalid"


def test_lexical_rehearsal_engine_answers_in_the_trained_format():
    ep = episode("Liver", "Heart", "Heart valve")
    (reply, _, _), = LexicalEngine().chat([decision_messages(ep)])
    d = parse_decision(reply, ep)
    assert d.valid and d.choice == "http://h#2"


def test_every_task_family_of_the_earlier_report_is_present():
    families = {t.column for t in tasks.tasks()}
    assert {"Anatomy", "Bio-ML", "MultiFarm", "DH", "DH-ML", "Biodiv", "ICD-10→11", "SNOMED→ICD-11"} <= families


def test_top1_follows_clf_py_precision_and_recall_at_1():
    """clf.py: only classes with a reference partner count; P@1 = hits / min(1, partners), R@1 = hits / partners."""
    partners = {"a": {"x"}, "b": {"y", "z"}, "c": {"w"}}
    choices = {"a": ["x"], "b": ["z"], "c": [], "d": ["q"]}  # c answered NONE; d has no partner: never scored
    s = AlignmentEvaluator().top1(choices, partners, trivial={("a", "x")})
    assert s["classes_scored"] == 3 and s["hits"] == 2 and s["Size"] == 2
    assert s["Precision"] == pytest.approx(2 / 3)                    # (1 + 1 + 0) / 3
    assert s["Recall"] == pytest.approx((1 + 0.5 + 0) / 3)           # b has two partners: 1/2
    assert s["Recall+"] == pytest.approx(1 / 2)                      # a is trivial; b hit, c missed

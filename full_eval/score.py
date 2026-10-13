"""Score one unit (model, task) and write its complete outputs.

    python -m full_eval.score grpo anatomy__mouse-human

How precision and recall are computed (the published protocols, checked 2026-10-13):
  OAEI (Anatomy, MultiFarm, DH, Biodiv, Complex, IoTSec):  P = |A∩R|/|A|, R = |A∩R|/|R|, F = 2PR/(P+R), with A the
      system's whole alignment and R the whole reference alignment. Only equivalence correspondences count (OAEI
      Anatomy and DH say so); vocabulary correspondences such as oboInOwl#Synonym = oboInOwl#Synonym are removed.
      Recall+ (Anatomy) = recall on the non-trivial references: those whose two entities do not share a normalised
      label (OAEI's trivial.rdf for Anatomy; normalised label equality elsewhere).
      DH reports per test case and the mean over test cases; MultiFarm and Bio-ML aggregate over tasks: we give the
      mean over pairs (macro) and the pooled counts (micro).
  Bio-ML 2026 global matching: the same P/R/F against the repaired reference; only its train and valid parts are
      public (the test part is kept by the organisers), so recall here is over the public references and precision is a
      lower bound (a correct mapping of a test source counts as wrong). alignment.rdf is the submission file.
  Coherence (Pellet/ELK) is not computed here ('N/A').
Our alignment A: every class gets C# or NONE; matches with confidence >= 0.5 (no confidence = 1.0) are kept, then a
greedy 1:1 extraction by confidence (oa_rlaif.ontology.alignment_io.select_equivalences), as in the earlier run.

Two views of every number:
  full     every reference correspondence, as OAEI scores it.
  heldout  the training links of our 80/20 split are set to null (neither right nor wrong), as Bio-ML's
           semi-supervised setting does with its training mappings; zero-shot pairs are entirely held out.
  Only the held-out view is free of training data: the models were trained on 80% of these references, which OAEI
  rules do not allow for an official entry. ICD-10 -> ICD-11: held-out = the held-out 20% of ICD-10 categories.
SNOMED CT -> ICD-11 has no gold: agreement with the links composed through WHO ICD-10 (SNOMED -> ICD-10 map, then the
WHO ICD-10 -> ICD-11 table), the same silver standard as the earlier 'bridge recovery' column, now over every concept.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

from oa_rlaif.eval.metrics import AlignmentEvaluator
from oa_rlaif.harness.prompts import fully_correct, parse_decision
from oa_rlaif.ontology.alignment_io import SSSOM_PREDICATE, select_equivalences, write_oaei_rdf, write_sssom
from oa_rlaif.ontology.models import Alignment

from .build_episodes import read_shard
from .schedule import LABELS, root
from .tasks import Task, is_meta, load_profile_any, reference_pairs, split_of, tasks, zero_shot_pairs

THRESHOLD = 0.5
TOOL = "oa-rlaif-e4b"


def card_field(card: str, name: str) -> str:
    return next((line[len(name) + 2:] for line in card.splitlines() if line.startswith(name + ": ")), "")


def load_unit(model: str, task: Task):
    """Every episode of the task with its reply, in shard order."""
    folder = root() / "episodes" / task.slug
    shards = sorted(json.loads((folder / "manifest.json").read_text())["shards"])
    replies_dir = root() / "replies" / model / task.slug
    for name in shards:
        with gzip.open(replies_dir / name.replace(".jsonl.gz", ".replies.jsonl.gz"), "rt", encoding="utf-8") as f:
            replies = {r["id"]: r for r in map(json.loads, f)}
        for e in read_shard(folder / name):
            yield e, replies[e["id"]]["reply"]


def decide(episode: dict, reply: str) -> dict:
    d = parse_decision(reply, episode)
    cards = {c["iri"]: c["card"] for c in episode["candidates"]}
    targets = [c for c in (d.choices or ()) if c != "NONE"] if d.valid else []
    return {"pair_id": episode["pair_id"], "source_iri": episode["source_iri"],
            "source_label": card_field(episode["source_card"], "name"),
            "decision": "invalid" if not d.valid else "NONE" if d.choice == "NONE" else "match",
            "targets": targets, "target_labels": [card_field(cards.get(t, ""), "name") for t in targets],
            "target_codes": [card_field(cards.get(t, ""), "codes") for t in targets],
            "relation": d.relation if d.valid else "", "confidence": d.confidence,
            "thought": reply.split("\n", 1)[0][:300], "error": d.error,
            "candidates": [c["iri"] for c in episode["candidates"]]}


def picks_of(rec: dict) -> list[Alignment]:
    if rec["decision"] != "match":
        return []
    confidence = rec["confidence"] if rec["confidence"] is not None else 1.0
    if confidence < THRESHOLD:
        return []
    relation = {"classified_under": "<"}.get(rec["relation"], rec["relation"] or "=")
    return [Alignment(rec["source_iri"], t, relation, confidence, TOOL) for t in rec["targets"]]


def write_decisions(path: Path, records: list[dict], kept: set) -> None:
    columns = ["pair_id", "source_iri", "source_label", "decision", "target_iri", "target_label", "relation",
               "confidence", "in_alignment", "thought", "error"]
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for r in records:
            for t, label in zip(r["targets"] or [""], r["target_labels"] or [""]):
                row = [r["pair_id"], r["source_iri"], r["source_label"], r["decision"], t, label, r["relation"],
                       "" if r["confidence"] is None else r["confidence"],
                       "yes" if (r["source_iri"], t) in kept else "no", r["thought"], r["error"]]
                f.write("\t".join(str(x).replace("\t", " ").replace("\n", " ") for x in row) + "\n")


def scores_dict(s: dict) -> dict:
    return {k: s[k] for k in ("Size", "Precision", "Recall", "F-Measure", "Recall+", "true_positive", "reference",
                              "predicted_flagged")}


def aggregate(rows: list[dict]) -> dict:
    """Mean over pairs (macro) and pooled counts (micro)."""
    if not rows:
        return {}
    macro = {k: sum(r[k] for r in rows) / len(rows) for k in ("Precision", "Recall", "F-Measure")}
    tp = sum(r["true_positive"] for r in rows)
    pred = sum(r["Size"] - r["predicted_flagged"] for r in rows)
    ref = sum(r["reference"] for r in rows)
    p, r = (tp / pred if pred else 0.0), (tp / ref if ref else 0.0)
    plus = [x["Recall+"] for x in rows if x["Recall+"] is not None]
    return {"pairs": len(rows), "macro": macro, "micro": {"Precision": p, "Recall": r, "F-Measure": 2 * p * r / (p + r) if p + r else 0.0},
            "macro_recall_plus": sum(plus) / len(plus) if plus else None}


# ---- ontology pairs ------------------------------------------------------------------------------------------------
class PairScorer:
    """OAEI scoring of a predicted set of equivalences for one pair, in both views (full and held-out)."""

    def __init__(self, spec):
        from oa_rlaif.eval.align import string_equiv
        from oa_rlaif.ontology.alignment_io import read_alignment

        self.spec = spec
        self.source, self.target = load_profile_any(spec.source), load_profile_any(spec.target)
        self.reference = {p for p in reference_pairs(spec, self.source, self.target) if not is_meta(p[0]) and not is_meta(p[1])}
        self.string_equiv = {p for p in string_equiv(self.source, self.target) if not is_meta(p[0]) and not is_meta(p[1])}
        if spec.trivial and spec.trivial.exists():
            self.trivial = {a.pair for a in read_alignment(spec.trivial) if a.atomic} & self.reference
        else:
            self.trivial = self.string_equiv & self.reference
        self.zero_shot = spec.pair_id in zero_shot_pairs()
        splits = split_of()
        self.flagged = set() if self.zero_shot else {p for p in self.reference if splits.get(p[0]) == "train"}

    def __call__(self, predicted: set) -> dict:
        predicted = {p for p in predicted if not is_meta(p[0]) and not is_meta(p[1])}
        evaluator = AlignmentEvaluator()
        return {"zero_shot": self.zero_shot,
                "full": scores_dict(evaluator.evaluate(predicted, self.reference, trivial=self.trivial)),
                "heldout": scores_dict(evaluator.evaluate(predicted, self.reference, trivial=self.trivial, flagged=self.flagged))}


def equivalences(alignment) -> set:
    return {a.pair for a in alignment if a.relation == "="}


def summarize(pairs: dict) -> dict:
    family_rows = [p for p in pairs.values() if not p["zero_shot"]]
    zero_rows = [p for p in pairs.values() if p["zero_shot"]]
    return {"pairs": pairs,
            "aggregate": {"full": aggregate([p["full"] for p in family_rows]),
                          "heldout": aggregate([p["heldout"] for p in family_rows])},
            "zero_shot": {"full": aggregate([p["full"] for p in zero_rows])} if zero_rows else {}}


def score_pairs(model: str, task: Task, by_pair: dict[str, list[dict]], out: Path) -> dict:
    pairs = {}
    for spec in sorted(task.specs, key=lambda s: s.pair_id):
        records = by_pair.get(spec.pair_id, [])
        alignment = select_equivalences([a for r in records for a in picks_of(r)])
        scorer = PairScorer(spec)
        aligned = {r["source_iri"] for r in records}
        pairs[spec.pair_id] = {
            **scorer(equivalences(alignment)), "sources": len(records),
            "decisions": {k: sum(r["decision"] == k for r in records) for k in ("match", "NONE", "invalid")},
            "reference_sources_not_aligned": len({s for s, _ in scorer.reference} - aligned)}
        folder = out / "pairs" / spec.pair_id.replace("/", "__")
        write_oaei_rdf(folder / "alignment.rdf", alignment)
        labels = {r["source_iri"]: r["source_label"] for r in records}
        tlabels = {t: lab for r in records for t, lab in zip(r["targets"], r["target_labels"])}
        write_sssom(folder / "alignment.sssom.tsv", ({
            "subject_id": a.source_iri, "subject_label": labels.get(a.source_iri, ""),
            "predicate_id": SSSOM_PREDICATE.get(a.relation, "skos:relatedMatch"), "object_id": a.target_iri,
            "object_label": tlabels.get(a.target_iri, ""), "mapping_justification": "semapv:SemanticSimilarityThresholdMatching",
            "confidence": round(a.confidence, 4), "mapping_tool": f"{TOOL} ({LABELS.get(model, model)})", "comment": ""}
            for a in alignment), {"mapping_set_id": f"https://github.com/AGR19/OA-RLAIF/full_eval/{model}/{spec.pair_id}",
                                  "mapping_tool": TOOL, "subject_source": spec.source.name, "object_source": spec.target.name})
    return summarize(pairs)


# ---- ICD tasks -----------------------------------------------------------------------------------------------------
def score_icd10(task: Task, records: list[dict], out: Path) -> dict:
    """Accuracy as in the earlier Table 1 (exact answer among the candidates; NONE right when no answer was
    retrieved) and P/R/F of (ICD-10 code, ICD-11 category) pairs against the WHO table."""
    gold = json.loads((root() / "episodes" / task.slug / "gold.json").read_text())
    out_rows = {}
    for view in ("full", "heldout"):
        rows = [r for r in records if view == "full" or gold[r["source_iri"]]["split"] == "test"]
        correct = 0
        predicted, reference = set(), set()
        for r in rows:
            targets = gold[r["source_iri"]]["targets"]
            present = [t for t in targets if t in r["candidates"]]
            if r["decision"] == "NONE":
                correct += not present
            elif r["decision"] == "match":
                correct += fully_correct(r["targets"], present)
                predicted.update((r["source_iri"], t) for t in r["targets"])
            reference.update((r["source_iri"], t) for t in targets)
        s = AlignmentEvaluator().evaluate(predicted, reference)
        out_rows[view] = {"codes": len(rows), "accuracy": correct / max(1, len(rows)),
                          "retrieval_ceiling": sum(any(t in r["candidates"] for t in gold[r["source_iri"]]["targets"])
                                                   for r in rows) / max(1, len(rows)), **scores_dict(s)}
    alignment = [a for r in records for a in picks_of(r)]
    write_oaei_rdf(out / "alignment.rdf", alignment)
    return {"pairs": {task.pair_ids[0]: out_rows}, "aggregate": out_rows}


def score_snomed_icd11(model: str, task: Task, records: list[dict], out: Path) -> dict:
    from oa_rlaif.io import read_jsonl
    from oa_rlaif.paths import data_dir

    strong, loose, test = defaultdict(set), defaultdict(set), set()
    for link in read_jsonl(data_dir() / "icd" / "triangulated.jsonl"):
        loose[link["snomed"]].add(link["icd11"])
        if link["leg_a"]["strong"] and link["leg_b"].startswith("strong"):
            strong[link["snomed"]].add(link["icd11"])
            if link["split"] == "test":
                test.add(link["snomed"])

    def agreement(links: dict, keep) -> dict:
        rows = [r for r in records if r["source_iri"] in links and keep(r)]
        hits = sum(bool(set(r["targets"]) & links[r["source_iri"]]) for r in rows)
        ceiling = sum(bool(set(r["candidates"]) & links[r["source_iri"]]) for r in rows)
        return {"concepts": len(rows), "agreement": hits / max(1, len(rows)), "retrieval_ceiling": ceiling / max(1, len(rows))}

    n = max(1, len(records))
    result = {"concepts": len(records),
              "mapped": sum(r["decision"] == "match" for r in records) / n,
              "none": sum(r["decision"] == "NONE" for r in records) / n,
              "invalid": sum(r["decision"] == "invalid" for r in records) / n,
              "strong_links": agreement(strong, lambda r: True),
              "strong_links_heldout": agreement(strong, lambda r: r["source_iri"] in test),
              "any_composed_link": agreement(loose, lambda r: True)}
    agrees = lambda r: ("strong" if set(r["targets"]) & strong.get(r["source_iri"], set()) else  # noqa: E731
                        "composed" if set(r["targets"]) & loose.get(r["source_iri"], set()) else
                        "no composed link" if r["source_iri"] not in loose else "differs")
    write_sssom(out / "icd11_snomed_complete.sssom.tsv", ({
        "subject_id": t, "subject_label": label, "predicate_id": "skos:narrowMatch", "object_id": r["source_iri"],
        "object_label": r["source_label"], "mapping_justification": "semapv:UnspecifiedMatching",
        "confidence": r["confidence"] if r["confidence"] is not None else 1.0,
        "mapping_tool": f"{TOOL} ({LABELS.get(model, model)})", "comment": f"composed-link check: {agrees(r)}"}
        for r in records if r["decision"] == "match" for t, label in zip(r["targets"], r["target_labels"])),
        {"mapping_set_id": f"https://github.com/AGR19/OA-RLAIF/full_eval/{model}/icd11_snomed_complete",
         "license": "SNOMED CT affiliate licence applies; private use only",
         "subject_source": "ICD-11 (WHO 2026-01 tables)", "object_source": "SNOMED CT US 2026-03",
         "comment": "Every SNOMED CT concept in ICD's scope, classified by the model: ICD-11 category skos:narrowMatch "
                    "SNOMED concept means the category classifies the concept (not clinical equivalence)."})
    with gzip.open(out / "snomed_to_icd11.tsv.gz", "wt", encoding="utf-8") as f:
        f.write("snomed_iri\tsnomed_term\tdecision\ticd11_uri\ticd11_code\ticd11_title\tconfidence\tcomposed_link_check\n")
        for r in records:
            for t, label, code in zip(r["targets"] or [""], r["target_labels"] or [""], r["target_codes"] or [""]):
                f.write("\t".join(str(x).replace("\t", " ") for x in (r["source_iri"], r["source_label"], r["decision"], t, code,
                                                                   label, "" if r["confidence"] is None else r["confidence"],
                                                                   agrees(r) if r["decision"] == "match" else "")) + "\n")
    return {"pairs": {task.pair_ids[0]: result}, "aggregate": result}


# ---- one unit ------------------------------------------------------------------------------------------------------
def score(model: str, slug: str) -> dict:
    task = next(t for t in tasks() if t.slug == slug)
    out = root() / "results" / model / slug
    out.mkdir(parents=True, exist_ok=True)
    records = [decide(e, reply) for e, reply in load_unit(model, task)]
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_pair[r["pair_id"]].append(r)
    if task.family == "icd10":
        result = score_icd10(task, records, out)
    elif task.family == "snomed_icd11":
        result = score_snomed_icd11(model, task, records, out)
    else:
        result = score_pairs(model, task, by_pair, out)
    kept = {(a.source_iri, a.target_iri) for r in records for a in picks_of(r)}
    write_decisions(out / "decisions.tsv.gz", records, kept)
    timing = unit_timing(model, slug)
    result.update(model=model, label=LABELS.get(model, model), task=slug, family=task.family, column=task.column,
                  decisions=len(records), invalid=sum(r["decision"] == "invalid" for r in records), **timing)
    (out / "scores.json").write_text(json.dumps(result, indent=1, default=str))
    return result


def unit_timing(model: str, slug: str) -> dict:
    """Inference wall time of the unit over all workers (first shard start to last shard end) and tokens."""
    starts, ends, busy, prompt, completion = [], [], 0.0, 0, 0
    for path in (root() / "progress").glob(f"{model}.*.json"):
        t = json.loads(path.read_text()).get("tasks", {}).get(slug)
        if t and t["start"]:
            starts.append(t["start"])
            ends.append(t["end"])
            busy += t["busy_seconds"]
            prompt += t["prompt_tokens"]
            completion += t["completion_tokens"]
    return {"runtime_s": round(max(ends) - min(starts)) if starts else None, "gpu_busy_s": round(busy),
            "prompt_tokens": prompt, "completion_tokens": completion}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="full_eval.score")
    parser.add_argument("model")
    parser.add_argument("task")
    args = parser.parse_args(argv)
    result = score(args.model, args.task)
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

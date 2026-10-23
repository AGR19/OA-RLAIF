"""Score one unit (method, task) and write its complete outputs.

    python -m full_eval.score grpo anatomy__mouse-human
    python -m full_eval.score all_merged-agent-think dh

No held-out view: no reference link is set aside or treated as null, and every score uses the whole reference. (The
models were trained on part of these references, the 80/20 split of the earlier run; the report says so. IoTSec -> UCO
and the zero-shot pairs were never trained on.) FULL_EVAL_SCOPE=reference (the default) runs the classes the Top-k
evaluation scores and computes only (2), the retrieval ceiling and (3); SNOMED CT -> ICD-11 keeps its complete map and
silver agreement. FULL_EVAL_SCOPE=all runs every class and adds (1), which needs the whole alignment.

1. OAEI scores of the whole alignment (the published protocol, as MELT computes it): P = |A∩R|/|A|, R = |A∩R|/|R|,
   F = 2PR/(P+R). A = the system's alignment: matches with confidence >= 0.5 (none given = 1.0), then a greedy 1:1
   extraction by confidence (oa_rlaif.ontology.alignment_io.select_equivalences). R = the whole reference. Only
   equivalence correspondences count (OAEI Anatomy and DH say so); vocabulary correspondences such as
   oboInOwl#Synonym = oboInOwl#Synonym are removed. A reference without any equivalence (IoTSec -> UCO: every link is
   'related') is used with all its links, and A then keeps every relation. Reference links that cite classes of
   neither ontology (FLOPO -> PTO: 14 PATO and PO classes) stay in R, as MELT reads the file. Recall+ = recall on the
   non-trivial references: OAEI's Anatomy measure (its trivial.rdf); no other track defines it ("–").
   Datasets with several pairs: the mean over pairs (macro) and the pooled counts (micro). Bio-ML: only its train and
   valid references are public (the organisers keep the test part), so recall is over the public references and
   precision is a lower bound.
2. The Top-k evaluation (k = 1, 3, 5): Precision@k, Recall@k and F-measure@k of DRAL-OA's clf.py
   (oa_rlaif.eval.metrics.AlignmentEvaluator.topk). Only the source classes that have a partner in the reference are
   scored; each class's ranked answers are the method's own choice first, then its other candidates in retrieval
   order (the episode's `ranking`); NONE or an invalid reply = nothing found. Per class c, with F_c its top k and R_c
   its reference partners: Precision@k = |F_c ∩ R_c| / min(k, |R_c|), Recall@k = |F_c ∩ R_c| / |R_c|, averaged over the
   scored classes; F-measure@k = 2PR / (P + R). Every reference link counts, whatever its relation (as in clf.py).
ICD-10 -> ICD-11 (reference: the WHO 10To11 tables): accuracy as in the earlier Table 1 (the exact answer among the
candidates; NONE is right when no answer was retrieved), P/R/F of the (code, category) pairs, and Top-k.
SNOMED CT -> ICD-11 has no reference: agreement with the links composed through WHO ICD-10 (SNOMED -> ICD-10 map, then
the WHO table), a silver standard, labelled so; Top-k against the strong composed links.
3. Behaviour of the method: decisions, NONE, invalid replies, tokens per class; thinking on: thinking text per class,
   answers written inside the thinking, replies cut at the token limit; agent: model calls, extra commands, format
   errors, exit statuses and seconds per class.

Outputs per unit (results/<task>/<method folder>/, e.g. results/anatomy__mouse-human/4-opsd-grpo/): scores.json, decisions.tsv.gz (every class run: decision, target, label,
confidence, THOUGHT), the alignment of the classes that ran (pairs/<pair>/alignment.rdf in OAEI format and
alignment.sssom.tsv: matches with confidence >= 0.5, greedy 1:1; ICD-10: alignment.rdf of every WHO code); SNOMED CT ->
ICD-11: the complete map (icd11_snomed_complete.sssom.tsv, snomed_to_icd11.tsv.gz).
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from oa_rlaif.eval.metrics import TOP_K, TOP_K_NAME, AlignmentEvaluator
from oa_rlaif.harness.prompts import fully_correct, parse_decision
from oa_rlaif.ontology.alignment_io import SSSOM_PREDICATE, select_equivalences, write_oaei_rdf, write_sssom
from oa_rlaif.ontology.models import Alignment

from .build_episodes import read_shard
from .schedule import label, method, results_dir, root
from .tasks import (Task, eval_scope, is_meta, load_profile_any, reference_classes, reference_pairs, tasks,
                    zero_shot_pairs)

THRESHOLD = 0.5
TOOL = "oa-rlaif-e4b"


def card_field(card: str, name: str) -> str:
    return next((line[len(name) + 2:] for line in card.splitlines() if line.startswith(name + ": ")), "")


def load_unit(tag: str, task: Task):
    """Every episode of the task with its reply row, in shard order."""
    folder = root() / "episodes" / task.slug
    shards = sorted(json.loads((folder / "manifest.json").read_text())["shards"])
    replies_dir = root() / "replies" / tag / task.slug
    for name in shards:
        with gzip.open(replies_dir / name.replace(".jsonl.gz", ".replies.jsonl.gz"), "rt", encoding="utf-8") as f:
            replies = {r["id"]: r for r in map(json.loads, f)}
        for e in read_shard(folder / name):
            yield e, replies[e["id"]]


def decide(episode: dict, row: dict) -> dict:
    """One class's decision, from a one-decision reply (C# or NONE) or from the agent's `om submit`."""
    cards = {c["iri"]: c["card"] for c in episode["candidates"]}
    stats = {k: row[k] for k in ("prompt_tokens", "completion_tokens", "finish_reason", "model_calls", "extra_commands",
                                 "format_errors", "exit", "seconds") if k in row}
    if "turns" in row:  # agent
        decision = {"match": "match", "NONE": "NONE"}.get(row["decision"], "invalid")
        targets = list(row["targets"]) if decision == "match" else []
        relation, confidence = row.get("relation") or "", row.get("confidence")
        thought, error = row.get("thought", ""), "" if decision != "invalid" else row.get("exit") or ""
        stats.update(thinking_chars=row.get("thinking_chars", 0), answer_in_thinking=row.get("answer_in_thinking", 0) > 0,
                     truncated=any(t.get("finish_reason") == "length" for t in row["turns"]))
    else:
        d = parse_decision(row["reply"], episode)
        decision = "invalid" if not d.valid else "NONE" if d.choice == "NONE" else "match"
        targets = [c for c in (d.choices or ()) if c != "NONE"] if d.valid else []
        relation, confidence = (d.relation if d.valid else ""), d.confidence
        thought, error = row["reply"].split("\n", 1)[0][:300], d.error
        if "finish_reason" in row:  # recorded for every reply since 2026-10-22 (thinking on: before that too)
            stats["truncated"] = row["finish_reason"] == "length"
        if "thinking" in row:
            stats.update(thinking_chars=len(row["thinking"]), answer_in_thinking=bool(row.get("answer_in_thinking")))
    return {"pair_id": episode["pair_id"], "source_iri": episode["source_iri"],
            "source_label": card_field(episode["source_card"], "name"), "decision": decision, "targets": targets,
            "target_labels": [card_field(cards.get(t, ""), "name") for t in targets],
            "target_codes": [card_field(cards.get(t, ""), "codes") for t in targets],
            "relation": relation, "confidence": confidence, "thought": thought, "error": error,
            "candidates": [c["iri"] for c in episode["candidates"]],
            "ranking": episode.get("ranking") or [c["iri"] for c in episode["candidates"]], "stats": stats}


def ranked_answers(rec: dict) -> list[str]:
    """The method's own choice first, then its other candidates in retrieval order; NONE / invalid = nothing."""
    if rec["decision"] != "match":
        return []
    first = rec["targets"][:1]
    return first + [c for c in rec["ranking"] if c not in first]


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
            for t, lab in zip(r["targets"] or [""], r["target_labels"] or [""]):
                row = [r["pair_id"], r["source_iri"], r["source_label"], r["decision"], t, lab, r["relation"],
                       "" if r["confidence"] is None else r["confidence"],
                       "yes" if (r["source_iri"], t) in kept else "no", r["thought"], r["error"]]
                f.write("\t".join(str(x).replace("\t", " ").replace("\n", " ") for x in row) + "\n")


# ---- aggregation -----------------------------------------------------------------------------------------------------
def oaei_dict(s: dict) -> dict:
    return {k: s[k] for k in ("Size", "Precision", "Recall", "F-Measure", "Recall+", "true_positive", "reference")}


def oaei_aggregate(rows: list[dict]) -> dict:
    """Mean over pairs (macro) and pooled counts (micro)."""
    if not rows:
        return {}
    macro = {k: sum(r[k] for r in rows) / len(rows) for k in ("Precision", "Recall", "F-Measure")}
    tp, pred, ref = (sum(r[k] for r in rows) for k in ("true_positive", "Size", "reference"))
    p, r = (tp / pred if pred else 0.0), (tp / ref if ref else 0.0)
    plus = [x["Recall+"] for x in rows if x["Recall+"] is not None]
    return {"pairs": len(rows), "macro": macro,
            "micro": {"Precision": p, "Recall": r, "F-Measure": 2 * p * r / (p + r) if p + r else 0.0},
            "macro_recall_plus": sum(plus) / len(plus) if plus else None}


def topk_aggregate(rows: list[dict]) -> dict:
    """Mean over pairs of Precision, Recall and F-measure at each k (pairs with no scored class are left out)."""
    rows = [r for r in rows if r.get("classes_scored")]
    if not rows:
        return {}
    return {"method": TOP_K_NAME, "pairs": len(rows), "classes_scored": sum(r["classes_scored"] for r in rows),
            "k": {k: {m: sum(r["k"][k][m] for r in rows) / len(rows) for m in ("Precision", "Recall", "F-measure")}
                  for k in TOP_K}}


def topk_of(records: list[dict], partners: dict) -> dict:
    return AlignmentEvaluator().topk({r["source_iri"]: ranked_answers(r) for r in records}, partners)


def summarize(pairs: dict) -> dict:
    """Dataset rows: zero-shot pairs inside a family task (e.g. FLOPO -> PTO in Biodiv) count only in zero_shot."""
    family = [p for p in pairs.values() if not p["zero_shot"]]
    zero = [p for p in pairs.values() if p["zero_shot"]]
    oaei = lambda rows: oaei_aggregate([p["oaei"] for p in rows if "oaei" in p])  # noqa: E731  (scope 'all' only)
    return {"pairs": pairs, "oaei": oaei(family), "topk": topk_aggregate([p["topk"] for p in family]),
            "zero_shot": {"oaei": oaei(zero), "topk": topk_aggregate([p["topk"] for p in zero])} if zero else {}}


def behaviour(records: list[dict], tag: str) -> dict:
    """What the method did, per class."""
    m, n = method(tag), max(1, len(records))
    stats = [r["stats"] for r in records]

    def mean(key):
        values = [s[key] for s in stats if s.get(key) is not None]
        return statistics.fmean(values) if values else None

    out = {"classes": len(records), "match": sum(r["decision"] == "match" for r in records) / n,
           "none": sum(r["decision"] == "NONE" for r in records) / n,
           "invalid": sum(r["decision"] == "invalid" for r in records) / n,
           "prompt_tokens_per_class": mean("prompt_tokens"), "completion_tokens_per_class": mean("completion_tokens")}
    if any("truncated" in s for s in stats):  # replies cut at the token limit (counted as invalid when unparsable)
        out["truncated"] = sum(bool(s.get("truncated")) for s in stats) / n
    if m.think:
        out.update(thinking_chars_per_class=mean("thinking_chars"),
                   answer_in_thinking=sum(bool(s.get("answer_in_thinking")) for s in stats) / n,
                   no_thinking=sum(not s.get("thinking_chars") for s in stats) / n)
    if m.agent:
        out.update(model_calls_per_class=mean("model_calls"), extra_commands_per_class=mean("extra_commands"),
                   classes_with_extra_commands=sum((s.get("extra_commands") or 0) > 0 for s in stats) / n,
                   format_errors_per_class=mean("format_errors"), seconds_per_class=mean("seconds"),
                   exit_statuses=dict(Counter(str(s.get("exit")) for s in stats).most_common()))
    return out


# ---- ontology pairs ----------------------------------------------------------------------------------------------------
class PairScorer:
    """One pair's reference: the Top-k partners, and (oaei=True) the OAEI scores of the whole alignment."""

    def __init__(self, spec, oaei: bool = True):
        from oa_rlaif.ontology.alignment_io import read_alignment

        self.spec = spec
        self.source, self.target = load_profile_any(spec.source), load_profile_any(spec.target)
        self.partner_map = reference_classes(spec, self.source, self.target)  # Top-k: the classes the episodes run
        self.all_links = {(s, t) for s, ts in self.partner_map.items() for t in ts}
        self.zero_shot = spec.pair_id in zero_shot_pairs()
        if not oaei:
            return
        keep = lambda pairs: {p for p in pairs if not is_meta(p[0]) and not is_meta(p[1])}  # noqa: E731
        whole = keep(reference_pairs(spec, self.source, self.target, keep_unresolved=True))
        equivalences = keep(reference_pairs(spec, self.source, self.target, equivalence_only=True, keep_unresolved=True))
        self.any_relation = not equivalences and bool(whole)  # e.g. IoTSec -> UCO: every link 'related'
        self.reference = whole if self.any_relation else equivalences  # OAEI: the whole reference, as MELT reads it
        self.unresolved = len(self.reference - self.all_links)
        # Recall+ is OAEI's Anatomy measure, defined by its trivial.rdf; no other track defines it, so it is "–" there
        self.trivial = ({a.pair for a in read_alignment(spec.trivial) if a.atomic} & self.reference
                        if spec.trivial and spec.trivial.exists() else None)

    def partners(self) -> dict:
        return self.partner_map

    def oaei(self, alignment) -> dict:
        predicted = {a.pair for a in alignment if self.any_relation or a.relation == "="}
        predicted = {p for p in predicted if not is_meta(p[0]) and not is_meta(p[1])}
        return oaei_dict(AlignmentEvaluator().evaluate(predicted, self.reference, trivial=self.trivial))


def score_pairs(tag: str, task: Task, by_pair: dict[str, list[dict]], out: Path) -> dict:
    whole = eval_scope() == "all"  # the OAEI scores and alignment files need every class aligned
    pairs = {}
    for spec in sorted(task.specs, key=lambda s: s.pair_id):
        records = by_pair.get(spec.pair_id, [])
        scorer = PairScorer(spec, oaei=whole)
        partners = scorer.partners()
        aligned = {r["source_iri"] for r in records}
        pairs[spec.pair_id] = {
            "zero_shot": scorer.zero_shot, "sources": len(records), "reference_links_scored": len(scorer.all_links),
            "reference_classes": len(partners), "reference_classes_not_aligned": len(set(partners) - aligned),
            # share of reference classes whose partner was among the candidates at all: no method can do better
            "retrieval_ceiling": sum(bool(set(r["candidates"]) & partners[r["source_iri"]]) for r in records
                                     if r["source_iri"] in partners) / max(1, len(partners)),
            "decisions": {k: sum(r["decision"] == k for r in records) for k in ("match", "NONE", "invalid")},
            "topk": topk_of(records, partners)}
        # the alignment file holds the classes that ran (scope 'reference': the reference classes, so it is not a
        # whole-ontology alignment and gets no OAEI score)
        alignment = select_equivalences([a for r in records for a in picks_of(r)])
        pairs[spec.pair_id]["alignment_size"] = len(alignment)
        if whole:
            pairs[spec.pair_id].update({
                "reference_links": len(scorer.reference), "reference_links_outside_the_ontologies": scorer.unresolved,
                "reference_relations": "all (no equivalences in the reference)" if scorer.any_relation else "equivalence",
                "oaei": scorer.oaei(alignment)})
        folder = out / "pairs" / spec.pair_id.replace("/", "__")
        write_oaei_rdf(folder / "alignment.rdf", alignment)
        labels = {r["source_iri"]: r["source_label"] for r in records}
        tlabels = {t: lab for r in records for t, lab in zip(r["targets"], r["target_labels"])}
        write_sssom(folder / "alignment.sssom.tsv", ({
            "subject_id": a.source_iri, "subject_label": labels.get(a.source_iri, ""),
            "predicate_id": SSSOM_PREDICATE.get(a.relation, "skos:relatedMatch"), "object_id": a.target_iri,
            "object_label": tlabels.get(a.target_iri, ""), "mapping_justification": "semapv:SemanticSimilarityThresholdMatching",
            "confidence": round(a.confidence, 4), "mapping_tool": f"{TOOL} ({label(tag)})", "comment": ""}
            for a in alignment), {"mapping_set_id": f"https://github.com/AGR19/OA-RLAIF/full_eval/{tag}/{spec.pair_id}",
                                  "mapping_tool": TOOL, "subject_source": spec.source.name, "object_source": spec.target.name})
    return summarize(pairs)


# ---- ICD tasks -------------------------------------------------------------------------------------------------------
def score_icd10(task: Task, records: list[dict], out: Path) -> dict:
    """Reference: the WHO ICD-10 -> ICD-11 tables (gold.json, written with the episodes); every code."""
    gold = json.loads((root() / "episodes" / task.slug / "gold.json").read_text())
    correct, predicted, reference = 0, set(), set()
    for r in records:
        targets = gold[r["source_iri"]]["targets"]
        present = [t for t in targets if t in r["candidates"]]
        if r["decision"] == "NONE":
            correct += not present
        elif r["decision"] == "match":
            correct += fully_correct(r["targets"], present)
            predicted.update((r["source_iri"], t) for t in r["targets"])
        reference.update((r["source_iri"], t) for t in targets)
    n = max(1, len(records))
    partners = {iri: set(g["targets"]) for iri, g in gold.items() if g["targets"]}
    pair = {"zero_shot": False, "codes": len(records), "reference_classes": len(partners),
            "retrieval_ceiling": sum(any(t in r["candidates"] for t in gold[r["source_iri"]]["targets"])
                                     for r in records) / n, "topk": topk_of(records, partners)}
    result = {"pairs": {task.pair_ids[0]: pair}, "topk": topk_aggregate([pair["topk"]]), "zero_shot": {}}
    if eval_scope() == "all":  # the earlier Table 1 accuracy and the OAEI scores (scope 'all' only)
        pair.update(accuracy=correct / n, oaei=oaei_dict(AlignmentEvaluator().evaluate(predicted, reference)))
        result.update(accuracy=pair["accuracy"], oaei=oaei_aggregate([pair["oaei"]]))
    write_oaei_rdf(out / "alignment.rdf", [a for r in records for a in picks_of(r)])  # every WHO code, either scope
    return result


def score_snomed_icd11(tag: str, task: Task, records: list[dict], out: Path) -> dict:
    """No reference exists: the links composed through WHO ICD-10 are a silver standard (strong = both legs strong)."""
    from oa_rlaif.io import read_jsonl
    from oa_rlaif.paths import data_dir

    strong, loose = defaultdict(set), defaultdict(set)
    for link in read_jsonl(data_dir() / "icd" / "triangulated.jsonl"):
        loose[link["snomed"]].add(link["icd11"])
        if link["leg_a"]["strong"] and link["leg_b"].startswith("strong"):
            strong[link["snomed"]].add(link["icd11"])

    def agreement(links: dict) -> dict:
        rows = [r for r in records if r["source_iri"] in links]
        hits = sum(bool(set(r["targets"]) & links[r["source_iri"]]) for r in rows)
        ceiling = sum(bool(set(r["candidates"]) & links[r["source_iri"]]) for r in rows)
        return {"concepts": len(rows), "agreement": hits / max(1, len(rows)), "retrieval_ceiling": ceiling / max(1, len(rows))}

    in_run = {r["source_iri"] for r in records}
    n = max(1, len(records))
    pair = {"zero_shot": False, "silver": True, "concepts": len(records),
            "mapped": sum(r["decision"] == "match" for r in records) / n,
            "none": sum(r["decision"] == "NONE" for r in records) / n,
            "invalid": sum(r["decision"] == "invalid" for r in records) / n,
            "strong_links": agreement(strong), "any_composed_link": agreement(loose),
            "topk": topk_of(records, {s: v for s, v in strong.items() if s in in_run})}
    agrees = lambda r: ("strong" if set(r["targets"]) & strong.get(r["source_iri"], set()) else  # noqa: E731
                        "composed" if set(r["targets"]) & loose.get(r["source_iri"], set()) else
                        "no composed link" if r["source_iri"] not in loose else "differs")
    write_sssom(out / "icd11_snomed_complete.sssom.tsv", ({
        "subject_id": t, "subject_label": lab, "predicate_id": "skos:narrowMatch", "object_id": r["source_iri"],
        "object_label": r["source_label"], "mapping_justification": "semapv:UnspecifiedMatching",
        "confidence": r["confidence"] if r["confidence"] is not None else 1.0,
        "mapping_tool": f"{TOOL} ({label(tag)})", "comment": f"composed-link check: {agrees(r)}"}
        for r in records if r["decision"] == "match" for t, lab in zip(r["targets"], r["target_labels"])),
        {"mapping_set_id": f"https://github.com/AGR19/OA-RLAIF/full_eval/{tag}/icd11_snomed_complete",
         "license": "SNOMED CT affiliate licence applies; private use only",
         "subject_source": "ICD-11 (WHO 2026-01 tables)", "object_source": "SNOMED CT US 2026-03",
         "comment": "Every SNOMED CT concept in ICD's scope, classified by the model: ICD-11 category skos:narrowMatch "
                    "SNOMED concept means the category classifies the concept (not clinical equivalence)."})
    with gzip.open(out / "snomed_to_icd11.tsv.gz", "wt", encoding="utf-8") as f:
        f.write("snomed_iri\tsnomed_term\tdecision\ticd11_uri\ticd11_code\ticd11_title\tconfidence\tcomposed_link_check\n")
        for r in records:
            for t, lab, code in zip(r["targets"] or [""], r["target_labels"] or [""], r["target_codes"] or [""]):
                f.write("\t".join(str(x).replace("\t", " ") for x in (r["source_iri"], r["source_label"], r["decision"], t, code,
                                                                   lab, "" if r["confidence"] is None else r["confidence"],
                                                                   agrees(r) if r["decision"] == "match" else "")) + "\n")
    return {"pairs": {task.pair_ids[0]: pair}, "silver": pair, "topk": topk_aggregate([pair["topk"]]), "zero_shot": {}}


# ---- one unit --------------------------------------------------------------------------------------------------------
def score(tag: str, slug: str) -> dict:
    task = next(t for t in tasks() if t.slug == slug)
    out = results_dir(tag, slug)
    out.mkdir(parents=True, exist_ok=True)
    records = [decide(e, row) for e, row in load_unit(tag, task)]
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_pair[r["pair_id"]].append(r)
    if task.family == "icd10":
        result = score_icd10(task, records, out)
    elif task.family == "snomed_icd11":
        result = score_snomed_icd11(tag, task, records, out)
    else:
        result = score_pairs(tag, task, by_pair, out)
    kept = {(a.source_iri, a.target_iri) for r in records for a in picks_of(r)}
    write_decisions(out / "decisions.tsv.gz", records, kept)
    m = method(tag)
    view = "full alignment" if eval_scope() == "all" else "reference classes (Top-k evaluation)"
    result.update(model=tag, method=m.base, thinking=m.think, agent=m.agent, label=label(tag), task=slug,
                  family=task.family, column=task.column, view=view, scope=eval_scope(), decisions=len(records),
                  behaviour=behaviour(records, tag), **unit_timing(tag, slug))
    (out / "scores.json").write_text(json.dumps(result, indent=1, default=str))
    return result


def unit_timing(tag: str, slug: str) -> dict:
    """Inference wall time of the unit over all workers (first shard start to last shard end) and tokens."""
    starts, ends, busy, prompt, completion = [], [], 0.0, 0, 0
    for path in (root() / "progress").glob(f"{tag}.*.json"):
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
    parser.add_argument("method")
    parser.add_argument("task")
    args = parser.parse_args(argv)
    result = score(args.method, args.task)
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

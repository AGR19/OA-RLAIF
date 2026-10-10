"""Full ontology-to-ontology alignment with the trained model (any size: retrieval keeps the model's input small).

    python -m oa_rlaif.eval.full_align --model outputs/merged/grpo --source <src.owl> --target <tgt.owl> \
        --name anatomy --reference <reference.rdf>
    python -m oa_rlaif.eval.full_align --model outputs/merged/grpo --icd11-to-snomed --name icd11_snomed

Pipeline: parse -> top-k candidates per source class (q-gram retrieval) -> model decision (C# or NONE, with
confidence) -> keep matches above the threshold -> greedy 1:1 extraction by confidence for equivalences ->
SSSOM TSV + OAEI RDF. With --reference, P/R/F1 are computed (diagnostic only when the pair was trained on).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..data.cards import card, children_index, names
from ..harness.prompts import decision_messages, parse_decision
from ..ontology.alignment_io import SSSOM_PREDICATE, orient, read_alignment, select_equivalences, write_oaei_rdf, write_sssom
from ..ontology.models import Alignment
from ..ontology.parser import OntologyParser
from ..paths import out_dir
from ..progress import stage
from ..retrieval.candidates import CandidateGenerator
from .metrics import AlignmentEvaluator
from .run_eval import Generator


def episodes_for(source, target, k: int, relation: str = "=", pair_id: str = "full") -> list[dict]:
    generator = CandidateGenerator(target)
    kids_s, kids_t = children_index(source), children_index(target)
    declared = sorted(i for i, e in source.entities.items() if e.declared)
    out = []
    with stage("full_align_retrieval", total=len(declared)) as bar:
        for iri in declared:
            cands = [c for c, _ in generator.candidates(source.entities[iri], k=k)]
            out.append({"id": iri, "pair_id": pair_id, "relation": relation, "source_iri": iri,
                        "source_card": card(source, iri, children=kids_s), "gold": [],
                        "candidates": [{"iri": c, "card": card(target, c, children=kids_t)} for c in cands]})
            bar.advance()
    return out


def extract(episodes, replies, threshold: float, one_to_one: bool) -> list[Alignment]:
    picks = []
    for ep, reply in zip(episodes, replies):
        d = parse_decision(reply, ep)
        if d.valid and d.choice != "NONE" and (d.confidence or 0.0) >= threshold:
            for choice in d.choices or (d.choice,):
                picks.append(Alignment(ep["source_iri"], choice, d.relation if d.relation != "classified_under" else "<",
                                       d.confidence or 0.0, "oa-rlaif"))
    return select_equivalences(picks) if one_to_one else picks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--trivial", type=Path, help="OAEI trivial alignment for Recall+ (default: trivial.rdf next to the reference)")
    parser.add_argument("--name", required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()
    started = time.time()
    source = OntologyParser().parse(args.source)
    target = OntologyParser().parse(args.target)
    episodes = episodes_for(source, target, args.k)
    generator = Generator(args.model, 256, args.tp)
    replies = generator.chat([decision_messages(e) for e in episodes])
    alignments = extract(episodes, replies, args.threshold, one_to_one=True)
    out = out_dir() / "full_align" / args.name
    out.mkdir(parents=True, exist_ok=True)
    write_oaei_rdf(out / "alignment.rdf", alignments)
    label = lambda profile, iri: names(profile.entities[iri])[0] if iri in profile.entities else ""  # noqa: E731
    write_sssom(out / "alignment.sssom.tsv", ({
        "subject_id": a.source_iri, "subject_label": label(source, a.source_iri),
        "predicate_id": SSSOM_PREDICATE.get(a.relation, "skos:relatedMatch"), "object_id": a.target_iri,
        "object_label": label(target, a.target_iri), "mapping_justification": "semapv:SemanticSimilarityThresholdMatching",
        "confidence": round(a.confidence, 4), "mapping_tool": "oa-rlaif-e4b", "comment": ""} for a in alignments),
        {"mapping_set_id": f"https://github.com/AGR19/OA-RLAIF/{args.name}", "mapping_tool": "oa-rlaif-e4b"})
    runtime = round(time.time() - started)
    summary = {"sources": len(episodes), "alignments": len(alignments), "runtime_s": runtime}
    if args.reference:
        reference = {orient(a, source, target).pair for a in read_alignment(args.reference) if a.atomic}
        trivial_path = args.trivial or args.reference.parent / "trivial.rdf"
        trivial = ({orient(a, source, target).pair for a in read_alignment(trivial_path) if a.atomic} & reference
                   if trivial_path.exists() else None)
        summary["scores"] = AlignmentEvaluator().evaluate({a.pair for a in alignments}, reference, matcher="oa-rlaif-e4b",
                                                          runtime=runtime, trivial=trivial)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

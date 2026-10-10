"""Align two ontologies with the trained model and emit what you ask for (runs locally; no external API).

    python -m oa_rlaif.eval.align --source mouse.owl --target human.owl \
        --backend openai --api-base http://127.0.0.1:11434/v1 --model oa-rlaif-e4b \     # Ollama (GGUF)
        --format oaei --format sssom --explain --reference reference.rdf --out outputs/align/anatomy

    python -m oa_rlaif.eval.align --source icd11.owl --target snomed.owl --backend vllm --model outputs/merged/grpo \
        --format sssom --out outputs/align/icd11_snomed                                   # no reference: no P/R/F

Outputs (any combination):
  --format oaei   alignment.rdf   Alignment API RDF/XML (OAEI submission format)
  --format sssom  alignment.sssom.tsv   mappings with predicate, confidence, justification
  --format tsv    alignment.tsv   source, target, relation, confidence
  --explain       explanations.jsonl   the model's THOUGHT evidence per mapping
  --reference     results.md / results.csv / results.tex: the OAEI results row
                  Matcher | Runtime (s) | Size | Precision | F-Measure | Recall | Recall+ | Coherent
                  Recall+ uses a StringEquiv baseline (normalised label equality) computed here, as OAEI does.
Coherent (+/-) is computed with or without a reference: ELK on source + target + alignment (tools/owl_verify).
Without a reference (e.g. ICD-11 <-> SNOMED) the report gives Size, Runtime, Coherent and, if --silver is given,
agreement with silver links (e.g. data/icd/triangulated.jsonl) as a proxy, never as precision.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from ..data.cards import card, children_index, kind_of, names
from ..data.ontology_profiles import build_profile, summary_text
from ..harness.prompts import decision_messages, parse_decision
from ..ontology.alignment_io import SSSOM_PREDICATE, orient, read_alignment, select_equivalences, write_oaei_rdf, write_sssom
from ..ontology.models import Alignment
from ..ontology.parser import OntologyParser
from ..paths import ROOT
from ..progress import stage
from ..retrieval.candidates import CandidateGenerator, normalize
from .metrics import AlignmentEvaluator


class OpenAIBackend:
    """Any local OpenAI-compatible server: Ollama serving the GGUF, or `vllm serve`."""

    def __init__(self, api_base: str, model: str, concurrency: int = 16):
        from openai import OpenAI

        self.client = OpenAI(base_url=api_base, api_key=os.environ.get("LOCAL_MODEL_KEY", "local"))
        self.model, self.concurrency = model, concurrency

    def chat(self, conversations: list[list[dict]]) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor

        def one(messages):
            r = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.0, max_tokens=256)
            return r.choices[0].message.content or ""

        with ThreadPoolExecutor(self.concurrency) as pool:
            return list(pool.map(one, conversations))


class LexicalBackend:
    """Sanity baseline (not a model): always submits the top retrieved candidate (C1)."""

    def chat(self, conversations):
        return ["THOUGHT: top lexical candidate.\n```mswea_bash_command\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && "
                "om submit C1 --relation = --confidence 0.9\n```" for _ in conversations]


def make_backend(kind: str, model: str, api_base: str | None, tp: int):
    if kind == "lexical":
        return LexicalBackend()
    if kind == "vllm":
        from .run_eval import Generator
        return Generator(model, 256, tp)
    return OpenAIBackend(api_base or "http://127.0.0.1:11434/v1", model)


def string_equiv(source, target) -> set[tuple[str, str]]:
    """OAEI's StringEquiv baseline: normalised label equality (used for Recall+)."""
    index: dict[str, set[str]] = {}
    for iri, e in target.entities.items():
        if e.declared:
            for label in e.labels:
                index.setdefault(normalize(label.text), set()).add(iri)
    pairs = set()
    for iri, e in source.entities.items():
        if e.declared:
            for label in e.labels:
                for t in index.get(normalize(label.text), ()):
                    pairs.add((iri, t))
    return pairs


def coherence(source_path: Path, target_path: Path, alignments: list[Alignment], workdir: Path) -> dict:
    from ..data.parse_verify import owlapi_command

    command = owlapi_command()
    if command is None:
        return {"coherent": None, "note": "OWL API tool not built"}
    tsv = workdir / "coherence_input.tsv"
    tsv.write_text("source_iri\ttarget_iri\trelation\n" + "".join(f"{a.source_iri}\t{a.target_iri}\t{a.relation}\n" for a in alignments))
    args = [command[0], "-Xmx12g", *command[1:], "--coherence", str(source_path), str(target_path), str(tsv)]
    result = subprocess.run(args, capture_output=True, text=True, timeout=7200)
    line = next((x for x in result.stdout.splitlines() if x.startswith("{")), "")
    return json.loads(line) if line else {"coherent": None, "note": (result.stderr or "no output")[-300:]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--backend", choices=["openai", "vllm", "lexical"], default="openai")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--format", action="append", choices=["oaei", "sssom", "tsv"], default=None)
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--silver", type=Path, help="silver links (JSONL with snomed/icd11 fields) for proxy agreement")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--name", default="OA-RLAIF-E4B")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-coherence", action="store_true")
    args = parser.parse_args()
    formats = args.format or ["oaei", "sssom"]
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    source, target = OntologyParser().parse(args.source), OntologyParser().parse(args.target)
    profile_s = summary_text(build_profile_from(args.source))
    profile_t = summary_text(build_profile_from(args.target))
    generator = CandidateGenerator(target)
    kids_s, kids_t = children_index(source), children_index(target)
    declared = sorted(i for i, e in source.entities.items() if e.declared)
    episodes = []
    with stage("align_retrieval", total=len(declared)) as bar:
        for iri in declared:
            kind = kind_of(source, iri)
            ranked = [c for c, _ in generator.candidates(source.entities[iri], k=args.k * 4) if kind_of(target, c) == kind][: args.k]
            episodes.append({"id": iri, "pair_id": f"{args.source.stem}->{args.target.stem}", "relation": "=",
                             "source_iri": iri, "source_card": card(source, iri, children=kids_s), "gold": [],
                             "source_profile": profile_s, "target_profile": profile_t,
                             "candidates": [{"iri": c, "card": card(target, c, children=kids_t)} for c in ranked]})
            bar.advance()
    backend = make_backend(args.backend, args.model, args.api_base, args.tp)
    replies = backend.chat([decision_messages(e) for e in episodes])

    picks, explanations = [], []
    for ep, reply in zip(episodes, replies):
        d = parse_decision(reply, ep)
        if d.valid and d.choice != "NONE" and (d.confidence if d.confidence is not None else 1.0) >= args.threshold:
            relation = {"classified_under": "<"}.get(d.relation, d.relation)
            for choice in d.choices or (d.choice,):
                picks.append(Alignment(ep["source_iri"], choice, relation, d.confidence if d.confidence is not None else 1.0, "oa-rlaif"))
                explanations.append({"source": ep["source_iri"], "target": choice, "relation": relation,
                                     "confidence": d.confidence, "thought": reply.split("\n", 1)[0]})
    alignments = select_equivalences(picks)  # a target serves one source; multi-target sources keep all picks
    runtime = time.time() - started

    label = lambda p, i: names(p.entities[i])[0] if i in p.entities else ""  # noqa: E731
    if "oaei" in formats:
        write_oaei_rdf(args.out / "alignment.rdf", alignments)
    if "sssom" in formats:
        write_sssom(args.out / "alignment.sssom.tsv", ({
            "subject_id": a.source_iri, "subject_label": label(source, a.source_iri),
            "predicate_id": SSSOM_PREDICATE.get(a.relation, "skos:relatedMatch"), "object_id": a.target_iri,
            "object_label": label(target, a.target_iri), "mapping_justification": "semapv:LexicalMatching",
            "confidence": round(a.confidence, 4), "mapping_tool": args.name, "comment": ""} for a in alignments),
            {"mapping_set_id": f"https://github.com/AGR19/OA-RLAIF/{args.source.stem}-{args.target.stem}",
             "mapping_tool": args.name, "subject_source": args.source.name, "object_source": args.target.name})
    if "tsv" in formats:
        (args.out / "alignment.tsv").write_text("source_iri\ttarget_iri\trelation\tconfidence\n" + "".join(
            f"{a.source_iri}\t{a.target_iri}\t{a.relation}\t{a.confidence:.4f}\n" for a in alignments))
    if args.explain:
        keep = {(a.source_iri, a.target_iri) for a in alignments}
        (args.out / "explanations.jsonl").write_text("".join(
            json.dumps(e) + "\n" for e in explanations if (e["source"], e["target"]) in keep))

    report = {"matcher": args.name, "runtime_s": round(runtime), "size": len(alignments), "sources": len(declared)}
    report["coherence"] = {"coherent": None} if args.no_coherence else coherence(args.source, args.target, alignments, args.out)
    coherent = report["coherence"].get("coherent")
    if args.reference:
        reference = {orient(a, source, target).pair for a in read_alignment(args.reference) if a.atomic}
        trivial = string_equiv(source, target) & reference
        scores = AlignmentEvaluator().evaluate({a.pair for a in alignments}, reference, matcher=args.name,
                                               runtime=round(runtime), trivial=trivial, coherent=coherent)
        report["scores"] = scores
        header = ["Matcher", "Runtime", "Size", "Precision", "F-Measure", "Recall", "Recall+", "Coherent"]
        row = [args.name, str(round(runtime)), str(len(alignments)), f"{scores['Precision']:.3f}",
               f"{scores['F-Measure']:.3f}", f"{scores['Recall']:.3f}",
               "-" if scores["Recall+"] is None else f"{scores['Recall+']:.3f}", scores["Coherent"]]
        (args.out / "results.md").write_text("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n| " + " | ".join(row) + " |\n")
        (args.out / "results.csv").write_text(",".join(header) + "\n" + ",".join(row) + "\n")
        (args.out / "results.tex").write_text(" & ".join(header) + r" \\ \hline" + "\n" + " & ".join(row) + r" \\" + "\n")
    if args.silver and args.silver.exists():
        silver = set()
        for line in args.silver.read_text().splitlines():
            link = json.loads(line)
            silver.add((link["icd11"], link["snomed"]))
            silver.add((link["snomed"], link["icd11"]))
        predicted = {a.pair for a in alignments}
        report["silver_agreement"] = {"predicted_in_silver": len(predicted & silver), "predicted": len(predicted),
                                      "note": "proxy only: silver links are composed through ICD-10, not gold"}
    (args.out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    return 0


def build_profile_from(path: Path) -> dict:
    """Modelling profile for an arbitrary ontology file (cached profile if it is a known raw file)."""
    from ..data.profiles import build_profiles, is_cached

    if not is_cached(path):
        build_profiles([path])
    return build_profile(path)


if __name__ == "__main__":
    raise SystemExit(main())

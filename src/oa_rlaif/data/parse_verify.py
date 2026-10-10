"""Independent verification that every ontology and every reference alignment is parsed correctly.

    python -m oa_rlaif.data.parse_verify            # all ontologies + references
    python -m oa_rlaif.data.parse_verify --skip-owlapi   # reference checks only (no Java)

1. Ontology cross-parse (what OAEI/OBO practice uses): each file is loaded a second time with the OWL API
   (Java; the library behind Protégé, ROBOT, HermiT and MELT's OntologyValidationService) and compared with
   our Python parser: declared classes (or skos:Concept members for SKOS vocabularies) must match exactly;
   label counts are compared as a ratio. ELK checks consistency and unsatisfiable classes (files < 300 MB).
2. Reference quality checks after Beisswanger & Hahn (J Biomed Semantics 2012, doi:10.1186/2041-1480-3-S1-S4):
   standard format, URIs, explicit/valid relation types, measure in [0, 1], duplicates, conflicting relations,
   missing classes (entity not in the given ontology version), non-class endpoints, multiple targets,
   self-mappings, trivial (label-identical) correspondences, and label-identical pairs with no correspondence
   (possible reference incompleteness; reported, never auto-added to gold).
Writes outputs/coverage/parse_verification.json and docs/PARSE_VERIFICATION.md; data/verify.py gates on it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..ontology.alignment_io import orient, read_alignment
from ..paths import ROOT, out_dir, relative_to_raw
from ..progress import stage
from ..retrieval.candidates import normalize
from .families import discover, ontology_files
from .profiles import load_profile

RELATIONS = {"=", "<", ">", "<=", ">=", "%", "HasInstance", "InstanceOf", "related", "classified_under"}
SKOS_HINTS = ("DH-benchmark",)


def owlapi_command() -> list[str] | None:
    """Built jar (Lambda: mvn package) or classes compiled against a local shaded OWL API jar (Mac)."""
    shaded = ROOT / "tools/owl_verify/target/owl-verify-1.0.jar"
    if shaded.exists():
        return ["java", "-cp", str(shaded), "OwlVerify"]
    classes = ROOT / "tools/owl_verify/target/classes"
    lib = os.environ.get("OWLAPI_JAR") or str((ROOT / "../tools/reasoner/target/reasoner-1.0.jar").resolve())
    if (classes / "OwlVerify.class").exists() and Path(lib).exists():
        return ["java", "-cp", f"{classes.resolve()}:{lib}", "OwlVerify"]
    return None


def heap_for(size: int) -> str:
    return "2g" if size < 50e6 else "8g" if size < 300e6 else os.environ.get("OWLAPI_MAX_HEAP", "12g")


def owlapi_parse(command: list[str], path: Path) -> dict:
    size = path.stat().st_size
    args = [command[0], f"-Xmx{heap_for(size)}", *command[1:], str(path.resolve())]
    if size < 300e6:
        args.append("--elk")
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    line = next((x for x in result.stdout.splitlines() if x.startswith("{")), "")
    if not line:
        tail = (result.stderr or "").strip().splitlines()[-1:] or ["no output"]
        return {"ok": False, "error": tail[0][:300]}
    return json.loads(line)


def ontology_checks(files: list[Path], use_owlapi: bool, workers: int) -> list[dict]:
    command = owlapi_command() if use_owlapi else None
    rows = []
    small = [f for f in files if f.stat().st_size < 300e6]
    big = [f for f in files if f.stat().st_size >= 300e6]
    with stage("verify_ontologies", total=len(files)) as bar:
        def one(path: Path) -> dict:
            profile = load_profile(path)
            ours_classes = profile.metadata.get("classes", 0)
            ours_properties = profile.metadata.get("properties", 0)
            ours_labels = sum(len(e.labels) for e in profile.entities.values())
            row = {"path": relative_to_raw(path), "bytes": path.stat().st_size, "ours_classes": ours_classes,
                   "ours_properties": ours_properties,
                   "ours_labels": ours_labels}
            if command is None:
                return {**row, "owlapi": None, "status": "not_run"}
            owl = owlapi_parse(command, path)
            row["owlapi"] = owl
            if not owl.get("ok"):
                return {**row, "status": "owlapi_failed"}
            theirs = owl["classes_declared"] + owl.get("skos_concepts", 0)
            row["class_diff"] = ours_classes - theirs
            row["property_diff"] = ours_properties - owl.get("object_properties", 0) - owl.get("data_properties", 0)
            row["label_ratio"] = round(ours_labels / owl["label_assertions"], 3) if owl.get("label_assertions") else None
            row["status"] = "match" if row["class_diff"] == 0 and row["property_diff"] == 0 else "mismatch"
            return row

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, f) for f in small]
            for future in as_completed(futures):
                rows.append(future.result())
                bar.advance()
        for path in big:  # one at a time: large heaps
            rows.append(one(path))
            bar.advance()
    return rows


def reference_checks(sample_label_pairs: int = 200_000) -> list[dict]:
    pairs = [p for p in discover() if p.ref_kind in ("simple", "bioml", "complex") and p.source and p.target]
    out = []
    cache: dict[str, object] = {}

    def profile(path):
        key = str(path)
        if key not in cache:
            if len(cache) > 60:
                cache.clear()
            cache[key] = load_profile(path)
        return cache[key]

    with stage("verify_references", total=len(pairs)) as bar:
        for pair in pairs:
            source, target = profile(pair.source), profile(pair.target)
            for ref in pair.references:
                checks = Counter()
                relations = defaultdict(set)
                targets_of = defaultdict(set)
                seen = set()
                parse_error = ""
                try:
                    cells = list(read_alignment(ref))
                except Exception as error:  # an unreadable reference is a hard failure
                    cells, parse_error = [], f"{type(error).__name__}: {error}"
                for a in cells:
                    checks["cells"] += 1
                    if not a.atomic:
                        checks["complex_cells"] += 1
                        continue
                    if not (a.source_iri.startswith(("http://", "https://", "urn:")) and a.target_iri.startswith(("http://", "https://", "urn:"))):
                        checks["non_uri_entities"] += 1
                    if not a.relation:
                        checks["missing_relation"] += 1
                    elif a.relation not in RELATIONS:
                        checks["invalid_relation"] += 1
                    if not 0.0 <= a.confidence <= 1.0:
                        checks["measure_out_of_range"] += 1
                    o = orient(a, source, target)
                    if o.status == "unresolved_endpoint":
                        missing_s = a.source_iri not in source.entities and a.source_iri not in target.entities
                        missing_t = a.target_iri not in target.entities and a.target_iri not in source.entities
                        checks["missing_source_class"] += missing_s
                        checks["missing_target_class"] += missing_t
                        continue
                    key = (o.source_iri, o.target_iri)
                    if key in seen:
                        checks["duplicates"] += 1
                    seen.add(key)
                    relations[key].add(o.relation)
                    if o.source_iri == o.target_iri:
                        checks["self_mappings"] += 1
                    if not (source.entities[o.source_iri].declared and target.entities[o.target_iri].declared):
                        checks["non_declared_endpoint"] += 1
                    if o.relation in ("=", "<=", ">="):
                        targets_of[o.source_iri].add(o.target_iri)
                    s_labels = {normalize(x.text) for x in source.entities[o.source_iri].labels}
                    t_labels = {normalize(x.text) for x in target.entities[o.target_iri].labels}
                    if s_labels & t_labels:
                        checks["trivial_label_identical"] += 1
                checks["conflicting_relations"] = sum(len(v) > 1 for v in relations.values())
                checks["multiple_targets"] = sum(len(v) > 1 for v in targets_of.values())
                # label identity but no correspondence (possible incompleteness): exact normalised label index join
                if len(source.entities) * len(target.entities) <= 4e10:
                    index = defaultdict(set)
                    for iri, e in target.entities.items():
                        if e.declared:
                            for label in e.labels:
                                index[normalize(label.text)].add(iri)
                    mapped = {s for s, _ in seen}
                    missing = 0
                    for iri, e in source.entities.items():
                        if e.declared and iri not in mapped:
                            if any(index.get(normalize(label.text)) for label in e.labels):
                                missing += 1
                    checks["label_identity_no_correspondence"] = missing
                atomic = checks["cells"] - checks["complex_cells"]
                missing = checks["missing_source_class"] + checks["missing_target_class"]
                hard = checks["invalid_relation"] + checks["measure_out_of_range"] + checks["non_uri_entities"]
                out.append({"pair_id": pair.pair_id, "reference": relative_to_raw(ref), "family": pair.family,
                            "parse_error": parse_error, **checks,
                            "missing_rate": round(missing / atomic, 4) if atomic else 0.0,
                            "status": "fail" if parse_error or hard else ("warn" if missing else "ok")})
            bar.advance()
    return out


def write_report(ontologies: list[dict], references: list[dict]) -> dict:
    status = Counter(r["status"] for r in ontologies)
    ref_status = Counter(r["status"] for r in references)
    by_family = defaultdict(Counter)
    for r in references:
        for k, v in r.items():
            if isinstance(v, int) and not isinstance(v, bool):
                by_family[r["family"]][k] += v
    report = {"ontologies": {"status": dict(status), "files": ontologies},
              "references": {"status": dict(ref_status), "by_family": {k: dict(v) for k, v in by_family.items()},
                             "files": references}}
    (out_dir() / "coverage").mkdir(parents=True, exist_ok=True)
    (out_dir() / "coverage" / "parse_verification.json").write_text(json.dumps(report, indent=1, default=str))
    lines = ["# Parse and reference verification (generated by `python -m oa_rlaif.data.parse_verify`)", "",
             "**Ontologies** are cross-parsed with the OWL API (Java) and compared with our parser; ELK checks "
             "consistency for files < 300 MB. **References** get the Beisswanger & Hahn (2012) quality checks "
             "(doi:10.1186/2041-1480-3-S1-S4) plus format, relation and measure validity.", "",
             f"Ontology files: {len(ontologies)} — " + ", ".join(f"{k}: {v}" for k, v in sorted(status.items())),
             f"Reference files: {len(references)} — " + ", ".join(f"{k}: {v}" for k, v in sorted(ref_status.items())),
             "", "## Ontologies with a class-count mismatch or OWL API failure", "",
             "| File | Ours | OWL API (classes + SKOS concepts) | Status | Note |", "|---|---:|---:|---|---|"]
    for r in sorted(ontologies, key=lambda r: r["path"]):
        if r["status"] != "match":
            owl = r.get("owlapi") or {}
            theirs = (owl.get("classes_declared", 0) + owl.get("skos_concepts", 0)) if owl.get("ok") else "—"
            lines.append(f"| {r['path']} | {r['ours_classes']:,} | {theirs if isinstance(theirs, str) else f'{theirs:,}'} | "
                         f"{r['status']} | {owl.get('error', '')[:80]} |")
    lines += ["", "## ELK consistency", ""]
    inconsistent = [r for r in ontologies if (r.get("owlapi") or {}).get("consistent_elk") is False]
    unsat = [r for r in ontologies if ((r.get("owlapi") or {}).get("unsatisfiable_elk") or 0) > 0]
    lines += [f"- inconsistent ontologies: {len(inconsistent)}", f"- ontologies with unsatisfiable classes: {len(unsat)}"]
    lines += ["", "## Reference quality checks per family (sums)", "",
              "| Family | Cells | Missing src | Missing tgt | Duplicates | Conflicting | Multiple targets | Trivial | Label-identical, unmapped |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for fam, c in sorted(by_family.items()):
        lines.append(f"| {fam} | {c['cells']:,} | {c['missing_source_class']:,} | {c['missing_target_class']:,} | "
                     f"{c['duplicates']:,} | {c['conflicting_relations']:,} | {c['multiple_targets']:,} | "
                     f"{c['trivial_label_identical']:,} | {c['label_identity_no_correspondence']:,} |")
    failed = [r for r in references if r["status"] == "fail"]
    if failed:
        lines += ["", "## Failed references", ""] + [f"- {r['reference']}: {r['parse_error'] or 'invalid relation/measure/URI'}" for r in failed]
    (ROOT / "docs" / "PARSE_VERIFICATION.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-owlapi", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args()
    files = ontology_files(discover())
    ontologies = ontology_checks(files, not args.skip_owlapi, args.workers)
    references = reference_checks()
    report = write_report(ontologies, references)
    print(json.dumps({"ontologies": report["ontologies"]["status"], "references": report["references"]["status"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build every derived dataset from the raw folder: episodes, facts, complex tasks, baselines, passages.

    python -m oa_rlaif.data.build            # full build (ontology profiles are cached)
    python -m oa_rlaif.data.build --only anatomy,dh   # subset of families (smoke runs)

Outputs (all under data/, gitignored): episodes/<family>.jsonl, facts/facts.jsonl, complex/complex.jsonl,
baselines/baselines.jsonl, knowledge/passages.jsonl, eval/skip_list.tsv, splits/components.tsv and
outputs/coverage/derived_counts.tsv (records derived from every raw file, used by data/verify.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

from ..io import atomic_open, exclusive_lock, read_tsv, read_yaml, write_tsv
from ..ontology.alignment_io import read_alignment
from ..paths import bioml_dir, config_dir, data_dir, out_dir, papers_dir, raw_dir, relative_to_raw
from ..progress import stage
from . import auxiliary, complex as complex_tasks_mod
from .coverage import build_manifest
from .episodes import SYSTEM_ERRORS, EpisodeConfig, bioml_candidates, build_pair_episodes, make_context
from .facts import build_facts, tbox_signature
from .families import PairSpec, discover, ontology_files
from .profiles import build_profiles, load_profile
from .splits import SplitAssigner

SIMPLE_KINDS = {"simple", "bioml"}


class Counts:
    def __init__(self):
        self.parsed: dict[str, int] = defaultdict(int)
        self.training: dict[str, int] = defaultdict(int)

    def add(self, source: str, parsed: int = 0, training: int = 0):
        self.parsed[source] += parsed
        self.training[source] += training


class ProfileCache:
    def __init__(self, limit: int = 80):
        self.limit, self.items = limit, {}

    def get(self, path: Path):
        key = str(path)
        if key not in self.items:
            if len(self.items) >= self.limit:
                self.items.pop(next(iter(self.items)))
            self.items[key] = load_profile(path)
        return self.items[key]

    def drop_large(self, threshold: int = 50_000):
        for key in [k for k, v in self.items.items() if len(v.entities) > threshold]:
            self.items.pop(key)


def _jsonl(path: Path):
    return atomic_open(path)  # <path>.tmp, renamed on success


def assign_splits(pairs: list[PairSpec], cfg: dict) -> SplitAssigner:
    assigner = SplitAssigner(cfg["test_fraction"], cfg["split_salt"])
    zero_shot = set(cfg.get("zero_shot_pairs", []))
    with stage("split_components", total=len(pairs)) as bar:
        for pair in pairs:
            if pair.ref_kind in SIMPLE_KINDS or pair.ref_kind == "complex":
                for ref in pair.references:
                    for a in read_alignment(ref):
                        if a.source_iri and a.target_iri and a.relation in ("=", "<=", ">="):
                            assigner.add_link(a.source_iri, a.target_iri)
                        if pair.pair_id in zero_shot:
                            for iri in (a.source_iri, a.target_iri):
                                if iri:
                                    assigner.force_test(iri)
            bar.advance()
    assigner.freeze()
    return assigner


ALL_STAGES = ("episodes", "facts", "complex", "baselines", "passages", "icd")


def _save_counts(name: str, counts: "Counts") -> None:
    path = out_dir() / "coverage" / "stage_counts" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"parsed": counts.parsed, "training": counts.training}))


def _merged_counts() -> "Counts":
    merged = Counts()
    for path in sorted((out_dir() / "coverage" / "stage_counts").glob("*.json")):
        data = json.loads(path.read_text())
        for k, v in data["parsed"].items():
            merged.add(k, parsed=v)
        for k, v in data["training"].items():
            merged.add(k, training=v)
    return merged


def build(only: set[str] | None = None, stages: set[str] | None = None) -> int:
    stages = stages or set(ALL_STAGES)
    cfg = read_yaml(config_dir() / "data.yaml")
    pairs = discover()
    if only:
        pairs = [p for p in pairs if p.family in only]
    biollm = raw_dir() / "OAEI-Bio-ML-main/biollm_scripts/biollm_result.tsv"
    for p in pairs:
        if p.pair_id == "bioml/NCIT-DOID" and biollm.exists():
            p.system_outputs.append(biollm)
    errors = build_profiles(ontology_files(pairs))
    if errors:
        print(json.dumps({"profile_errors": errors}, indent=1), file=sys.stderr)
    assigner = assign_splits(pairs, cfg)
    ep_cfg = EpisodeConfig(seed=cfg["seed"], **cfg["episodes"])
    profiles = ProfileCache()
    generators: dict = {}
    base = data_dir()

    # ---------------- episodes (simple alignments) ----------------
    counts = Counts()
    simple = [p for p in pairs if p.ref_kind in SIMPLE_KINDS] if "episodes" in stages else []
    handles: dict[str, object] = {}
    summary = defaultdict(lambda: defaultdict(int))
    zero_shot_pairs = set(cfg.get("zero_shot_pairs", []))
    # One handle per family, opened once. (A previous version called setdefault(family, open(path, "w")), which
    # reopened and truncated the file for every pair and left zero-filled holes in multi-pair families.)
    with ExitStack() as files, stage("episodes", total=len(simple)) as bar:
        for pair in sorted(simple, key=lambda p: (p.family, str(p.target))):
            if pair.source is None or pair.target is None:
                bar.advance()
                continue
            source, target = profiles.get(pair.source), profiles.get(pair.target)
            ctx = make_context(pair, source, target, generators, ep_cfg)
            pool = bioml_candidates(pair) if pair.family == "bioml" else None
            episodes, file_counts, _gold = build_pair_episodes(ctx, assigner.split, ep_cfg, pool)
            if pair.pair_id in zero_shot_pairs:  # every episode of a zero-shot pair is test, labeled or not
                for e in episodes:
                    e["split"] = "test"
            for key, n in file_counts.items():
                counts.add(key, parsed=n)
            for path in pair.bioml_cands.values():
                counts.add(relative_to_raw(path), parsed=sum(1 for _ in read_tsv(path)))
            if pair.family not in handles:
                handles[pair.family] = files.enter_context(_jsonl(base / "episodes" / f"{pair.family}.jsonl"))
            handle = handles[pair.family]
            for e in episodes:
                handle.write(json.dumps(e, ensure_ascii=False) + "\n")
                summary[pair.family][f"{e['split']}/{e['kind']}"] += 1
                for ref in e["provenance"]["references"]:
                    counts.add(ref, training=1)
                if e["hard_negatives"]:
                    for sys_file in e["provenance"]["systems"]:
                        counts.add(sys_file, training=1)
            if pair.family == "bioml":
                profiles.drop_large()
                generators.clear()
            bar.advance()
    if "episodes" in stages:
        with _jsonl(base / "baselines" / "system_errors.jsonl") as handle:
            for record in SYSTEM_ERRORS:
                handle.write(json.dumps(record) + "\n")
        _save_counts("episodes", counts)

    # ---------------- ontology facts ----------------
    counts = Counts()
    files = ontology_files(pairs) if "facts" in stages else []
    seen_tbox: dict[str, str] = {}
    caps = cfg["facts"]["caps"]
    with (_jsonl(base / "facts" / "facts.jsonl") if files else open(os.devnull, "w")) as handle, stage("facts", total=len(files)) as bar:
        for path in files:
            profile = profiles.get(path)
            rel = relative_to_raw(path)
            counts.add(rel, parsed=len(profile.entities))
            signature = tbox_signature(profile)
            if signature in seen_tbox:  # populated variants share their TBox: facts once, profile still used
                counts.add(rel, training=0)
                bar.advance()
                continue
            seen_tbox[signature] = rel
            cap = next((v for k, v in caps.items() if k.lower() in path.name.lower()), cfg["facts"]["default_cap"])
            facts = build_facts(profile, cap, assigner.split, cfg["seed"])
            for fact in facts:
                handle.write(json.dumps(fact, ensure_ascii=False) + "\n")
            counts.add(rel, training=len(facts))
            summary["facts"]["total"] += len(facts)
            if len(profile.entities) > 50_000:
                profiles.drop_large()
            bar.advance()

    if "facts" in stages:
        _save_counts("facts", counts)

    # ---------------- complex alignments ----------------
    counts = Counts()
    complex_pairs = [p for p in pairs if p.ref_kind in ("complex", "complex_owl")] if "complex" in stages else []
    with (_jsonl(base / "complex" / "complex.jsonl") if complex_pairs else open(os.devnull, "w")) as handle, stage("complex", total=len(complex_pairs)) as bar:
        for pair in complex_pairs:
            if pair.ref_kind == "complex_owl":
                tasks, n = complex_tasks_mod.logical_definition_tasks(pair.references[0], assigner.split)
                counts.add(relative_to_raw(pair.references[0]), parsed=n, training=len(tasks))
            else:
                src = profiles.get(pair.source) if pair.source else None
                tgt = profiles.get(pair.target) if pair.target else None
                tasks, file_counts = complex_tasks_mod.complex_tasks(pair, src, tgt, assigner.split)
                for key, n in file_counts.items():
                    counts.add(key, parsed=n, training=sum(1 for t in tasks if t["provenance"] == key))
            for task in tasks:
                handle.write(json.dumps(task, ensure_ascii=False) + "\n")
            summary["complex"][pair.ref_kind] += len(tasks)
            bar.advance()

    if "complex" in stages:
        _save_counts("complex", counts)
        summary["complex_excluded_relations"] = dict(complex_tasks_mod.EXCLUDED_RELATIONS)

    # ---------------- baselines, passages, eval skip list ----------------
    counts = Counts()
    manifest = build_manifest()

    def raw_path(row):
        return {"raw": raw_dir(), "bioml": bioml_dir(), "papers": papers_dir()}[row["source"]] / row["path"]

    metric_files = [r for r in manifest if r["role"] == "baseline_metrics"] if "baselines" in stages else []
    with (_jsonl(base / "baselines" / "baselines.jsonl") if metric_files else open(os.devnull, "w")) as handle, stage("baselines", total=len(metric_files)) as bar:
        for row in metric_files:
            try:
                rows = auxiliary.baseline_rows(raw_path(row))
            except Exception as error:  # recorded, never hidden
                rows = []
                print(f"baseline parse failed {row['path']}: {error}", file=sys.stderr)
            for r in rows:
                handle.write(json.dumps(r, ensure_ascii=False) + "\n")
            counts.add(relative_to_raw(raw_path(row)), parsed=len(rows), training=0)
            summary["baselines"]["rows"] += len(rows)
            bar.advance()

    if "baselines" in stages:
        skip = raw_dir() / "OAEI-Bio-ML-main/melt_scripts/test_map_to_skip_melt_eval.rdf"
        if skip.exists():
            cells = [(a.source_iri, a.target_iri) for a in read_alignment(skip)]
            write_tsv(base / "eval" / "skip_list.tsv", ["source_iri", "target_iri"], cells)
            counts.add(relative_to_raw(skip), parsed=len(cells))
        # OAEI trivial alignments (e.g. Anatomy trivial.rdf): the label-identity baseline behind Recall+ in evaluation
        for pair in pairs:
            if pair.trivial is not None and pair.trivial.exists():
                trivial = [(a.source_iri, a.target_iri) for a in read_alignment(pair.trivial)]
                write_tsv(base / "eval" / f"trivial_{pair.family}.tsv", ["source_iri", "target_iri"], trivial)
                counts.add(relative_to_raw(pair.trivial), parsed=len(trivial))
        _save_counts("baselines", counts)
    counts = Counts()
    knowledge_files = [r for r in manifest if r["role"] == "knowledge_passage"] if "passages" in stages else []
    pcfg = cfg["passages"]
    with (_jsonl(base / "knowledge" / "passages.jsonl") if knowledge_files else open(os.devnull, "w")) as handle, stage("passages", total=len(knowledge_files)) as bar:
        for row in knowledge_files:
            items = auxiliary.passages(raw_path(row), pcfg["size_chars"])
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            counts.add(relative_to_raw(raw_path(row)), parsed=len(items), training=len(items))
            summary["passages"][row["family"]] += len(items)
            bar.advance()

    if "passages" in stages:
        _save_counts("passages", counts)

    # ---------------- ICD bridge + SNOMED RF2 (step 8 module) ----------------
    try:
        from . import icd_triangulation
    except ImportError:
        icd_triangulation = None
    if icd_triangulation is not None and (not only or "icd" in only) and "icd" in stages:
        counts = Counts()
        for key, (parsed, training) in icd_triangulation.build(base, assigner).items():
            counts.add(key, parsed=parsed, training=training)
        _save_counts("icd", counts)
    counts = _merged_counts()

    write_tsv(base / "splits" / "components.tsv", ["iri", "component", "split"],
              ((iri, assigner.component(iri), assigner.split(iri)) for iri in sorted(assigner.uf.parent)))
    write_tsv(out_dir() / "coverage" / "derived_counts.tsv", ["path", "parsed_records", "training_records"],
              sorted((k, counts.parsed[k], counts.training.get(k, 0)) for k in set(counts.parsed) | set(counts.training)))
    (out_dir() / "coverage" / "build_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma-separated families (smoke runs)")
    parser.add_argument("--stages", default="", help=f"comma-separated subset of {','.join(ALL_STAGES)} (resume)")
    args = parser.parse_args()
    with exclusive_lock(data_dir() / ".build.lock", "data build"):
        return build({x for x in args.only.split(",") if x} or None, {x for x in args.stages.split(",") if x} or None)


if __name__ == "__main__":
    raise SystemExit(main())

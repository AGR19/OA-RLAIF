"""Build one decision episode per class for every task: the prompts the models answer (CPU only).

    python -m full_eval.build_episodes                 # every task, in the run's priority order
    python -m full_eval.build_episodes --task anatomy__mouse-human --limit 50   # a quick check

Each episode is what training and the earlier evaluation used: the source card, k candidate cards (anatomy k=40, else
k=20; q-gram retrieval, multilingual e5 retrieval for MultiFarm and DH multilingual, ICD-11 retrieval with ancestor
expansion for the ICD tasks), the pair id and both modelling profiles. Which source classes get an episode is
FULL_EVAL_SCOPE (full_eval/tasks.py): 'reference' (default) = every source class with a partner in the reference, the
classes the Top-k evaluation scores (SNOMED CT -> ICD-11 stays the complete map of ICD's scope, ICD-10 -> ICD-11 every
WHO code); 'all' = every source class. Either way, no held-out view: all of those classes run. Candidates come from
retrieval over the whole target (Bio-ML's local 100-candidate pools contain the answer, so they are not used); the
answer is never inserted into the candidates and the reference is never shown; deprecated and vocabulary entities are
neither sources nor candidates.

Output: outputs/full_eval/episodes/<task>/<shard>.jsonl.gz (2,000 episodes each), gold.json for the ICD tasks,
manifest.json (counts per pair, entities left out and why, the scope) and READY when the task is complete.
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from oa_rlaif.paths import out_dir

from .schedule import progress_text
from .tasks import (GIANT, SNOMED_ICD_ROOTS, Task, deprecated_iris, eval_scope, is_meta, load_profile_any,
                    reference_classes, scope)

SHARD = 2000
_W: dict = {}  # worker state, set in the parent before forking (shared copy-on-write)
_DONE = None  # episodes finished by the ICD workers (a shared counter, set in the parent before forking)
REPORT_S = 20  # seconds between progress lines


def episodes_dir(task: Task) -> Path:
    return out_dir() / "full_eval" / "episodes" / task.slug


def write_shard(path: Path, episodes: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=3) as f:
        for e in episodes:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_shard(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_workers() -> int:
    """Worker processes for retrieval: forked workers slowly copy the parent's large indexes (SNOMED CT, NCIT), so the
    count is capped (FULL_EVAL_BUILD_WORKERS, default 16) and kept to one per 24 GB of RAM."""
    cpus = max(1, (os.cpu_count() or 2) - 2)
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        ram_gb = 64
    return max(1, min(cpus, int(os.environ.get("FULL_EVAL_BUILD_WORKERS", "16")), int(ram_gb // 24) or 1))


def config() -> dict:
    from oa_rlaif.io import read_yaml
    from oa_rlaif.paths import config_dir

    return read_yaml(config_dir() / "data.yaml")["episodes"]


# ---- ontology pairs ------------------------------------------------------------------------------------------------
def _pair_chunk(args) -> tuple[str, int]:
    """Worker: episodes for a slice of one pair's sources, written straight to its shard."""
    from oa_rlaif.data.cards import card, kind_of
    from oa_rlaif.data.episodes import episode_id

    shard_path, iris = args
    w = _W
    spec, source, target = w["spec"], w["source"], w["target"]
    out = []
    for iri in iris:
        kind = kind_of(source, iri)
        if w["semantic"] is not None:
            ranked = w["semantic"].get(iri, [])
        else:
            ranked = w["generator"].candidates(source.entities[iri], k=w["k"] * 4)
        ranking = [c for c, _ in ranked if c in w["targets"] and kind_of(target, c) == kind][: w["k"]]
        cands = list(ranking)
        random.Random(episode_id(spec.pair_id, iri, "full")).shuffle(cands)  # shown shuffled: position carries no signal
        out.append({"id": episode_id(spec.pair_id, iri, "full"), "family": spec.family, "pair_id": spec.pair_id,
                    "relation": "=", "source_iri": iri,
                    "source_card": card(source, iri, spec.source_lang, w["kids_s"]),
                    "source_profile": w["profiles"][0], "target_profile": w["profiles"][1],
                    "candidates": [{"iri": c, "card": card(target, c, spec.target_lang, w["kids_t"])} for c in cands],
                    "ranking": ranking})  # retrieval order (not shown to the model): Top-3/Top-5 after its choice
    write_shard(Path(shard_path), out)
    return shard_path, len(out)


def semantic_candidates(source, target, iris: list[str], targets: set[str], model: str, k: int) -> dict[str, list]:
    """The same ranking as oa_rlaif.retrieval.semantic.SemanticIndex.search (e5 'query:' text, exact cosine over the
    target's declared entities), with every source query encoded in one batch."""
    import numpy as np

    from oa_rlaif.retrieval.semantic import SemanticIndex, encoder, entity_text

    if not iris:  # a pair with no source to align (e.g. a MultiFarm reference none of whose cells resolves)
        return {}
    index = SemanticIndex(target, model, cache_key=None)
    queries = encoder(model).encode([entity_text(source.entities[i]) for i in iris], batch_size=256,
                                    normalize_embeddings=True, show_progress_bar=False).astype("float32")
    scores = queries @ index.vectors.T
    out = {}
    for row, iri in zip(scores, iris):
        top = np.argsort(-row)[: k * 4]
        out[iri] = [(index.ids[j], float(row[j])) for j in top]
    return out


def file_profiles(spec) -> tuple[str, str]:
    """Modelling-profile texts for a pair that data/ontology_profiles.json does not list (the biodiv-2026 test cases,
    added after training): built from the files, as the om_agent console does. Every prompt carries both profiles."""
    from oa_rlaif.data.ontology_profiles import summary_text
    from oa_rlaif.eval.align import build_profile_from

    return summary_text(build_profile_from(spec.source)), summary_text(build_profile_from(spec.target))


def sources_to_align(task: Task, spec, source, target, dep_s: set[str]) -> tuple[list[str], dict]:
    """The source classes that get an episode, sorted, and the manifest counts. scope() leaves out vocabulary and
    deprecated entities; classes_only tracks match classes; FULL_EVAL_SCOPE=reference keeps the classes with a
    reference partner (deprecated reference sources stay out and count as misses in the Top-k evaluation)."""
    from oa_rlaif.data.cards import kind_of

    iris = scope(source, dep_s)
    if task.classes_only:  # OAEI parameters.xml: classes are matched, properties are not
        iris = [i for i in iris if kind_of(source, i) == "class"]
    info: dict = {"scope": eval_scope()}
    if eval_scope() == "reference":
        partners = reference_classes(spec, source, target)
        in_scope = set(iris)
        iris = [i for i in iris if i in partners]
        info.update(reference_classes=len(partners), reference_classes_left_out=len(set(partners) - in_scope))
    return iris, info


def build_pairs(task: Task, limit: int, workers: int, pair_profiles: dict) -> dict:
    from oa_rlaif.data.cards import children_index
    from oa_rlaif.retrieval.candidates import CandidateGenerator

    cfg = config()
    folder = episodes_dir(task)
    manifest: dict = {"task": task.slug, "pairs": {}, "shards": {}}
    for p_index, spec in enumerate(sorted(task.specs, key=lambda s: s.pair_id)):
        source, target = load_profile_any(spec.source), load_profile_any(spec.target)
        dep_s, dep_t = deprecated_iris(spec.source), deprecated_iris(spec.target)
        iris, info = sources_to_align(task, spec, source, target, dep_s)
        targets = set(scope(target, dep_t))
        if task.classes_only:  # OAEI parameters.xml: classes are matched, properties are not
            from oa_rlaif.data.cards import kind_of

            targets = {t for t in targets if kind_of(target, t) == "class"}
        declared = sum(1 for e in source.entities.values() if e.declared)
        manifest["pairs"][spec.pair_id] = {
            "source": spec.source.name, "target": spec.target.name, "sources": len(iris),
            "declared": declared, "left_out_deprecated": sum(1 for i in dep_s if i in source.entities),
            "left_out_vocabulary": sum(1 for i, e in source.entities.items() if e.declared and is_meta(i)),
            "target_entities": len(targets), "classes_only": task.classes_only, **info}
        if limit:
            iris = iris[:limit]
        k = cfg.get("k_by_family", {}).get(spec.family, cfg.get("k", 20))
        mode = cfg.get("retrieval_by_family", {}).get(spec.family, "lexical")
        _W.clear()
        _W.update(spec=spec, source=source, target=target, targets=targets, k=k,
                  kids_s=children_index(source), kids_t=children_index(target),
                  profiles=pair_profiles.get(spec.pair_id) or file_profiles(spec), semantic=None, generator=None)
        if mode == "semantic":
            _W["semantic"] = semantic_candidates(source, target, iris, targets, cfg.get("semantic_model"), k)
        else:
            _W["generator"] = CandidateGenerator(target)
        chunks = [(str(folder / f"{p_index:04d}-{c:05d}.jsonl.gz"), iris[i:i + SHARD])
                  for c, i in enumerate(range(0, len(iris), SHARD))]
        started, results = time.time(), []
        if len(iris) > 4 * SHARD and workers > 1:
            with mp.get_context("fork").Pool(workers) as pool:
                for result in pool.imap_unordered(_pair_chunk, chunks, chunksize=1):
                    results.append(result)
                    done = sum(n for _, n in results)
                    print("  " + progress_text(spec.pair_id, done, len(iris), done / max(1e-9, time.time() - started),
                                               "episodes"), flush=True)
        else:
            results = [_pair_chunk(c) for c in chunks]
        for path, n in results:
            manifest["shards"][Path(path).name] = n
        print(f"  {spec.pair_id}: {len(iris):,} episodes", flush=True)
    return manifest


# ---- ICD tasks ------------------------------------------------------------------------------------------------------
def _icd_chunk(args) -> list[dict]:
    """Worker: data/icd_triangulation._episode_chunk for whole-map episodes (no answer ever inserted), keeping the
    retrieval order as `ranking`. The shown order is shuffled with the same seed as in training."""
    import hashlib

    from oa_rlaif.data import icd_triangulation as icd
    from oa_rlaif.data.cards import card
    from oa_rlaif.retrieval.candidates import CandidateGenerator

    icd11, ancestors, items, k = args
    generator = CandidateGenerator(icd11)
    out = []
    for i, (pair_id, _kind, entity, parent, source_card, _gold, _split, extra) in enumerate(items, 1):
        if _DONE is not None and i % 25 == 0:
            with _DONE.get_lock():
                _DONE.value += 25
        ranking = icd.icd_candidates(generator, ancestors, entity, parent, k)
        key = hashlib.sha1(f"{pair_id}|{entity.iri}".encode()).hexdigest()
        order = list(ranking)
        random.Random(key).shuffle(order)
        out.append({"id": key[:16], "family": "icd", "pair_id": pair_id, "relation": extra.get("relation", "classified_under"),
                    "source_iri": entity.iri, "source_card": source_card,
                    "candidates": [{"iri": c, "card": card(icd11, c)} for c in order], "ranking": ranking})
    return out


def _icd_episodes(icd11, ancestors, items: list, k: int, workers: int, name: str = "") -> list[dict]:
    """ICD episodes on `workers` forked processes, with a progress line every REPORT_S seconds."""
    global _DONE
    from concurrent.futures import ProcessPoolExecutor, wait

    size = max(1, len(items) // (workers * 4))
    chunks = [items[i:i + size] for i in range(0, len(items), size)]
    context = mp.get_context("fork")  # as build_pairs
    _DONE = context.Value("i", 0)
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [pool.submit(_icd_chunk, (icd11, ancestors, chunk, k)) for chunk in chunks]
        while wait(futures, timeout=REPORT_S).not_done:
            done = _DONE.value
            print("  " + progress_text(name, done, len(items), done / max(1e-9, time.time() - started), "episodes"),
                  flush=True)
        episodes = [e for f in futures for e in f.result()]  # in submission order, as before
    _DONE = None
    return episodes


def foundation_targets(table_info: dict, owl) -> dict:
    """Every entity of the ICD-11 Foundation (icd11.owl) as a possible target: a full alignment to ICD-11, not only
    the entities the WHO ICD-10 <-> ICD-11 tables name. Table entries keep their MMS code and title."""
    from oa_rlaif.data.cards import names

    out = {}
    for uri, e in owl.entities.items():
        if e.declared and uri.startswith("http://id.who.int/icd/entity/"):
            out[uri] = table_info.get(uri) or {"code": next(iter(sorted(e.codes)), ""), "title": names(e)[0], "chapter": ""}
    return out


def build_icd(task: Task, limit: int, pair_profiles: dict) -> dict:
    """WHO ICD-10 -> ICD-11 (every ICD-10 code in the WHO tables) or SNOMED CT -> ICD-11 (every active concept in
    ICD's scope), with the training-time ICD-11 retrieval (data/icd_triangulation.py) and its source cards. The
    targets are every ICD-11 Foundation entity (FULL_EVAL_ICD11_TARGETS=foundation, the default; 71,175 in icd11.owl);
    `tables` keeps the training-time set, the entities the WHO tables name (48,844)."""
    from oa_rlaif.data import icd_triangulation as icd
    from oa_rlaif.data.cards import card
    from oa_rlaif.ontology.models import Entity, Label

    counts: dict = defaultdict(int)
    leg_b, icd10_title, icd11_info = icd.read_who(counts)
    owl = load_profile_any(icd.icd_dir() / "icd11.owl")
    table_targets = len(icd11_info)
    if os.environ.get("FULL_EVAL_ICD11_TARGETS", "foundation") == "foundation":
        icd11_info = foundation_targets(icd11_info, owl)
    icd11 = icd._icd11_profile(icd11_info, owl)
    ancestors = icd._ancestors(icd11_info, owl, icd.EXPANSION["depth"])
    items, gold, info = [], {}, {}
    pair_id = task.pair_ids[0]
    if task.family == "icd10":
        for code, targets in sorted(leg_b.items()):
            e = Entity(icd.ICD10 + code, declared=True)
            e.labels.add(Label(icd10_title.get(code, code), "en", "label"))
            e.codes.add(code)
            category, parent = code.split(".")[0], None
            source_card = f"IRI: {icd.ICD10 + code}\nname: {icd10_title.get(code, code)}"
            if category != code and category in icd10_title:
                parent = Entity(icd.ICD10 + category, declared=True)
                parent.labels.add(Label(icd10_title[category], "en", "label"))
                source_card += f"\nparents: {icd10_title[category]}"
            source_card += f"\ncodes: {code}"
            items.append((pair_id, "full", e, parent, source_card, [], "full", {"relation": "classified_under"}))
            gold[icd.ICD10 + code] = {"targets": sorted(targets), "split": icd.bridge_split(code)}
        info = {"source": "WHO ICD-10 codes in the 10To11 tables", "sources": len(items)}
    else:
        sct = icd.Snomed(counts, sample_concepts=0, seed=13)
        concept_file = next((icd.rf2_dir() / "Snapshot" / "Terminology").glob("sct2_Concept_*"))
        active = [row["id"] for row in icd._rf2(concept_file) if row["active"] == "1"]
        sct.load(set(active))
        all_scope = os.environ.get("FULL_EVAL_SNOMED_SCOPE", "icd") == "all"
        if all_scope:
            chosen = sorted(active)
        else:
            children: dict[str, set[str]] = defaultdict(set)
            for cid, parents in sct.parents.items():
                for p in parents:
                    children[p[len(icd.SCT):] if p.startswith(icd.SCT) else p].add(cid)
            chosen, stack = set(), list(SNOMED_ICD_ROOTS)
            while stack:
                node = stack.pop()
                for child in children.get(node, ()):
                    if child not in chosen and child in sct.active:
                        chosen.add(child)
                        stack.append(child)
            chosen = sorted(chosen | set(SNOMED_ICD_ROOTS))
        for cid in chosen:
            iri = icd.SCT + cid
            entity = sct.entity(cid)
            context = {iri: entity}
            parent = Entity(iri + "#parents", declared=True)
            for p in entity.parents:
                context[p] = Entity(p, declared=True)
                context[p].labels.add(Label(sct.name(p[len(icd.SCT):]), "en", "prefLabel"))
                parent.labels.add(Label(sct.name(p[len(icd.SCT):]), "en", "label"))
            source_card = card(icd._Profile(context), iri)
            items.append((pair_id, "full", entity, parent, source_card, [], "full", {"relation": "classified_under"}))
        info = {"source": "SNOMED CT US 2026-03 RF2, active concepts", "scope": "all active concepts" if all_scope else
                "ICD's scope: descendants of " + ", ".join(SNOMED_ICD_ROOTS.values()),
                "active_concepts": len(active), "sources": len(items)}
    info.update(targets="every ICD-11 Foundation entity" if len(icd11_info) > table_targets else
                "the ICD-11 entities the WHO tables name", target_entities=len(icd11_info))
    if limit:
        items = items[:limit]
    episodes = _icd_episodes(icd11, ancestors, items, k=20, workers=build_workers(), name=pair_id)
    profiles = pair_profiles.get(pair_id, ("", ""))
    folder = episodes_dir(task)
    manifest: dict = {"task": task.slug, "pairs": {pair_id: info}, "shards": {}}
    keep = ("id", "family", "pair_id", "relation", "source_iri", "source_card", "candidates", "ranking")
    for c, i in enumerate(range(0, len(episodes), SHARD)):
        chunk = [{**{k: e[k] for k in keep}, "source_profile": profiles[0], "target_profile": profiles[1]}
                 for e in episodes[i:i + SHARD]]
        name = f"0000-{c:05d}.jsonl.gz"
        write_shard(folder / name, chunk)
        manifest["shards"][name] = len(chunk)
    if gold:
        (folder / "gold.json").write_text(json.dumps(gold))
    print(f"  {pair_id}: {len(episodes):,} episodes", flush=True)
    return manifest


def built_scope(task: Task) -> str | None:
    """The FULL_EVAL_SCOPE a task's finished episodes were built with (before the switch existed: 'all')."""
    manifest = episodes_dir(task) / "manifest.json"
    if not (manifest.parent / "READY").exists():
        return None
    return json.loads(manifest.read_text()).get("scope", "all")


def stale_tasks(task_list: list[Task]) -> list[str]:
    """Tasks whose finished episodes belong to the other scope: their replies and scores would not match this run."""
    return [t.slug for t in task_list if built_scope(t) not in (None, eval_scope())]


def build(task: Task, limit: int = 0, workers: int | None = None) -> dict:
    from oa_rlaif.data.render import pair_profiles

    folder = episodes_dir(task)
    if built_scope(task) not in (None, eval_scope()):
        raise SystemExit(f"{task.slug}: the episodes in {folder} were built with FULL_EVAL_SCOPE={built_scope(task)}; "
                         f"use a new OA_OUT_DIR for FULL_EVAL_SCOPE={eval_scope()}")
    if (folder / "READY").exists() and not limit:
        return json.loads((folder / "manifest.json").read_text())
    folder.mkdir(parents=True, exist_ok=True)
    for old in [*folder.glob("*.jsonl.gz"), folder / "FAILED"]:
        old.unlink(missing_ok=True)
    started = time.time()
    workers = workers or build_workers()
    profiles = pair_profiles()
    manifest = build_icd(task, limit, profiles) if task.family in ("icd10", "snomed_icd11") else \
        build_pairs(task, limit, workers, profiles)
    manifest.update(episodes=sum(manifest["shards"].values()), seconds=round(time.time() - started), limit=limit,
                    giant=task.slug in GIANT, scope=eval_scope())
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if not limit:
        (folder / "READY").write_text(str(time.time()))
    return manifest


def main(argv: list[str] | None = None) -> int:
    from .schedule import selected_tasks, task_order

    parser = argparse.ArgumentParser(prog="full_eval.build_episodes")
    parser.add_argument("--task", action="append", help="task slug (default: every task, in priority order)")
    parser.add_argument("--limit", type=int, default=0, help="first N sources per pair (a check; not marked READY)")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args(argv)
    import traceback

    failures = 0
    for task in task_order(args.task or selected_tasks()):
        print(f"[episodes] {task.slug}", flush=True)
        try:
            m = build(task, args.limit, args.workers or None)
        except Exception:  # one task failing must not stop the others; the run reports it
            failures += 1
            episodes_dir(task).mkdir(parents=True, exist_ok=True)
            (episodes_dir(task) / "FAILED").write_text(traceback.format_exc())
            print(traceback.format_exc(), flush=True)
            continue
        print(f"[episodes] {task.slug}: {m['episodes']:,} episodes in {m.get('seconds', 0)} s", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

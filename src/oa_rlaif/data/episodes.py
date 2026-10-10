"""Alignment decision episodes: one per (pair, source entity) with up to K candidate targets.

The model sees the source card and K candidate cards (shuffled, so position carries no signal) and must pick
the gold target or NONE. Training episodes add deliberate-NONE cases and hard negatives taken from other
matchers' false positives; test episodes keep retrieval exactly as it would be at inference time.
"""

from __future__ import annotations

import ast
import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

from ..io import read_tsv
from ..ontology.alignment_io import orient, read_alignment
from ..paths import relative_to_raw
from ..retrieval.candidates import CandidateGenerator, grams
from .cards import card, children_index, kind_of
from .families import PairSpec


SYSTEM_ERRORS: list[dict] = []  # malformed system outputs, written to data/baselines/system_errors.jsonl


@dataclass
class EpisodeConfig:
    k: int = 20
    deliberate_none_rate: float = 0.10  # training only
    insert_missing_gold_rate: float = 0.70  # training only: put a retrieval-missed gold back in
    max_hard_negatives: int = 3
    unlabeled_rate: float = 0.10  # unlabeled sources per labeled episode (GRPO DRAL-only reward)
    seed: int = 13
    k_by_family: dict = field(default_factory=dict)        # e.g. {"anatomy": 40}
    retrieval_by_family: dict = field(default_factory=dict)  # family -> "lexical" | "semantic" | "fused"
    semantic_model: str = "intfloat/multilingual-e5-small"


@dataclass
class PairContext:
    spec: PairSpec
    source: object
    target: object
    generator: CandidateGenerator
    children_s: dict = field(default_factory=dict)
    children_t: dict = field(default_factory=dict)
    semantic: object = None  # retrieval.semantic.SemanticIndex for cross-lingual families
    children_t: dict = field(default_factory=dict)


def episode_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def gold_links(spec: PairSpec, source, target) -> tuple[dict[str, set[str]], dict[str, str], dict[str, int]]:
    """Gold targets per source IRI, the relation per source, and per-file parsed record counts."""
    gold: dict[str, set[str]] = defaultdict(set)
    relation: dict[str, str] = {}
    counts: dict[str, int] = defaultdict(int)
    for ref in spec.references:
        for a in read_alignment(ref):
            counts[relative_to_raw(ref)] += 1
            if not a.atomic:
                continue
            a = orient(a, source, target)
            if a.status == "unresolved_endpoint":
                counts[f"unresolved::{relative_to_raw(ref)}"] += 1
                continue
            gold[a.source_iri].add(a.target_iri)
            relation[a.source_iri] = {"<=": "=", ">=": "="}.get(a.relation, a.relation)
    return gold, relation, counts


def system_negatives(spec: PairSpec, source, target, gold) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Other matchers' predictions that are NOT in the gold set: hard negatives per source entity."""
    negatives: dict[str, list[str]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for path in spec.system_outputs:
        key = relative_to_raw(path)
        if path.suffix == ".tsv":  # Bio-ML ranking output: (SrcEntity, TgtEntity, [(cand, score, is_gold)])
            for row in read_tsv(path):
                counts[key] += 1
                for cand, _score, flag in ast.literal_eval(row["TgtCandidates"])[:5]:
                    if not flag and cand not in gold.get(row["SrcEntity"], ()):
                        negatives[row["SrcEntity"]].append(cand)
            continue
        try:
            for a in read_alignment(path):
                counts[key] += 1
                if not a.atomic:
                    continue
                a = orient(a, source, target)
                if a.status != "unresolved_endpoint" and a.target_iri not in gold.get(a.source_iri, ()):
                    negatives[a.source_iri].append(a.target_iri)
        except etree.XMLSyntaxError:
            # e.g. a matcher service that returned the text "error": a real failed run, kept as a baseline record
            counts[key] += 1
            SYSTEM_ERRORS.append({"source": key, "pair_id": spec.pair_id, "outcome": "system_error",
                                  "content_head": path.read_bytes()[:80].decode("utf-8", "replace")})
    return negatives, counts


def bioml_candidates(spec: PairSpec) -> dict[str, list[str]]:
    pools: dict[str, list[str]] = {}
    for _split, path in spec.bioml_cands.items():
        for row in read_tsv(path):
            pools[row["SrcEntity"]] = list(ast.literal_eval(row["TgtCandidates"]))
    return pools


_GRAM_CACHE: dict[int, dict[str, frozenset]] = {}


def _entity_grams(profile, iri: str) -> frozenset:
    cache = _GRAM_CACHE.setdefault(id(profile), {})
    if iri not in cache:
        entity = profile.entities.get(iri)
        cache[iri] = frozenset(g for label in (entity.labels if entity else ()) for g in grams(label.text))
    return cache[iri]


def _rank_pool(ctx: PairContext, source_iri: str, pool: list[str]) -> list[str]:
    """Order an external candidate pool (e.g. Bio-ML's 100 candidates) by direct q-gram Jaccard to the source.

    Scores only the pool members (not a full-ontology search); ties keep the pool's original order.
    """
    source = _entity_grams(ctx.source, source_iri)
    in_target = [c for c in pool if c in ctx.target.entities]

    def score(c):
        target = _entity_grams(ctx.target, c)
        union = len(source | target)
        return len(source & target) / union if union else 0.0

    return sorted(in_target, key=lambda c: -score(c))


def build_pair_episodes(ctx: PairContext, split_of, cfg: EpisodeConfig, bioml_pool=None):
    spec = ctx.spec
    gold, relation, counts = gold_links(spec, ctx.source, ctx.target)
    negatives, sys_counts = system_negatives(spec, ctx.source, ctx.target, gold)
    counts.update(sys_counts)
    rng = random.Random(f"{cfg.seed}|{spec.pair_id}")
    episodes = []

    k = cfg.k_by_family.get(spec.family, cfg.k)
    mode = cfg.retrieval_by_family.get(spec.family, "lexical")

    def candidates_for(source_iri: str) -> list[str]:
        if bioml_pool and source_iri in bioml_pool:
            return _rank_pool(ctx, source_iri, bioml_pool[source_iri])[:k]
        # classes are matched with classes, properties with properties (same kind as the source)
        kind = kind_of(ctx.source, source_iri)
        entity = ctx.source.entities[source_iri]
        ranked = ctx.generator.candidates(entity, k=k * 4) if mode != "semantic" else []
        if mode in ("semantic", "fused") and ctx.semantic is not None:
            from ..retrieval.semantic import fuse
            semantic = ctx.semantic.search(entity, k=k * 4)
            ranked = semantic if mode == "semantic" else fuse(ranked, semantic)
        return [iri for iri, _ in ranked if kind_of(ctx.target, iri) == kind][:k]

    def emit(source_iri: str, golds: set[str], kind: str):
        split = split_of(source_iri)
        cands = candidates_for(source_iri)
        none_reason = ""
        used_negatives = []
        if split == "train" and kind == "gold":
            missing = [g for g in sorted(golds) if g not in cands and g in ctx.target.entities]
            if missing and rng.random() < cfg.insert_missing_gold_rate:
                cands = cands[: k - len(missing)] + missing
            hard = [n for n in dict.fromkeys(negatives.get(source_iri, [])) if n in ctx.target.entities
                    and n not in cands][: cfg.max_hard_negatives]
            if hard:
                keep = [c for c in cands if c in golds]
                rest = [c for c in cands if c not in golds][: k - len(keep) - len(hard)]
                cands, used_negatives = keep + rest + hard, hard
            if rng.random() < cfg.deliberate_none_rate:
                cands = [c for c in cands if c not in golds]
                none_reason = "deliberate"
        present = [g for g in sorted(golds) if g in cands]
        if kind == "gold" and not present and not none_reason:
            none_reason = "retrieval_miss"
        order = list(cands)
        random.Random(episode_id(spec.pair_id, source_iri, kind)).shuffle(order)
        episodes.append({
            "id": episode_id(spec.pair_id, source_iri, kind),
            "family": spec.family, "pair_id": spec.pair_id, "split": split, "kind": kind,
            "source_iri": source_iri,
            "source_card": card(ctx.source, source_iri, spec.source_lang, ctx.children_s),
            "candidates": [{"iri": c, "card": card(ctx.target, c, spec.target_lang, ctx.children_t)} for c in order],
            "gold": present if kind == "gold" else [],
            "gold_all": sorted(golds) if kind == "gold" else [],
            "relation": relation.get(source_iri, "="),
            "none_reason": none_reason,
            "hard_negatives": used_negatives,
            "source_lang": spec.source_lang, "target_lang": spec.target_lang,
            "provenance": {"references": [relative_to_raw(r) for r in spec.references],
                           "systems": sorted({relative_to_raw(p) for p in spec.system_outputs})},
        })

    for source_iri in sorted(gold):
        if source_iri in ctx.source.entities:
            emit(source_iri, gold[source_iri], "gold")
    # Unlabeled sources (no reference correspondence in this pair): DRAL-only reward in GRPO, never gold.
    declared = sorted(i for i, e in ctx.source.entities.items() if e.declared and i not in gold)
    pool = [i for i in declared if bioml_pool is None or i in bioml_pool]
    for source_iri in rng.sample(pool, min(len(pool), int(len(gold) * cfg.unlabeled_rate))):
        emit(source_iri, set(), "unlabeled")
    return episodes, counts, gold


def make_context(spec: PairSpec, source, target, generator_cache: dict, cfg: EpisodeConfig | None = None) -> PairContext:
    key = id(target)
    if key not in generator_cache:
        generator_cache.clear() if len(generator_cache) > 64 else None
        generator_cache[key] = CandidateGenerator(target)
    semantic = None
    if cfg is not None and cfg.retrieval_by_family.get(spec.family, "lexical") != "lexical":
        from ..retrieval.semantic import SemanticIndex
        semantic = SemanticIndex(target, cfg.semantic_model, cache_key=f"{spec.target.parent.name}-{spec.target.stem}")
    return PairContext(spec, source, target, generator_cache[key], children_index(source), children_index(target),
                       semantic)


def unused(path: Path) -> None:  # pragma: no cover - placeholder to keep imports explicit
    return None

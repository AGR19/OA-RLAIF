"""Two matcher variants built from the models' own alignments (no extra GPU time), reported as separate rows.

    python -m full_eval.variants

  grpo_anchors       GRPO's alignment + exact-label anchors: every pair whose two entities share a normalised label
                     (OAEI's StringEquiv). Exact-label anchors are a standard first step of OAEI matchers (LogMap's
                     anchors; MILA's "identify" step before the LLM).
  ensemble_anchors   the matches GRPO and All merged agree on + the same anchors (agreement of two models removes most
                     of the false matches of classes that have no counterpart).

Why these two: on the EARLIER run's whole-Anatomy alignments (k=20, no profiles), GRPO alone scored F 0.883, GRPO +
anchors 0.904 and (GRPO ∩ All merged) + anchors 0.908 against the full reference (measured 2026-10-13). They were
chosen after that measurement, so they are reported as variants next to the plain model rows, never instead of them,
and with the held-out view (training links null) like every other row.
"""

from __future__ import annotations

import json
import sys

from oa_rlaif.ontology.alignment_io import read_alignment, write_oaei_rdf
from oa_rlaif.ontology.models import Alignment

from .schedule import LABELS, root
from .score import PairScorer, summarize
from .tasks import tasks

VARIANTS = {"grpo_anchors": ("GRPO + exact-label anchors", ["grpo"]),
            "ensemble_anchors": ("GRPO ∩ All merged + exact-label anchors", ["grpo", "all_merged"])}
LABELS.update({k: v[0] for k, v in VARIANTS.items()})


def model_pairs(model: str, slug: str, pair_id: str) -> set | None:
    path = root() / "results" / model / slug / "pairs" / pair_id.replace("/", "__") / "alignment.rdf"
    if not path.exists():
        return None
    return {a.pair for a in read_alignment(path) if a.atomic and a.relation == "="}


def build_variant(tag: str, slug: str) -> dict | None:
    task = next(t for t in tasks() if t.slug == slug)
    label, models = VARIANTS[tag]
    if not task.specs or not all((root() / "results" / m / slug / "scores.json").exists() for m in models):
        return None
    pairs = {}
    for spec in sorted(task.specs, key=lambda s: s.pair_id):
        sets = [model_pairs(m, slug, spec.pair_id) for m in models]
        if any(s is None for s in sets):
            return None
        scorer = PairScorer(spec)
        agreed = set.intersection(*sets)
        predicted = agreed | scorer.string_equiv
        pairs[spec.pair_id] = {**scorer(predicted), "anchors": len(scorer.string_equiv), "model_pairs": len(agreed)}
        write_oaei_rdf(root() / "results" / tag / slug / "pairs" / spec.pair_id.replace("/", "__") / "alignment.rdf",
                       [Alignment(s, t, "=", 1.0, tag) for s, t in sorted(predicted)])
    result = summarize(pairs)
    result.update(model=tag, label=label, task=slug, family=task.family, column=task.column, built_from=models,
                  decisions=None, invalid=None, runtime_s=None)
    (root() / "results" / tag / slug / "scores.json").write_text(json.dumps(result, indent=1, default=str))
    return result


def main() -> int:
    for tag in VARIANTS:
        for task in tasks():
            if task.family in ("icd10", "snomed_icd11"):
                continue
            if build_variant(tag, task.slug):
                print(f"[variants] {tag} {task.slug}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

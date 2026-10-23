"""Each scored class's reference partners, the labels of the Top-k evaluation (full_eval.score uses the same rules):
pair tasks -> full_eval.tasks.reference_classes (through PairScorer), ICD-10 -> ICD-11 -> the WHO tables (the
episodes' gold.json), SNOMED CT -> ICD-11 -> the strong ICD-10-composed links (silver). Used by the threshold fit of
arm C (dm_harness.calibrate) and the report's tables that recompute scores (dm_harness.report)."""

from __future__ import annotations

import json
from collections import defaultdict

from full_eval.schedule import root


def partners(task) -> dict[str, dict[str, set[str]]]:
    """pair id -> source IRI -> its partners."""
    if task.family == "icd10":
        gold = json.loads((root() / "episodes" / task.slug / "gold.json").read_text())
        return {task.pair_ids[0]: {k: set(v["targets"]) for k, v in gold.items() if v["targets"]}}
    if task.family == "snomed_icd11":
        from oa_rlaif.io import read_jsonl
        from oa_rlaif.paths import data_dir

        strong: dict[str, set[str]] = defaultdict(set)
        for link in read_jsonl(data_dir() / "icd" / "triangulated.jsonl"):
            if link["leg_a"]["strong"] and link["leg_b"].startswith("strong"):
                strong[link["snomed"]].add(link["icd11"])
        return {task.pair_ids[0]: dict(strong)}
    from full_eval.score import PairScorer

    return {s.pair_id: PairScorer(s, oaei=False).partners() for s in task.specs}


def correct(answer: list[str] | None, partner_set: set[str] | None) -> bool | None:
    """Top-1 correctness of an answer (its first target; NONE or invalid = []); None when the class is not scored."""
    if not partner_set:
        return None
    return bool(answer) and answer[0] in partner_set

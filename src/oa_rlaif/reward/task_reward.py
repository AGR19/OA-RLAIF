"""R_task for one decision episode: gold match + DRAL-OA Eq. 1, gated by a strict format check.

    labeled  (kind=gold):      R_task = 0.5 * F1(picks, gold) [NONE scores 1 when gold is absent] + 0.5 * r_DRAL
                               (one correct pick for one gold = 1; extra or missing picks lower F1, so listing
                               several targets only pays when the source really has several; r_DRAL = mean over picks)
    silver   (kind=silver):    same formula, then x 0.5 (composed ICD links are evidence, not gold)
    unlabeled(kind=unlabeled): R_task = 0.3 * r_DRAL   (semi-supervised, as in DRAL-OA; no gold term)
    invalid format / out-of-range index:  R_task = 0  (and the judge term is also 0)
If r_DRAL cannot be computed (entity not embedded) the gold term alone is used, rescaled to the same range.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..harness.prompts import parse_decision


@dataclass
class TaskRewardConfig:
    gold_weight: float = 0.5
    dral_weight: float = 0.5
    silver_scale: float = 0.5
    unlabeled_dral_weight: float = 0.3


def correct(choices, episode: dict) -> float:
    """F1 between the submitted targets and the gold targets present among the candidates."""
    choices = (choices,) if isinstance(choices, str) else tuple(choices)
    gold = set(episode.get("gold") or [])
    if choices == ("NONE",):
        return 1.0 if not gold else 0.0
    hits = len(gold & set(choices))
    if not hits:
        return 0.0
    precision, recall = hits / len(choices), hits / len(gold)
    return 2 * precision * recall / (precision + recall)


def task_reward(completion: str, episode: dict, scorer=None, cfg: TaskRewardConfig | None = None) -> dict:
    cfg = cfg or TaskRewardConfig()
    decision = parse_decision(completion, episode)
    if not decision.valid:
        return {"valid": False, "gold": 0.0, "dral": 0.0, "task": 0.0, "error": decision.error}
    candidates = [c["iri"] for c in episode["candidates"]]
    dral = None
    if scorer is not None:
        values = [scorer.reward(episode["source_iri"], candidates, c) for c in (decision.choices or (decision.choice,))]
        values = [v for v in values if v is not None]
        dral = sum(values) / len(values) if values else None
    kind = episode.get("kind", "gold")
    if kind == "unlabeled":
        task = cfg.unlabeled_dral_weight * (dral if dral is not None else 0.0)
        return {"valid": True, "gold": None, "dral": dral, "task": task, "choice": decision.choice}
    gold = correct(decision.choices or (decision.choice,), episode)
    if dral is None:
        task = gold * (cfg.gold_weight + cfg.dral_weight)
    else:
        task = cfg.gold_weight * gold + cfg.dral_weight * dral
    if kind == "silver":
        task *= cfg.silver_scale
    return {"valid": True, "gold": gold, "dral": dral, "task": task, "choice": decision.choice,
            "choices": list(decision.choices)}

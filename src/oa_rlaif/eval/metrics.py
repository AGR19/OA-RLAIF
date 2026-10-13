"""OAEI-compatible set and fixed-pool ranking metrics (vendored from the parent repo
src/training/eval_metrics.py; tested there for parity with the Bio-ML scoring kit)."""

from __future__ import annotations

RESULT_COLUMNS = [
    "Matcher",
    "Runtime",
    "Size",
    "Precision",
    "F-Measure",
    "Recall",
    "Recall+",
    "Coherent",
]


class AlignmentEvaluator:
    def evaluate(
        self,
        predicted,
        reference,
        *,
        matcher="oa-rlaif-e4b",
        runtime=None,
        trivial=None,
        coherent=None,
        flagged=(),
    ):
        predicted = set(predicted)
        reference = set(reference)
        flagged = set(flagged)
        scored = predicted - flagged
        positive = reference - flagged
        true = len(scored & positive)
        precision = true / len(scored) if scored else 0.0
        recall = true / len(positive) if positive else 0.0
        nontrivial = positive - set(trivial) if trivial is not None else None
        return {
            "Matcher": matcher,
            "Runtime": runtime,
            "Size": len(predicted),
            "Precision": precision,
            "F-Measure": 2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
            "Recall": recall,
            "Recall+": len(scored & nontrivial) / len(nontrivial) if nontrivial else None,
            "Coherent": "+" if coherent is True else "-" if coherent is False else "N/A",
            "true_positive": true,
            "reference": len(positive),
            "predicted_flagged": len(predicted & flagged),
        }

    def ranking(self, rankings, golds):
        reciprocal = []
        hits = {k: [] for k in [1, 5, 10]}
        for key, gold in golds.items():
            ranked = rankings.get(key, [])
            rank = ranked.index(gold) + 1 if gold in ranked else None
            reciprocal.append(1 / rank if rank else 0.0)
            for k in hits:
                hits[k].append(float(rank is not None and rank <= k))
        return {
            "mrr": sum(reciprocal) / len(reciprocal) if reciprocal else 0.0,
            "queries": len(golds),
            **{f"hits_at_{k}": sum(v) / len(v) if v else 0.0 for k, v in hits.items()},
        }

    def top1(self, choices, partners, trivial=None, matcher="oa-rlaif-e4b", runtime=None):
        """DRAL-OA's clf.py regime (precision_at_k / recall_at_k at k = 1), the scoring the user asked for.

        Only source classes that have a partner in the reference are scored (clf.py's link_labels); classes without
        one are not evaluated. `found` is the model's own top-1 choice for the class (its first submitted target;
        NONE or an invalid reply = nothing found). Per class: precision@1 = hits / min(1, partners) and
        recall@1 = hits / partners; both are averaged over the scored classes (macro), as clf.py does.
        choices: source -> ordered list of chosen targets; partners: source -> set of reference targets;
        trivial: set of (source, target) reference pairs whose names match (Recall+@1 = hit rate on the classes
        none of whose reference pairs is trivial)."""
        precision, recall, hits, answered, plus = [], [], 0, 0, []
        for source, targets in partners.items():
            found = list(choices.get(source, []))[:1]
            n = sum(1 for t in targets if t in found)
            precision.append(n / min(1, len(targets)))
            recall.append(n / len(targets))
            hits += n > 0
            answered += bool(found)
            if trivial is not None and not any((source, t) in trivial for t in targets):
                plus.append(n > 0)
        p = sum(precision) / len(precision) if precision else 0.0
        r = sum(recall) / len(recall) if recall else 0.0
        return {"Matcher": matcher, "Runtime": runtime, "Size": answered, "Precision": p,
                "F-Measure": 2 * p * r / (p + r) if p + r else 0.0, "Recall": r,
                "Recall+": sum(plus) / len(plus) if plus else None, "classes_scored": len(partners), "hits": hits,
                "regime": "clf.py top-1 (reference classes only, model's own choice)"}

    def candidate_recall(self, candidates, reference):
        gold = set(reference)
        recovered = sum(t in {x[0] for x in candidates.get(s, [])} for s, t in gold)
        return recovered / len(gold) if gold else 0.0

    def normalized_trivial(self, source, target, reference):
        from ..retrieval.candidates import normalize

        return {
            (s, t)
            for s, t in reference
            if s in source.entities
            and t in target.entities
            and {normalize(label.text) for label in source.entities[s].labels}
            & {normalize(label.text) for label in target.entities[t].labels}
        }

"""OAEI-compatible set and fixed-pool ranking metrics (vendored from the parent repo
src/training/eval_metrics.py; tested there for parity with the Bio-ML scoring kit)."""

from __future__ import annotations

TOP_K = (1, 3, 5)
TOP_K_NAME = "Top-k evaluation (k = 1, 3, 5)"  # Precision@k, Recall@k, F-measure@k; the name in tables and charts

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

    def topk(self, rankings, partners, ks=TOP_K):
        """Top-k evaluation (k = 1, 3, 5): Precision@k, Recall@k and F-measure@k, the per-class top-k precision and
        recall of the DRAL-OA evaluation code (clf.py precision_at_k / recall_at_k).

        Only source classes that have a partner in the reference are scored (partners: class -> set of reference
        targets); `rankings` gives each class's ranked answers (the model's own choice first; NONE or an invalid reply
        = an empty list). Per class c, with F_c the top k and R_c its partners:
            Precision@k = |F_c ∩ R_c| / min(k, |R_c|)        Recall@k = |F_c ∩ R_c| / |R_c|
        each averaged over the scored classes; F-measure@k = 2PR / (P + R) of the two averages."""
        out = {"method": TOP_K_NAME, "classes_scored": len(partners), "k": {}}
        for k in ks:
            precision, recall, hits = [], [], 0
            for source, targets in partners.items():
                found = list(rankings.get(source, []))[:k]
                n = sum(1 for t in targets if t in found)
                precision.append(n / min(k, len(targets)))
                recall.append(n / len(targets))
                hits += n > 0
            p = sum(precision) / len(precision) if precision else 0.0
            r = sum(recall) / len(recall) if recall else 0.0
            out["k"][k] = {"Precision": p, "Recall": r, "F-measure": 2 * p * r / (p + r) if p + r else 0.0,
                           "classes_with_a_hit": hits}
        return out

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

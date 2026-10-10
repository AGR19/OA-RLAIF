"""Batch evaluation with local vLLM: held-out decisions, zero-shot pairs and closed-book knowledge.

    python -m oa_rlaif.eval.run_eval --model google/gemma-4-E4B-it --tag base
    python -m oa_rlaif.eval.run_eval --model outputs/merged/sdft --tag sdft
    python -m oa_rlaif.eval.run_eval --model outputs/merged/grpo --tag grpo

Decision metrics per eval set: accuracy (correct target or correct NONE), OAEI precision/recall/F1 over the
predicted correspondences (NONE = no prediction) against all gold correspondences of the evaluated sources,
Recall+ for Anatomy (non-trivial references), invalid-reply rate and NONE rate. Closed-book: token F1 to the
reference answer. Results go to outputs/eval/<tag>/metrics.json and are compared in docs/RESULTS.md.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from ..harness.prompts import KNOWLEDGE_SYSTEM, decision_messages, parse_decision
from ..io import read_jsonl
from ..ontology.alignment_io import read_alignment
from ..paths import data_dir, out_dir, raw_dir
from ..progress import stage
from .metrics import AlignmentEvaluator


def token_f1(prediction: str, reference: str) -> float:
    p, r = re.findall(r"\w+", prediction.lower()), re.findall(r"\w+", reference.lower())
    if not p or not r:
        return 0.0
    common = sum(min(p.count(w), r.count(w)) for w in set(p))
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


class Generator:
    def __init__(self, model: str, max_new_tokens: int, tensor_parallel: int = 1):
        from vllm import LLM, SamplingParams

        self.llm = LLM(model=model, tensor_parallel_size=tensor_parallel, max_model_len=8192, limit_mm_per_prompt={"image": 0, "audio": 0})
        self.params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    def chat(self, conversations: list[list[dict]]) -> list[str]:
        outputs = self.llm.chat(conversations, self.params, chat_template_kwargs={"enable_thinking": False})
        return [o.outputs[0].text for o in outputs]


def decision_metrics(episodes: list[dict], replies: list[str], trivial: set | None = None) -> dict:
    predicted, reference = set(), set()
    correct = invalid = none = 0
    for ep, reply in zip(episodes, replies):
        for g in ep.get("gold_all") or []:
            reference.add((ep["source_iri"], g))
        d = parse_decision(reply, ep)
        if not d.valid:
            invalid += 1
            continue
        if d.choice == "NONE":
            none += 1
            correct += not ep.get("gold")
        else:
            for choice in d.choices or (d.choice,):
                predicted.add((ep["source_iri"], choice))
            correct += set(d.choices or (d.choice,)) == set(ep.get("gold") or [])  # exact set match
    scores = AlignmentEvaluator().evaluate(predicted, reference, trivial=trivial)
    n = max(1, len(episodes))
    return {"episodes": len(episodes), "accuracy": correct / n, "invalid_rate": invalid / n, "none_rate": none / n,
            "precision": scores["Precision"], "recall": scores["Recall"], "f1": scores["F-Measure"],
            "recall_plus": scores["Recall+"], "predicted": scores["Size"], "reference": scores["reference"]}


def anatomy_trivial() -> set:
    path = raw_dir() / "anatomy-dataset-2" / "trivial.rdf"
    return {(a.source_iri, a.target_iri) for a in read_alignment(path)} if path.exists() else set()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--cap", type=int, default=2000, help="max decisions per eval set")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    gen = Generator(args.model, args.max_new_tokens, args.tp)
    results: dict = {"model": args.model, "tag": args.tag, "decisions": {}, "closed_book": {}}
    eval_dir = data_dir() / "eval"
    files = sorted(eval_dir.glob("*.jsonl"))
    out = out_dir() / "eval" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    with stage(f"eval_{args.tag}", total=len(files)) as bar:
        for path in files:
            rows = list(read_jsonl(path))
            random.Random(11).shuffle(rows)
            rows = rows[: args.cap]
            if path.stem.startswith("closed_book"):
                replies = gen.chat([[{"role": "system", "content": KNOWLEDGE_SYSTEM},
                                     {"role": "user", "content": r["prompt"]}] for r in rows])
                scores = [token_f1(reply, r["answer"]) for r, reply in zip(rows, replies)]
                results["closed_book"][path.stem] = {"n": len(rows), "token_f1": sum(scores) / max(1, len(scores))}
            else:
                replies = gen.chat([decision_messages(e) for e in rows])
                trivial = anatomy_trivial() if "anatomy" in path.stem else None
                results["decisions"][path.stem] = decision_metrics(rows, replies, trivial)
                by_family = defaultdict(list)
                for e, reply in zip(rows, replies):
                    by_family[e["family"]].append((e, reply))
            with (out / f"{path.stem}.predictions.jsonl").open("w") as f:
                for r, reply in zip(rows, replies):
                    f.write(json.dumps({"id": r["id"], "reply": reply}) + "\n")
            bar.advance()
    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

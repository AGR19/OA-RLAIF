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
import os
import random
import re
from collections import defaultdict
from pathlib import Path

from ..harness.prompts import KNOWLEDGE_SYSTEM, decision_messages, fully_correct, parse_decision
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


class HFGenerator:
    """Smoke test only (OA_SMOKE=1): greedy generation with transformers on CPU instead of vLLM."""

    def __init__(self, model: str, max_new_tokens: int):
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForImageTextToText.from_pretrained(model)
        self.max_new_tokens = max_new_tokens

    def chat(self, conversations: list[list[dict]]) -> list[str]:
        out = []
        for messages in conversations:
            ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                                     enable_thinking=False)
            ids = ids["input_ids"] if hasattr(ids, "keys") else ids
            gen = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            out.append(self.tokenizer.decode(gen[0, ids.shape[1]:], skip_special_tokens=True))
        return out


class Generator:
    def __new__(cls, model: str, max_new_tokens: int, tensor_parallel: int = 1):
        import os
        if os.environ.get("OA_SMOKE") == "1":
            return HFGenerator(model, min(max_new_tokens, 24))
        return super().__new__(cls)

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
            correct += fully_correct(d.choices or (d.choice,), ep.get("gold"))  # exact, under the 3-target limit
    scores = AlignmentEvaluator().evaluate(predicted, reference, trivial=trivial)
    n = max(1, len(episodes))
    return {"episodes": len(episodes), "accuracy": correct / n, "invalid_rate": invalid / n, "none_rate": none / n,
            "precision": scores["Precision"], "recall": scores["Recall"], "f1": scores["F-Measure"],
            "recall_plus": scores["Recall+"], "predicted": scores["Size"], "reference": scores["reference"]}


def icd_metrics(rows: list[dict], replies: list[str], out: Path) -> dict:
    """Per ICD pair. For SNOMED -> ICD-11 also the bridge recovery: the share of held-out SNOMED concepts whose
    predicted ICD-11 category is one of its composed links (retrieval misses count as failures), the retrieval
    ceiling, and the predictions as SSSOM (ICD-11 skos:narrowMatch SNOMED), i.e. the predicted ICD-11<->SNOMED map."""
    from ..ontology.alignment_io import write_sssom

    metrics = {}
    for pair in sorted({e["pair_id"] for e in rows}):
        sub = [(e, r) for e, r in zip(rows, replies) if e["pair_id"] == pair]
        m = decision_metrics([e for e, _ in sub], [r for _, r in sub])
        if pair.endswith("snomed-to-icd11"):
            hits, links = 0, []
            for e, reply in sub:
                d = parse_decision(reply, e)
                picks = [c for c in (d.choices or (d.choice,)) if d.valid and c and c != "NONE"]
                hits += bool(set(picks) & set(e.get("gold_all") or []))
                name = lambda card: next((l[6:] for l in card.splitlines() if l.startswith("name: ")), "")  # noqa: E731
                cards = {c["iri"]: c["card"] for c in e["candidates"]}
                links += [{"subject_id": p, "subject_label": name(cards.get(p, "")), "predicate_id": "skos:narrowMatch",
                           "object_id": e["source_iri"], "object_label": name(e["source_card"]),
                           "mapping_justification": "semapv:UnspecifiedMatching",
                           "confidence": d.confidence if d.confidence is not None else 1.0,
                           "mapping_tool": "oa-rlaif-e4b", "comment": "held-out ICD-10 categories"} for p in picks]
            m["bridge_recovery"] = hits / max(1, len(sub))
            m["retrieval_ceiling"] = sum(bool(e.get("gold")) for e, _ in sub) / max(1, len(sub))
            write_sssom(out / "icd11_snomed_predicted.sssom.tsv", links,
                        {"mapping_set_id": "https://github.com/AGR19/OA-RLAIF/icd11_snomed_predicted",
                         "license": "SNOMED CT affiliate licence applies; private use only",
                         "comment": "Model predictions for held-out SNOMED concepts; ICD-11 category classifies the concept."})
        metrics[f"heldout_icd:{pair.split('/')[-1]}"] = m
    return metrics


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
            if path.stem != "heldout_icd" or os.environ.get("OA_SMOKE") == "1":  # ICD set: in full on the GPU box
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
                if path.stem == "heldout_icd":
                    results["decisions"].update(icd_metrics(rows, replies, out))
            with (out / f"{path.stem}.predictions.jsonl").open("w") as f:
                for r, reply in zip(rows, replies):
                    f.write(json.dumps({"id": r["id"], "reply": reply}) + "\n")
            bar.advance()
    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

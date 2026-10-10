"""Knowledge questions for SDFT from every knowledge passage (datasource, DRAL-OA papers, benchmark docs).

    python -m oa_rlaif.data.datasource --mode api        # A4B via JUDGE_API_* writes grounded Q&A (recommended)
    python -m oa_rlaif.data.datasource --mode heuristic  # offline fallback: definitional/cloze questions

Each question is answered closed-book by the student; the frozen teacher sees the source passage as privileged
context, so after SDFT the knowledge lives in the weights. API answers are kept only if they are grounded in the
passage (token overlap check). Every passage is trained on; about 20% of each passage's QUESTIONS are held out
for the closed-book evaluation (question-level split), so no chapter of the datasource is withheld from training.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path

from ..api_client import RateLimiter, answer_text, chat, settings
from ..budget import BudgetGuard
from ..io import read_jsonl
from ..paths import data_dir
from ..progress import stage
from ..retrieval.candidates import normalize
from .splits import stable_fraction

GENERATE = (
    "You write study questions for a model that must learn this material without seeing it again.\n"
    "From the PASSAGE, write {n} question-answer pairs that test understanding of its ontology-matching "
    "concepts (definitions, techniques, measures, systems, trade-offs). Each answer must be fully supported by "
    "the passage, 1-3 sentences, and self-contained. Do not ask about page layout, figures or citations.\n"
    'Return only JSON: [{{"question": "...", "answer": "..."}}]\n\nPASSAGE\n{passage}')


def grounded(answer: str, passage: str, threshold: float = 0.6) -> bool:
    words = [w for w in normalize(answer).split() if len(w) > 3]
    if not words:
        return False
    text = normalize(passage)
    return sum(w in text for w in words) / len(words) >= threshold


def heuristic_questions(passage: dict, limit: int = 3) -> list[dict]:
    """Offline fallback: '<Term> is/are ...' sentences become definition questions."""
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", passage["text"]):
        match = re.match(r"^(?:An?|The)?\s*([A-Z][\w\- ]{2,60}?)\s+(is|are|consists of|refers to)\s+(.{20,300}?)\.$", sentence.strip())
        if match:
            term = match.group(1).strip()
            out.append({"question": f"In ontology matching, what {('is' if match.group(2) != 'are' else 'are')} {term.lower()}?",
                        "answer": sentence.strip()})
        if len(out) == limit:
            break
    return out


async def api_questions(passages: list[dict], n: int, guard: BudgetGuard, cache: Path) -> dict[str, list[dict]]:
    """One generation call per passage; each result is cached on disk, so an interrupted run resumes."""
    from openai import AsyncOpenAI

    cfg = settings()
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=600.0, max_retries=0)
    model = cfg["model"]
    limiter = RateLimiter(cfg["rpm"])
    price_in = float(os.environ.get("JUDGE_USD_PER_MTOK_IN", "0") or 0)
    price_out = float(os.environ.get("JUDGE_USD_PER_MTOK_OUT", "0") or 0)
    semaphore = asyncio.Semaphore(int(os.environ.get("JUDGE_CONCURRENCY", "8")))
    cache.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict]] = {}
    todo = []
    for p in passages:
        path = cache / f"{p['id']}.json"
        if path.exists():
            results[p["id"]] = json.loads(path.read_text())["pairs"]
        else:
            todo.append(p)

    async def one(p):
        async with semaphore:
            try:
                r = await chat(client, limiter, cfg["retries"], model=model, temperature=0.2,
                               max_tokens=cfg["max_tokens"],
                               messages=[{"role": "user", "content": GENERATE.format(n=n, passage=p["text"])}])
            except Exception as error:
                return p, None, str(error)[:300]
        if r.usage and (price_in or price_out):
            guard.add_api("question_generation", r.usage.prompt_tokens / 1e6 * price_in
                          + r.usage.completion_tokens / 1e6 * price_out, r.usage.prompt_tokens, r.usage.completion_tokens)
        text = answer_text(r.choices[0].message.content or "")
        match = re.search(r"\[.*\]", text, re.S)
        try:
            pairs = json.loads(match.group(0)) if match else []
        except json.JSONDecodeError:
            pairs = []
        return p, pairs, None

    with stage("datasource_api", total=len(todo)) as bar:
        for future in asyncio.as_completed([one(p) for p in todo]):
            p, pairs, error = await future
            if error is None:
                (cache / f"{p['id']}.json").write_text(json.dumps({"model": model, "pairs": pairs}, ensure_ascii=False))
                results[p["id"]] = pairs
            else:
                results[p["id"]] = [{"error": error}]
            bar.advance()
    return results


def _tokens(text: str) -> set[str]:
    return {w for w in normalize(text).split() if len(w) > 3}


def build(mode: str, per_passage: int = 5, test_fraction: float = 0.2, limit: int | None = None) -> dict:
    """Question-level split: every passage is trained on; ~test_fraction of each passage's questions are held out.

    A held-out question is dropped if it overlaps a training question from the same passage (Jaccard > 0.5), so
    the closed-book test asks about the material, not a paraphrase of a training prompt.
    """
    passages = list(read_jsonl(data_dir() / "knowledge" / "passages.jsonl"))[:limit]
    guard = BudgetGuard()
    cache = data_dir() / "knowledge" / "generated" / re.sub(r"[^\w.-]", "_", settings()["model"])
    generated = asyncio.run(api_questions(passages, per_passage, guard, cache)) if mode == "api" else {}
    counts = {"kept_train": 0, "kept_test": 0, "rejected_ungrounded": 0, "near_duplicate_test": 0, "api_errors": 0}
    units: dict[str, dict[str, int]] = {}
    out_path = data_dir() / "knowledge" / "questions.jsonl"
    with out_path.open("w", encoding="utf-8") as f, stage("datasource_questions", total=len(passages)) as bar:
        for p in passages:
            pairs = generated.get(p["id"], []) if mode == "api" else heuristic_questions(p)
            good = []
            for i, qa in enumerate(pairs):
                if "error" in qa:
                    counts["api_errors"] += 1
                    continue
                question, answer = str(qa.get("question", "")).strip(), str(qa.get("answer", "")).strip()
                if not question or not answer or (mode == "api" and not grounded(answer, p["text"])):
                    counts["rejected_ungrounded"] += 1
                    continue
                qid = hashlib.sha1(f"{p['id']}|{i}|{question}".encode()).hexdigest()[:16]
                good.append((qid, question, answer))
            held = {qid for qid, _, _ in good if stable_fraction(qid, "datasource-question") < test_fraction}
            if good and len(held) == len(good):  # every passage keeps at least one training question
                held.discard(good[0][0])
            train_tokens = [_tokens(q + " " + a) for qid, q, a in good if qid not in held]
            unit = p.get("unit") or p["source"]
            tally = units.setdefault(f"{p['source']} :: {unit}", {"passages": 0, "train": 0, "test": 0})
            tally["passages"] += 1
            for qid, question, answer in good:
                split = "test" if qid in held else "train"
                if split == "test":
                    tok = _tokens(question + " " + answer)
                    if any(len(tok & t) / max(1, len(tok | t)) > 0.5 for t in train_tokens):
                        counts["near_duplicate_test"] += 1
                        continue
                f.write(json.dumps({
                    "id": qid, "task": "knowledge_qa", "source": p["source"], "unit": unit, "passage_id": p["id"],
                    "split": split, "prompt": question, "answer": answer,
                    "privileged_context": f"Source passage:\n{p['text']}", "mode": mode,
                    "generator": settings()["model"] if mode == "api" else "heuristic"}, ensure_ascii=False) + "\n")
                counts[f"kept_{split}"] += 1
                tally[split] += 1
            bar.advance()
    missing = [u for u, t in units.items() if t["passages"] and (t["train"] == 0)]
    report = {"passages": len(passages), **counts, "mode": mode, "api_usd": round(guard.api_usd(), 4),
              "units_without_training_questions": missing, "units": units}
    (data_dir() / "knowledge" / "question_coverage.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return {k: v for k, v in report.items() if k != "units"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["api", "heuristic"], default="api")
    parser.add_argument("--per-passage", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None, help="first N passages only (smoke test)")
    args = parser.parse_args()
    if args.mode == "api" and not os.environ.get("JUDGE_API_KEY"):
        raise SystemExit("JUDGE_API_KEY is not set; fill .env or use --mode heuristic")
    print(json.dumps(build(args.mode, args.per_passage, args.test_fraction, args.limit), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

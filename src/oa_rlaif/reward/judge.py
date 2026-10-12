"""A4B judge (Gemma 4 26B-A4B) over any OpenAI-compatible API, scoring a decision 0-10 against a rubric.

The judge never sees gold. It sees the task, the candidates and the model's reply, and grades the reasoning:
evidence cited correctly, thought consistent with the submitted choice, no invented facts or identifiers, and
calibrated confidence. Calls are cached on disk (sqlite) by content hash, run at temperature 0, and fall back to
a neutral 5 on timeout or parse failure (counted, never silent). Spend is recorded in the budget ledger.

Quota control: `fraction` judges a deterministic subset of episodes (all generations of a selected prompt, so the
group-relative advantage stays consistent); unselected episodes get a constant, which cancels inside a GRPO group.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import threading
from pathlib import Path

from ..api_client import RateLimiter, answer_text, chat, settings
from ..budget import BudgetGuard
from ..harness.prompts import decision_user
from ..paths import ROOT, out_dir

# 'SCORE: 7', 'Score: 7.5', '**SCORE:** 8', 'score = 9/10'; a ':' or '=' is required ('a score of 3' does not count)
SCORE = re.compile(r"SCORE[\s*_]{0,3}[:=][\s*_]{0,3}(10(?:\.0+)?|[0-9](?:\.[0-9]+)?)", re.IGNORECASE)


def rubric() -> str:
    return (ROOT / "configs" / "judge_rubric.md").read_text()


class Judge:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 cache_path: Path | None = None, concurrency: int = 16, timeout: float = 180.0,
                 guard: BudgetGuard | None = None, fraction: float = 1.0):
        from openai import AsyncOpenAI

        cfg = settings()
        self.model = model or cfg["model"]
        self.client = AsyncOpenAI(base_url=base_url or cfg["base_url"], api_key=api_key or cfg["api_key"],
                                  timeout=timeout, max_retries=0)
        self.rpm, self.max_tokens, self.retries = cfg["rpm"], cfg["max_tokens"], cfg["retries"]
        self.semaphore = asyncio.Semaphore(concurrency)
        self.fraction = fraction
        self.unselected = 0
        self._next_start = 0.0
        self.cache_path = cache_path or out_dir() / "judge_cache.sqlite"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with sqlite3.connect(self.cache_path, timeout=30) as db:
            db.execute("create table if not exists judge (key text primary key, score real, raw text)")
        self.guard = guard or BudgetGuard()
        self.price_in = float(os.environ.get("JUDGE_USD_PER_MTOK_IN", "0") or 0)
        self.price_out = float(os.environ.get("JUDGE_USD_PER_MTOK_OUT", "0") or 0)
        self.fallbacks = 0
        self.cache_errors = 0
        self._rubric = rubric()

    def selected(self, episode: dict) -> bool:
        """Deterministic per episode, so every process judges the same prompts (whole GRPO groups)."""
        if self.fraction >= 1.0:
            return True
        return int(hashlib.sha256(f"judge|{episode['id']}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < self.fraction

    def _key(self, episode: dict, completion: str) -> str:
        return hashlib.sha256(f"{self.model}|{episode['id']}|{completion}".encode()).hexdigest()

    # The cache is shared by the training processes (one per GPU): wait for SQLite's lock, and never let a cache
    # problem stop training (a miss only means one more judge call).
    def _cached(self, key: str):
        try:
            with self._lock, sqlite3.connect(self.cache_path, timeout=30) as db:
                row = db.execute("select score from judge where key = ?", (key,)).fetchone()
        except sqlite3.Error:
            self.cache_errors += 1
            return None
        return row[0] if row else None

    def _store(self, key: str, score: float, raw: str) -> None:
        try:
            with self._lock, sqlite3.connect(self.cache_path, timeout=30) as db:
                db.execute("insert or replace into judge values (?, ?, ?)", (key, score, raw))
        except sqlite3.Error:
            self.cache_errors += 1

    async def score(self, episode: dict, completion: str, limiter: RateLimiter | None = None) -> float:
        if not self.selected(episode):
            self.unselected += 1
            return 5.0
        key = self._key(episode, completion)
        cached = self._cached(key)
        if cached is not None:
            return cached
        messages = [{"role": "system", "content": self._rubric},
                    {"role": "user", "content": f"TASK\n{decision_user(episode)}\n\nREPLY TO GRADE\n{completion}\n\n"
                                                "Grade the reply. End with a line 'SCORE: <0-10>'."}]
        async with self.semaphore:
            try:
                response = await chat(self.client, limiter, self.retries, model=self.model, messages=messages,
                                      temperature=0.0, max_tokens=self.max_tokens)
            except Exception:
                self.fallbacks += 1
                return 5.0
        raw = response.choices[0].message.content or ""
        text = answer_text(raw)
        usage = getattr(response, "usage", None)
        if usage is not None and (self.price_in or self.price_out):
            self.guard.add_api("judge", usage.prompt_tokens / 1e6 * self.price_in
                               + usage.completion_tokens / 1e6 * self.price_out,
                               usage.prompt_tokens, usage.completion_tokens)
        matches = SCORE.findall(text) or SCORE.findall(raw)
        if not matches:
            self.fallbacks += 1
            return 5.0
        value = max(0.0, min(10.0, float(matches[-1])))
        self._store(key, value, json.dumps(text[-400:]))
        return value

    def score_batch(self, episodes: list[dict], completions: list[str]) -> list[float]:
        async def run():
            limiter = RateLimiter(self.rpm, self._next_start)  # created inside the running loop
            try:
                return await asyncio.gather(*(self.score(e, c, limiter) for e, c in zip(episodes, completions)))
            finally:
                self._next_start = limiter.next_start
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():  # called from inside an event loop: run in a worker thread
            result: list = []
            thread = threading.Thread(target=lambda: result.extend(asyncio.run(run())))
            thread.start()
            thread.join()
            return result
        return asyncio.run(run())

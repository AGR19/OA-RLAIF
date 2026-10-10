"""The one remote dependency: the A4B (Gemma 4 26B-A4B) OpenAI-compatible API, shared by the judge and datasource.

Settings come from .env:
  JUDGE_API_BASE    e.g. https://generativelanguage.googleapis.com/v1beta/openai/  (Google AI Studio)
  JUDGE_MODEL       e.g. gemma-4-26b-a4b-it  (AI Studio name; OpenRouter-style "google/...:free" names 404 there)
  JUDGE_RPM         client-side requests/minute across all processes (AI Studio free tier: check your quota page)
  JUDGE_MAX_TOKENS  completion budget; Gemma 4 on AI Studio always thinks first, so keep this large
  JUDGE_EXTRA_BODY  JSON merged into each request, e.g. {"chat_template_kwargs": {"enable_thinking": false}} for the
                    local vLLM judge on Lambda (scripts/_common.sh judge_up sets all of these for GRPO)

Gemma 4 on AI Studio returns its reasoning inline as <thought>...</thought> before the answer and does not accept a
thinking budget; `answer_text` strips it. Rate-limit (429) and server errors are retried with exponential backoff.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time

from . import paths  # noqa: F401  (loads .env so settings() sees JUDGE_*)

THOUGHT = re.compile(r"<(thought|think)>.*?(?:</\1>|$)", re.S)


def answer_text(text: str) -> str:
    """The reply without inline reasoning; falls back to the reasoning when no answer follows it (truncation)."""
    answer = THOUGHT.sub("", text or "").strip()
    if answer:
        return answer
    inner = re.search(r"<(thought|think)>(.*?)(?:</\1>|$)", text or "", re.S)
    return inner.group(2).strip() if inner else ""


def settings() -> dict:
    world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    return {"base_url": os.environ.get("JUDGE_API_BASE") or None,
            "api_key": os.environ.get("JUDGE_API_KEY") or "missing",
            "model": os.environ.get("JUDGE_MODEL", "gemma-4-26b-a4b-it"),
            "rpm": float(os.environ.get("JUDGE_RPM", "0") or 0) / world,  # each trainer process gets its share
            "max_tokens": int(os.environ.get("JUDGE_MAX_TOKENS", "4096")),
            "retries": int(os.environ.get("JUDGE_RETRIES", "6")),
            "extra_body": json.loads(os.environ.get("JUDGE_EXTRA_BODY") or "{}")}


class RateLimiter:
    """Spaces request starts at least 60/rpm seconds apart (rpm <= 0 disables it)."""

    def __init__(self, rpm: float, next_start: float = 0.0):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.next_start = next_start  # carried across event loops by the caller (asyncio.Lock is per loop)
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        if not self.interval:
            return
        async with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_start - now)
            self.next_start = max(now, self.next_start) + self.interval
        if delay:
            await asyncio.sleep(delay)


async def chat(client, limiter: RateLimiter | None, retries: int = 6, **kwargs):
    """chat.completions.create with retries on 429 / 5xx / timeouts (2, 4, 8, ... s backoff, capped at 60 s)."""
    from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

    extra = settings()["extra_body"]
    if extra and "extra_body" not in kwargs:
        kwargs["extra_body"] = extra

    for attempt in range(retries + 1):
        if limiter is not None:
            await limiter.wait()
        try:
            return await client.chat.completions.create(**kwargs)
        except (RateLimitError, APITimeoutError, APIConnectionError) as error:
            last = error
        except APIStatusError as error:
            if error.status_code < 500:
                raise
            last = error
        if attempt < retries:
            await asyncio.sleep(min(60.0, 2.0 ** (attempt + 1)))
    raise last

"""Arm A: pplx-decider alone answers every class (one candidate question per class; dm_harness.questions).

    python -m dm_harness.base_run --tasks anatomy__mouse-human,dh [--url http://127.0.0.1:8090] [--concurrency 512]

Reads the built episodes (full_eval.build_episodes), sends every class of a shard to the System-One server at once
(the server packs the waiting requests into GPU batches) and writes replies/decider/<task>/<shard>.replies.jsonl.gz,
which full_eval.score scores like any other method (tag "decider"). A shard whose replies exist is skipped, so a
stopped run continues; outputs/full_eval/STOP stops it between shards (the money guard, exit 3).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from full_eval.build_episodes import read_shard
from full_eval.infer import write_replies
from full_eval.schedule import root

from .questions import read, reply_row, request

TAG = "decider"


async def ask(session, url: str, episode: dict, sem: asyncio.Semaphore) -> dict:
    body = json.dumps(request(episode))
    for attempt in range(4):
        async with sem:
            t0 = time.monotonic()
            try:
                async with session.post(f"{url}/v1/systemone", data=body,
                                        headers={"Content-Type": "application/json"}) as r:
                    data = await r.json(content_type=None)
                    if r.status == 200:
                        return reply_row(episode, read(episode, data), time.monotonic() - t0, data.get("usage"))
                    error = f"HTTP {r.status}: {str(data)[:200]}"
            except Exception as e:  # noqa: BLE001 - retried, then recorded as no submission
                error = f"{type(e).__name__}: {e}"
        await asyncio.sleep(2 ** attempt)
    return {"id": episode["id"], "decision": "no submission", "targets": [], "relation": None, "confidence": None,
            "exit": f"decider_error: {error}"[:300], "model_calls": 0, "turns": [], "seconds": 0.0}


async def run_shard(url: str, episodes: list[dict], concurrency: int) -> list[dict]:
    from aiohttp import ClientSession, ClientTimeout, TCPConnector

    sem = asyncio.Semaphore(concurrency)
    async with ClientSession(timeout=ClientTimeout(total=1800), connector=TCPConnector(limit=concurrency)) as session:
        return await asyncio.gather(*(ask(session, url, e, sem) for e in episodes))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dm_harness.base_run")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--concurrency", type=int, default=512)
    args = ap.parse_args(argv)
    progress = root() / "progress" / f"{TAG}.0.json"
    progress.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(progress.read_text()) if progress.exists() else {"tasks": {}}
    for slug in [s for s in args.tasks.split(",") if s]:
        folder = root() / "episodes" / slug
        while not (folder / "READY").exists():
            if (folder / "FAILED").exists():
                print(f"[decider] {slug}: episodes failed; skipped", flush=True)
                break
            time.sleep(10)
        if not (folder / "READY").exists():
            continue
        shards = sorted(json.loads((folder / "manifest.json").read_text())["shards"])
        out = root() / "replies" / TAG / slug
        out.mkdir(parents=True, exist_ok=True)
        timing = state["tasks"].setdefault(slug, {"decisions": 0, "busy_seconds": 0.0, "prompt_tokens": 0,
                                                  "completion_tokens": 0, "start": None, "end": None})
        for name in shards:
            target = out / name.replace(".jsonl.gz", ".replies.jsonl.gz")
            if target.exists():
                continue
            if (root() / "STOP").exists():
                return 3
            episodes = read_shard(folder / name)
            started = time.time()
            timing["start"] = timing["start"] or started
            rows = asyncio.run(run_shard(args.url, episodes, args.concurrency))
            write_replies(target, rows)
            timing["decisions"] += len(rows)
            timing["busy_seconds"] += time.time() - started
            timing["prompt_tokens"] += sum(r.get("prompt_tokens") or 0 for r in rows)
            timing["end"] = time.time()
            state["task"], state["in_shard"] = slug, 0
            progress.write_text(json.dumps(state))
            print(f"[decider] {slug}/{name}: {len(rows)} classes, {len(rows) / max(1e-9, time.time() - started):.1f}/s",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Arm B's worker: runs dm_harness.harness.HarnessMatcher over every class of the given tasks, through the running
JevAgentHarness proxy (dm_harness.run starts the decision model, vLLM and the proxy first).

    python -m dm_harness.harness_run --tasks icd__snomed-to-icd11,cyber__iotsec-uco

Replies: replies/jah-agent/<task>/<shard>.replies.jsonl.gz (the agent-row layout full_eval.score reads, plus the
guard's record), progress/jah-agent.0.json (timings, and how far the current shard is, for the progress bar). A shard
whose replies exist is skipped; outputs/full_eval/STOP stops it between shards (exit 3). The agent's commands run in
the user sandbox on the GPU box (OA_AGENT_ENV=user), locally in the Mac rehearsal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from full_eval.agent_run import row_of
from full_eval.build_episodes import read_shard
from full_eval.infer import write_replies
from full_eval.schedule import root

from .harness import HarnessMatcher
from .stack import fake

TAG = "jah-agent"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dm_harness.harness_run")
    ap.add_argument("--tasks", required=True)
    args = ap.parse_args(argv)
    sandbox = os.environ.get("OA_AGENT_ENV") or ("local" if fake() else "user")
    workers = int(os.environ.get("DM_AGENT_WORKERS", "8" if fake() else "256"))
    matcher = HarnessMatcher(root() / "agent_tasks" / TAG, workers, sandbox, save_trajectory=False,
                             keep_task_dirs=False)
    progress = root() / "progress" / f"{TAG}.0.json"
    progress.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(progress.read_text()) if progress.exists() else {"tasks": {}}
    state.update(tag=TAG, sandbox=sandbox, workers=workers)
    with ThreadPoolExecutor(workers) as pool:
        for slug in [s for s in args.tasks.split(",") if s]:
            folder = root() / "episodes" / slug
            while not (folder / "READY").exists():
                if (root() / "STOP").exists():
                    return 3
                if (folder / "FAILED").exists():
                    break
                time.sleep(10)
            if not (folder / "READY").exists():
                continue
            shards = sorted(json.loads((folder / "manifest.json").read_text())["shards"])
            out = root() / "replies" / TAG / slug
            out.mkdir(parents=True, exist_ok=True)
            state["task"] = slug
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
                records = []
                for i, record in enumerate(pool.map(matcher.run_episode, episodes), 1):
                    records.append(record)
                    if i % 50 == 0:
                        state["in_shard"] = i
                        progress.write_text(json.dumps(state))
                state["in_shard"] = 0
                write_replies(target, [row_of(e, r) for e, r in zip(episodes, records)])
                timing["decisions"] += len(records)
                timing["busy_seconds"] += time.time() - started
                timing["prompt_tokens"] += sum(r.get("prompt_tokens") or 0 for r in records)
                timing["completion_tokens"] += sum(r.get("completion_tokens") or 0 for r in records)
                timing["end"] = time.time()
                progress.write_text(json.dumps(state))
                print(f"[harness] {slug}/{name}: {len(records)} classes, "
                      f"{len(records) / max(1e-9, time.time() - started):.1f}/s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

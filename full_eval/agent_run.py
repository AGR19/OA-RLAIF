"""GPU worker for the agent method, All merged + mini-SWE-agent: the om_agent/ harness of the local console.

    CUDA_VISIBLE_DEVICES=1 python -m full_eval.agent_run --tag all_merged-agent --tasks anatomy__mouse-human --rank 1 --world 4
    CUDA_VISIBLE_DEVICES=1 python -m full_eval.agent_run --tag all_merged-agent-think --tasks dh --rank 1 --world 4

The worker starts `vllm serve` for the All merged weights on its GPU (port 8100 + rank, served as oa-rlaif-e4b,
context 32,768) and runs om_agent's Matcher, the same agent as the local console (om_agent/test_matching_ai.py), over
its share of every episode shard: one mini-SWE-agent episode per class, FULL_EVAL_AGENT_WORKERS (64) at a time.
Each episode: the scripted `om profile`, `om show <class>`, `om candidates <class> --k N`, then the model with the
trained agent prompt (step limit 6) until `om submit`. The episodes and candidates are the ones every other method
gets (full_eval/build_episodes.py), so the comparison with All merged is like for like.

The agent's commands run in the user sandbox (OA_AGENT_ENV=user, set up by scripts/sandbox_setup.sh: the
unprivileged user oa-agent, a task folder outside the repo with the public episode only, the repo closed to it). The
agent is never given a reference and never asks for one: the references are read from the raw folders afterwards, by
full_eval/score.py, for scoring only.

Replies: replies/<tag>/<task>/<shard>.replies.jsonl.gz, one row per class: the submission (decision, targets,
relation, confidence), exit status, the agent's own commands, model calls, format errors, tokens, seconds, and each
model turn (answer, thinking, finish reason). Trajectories are not kept per class; the turns plus the episode
reproduce them (the `om` outputs are deterministic). FULL_EVAL_FAKE=lexical (Mac rehearsal): no vLLM; a lexical
matcher answers in the model's format through LiteLLM's mock response, so every other step runs for real.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .build_episodes import read_shard
from .infer import LexicalEngine, write_replies
from .schedule import method, root

SERVED = "oa-rlaif-e4b"
MAX_MODEL_LEN = int(os.environ.get("FULL_EVAL_AGENT_MAX_LEN", "32768"))


def serve(model_dir: Path, tag: str, rank: int) -> tuple[subprocess.Popen, str]:
    """`vllm serve` on this worker's GPU; returns once it serves the model (15 minutes at most)."""
    port = 8100 + rank
    log = open(root() / "logs" / f"vllm_{tag}_{rank}.log", "a")
    args = [str(Path(sys.executable).parent / "vllm"), "serve", str(model_dir), "--port", str(port),
            "--served-model-name", SERVED, "--max-model-len", str(MAX_MODEL_LEN),
            "--gpu-memory-utilization", os.environ.get("FULL_EVAL_GPU_UTIL", "0.88"),
            "--max-num-seqs", os.environ.get("FULL_EVAL_AGENT_MAX_SEQS", "256"),
            "--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0})]
    quant = os.environ.get("FULL_EVAL_QUANT")
    if quant:
        args += ["--quantization", quant]
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
    base = f"http://127.0.0.1:{port}"
    for _ in range(180):
        if proc.poll() is not None:
            raise SystemExit(f"vllm serve stopped during start-up; see {log.name}")
        try:
            with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as r:
                if SERVED in r.read().decode():
                    return proc, f"{base}/v1"
        except OSError:
            pass
        time.sleep(5)
    proc.terminate()
    raise SystemExit(f"vllm serve did not start within 15 minutes; see {log.name}")


def fake_model_class(base, think: bool):
    """FULL_EVAL_FAKE=lexical: the model class with LiteLLM's mock response in place of the server. The reply is the
    lexical matcher's (full_eval.infer.LexicalEngine) for the episode named in the task message."""
    import litellm

    from oa_rlaif.harness.prompts import decision_messages

    engine, episodes = LexicalEngine(think), {}

    class FakeModel(base):
        def _query(self, messages, **kwargs):
            task = next(m["content"] for m in messages if m["role"] == "user")
            iri, pair = re.match(r"Align the source entity (\S+) \((.+)\)\.", task).groups()
            reply = engine.chat([decision_messages(episodes[(iri, pair)])])[0][0]
            return litellm.completion(model="openai/fake", messages=messages, mock_response=reply)

    return FakeModel, episodes


def row_of(episode: dict, record: dict) -> dict:
    record = dict(record)
    record.pop("source", None)
    return {"id": episode["id"], **record}


def main(argv: list[str] | None = None) -> int:
    from om_agent.test_matching_ai import Matcher

    from .models import model_path

    parser = argparse.ArgumentParser(prog="full_eval.agent_run")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    args = parser.parse_args(argv)
    m = method(args.tag)
    if not m.agent:
        raise SystemExit(f"{args.tag} is not the agent method: full_eval.infer runs it")
    fake = os.environ.get("FULL_EVAL_FAKE") == "lexical"
    sandbox = os.environ.get("OA_AGENT_ENV") or ("local" if fake else "user")
    workers = int(os.environ.get("FULL_EVAL_AGENT_WORKERS", "8" if fake else "64"))
    (root() / "logs").mkdir(parents=True, exist_ok=True)
    started = time.time()
    server, api_base = (None, None) if fake else serve(model_path(m.model), args.tag, args.rank)
    matcher = Matcher(SERVED, root() / "agent_tasks" / args.tag, workers, sandbox, think=m.think, backend="vllm",
                      api_base=api_base, save_trajectory=False, keep_task_dirs=False)
    known: dict = {}
    if fake:
        matcher.Model, known = fake_model_class(matcher.Model, m.think)
    progress = root() / "progress" / f"{args.tag}.{args.rank}.json"
    progress.parent.mkdir(parents=True, exist_ok=True)
    state = {"tag": args.tag, "rank": args.rank, "load_seconds": round(time.time() - started, 1), "decisions": 0,
             "busy_seconds": 0.0, "task": None, "started": time.time(), "tasks": {}, "sandbox": sandbox}
    if progress.exists():  # a restarted worker keeps the earlier per-task timings
        state["tasks"] = json.loads(progress.read_text()).get("tasks", {})
    progress.write_text(json.dumps(state))
    try:
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
                out_dir = root() / "replies" / args.tag / slug
                out_dir.mkdir(parents=True, exist_ok=True)
                state["task"] = slug
                timing = state["tasks"].setdefault(slug, {"decisions": 0, "busy_seconds": 0.0, "prompt_tokens": 0,
                                                          "completion_tokens": 0, "start": None, "end": None})
                for name in shards[args.rank::args.world]:
                    target = out_dir / name.replace(".jsonl.gz", ".replies.jsonl.gz")
                    if target.exists():
                        continue
                    if (root() / "STOP").exists():
                        return 3
                    episodes = read_shard(folder / name)
                    if fake:
                        known.update({(e["source_iri"], e["pair_id"]): e for e in episodes})
                    shard_start = time.time()
                    timing["start"] = timing["start"] or shard_start
                    records = list(pool.map(matcher.run_episode, episodes))
                    write_replies(target, [row_of(e, r) for e, r in zip(episodes, records)])
                    spent = time.time() - shard_start
                    state["decisions"] += len(episodes)
                    state["busy_seconds"] += spent
                    timing["decisions"] += len(episodes)
                    timing["busy_seconds"] += spent
                    timing["prompt_tokens"] += sum(r["prompt_tokens"] for r in records)
                    timing["completion_tokens"] += sum(r["completion_tokens"] for r in records)
                    timing["end"] = time.time()
                    progress.write_text(json.dumps(state))
        state["finished"] = time.time()
        progress.write_text(json.dumps(state))
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""GPU worker: one model on one GPU answers its share of every episode shard of the given tasks (vLLM, offline).

    CUDA_VISIBLE_DEVICES=2 python -m full_eval.infer --tag grpo --tasks anatomy__mouse-human,dh --rank 2 --world 4
    python -m full_eval.infer --tag grpo --verify          # the rebuilt model against the earlier run's predictions

Generation is the earlier evaluation's (oa_rlaif.eval.run_eval.Generator): the trained chat format with thinking off,
temperature 0, at most 256 new tokens, context 8,192. Worker r takes shards r, r+world, ... of each task; a shard whose
replies file exists is skipped, so a stopped run continues where it stopped. The worker waits for a task's episodes
(READY) and stops between shards when outputs/full_eval/STOP exists (the money guard).
Replies: outputs/full_eval/replies/<tag>/<task>/<shard>.replies.jsonl.gz (id, reply, prompt and completion tokens).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

from .build_episodes import read_shard
from .schedule import root

MAX_NEW_TOKENS = 256


class LexicalEngine:
    """FULL_EVAL_FAKE=lexical (Mac rehearsal, no GPU): a lexical matcher in the model's reply format. It picks the
    candidate whose name and synonyms share the most label 3-grams with the source (Jaccard), or NONE below 0.5.
    Exercises every step except vLLM; its scores are a sanity check, never a model result."""

    load_seconds = 0.0

    @staticmethod
    def _names(block: str) -> list[str]:
        out = []
        for line in block.splitlines():
            if line.startswith("name: "):
                out.append(line[6:])
            elif line.startswith("synonyms: "):
                out += [s.split(" @")[0] for s in line[10:].split("; ")]
        return out

    def chat(self, conversations):
        import re

        from oa_rlaif.retrieval.candidates import grams

        replies = []
        for messages in conversations:
            text = messages[-1]["content"]
            source_block, _, cand_text = text.partition("\nCANDIDATES\n")
            source = set().union(*(grams(n) for n in self._names(source_block.split("SOURCE\n", 1)[-1]))) or {""}
            best, best_i = 0.0, None
            for i, block in enumerate(re.split(r"\n?\[C\d+\]\n", "\n" + cand_text)[1:], 1):
                for name in self._names(block):
                    g = grams(name)
                    score = len(source & g) / max(1, len(source | g))
                    if score > best:
                        best, best_i = score, i
            choice = f"C{best_i}" if best_i and best >= 0.5 else "NONE"
            replies.append((f"THOUGHT: lexical 3-gram overlap {best:.2f}.\n```mswea_bash_command\necho "
                            f"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && om submit {choice} --relation = --confidence "
                            f"{max(best, 0.5):.2f}\n```", 0, 0))
        return replies


class Engine:
    def __new__(cls, model: str):
        if os.environ.get("FULL_EVAL_FAKE") == "lexical":
            return LexicalEngine()
        return super().__new__(cls)

    def __init__(self, model: str):
        from vllm import LLM, SamplingParams

        started = time.time()
        quant = os.environ.get("FULL_EVAL_QUANT") or None  # e.g. fp8: faster, but not the evaluated bf16 weights
        self.llm = LLM(model=model, tensor_parallel_size=1, max_model_len=8192, quantization=quant,
                       limit_mm_per_prompt={"image": 0, "audio": 0},
                       gpu_memory_utilization=float(os.environ.get("FULL_EVAL_GPU_UTIL", "0.88")))
        self.params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
        self.load_seconds = time.time() - started

    def chat(self, conversations: list[list[dict]]):
        outputs = self.llm.chat(conversations, self.params, chat_template_kwargs={"enable_thinking": False},
                                use_tqdm=False)
        return [(o.outputs[0].text, len(o.prompt_token_ids or []), len(o.outputs[0].token_ids)) for o in outputs]


def write_replies(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=3) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def verify(engine: Engine, tag: str) -> dict:
    """Answer the earlier held-out Anatomy set and compare with that run's predictions for the same model: a check
    that the rebuilt (re-merged) weights are the evaluated ones. Exact agreement is expected up to rare ties."""
    from oa_rlaif.harness.prompts import decision_messages, parse_decision
    from oa_rlaif.io import read_jsonl
    from oa_rlaif.paths import data_dir

    episodes_path = data_dir() / "eval" / "heldout_anatomy.jsonl"
    earlier_path = root() / "earlier" / tag / "heldout_anatomy.predictions.jsonl"
    if not episodes_path.exists() or not earlier_path.exists():
        return {"tag": tag, "checked": 0, "note": "earlier predictions or episodes not downloaded"}
    episodes = list(read_jsonl(episodes_path))
    earlier = {r["id"]: r["reply"] for r in read_jsonl(earlier_path)}
    episodes = [e for e in episodes if e["id"] in earlier]
    replies = engine.chat([decision_messages(e) for e in episodes])
    same_text = same_decision = 0
    for e, (reply, _, _) in zip(episodes, replies):
        same_text += reply.strip() == earlier[e["id"]].strip()
        a, b = parse_decision(reply, e), parse_decision(earlier[e["id"]], e)
        same_decision += (a.valid, a.choices) == (b.valid, b.choices)
    n = max(1, len(episodes))
    return {"tag": tag, "checked": len(episodes), "same_reply": round(same_text / n, 4),
            "same_decision": round(same_decision / n, 4)}


def main(argv: list[str] | None = None) -> int:
    from oa_rlaif.harness.prompts import decision_messages

    from .models import model_path

    parser = argparse.ArgumentParser(prog="full_eval.infer")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--verify", action="store_true", help="only the check against the earlier predictions")
    parser.add_argument("--verify-first", action="store_true", help="the check, then the tasks (rank 0 of a rebuilt model)")
    args = parser.parse_args(argv)
    engine = Engine(str(model_path(args.tag)))
    progress = root() / "progress" / f"{args.tag}.{args.rank}.json"
    progress.parent.mkdir(parents=True, exist_ok=True)
    state = {"tag": args.tag, "rank": args.rank, "load_seconds": round(engine.load_seconds, 1), "decisions": 0,
             "busy_seconds": 0.0, "task": None, "started": time.time(), "tasks": {}}
    if progress.exists():  # a restarted worker keeps the earlier per-task timings
        state["tasks"] = json.loads(progress.read_text()).get("tasks", {})
    progress.write_text(json.dumps(state))
    if args.verify or (args.verify_first and not (root() / f"verify_{args.tag}.json").exists()):
        result = verify(engine, args.tag)
        (root() / f"verify_{args.tag}.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result), flush=True)
        if args.verify:
            return 0
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
            started = time.time()
            timing["start"] = timing["start"] or started
            replies = engine.chat([decision_messages(e) for e in episodes])
            write_replies(target, [{"id": e["id"], "reply": r, "prompt_tokens": p, "completion_tokens": c}
                                   for e, (r, p, c) in zip(episodes, replies)])
            spent = time.time() - started
            state["decisions"] += len(episodes)
            state["busy_seconds"] += spent
            timing["decisions"] += len(episodes)
            timing["busy_seconds"] += spent
            timing["prompt_tokens"] += sum(p for _, p, _ in replies)
            timing["completion_tokens"] += sum(c for _, _, c in replies)
            timing["end"] = time.time()
            progress.write_text(json.dumps(state))
    state["finished"] = time.time()
    progress.write_text(json.dumps(state))
    return 0


if __name__ == "__main__":
    sys.exit(main())

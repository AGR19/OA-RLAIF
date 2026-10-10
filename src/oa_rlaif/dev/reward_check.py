"""Check the GRPO reward path end to end: valid gold replies through the reward functions, judge call included.

Invalid replies never reach the judge (they score 0), so a model that writes only invalid replies (the random smoke
model, or an early GRPO step) does not prove the judge works. This check builds the rewards exactly as
train/grpo.py does (DRAL scorer + Judge from the JUDGE_* environment) and scores the gold reply of real GRPO
episodes. Used by scripts/smoke_mac.sh (stub judge) and scripts/35_dry_run.sh (local A4B judge, before GRPO).

    python -m oa_rlaif.dev.reward_check --n 8
"""

from __future__ import annotations

import argparse
import json

from ..budget import BudgetGuard
from ..harness.prompts import gold_answer
from ..paths import data_dir
from ..reward.combined import make_reward_functions
from ..reward.dral_reward import DralScorer
from ..train.common import grpo_paths, load_config, load_jsonl_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4)
    args = parser.parse_args()
    cfg = load_config()
    from ..reward.judge import Judge

    train = load_jsonl_dataset(grpo_paths()[0]).select(range(args.n))
    scorer = DralScorer(sorted((data_dir() / "dral").glob("*.npz")), temperature=cfg["grpo"]["dral_temperature"])
    judge = Judge(guard=BudgetGuard(), fraction=1.0)
    (task_fn, judge_fn), weights = make_reward_functions(scorer=scorer, judge=judge)
    episodes = [json.loads(r) if isinstance(r, str) else r for r in train["episode"]]
    completions = [[{"role": "assistant", "content": gold_answer(e)}] for e in episodes]
    task = task_fn(train["prompt"], completions, train["episode"])
    judged = judge_fn(train["prompt"], completions, train["episode"])
    total = [weights[0] * t + weights[1] * j for t, j in zip(task, judged)]
    print(json.dumps({"task": task, "judge": judged, "total": total, "judge_fallbacks": judge.fallbacks}))
    if judge.fallbacks:
        raise SystemExit(f"judge path failed: {judge.fallbacks} judge call(s) fell back (server down or unparsable reply)")
    if not any(j > 0 for j in judged):
        raise SystemExit("judge path failed: no valid gold reply got a judge score above 0")
    if not all(t > 0 for t in task):
        raise SystemExit("task reward failed: a gold reply did not get a positive task reward")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

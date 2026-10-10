"""TRL GRPO reward functions: R = 0.8 * R_task + 0.2 * J / 10, logged as separate terms.

TRL calls each function as f(prompts, completions, **dataset_columns) and combines them with
GRPOConfig.reward_weights. We pass the episode JSON in the `episode` column and use weights [0.8, 0.2]:
  task_reward_fn  -> R_task (gold match + DRAL-OA Eq. 1), 0 for invalid replies
  judge_reward_fn -> J / 10 from the A4B judge, 0 for invalid replies (no credit for good prose around a broken
                     submission)
Diagnostics (gold, dral, validity) are collected for logging and the reward-hacking monitor.
"""

from __future__ import annotations

import json
from collections import deque

from ..harness.prompts import parse_decision
from .task_reward import TaskRewardConfig, task_reward

RECENT: deque = deque(maxlen=4096)  # (episode_id, kind, valid, gold, dral, task, judge) for monitoring


def _text(completion) -> str:
    if isinstance(completion, list):  # conversational completions: [{"role": "assistant", "content": ...}]
        return completion[-1]["content"] if completion else ""
    return completion


def make_reward_functions(scorer=None, judge=None, cfg: TaskRewardConfig | None = None):
    cfg = cfg or TaskRewardConfig()

    def task_reward_fn(prompts, completions, episode, **kwargs):
        rewards = []
        for completion, raw in zip(completions, episode):
            ep = json.loads(raw) if isinstance(raw, str) else raw
            result = task_reward(_text(completion), ep, scorer, cfg)
            RECENT.append((ep["id"], ep.get("kind"), result["valid"], result.get("gold"), result.get("dral"),
                           result["task"], None))
            rewards.append(float(result["task"]))
        return rewards

    def judge_reward_fn(prompts, completions, episode, **kwargs):
        episodes = [json.loads(raw) if isinstance(raw, str) else raw for raw in episode]
        texts = [_text(c) for c in completions]
        valid = [parse_decision(t, e).valid for t, e in zip(texts, episodes)]
        if judge is None:
            return [0.5 if v else 0.0 for v in valid]
        todo = [(e, t) for e, t, v in zip(episodes, texts, valid) if v]
        scores = iter(judge.score_batch([e for e, _ in todo], [t for _, t in todo])) if todo else iter(())
        return [next(scores) / 10.0 if v else 0.0 for v in valid]

    task_reward_fn.__name__ = "task_reward"
    judge_reward_fn.__name__ = "judge_reward"
    return [task_reward_fn, judge_reward_fn], [0.8, 0.2]

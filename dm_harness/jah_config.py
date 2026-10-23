"""The JevAgentHarness proxy's configuration for arm B (written to outputs/full_eval/jah_config.yaml).

One upstream, vLLM with the All-merged weights; two tiers on it that differ only in thinking (the tier's `extra` is
merged into every request body, so it switches Gemma 4's chat template). The decision model is pplx-decider behind
the local System-One server. The thresholds are fixed before the run (the user's decision, 2026-10-23): routing
tau = 0.75, the repository's default; no reference answer is used to choose them.

  jah-auto    routed: pplx-decider answers "routine or hard?" before each model call; hard -> strong (thinking on)
  jah-strong  every step on the strong tier (the NONE/invalid guard's redo, dm_harness.harness)
  jah-fast    every step on the fast tier
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .stack import DECIDER_PORT, JAH_PORT, SERVED, VLLM_PORT

ROUTING_TAU = 0.75
GUARD_TAU = 0.5  # the NONE/invalid guard (dm_harness.harness), fixed before the run as well
THINK_TOKENS = 4096


def config(log_dir: Path, tau: float = ROUTING_TAU) -> dict[str, Any]:
    tier = lambda think: {  # noqa: E731
        "upstream": "vllm", "model": SERVED,
        "extra": {"chat_template_kwargs": {"enable_thinking": think},
                  # thinking on keeps Gemma 4's channel markers, so the agent can split the thinking off
                  "skip_special_tokens": not think}}
    return {
        "upstreams": {"vllm": {"type": "openai", "base_url": f"http://127.0.0.1:{VLLM_PORT}/v1", "key_env": "",
                               "timeout_s": 900.0, "retries": 3}},
        "tiers": {"strong": tier(True), "fast": tier(False)},
        "router": {
            "decision": {"type": "jev", "provider": "local", "base_url": f"http://127.0.0.1:{DECIDER_PORT}",
                         "model": "pplx-decider-v1.2", "timeout_s": 120.0},
            "tau": tau,
            # the agent's first model call comes after the three scripted steps (om profile / show / candidates),
            # so it is step 3 and is routed by the decision model
            "first_steps_strong": 1,
            "verify_finish": True,  # needs native tool calls; the text-based agent's submission is guarded in harness.py
            "on_error": "strong",
            "decision_timeout_s": 120.0,
        },
        "models": {"jah-auto": {"policy": "route"}, "jah-strong": {"tier": "strong"}, "jah-fast": {"tier": "fast"}},
        "listen": {"host": "127.0.0.1", "port": JAH_PORT},
        "log_dir": str(log_dir),
    }


def write(path: Path, log_dir: Path, tau: float = ROUTING_TAU) -> Path:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config(log_dir, tau), sort_keys=False))
    return path

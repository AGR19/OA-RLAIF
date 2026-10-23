"""The JevAgentHarness proxy's configuration for arm B (written to outputs/full_eval/jah_config.yaml).

One upstream, vLLM with the All-merged weights; two tiers on it that differ only in thinking (the tier's `extra` is
merged into every request body, so it switches Gemma 4's chat template). The decision model is pplx-decider behind
the local System-One server. Arm B's thresholds are fixed before the run (the user's decision, 2026-10-23): routing
tau = 0.75, the repository's default, and the guard's 0.5. Arm C's are fitted on a calibration sample
(dm_harness.calibrate) and frozen: `jah-fit` carries the fitted routing tau.

  jah-auto    routed: pplx-decider answers "routine or hard?" before each model call; hard -> strong (thinking on)
  jah-strong  every step on the strong tier (the NONE/invalid guard's redo, dm_harness.harness)
  jah-fast    every step on the fast tier
  jah-shadow  calibration (dm_harness.calibrate): every step on the strong tier, while the routing question is asked
              and its p(hard) logged (routing: shadow)
  jah-fit     arm C: routed at the threshold fitted on the calibration sample (written once the fit exists)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .stack import DECIDER_PORT, JAH_PORT, SERVED, VLLM_PORT

ROUTING_TAU = 0.75
GUARD_TAU = 0.5  # the NONE/invalid guard (dm_harness.harness), fixed before the run as well
THINK_TOKENS = 4096


def config(log_dir: Path, tau: float = ROUTING_TAU, fitted_tau: float | None = None) -> dict[str, Any]:
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
        "models": {"jah-auto": {"policy": "route"}, "jah-strong": {"tier": "strong"}, "jah-fast": {"tier": "fast"},
                   "jah-shadow": {"policy": "route", "decisions": {"routing": {"mode": "shadow"}}},
                   **({"jah-fit": {"policy": "route", "tau": fitted_tau}} if fitted_tau is not None else {})},
        "listen": {"host": "127.0.0.1", "port": JAH_PORT},
        "log_dir": str(log_dir),
    }


def write(path: Path, log_dir: Path, tau: float = ROUTING_TAU, fitted_tau: float | None = None) -> Path:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config(log_dir, tau, fitted_tau), sort_keys=False))
    return path

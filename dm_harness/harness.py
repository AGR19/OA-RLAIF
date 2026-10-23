"""Arm B's agent: om_agent's Matcher (the local console's mini-SWE-agent) behind the JevAgentHarness proxy, plus the
NONE/invalid guard.

Each class runs as one agent episode with the model name `jah-auto`: the proxy asks pplx-decider before the model
call whether the step is routine (fast tier, thinking off) or hard (strong tier, thinking on). The class's calls go to
the proxy's per-session URL `/s/<episode id>/v1` (a redo: `/s/<episode id>-redo/v1`), so the proxy's step log
(session = "<model name>/<episode id>") joins to the class without logging any prompt text.

The guard (the paper's GUARD mode at the candidate decision; the proxy's own finish check needs native tool calls,
which this text-based agent does not send): when the submission is NONE, or the episode ends without a valid
submission, pplx-decider answers the candidate question (dm_harness.questions).
  NONE,    best candidate p >= GUARD_TAU   -> the class is redone on the strong tier (`jah-strong`) with a note naming
                                              that candidate; Gemma decides again
  NONE,    p < GUARD_TAU                   -> NONE stays
  invalid                                  -> redone on the strong tier (with the note when p >= GUARD_TAU, as the
                                              proxy's fail-safe does for errors); still invalid -> pplx-decider's pick
The row keeps the first attempt and the guard's record, so the report can say what the guard changed.
"""

from __future__ import annotations

import json
import urllib.request

from om_agent.test_matching_ai import Matcher

from .jah_config import GUARD_TAU
from .questions import NONE, option_names, read, request
from .stack import DECIDER_PORT, JAH_PORT

VALID = ("match", "NONE")


def decide(episode: dict, url: str = f"http://127.0.0.1:{DECIDER_PORT}", timeout: float = 300) -> dict:
    data = json.dumps(request(episode)).encode()
    req = urllib.request.Request(f"{url}/v1/systemone", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return read(episode, json.loads(r.read()))


def hint(episode: dict, option: str) -> str:
    """The note added to the task text of a redo: the candidate the decision model finds likely, by the name the agent
    sees (C#) and its label."""
    names = option_names(episode)
    card = episode["candidates"][names.index(option)]["card"]
    label = next((line[6:] for line in card.splitlines() if line.startswith("name: ")), "")
    return (f"NOTE: a second check of the candidates suggests candidate {option} ({label}) may be the target. Look at "
            "it again before you submit; submit NONE only if no candidate fits.")


class HarnessMatcher(Matcher):
    def __init__(self, out, workers: int, sandbox: str, proxy: str = f"http://127.0.0.1:{JAH_PORT}",
                 decider_url: str = f"http://127.0.0.1:{DECIDER_PORT}", guard_tau: float = GUARD_TAU, **kwargs):
        # think=True: max_tokens 4096 for the strong tier; the proxy's tier decides thinking on or off
        super().__init__("jah-auto", out, workers, sandbox=sandbox, think=True, backend="vllm",
                         api_base=f"{proxy}/v1", **kwargs)
        self.proxy, self.decider_url, self.guard_tau = proxy, decider_url, guard_tau

    def session(self, name: str) -> dict:
        """The proxy's per-session URL: its step log then names the session after the class."""
        return {"api_base": f"{self.proxy}/s/{name}/v1"}

    def run_episode(self, episode: dict, on_message=None, approve=None, model_name=None, model_kwargs=None,
                    note: str = "") -> dict:
        if model_name or note:  # a redo (below), or a caller's own override: no guard around it
            return super().run_episode(episode, on_message, approve, model_name, model_kwargs, note)
        first = super().run_episode(episode, model_kwargs=self.session(episode["id"]))
        if first["decision"] == "match":
            return first
        invalid = first["decision"] not in VALID
        try:
            check = decide(episode, self.decider_url)
        except Exception as error:  # noqa: BLE001 - no decision model: the first answer stands (fail-safe)
            return {**first, "guard": {"reason": "invalid" if invalid else "none", "outcome": "decider_error",
                                       "error": f"{type(error).__name__}: {error}"[:200]}}
        option, p = check["best_candidate"], check["p_best_candidate"]
        guard = {"reason": "invalid" if invalid else "none", "suggested": option, "p": round(p, 4),
                 "first": {k: first[k] for k in ("decision", "targets", "exit")}}
        if not invalid and (option is None or p < self.guard_tau):
            return {**first, "guard": {**guard, "outcome": "kept_none"}}
        redo = super().run_episode(episode, model_name="jah-strong", model_kwargs=self.session(episode["id"] + "-redo"),
                                   note=hint(episode, option) if option and p >= self.guard_tau else "")
        merged = {**redo, "model_calls": first["model_calls"] + redo["model_calls"],
                  "seconds": round(first["seconds"] + redo["seconds"], 2),
                  "prompt_tokens": first["prompt_tokens"] + redo["prompt_tokens"],
                  "completion_tokens": first["completion_tokens"] + redo["completion_tokens"],
                  "thinking_chars": first["thinking_chars"] + redo["thinking_chars"],
                  "turns": first["turns"] + redo["turns"]}
        if redo["decision"] in VALID:
            return {**merged, "guard": {**guard, "outcome": f"redo_{redo['decision'].lower()}"}}
        pick = check["pick"]  # still invalid: the decision model's own answer (its most likely option)
        return {**merged, "decision": "NONE" if pick == NONE else "match", "targets": [] if pick == NONE else [pick],
                "relation": episode.get("relation", "="), "confidence": round(check["p"], 4),
                "guard": {**guard, "outcome": "decider_pick"}}

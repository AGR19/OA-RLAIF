"""DM_FAKE=lexical (Mac rehearsal): an OpenAI-compatible stand-in for vLLM behind the JevAgentHarness proxy.

    python -m dm_harness.fake_upstream --port 8100

POST /v1/chat/completions answers the agent's model call in the trained reply format with full_eval.infer's lexical
matcher, for the episode named in the task message ("Align the source entity <iri> (<pair>)"), read from the built
episodes. Thinking on (the strong tier's chat_template_kwargs) wraps the reply in Gemma 4's thinking channel, as
vLLM returns it. A redo with the guard's hint ("... suggests candidate C7 ...") submits that candidate, so the redo
path runs too. Its answers are a rehearsal of the plumbing, never a result.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from typing import Any

from full_eval.infer import LexicalEngine
from full_eval.schedule import root
from oa_rlaif.harness.prompts import decision_messages

try:  # module level, so FastAPI can resolve the (string) annotation under `from __future__ import annotations`
    from fastapi import Request
except ImportError:  # pragma: no cover - the server extra is not installed
    Request = Any  # type: ignore[misc,assignment]

TASK = re.compile(r"Align the source entity (\S+) \((.+?)\)\.")
HINT = re.compile(r"suggests candidate (C\d+)")
_EPISODES: dict[tuple[str, str], dict] = {}


def episode(iri: str, pair: str) -> dict | None:
    if not _EPISODES:
        for path in (root() / "episodes").glob("*/*.jsonl.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    e = json.loads(line)
                    _EPISODES[(e["source_iri"], e["pair_id"])] = e
    return _EPISODES.get((iri, pair))


def answer(body: dict) -> str:
    think = bool((body.get("chat_template_kwargs") or {}).get("enable_thinking"))
    text = "\n".join(str(m.get("content") or "") for m in body.get("messages", []) if m.get("role") == "user")
    found = TASK.search(text)
    ep = episode(*found.groups()) if found else None
    if ep is None:
        return "THOUGHT: no episode.\n```mswea_bash_command\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && om submit NONE --relation = --confidence 0.5\n```"
    hint = HINT.search(text)
    if hint:
        reply = (f"THOUGHT: the second check points to {hint.group(1)}; it fits.\n```mswea_bash_command\n"
                 f"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && om submit {hint.group(1)} --relation "
                 f"{ep.get('relation', '=')} --confidence 0.8\n```")
        return f"<|channel>thought\nthe hint.<channel|>{reply}" if think else reply
    (raw, *_), = LexicalEngine(think).chat([decision_messages(ep)])
    return raw


def build_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.get("/v1/models")
    def models() -> dict:
        return {"data": [{"id": "oa-rlaif-e4b", "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(req: Request) -> JSONResponse:
        body = await req.json()
        content = answer(body)
        return JSONResponse({
            "id": f"fake-{time.time_ns()}", "object": "chat.completion", "created": int(time.time()),
            "model": body.get("model", "oa-rlaif-e4b"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": sum(len(str(m.get("content") or "")) for m in body.get("messages", [])) // 4,
                      "completion_tokens": len(content) // 4,
                      "total_tokens": 0}})

    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    ap = argparse.ArgumentParser(prog="dm_harness.fake_upstream")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args(argv)
    uvicorn.run(build_app(), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())

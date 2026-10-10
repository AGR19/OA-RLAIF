#!/usr/bin/env python3
"""Stand-in OpenAI-compatible server for the Mac smoke test (stdlib only).

- A4B judge requests (they contain "REPLY TO GRADE") get "SCORE: 7".
- Agent requests get a scripted tool session: om profile -> om show -> om candidates -> om submit C1,
  so the real mini-SWE-agent loop, LiteLLM client, Docker sandbox and `om` tool are exercised end to end.

    python3 scripts/dev/stub_openai.py 8099
"""

import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def fence(cmd: str, thought: str) -> str:
    return f"THOUGHT: {thought}\n```mswea_bash_command\n{cmd}\n```"


def reply(messages: list[dict]) -> str:
    text = "\n".join(str(m.get("content", "")) for m in messages)
    if "REPLY TO GRADE" in text:
        return "The evidence cited matches the candidate card.\nSCORE: 7"
    match = re.search(r"Align source entity (\S+) \(", text)
    iri = match.group(1) if match else "x"
    turns = sum(1 for m in messages if m.get("role") == "assistant")
    script = [fence("om profile", "check the modelling profiles"), fence(f"om show {iri}", "read the source card"),
              fence(f"om candidates {iri} --k 20", "list the candidates"),
              fence("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && om submit C1 --relation = --confidence 0.8",
                    "C1 matches")]
    return script[min(turns, len(script) - 1)]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body: dict, code: int = 200):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": "stub", "object": "model"}]} if "models" in self.path else {"ok": True})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        content = reply(body.get("messages", []))
        self._send({"id": f"stub-{time.time_ns()}", "object": "chat.completion", "created": int(time.time()),
                    "model": body.get("model", "stub"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}})


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8099), Handler).serve_forever()

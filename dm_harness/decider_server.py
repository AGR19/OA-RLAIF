"""pplx-decider-v1.2 behind the Jev System-One wire format (POST /v1/systemone), one worker process per GPU, with
concurrent requests packed into batches.

    .venv-decider/bin/python -m dm_harness.decider_server --checkpoint outputs/dm_models/pplx-decider-v1.2 \\
        --devices 0,1,2,3 --port 8090
    DM_FAKE=lexical .venv/bin/python -m dm_harness.decider_server --port 8090      # Mac rehearsal (no GPU)

JevAgentHarness's decision backend (`{type: jev, provider: local, base_url: http://127.0.0.1:8090}`) and this
repo's clients (dm_harness.base_run, dm_harness.harness) send

    {"state": str, "model": str, "questions": {id: {"type": "choice", "instructions": str, "criteria": {...}}}}

pplx_decider takes the same schema without "model" (it rejects unknown keys), so the request is cleaned and passed
on; its answer {"answers": {id: {"choice", "probabilities", "confidence"}}, "usage": {...}} is what the proxy reads.
The model card's fast path (`fast=True`: packing, compiled layers, CUDA graphs) runs on each GPU; batches are formed
from the requests that wait while a GPU is busy (up to --batch requests).

Endpoints: POST /v1/systemone, GET /v1/models, GET /health, GET /stats (requests, mean latency, requests/s).
DM_FAKE=lexical answers with a lexical stand-in (label 3-gram overlap for candidate choices; 'hard' when the state
shows an error or a format problem): every step but the model runs for real.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

try:  # module level, so FastAPI can resolve the (string) annotation under `from __future__ import annotations`
    from fastapi import Request
except ImportError:  # pragma: no cover - the server extra is not installed
    Request = Any  # type: ignore[misc,assignment]

MODEL = "pplx-decider-v1.2"
REQUEST_KEYS = ("state", "questions", "images")
QUESTION_KEYS = ("type", "instructions", "criteria")


def clean(body: dict[str, Any]) -> dict[str, Any]:
    """The request as pplx_decider accepts it: only state, questions (type, instructions, criteria) and images."""
    out = {k: body[k] for k in REQUEST_KEYS if k in body}
    out["questions"] = {qid: {k: q[k] for k in QUESTION_KEYS if k in q} for qid, q in (body.get("questions") or {}).items()}
    return out


# ---- the lexical stand-in (DM_FAKE=lexical) ---------------------------------------------------------------------------
def _grams(text: str) -> set[str]:
    t = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    return {t[i:i + 3] for i in range(max(0, len(t) - 2))}


def _name(text: str) -> str:
    m = re.search(r"name:\s*([^|\n]+)", text or "")
    return m.group(1).strip() if m else (text or "")[:80]


def fake_answer(request: dict[str, Any]) -> dict[str, Any]:
    state = str(request.get("state", ""))
    answers = {}
    for qid, q in request["questions"].items():
        criteria = q.get("criteria") or {}
        if q.get("type") == "noul":
            answers[qid] = {"type": "noul", "noul": 0.5}
            continue
        options = list(criteria) if isinstance(criteria, dict) else list(criteria)
        if set(options) == {"routine", "hard"}:
            hard = 0.85 if re.search(r"error|format|traceback|invalid", state, re.I) else 0.2
            probs = {"routine": 1 - hard, "hard": hard}
        else:
            source = _grams(_name(state.split("SOURCE ENTITY:", 1)[-1]))
            scores = {}
            for o in options:
                if o == "NONE":
                    scores[o] = 0.25
                else:
                    g = _grams(_name(str(criteria.get(o) or o)))
                    scores[o] = len(source & g) / max(1, len(source | g))
            total = sum(v ** 2 for v in scores.values()) or 1.0
            probs = {o: v ** 2 / total for o, v in scores.items()}
        best = max(probs, key=probs.get)
        answers[qid] = {"type": "choice", "choice": best, "confidence": probs[best], "probabilities": probs}
    return {"answers": answers, "usage": {"input_tokens": len(state) // 4, "output_tokens": 0, "sequences": 1}}


# ---- GPU workers ------------------------------------------------------------------------------------------------------
def _worker(device: str, checkpoint: str, fast: bool, conn) -> None:  # pragma: no cover - GPU only
    """One GPU: load the decider once, then answer batches until told to stop."""
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    sys.path.insert(0, str(Path(checkpoint) / "source" / "src"))
    from pplx_decider import Decider

    decider = Decider.from_pretrained(checkpoint, device="cuda", fast=fast)
    if fast:
        decider.warmup()
    conn.send({"ready": device})
    while True:
        batch = conn.recv()
        if batch is None:
            break
        try:
            conn.send(decider.decide(batch))
        except Exception:  # noqa: BLE001 - one bad request must not fail its batch: answer one by one
            out = []
            for req in batch:
                try:
                    out.append(decider.decide(req))
                except Exception as error:  # noqa: BLE001
                    out.append({"error": f"{type(error).__name__}: {error}"[:300]})
            conn.send(out)


class Pool:
    """One worker process per GPU; each takes the requests waiting in the shared queue as one batch."""

    def __init__(self, devices: list[str], checkpoint: str, fast: bool, batch: int):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self.batch = batch
        self.workers = []
        for device in devices:
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(device, checkpoint, fast, child), daemon=True)
            proc.start()
            self.workers.append((proc, parent))
        for proc, conn in self.workers:
            msg = conn.recv()  # waits for the model to load (and compile)
            print(f"[decider] GPU {msg['ready']} ready", flush=True)
        self.queue: asyncio.Queue | None = None

    async def start(self) -> None:
        self.queue = asyncio.Queue()
        for _, conn in self.workers:
            asyncio.get_running_loop().create_task(self._serve(conn))

    async def _serve(self, conn) -> None:
        loop = asyncio.get_running_loop()
        while True:
            items = [await self.queue.get()]
            while len(items) < self.batch and not self.queue.empty():
                items.append(self.queue.get_nowait())

            def call(reqs=[r for r, _ in items]):
                conn.send(reqs)
                return conn.recv()

            try:
                results = await loop.run_in_executor(None, call)
            except Exception as error:  # noqa: BLE001
                results = [{"error": f"{type(error).__name__}: {error}"[:300]}] * len(items)
            for (_, fut), res in zip(items, results):
                if not fut.done():
                    fut.set_result(res)

    async def answer(self, request: dict[str, Any]) -> dict[str, Any]:
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((request, fut))
        return await fut


def build_app(pool: Pool | None):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    @asynccontextmanager
    async def lifespan(_app):
        if pool is not None:
            await pool.start()  # the batching loops run on the server's event loop
        yield

    app = FastAPI(lifespan=lifespan)
    stats = {"requests": 0, "errors": 0, "seconds": 0.0, "started": time.time()}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "model": MODEL, "workers": len(pool.workers) if pool else 0, "fake": pool is None}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"data": [{"id": MODEL, "object": "model"}]}

    @app.get("/stats")
    def get_stats() -> dict[str, Any]:
        n = max(1, stats["requests"])
        return {**stats, "mean_latency_s": stats["seconds"] / n,
                "requests_per_s": stats["requests"] / max(1e-9, time.time() - stats["started"])}

    @app.post("/v1/systemone")
    async def systemone(req: Request) -> JSONResponse:
        t0 = time.monotonic()
        try:
            body = clean(await req.json())
        except Exception as error:  # noqa: BLE001
            return JSONResponse({"error": f"bad request: {error}"}, status_code=400)
        out = fake_answer(body) if pool is None else await pool.answer(body)
        stats["requests"] += 1
        stats["seconds"] += time.monotonic() - t0
        if "error" in out:
            stats["errors"] += 1
            return JSONResponse(out, status_code=400 if "RequestError" in out["error"] else 500)
        out.setdefault("usage", {})["cost"] = 0.0  # self-hosted: the proxy logs GPU time, not a price
        out["model"] = MODEL
        return JSONResponse(out)

    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    ap = argparse.ArgumentParser(prog="dm_harness.decider_server")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--devices", default="0")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--batch", type=int, default=int(os.environ.get("DM_DECIDER_BATCH", "64")))
    ap.add_argument("--no-fast", action="store_true", help="the reference path (slower; bit-stable)")
    args = ap.parse_args(argv)
    fake = os.environ.get("DM_FAKE") == "lexical"
    pool = None if fake else Pool([d for d in args.devices.split(",") if d], args.checkpoint, not args.no_fast,
                                  args.batch)
    print(f"[decider] {'lexical stand-in' if fake else MODEL} on port {args.port}", flush=True)
    uvicorn.run(build_app(pool), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())

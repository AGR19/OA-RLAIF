#!/usr/bin/env python3
"""Live dashboard for scripts/lambda_all.sh (standard library only, so it also works before the venv exists).

Reads outputs/stages/ (written by lambda_all.sh) and outputs/progress.json (written by every training and evaluation
stage) and prints: the overall bar, the current step and its own bar, elapsed time, ETA, a spend estimate and the
last log line of the running step.

    python3 scripts/dashboard.py            # one frame (lambda_all.sh redraws it every few seconds)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES_DIR = ROOT / "outputs" / "stages"
RATE = float(os.environ.get("OA_GPU_USD_PER_HOUR", "27.16"))

# (name, estimated minutes, what happens). Estimates feed the ETA until real progress is known.
PLAN = [
    ("setup", 25, "install packages, download Gemma 4 E4B + A4B judge, Docker sandbox, llama.cpp"),
    ("data", 5, "download the private datasets from Hugging Face"),
    ("check_env", 5, "check GPUs, libraries, data and the Docker agent sandbox"),
    ("gguf_check", 15, "convert the base model to GGUF to prove export works"),
    ("dry_run", 35, "time SDFT and GRPO (with judge), plan the step counts"),
    ("baseline", 45, "evaluate the base E4B (batch + agent)"),
    ("sdft", 330, "Stage 1: on-policy self-distillation (knowledge into weights)"),
    ("eval_sdft", 60, "evaluate the SDFT model (batch + agent)"),
    ("grpo", 330, "Stage 2: GRPO with DRAL-OA reward + A4B judge (RLAIF)"),
    ("eval_grpo", 90, "evaluate GRPO: held-out, zero-shot, full Anatomy OAEI row, agent"),
    ("report", 2, "results tables: base vs SDFT vs GRPO, agents, OAEI Anatomy row"),
    ("export", 30, "merge and convert to GGUF Q4_K_M + Ollama Modelfile"),
    ("push", 15, "upload GGUF, adapters and results to private Hugging Face"),
]


def fmt(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    seconds = max(0, int(seconds))
    h, m = divmod(seconds // 60, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {seconds % 60:02d}s"


def bar(fraction: float, width: int = 32) -> str:
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return "█" * filled + "░" * (width - filled)


def read(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def stage_fraction(started: float, minutes: float, now: float) -> tuple[float, str]:
    """Progress of the running step: the newest progress.json record started after it, else elapsed / estimate."""
    progress = (read(ROOT / "outputs" / "progress.json", {}) or {}).get("stages", {})
    live = [v for v in progress.values() if v.get("status") == "running" and v.get("started", 0) >= started - 5
            and v.get("fraction") is not None]
    if live:
        newest = max(live, key=lambda v: v["started"])
        return float(newest["fraction"]), "measured"
    return min(0.95, (now - started) / (minutes * 60)), "estimated"


def last_log_line(name: str) -> str:
    path = ROOT / "outputs" / "logs" / f"stage_{name}.log"
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4000))
            tail = f.read().decode("utf-8", "ignore").replace("\r", "\n").strip().splitlines()
        return tail[-1][:110] if tail else ""
    except OSError:
        return ""


def frame(now: float | None = None) -> str:
    now = now or time.time()
    run_start = float((STAGES_DIR / "run.start").read_text()) if (STAGES_DIR / "run.start").exists() else now
    total = sum(m for _, m, _ in PLAN)
    done_minutes, current, lines = 0.0, None, []
    for i, (name, minutes, what) in enumerate(PLAN, 1):
        if (STAGES_DIR / f"{name}.done").exists():
            done_minutes += minutes
            status = "done"
        elif (STAGES_DIR / f"{name}.failed").exists():
            status = "FAILED"
        elif (STAGES_DIR / f"{name}.start").exists() and current is None:
            current = (i, name, minutes, what, float((STAGES_DIR / f"{name}.start").read_text()))
            status = "running"
        else:
            status = "waiting"
        mark = {"done": "✔", "running": "▶", "FAILED": "✖", "waiting": "·"}[status]
        lines.append(f"  {mark} {i:2d}. {name:<11} {status:<8} {what}")
    fraction, how, step_line = done_minutes / total, "", ""
    if current:
        i, name, minutes, what, started = current
        f, how = stage_fraction(started, minutes, now)
        fraction = (done_minutes + f * minutes) / total
        step_left = (now - started) * (1 - f) / f if f > 0.02 else minutes * 60
        step_line = (f"STEP {i}/{len(PLAN)}  {name}: {what}\n"
                     f"     [{bar(f)}] {f:6.1%}  elapsed {fmt(now - started)}  left ~{fmt(step_left)} ({how})\n"
                     f"     log: {last_log_line(name)}")
    remaining = (1 - fraction) * total * 60
    elapsed = now - run_start
    head = (f"OA-RLAIF on Lambda · {time.strftime('%H:%M:%S')} · logs in outputs/logs/stage_<step>.log\n\n"
            f"TOTAL [{bar(fraction, 40)}] {fraction:6.1%}   elapsed {fmt(elapsed)}   ETA ~{fmt(remaining)}"
            f"   GPU spend so far ≈ ${elapsed / 3600 * RATE:,.0f} (stop at $450)\n")
    return head + ("\n" + step_line + "\n" if step_line else "") + "\n" + "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--clear":
        sys.stdout.write("\033[2J\033[H")
    print(frame())

"""Progress bars, ETAs and a whole-pipeline status view.

Every script wraps its work in `stage(...)`. A stage shows a live bar with rate and ETA, and its measured
duration is saved to outputs/progress.json. `python -m oa_rlaif.progress status` (or scripts/status.sh)
prints the overall pipeline bar: completed stages, the running stage, and an ETA for everything left.
Remaining estimates are rescaled by how fast completed stages actually ran compared with their estimates.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path

from .budget import BudgetGuard
from .paths import out_dir

# Default estimates in minutes (Mac build stages run free; Lambda stages are billable).
PIPELINE: list[tuple[str, str, float]] = [
    ("prepare_data", "mac", 90),
    ("datasource", "mac", 40),
    ("icd_triangulation", "mac", 30),
    ("fit_dral", "mac", 45),
    ("verify_and_push_data", "mac", 20),
    ("check_env", "lambda", 10),
    ("dry_run", "lambda", 30),
    ("baseline", "lambda", 45),
    ("sdft", "lambda", 360),
    ("eval_sdft", "lambda", 45),
    ("grpo", "lambda", 360),
    ("eval_grpo", "lambda", 75),
    ("export", "lambda", 30),
    ("push_models", "lambda", 10),
]


def _state_path() -> Path:
    return out_dir() / "progress.json"


def _load() -> dict:
    path = _state_path()
    return json.loads(path.read_text()) if path.exists() else {"stages": {}}


def _save(state: dict) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(path)


def _update(name: str, **fields) -> dict:
    """Read-merge-write one stage entry so concurrent processes don't drop each other's stages."""
    state = _load()
    state["stages"].setdefault(name, {}).update(fields)
    _save(state)
    return state


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = int(max(0, seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {secs:02d}s"


def _estimate_minutes(name: str) -> float:
    return next((m for n, _, m in PIPELINE if n == name), 30.0)


def overall(state: dict | None = None) -> dict:
    """Overall completion fraction and ETA (seconds) for the remaining pipeline."""
    state = state or _load()
    stages = state["stages"]
    done_est = done_real = 0.0
    for name, _, minutes in PIPELINE:
        info = stages.get(name, {})
        if info.get("status") == "done" and info.get("seconds"):
            done_est += minutes * 60
            done_real += info["seconds"]
    speed = (done_real / done_est) if done_est else 1.0  # >1 means slower than estimated
    remaining = 0.0
    total = sum(m for _, _, m in PIPELINE) * 60
    finished = 0.0
    for name, _, minutes in PIPELINE:
        info = stages.get(name, {})
        if info.get("status") == "done":
            finished += minutes * 60
            continue
        left = minutes * 60 * speed
        if info.get("status") == "running" and info.get("fraction") is not None:
            left *= max(0.0, 1.0 - info["fraction"])
            finished += minutes * 60 * info["fraction"]
        remaining += left
    return {"fraction": finished / total if total else 0.0, "eta_seconds": remaining, "speed_factor": speed}


class _Plain:
    """Fallback renderer when rich is unavailable or output is not a terminal."""

    def __init__(self, name: str, total: float | None):
        self.name, self.total, self.done, self.start, self.last = name, total, 0.0, time.time(), 0.0

    def advance(self, n: float = 1.0) -> None:
        self.done += n
        now = time.time()
        if now - self.last >= 10 or (self.total and self.done >= self.total):
            self.last = now
            if self.total:
                fraction = min(1.0, self.done / self.total)
                elapsed = now - self.start
                eta = elapsed / fraction - elapsed if fraction else None
                bar = "█" * int(fraction * 30) + "░" * (30 - int(fraction * 30))
                print(f"[{self.name}] {bar} {fraction:6.1%}  {self.done:.0f}/{self.total:.0f}"
                      f"  elapsed {fmt_duration(elapsed)}  ETA {fmt_duration(eta)}", file=sys.stderr, flush=True)
            else:
                print(f"[{self.name}] {self.done:.0f} done  elapsed {fmt_duration(now - self.start)}",
                      file=sys.stderr, flush=True)


class StageHandle:
    def __init__(self, name: str, total: float | None, bar, task, state: dict, guard: BudgetGuard | None):
        self.name, self.total, self._bar, self._task, self._state, self._guard = name, total, bar, task, state, guard
        self._done = 0.0
        self._last_save = 0.0

    def set_total(self, total: float) -> None:
        self.total = total
        if self._task is not None:
            self._bar.update(self._task, total=total)
        else:
            self._bar.total = total

    def advance(self, n: float = 1.0) -> None:
        self._done += n
        if self._task is not None:
            self._bar.advance(self._task, n)
        else:
            self._bar.advance(n)
        now = time.time()
        if now - self._last_save > 15:
            self._last_save = now
            _update(self.name, fraction=min(1.0, self._done / self.total) if self.total else None)
            if self._guard is not None:
                self._guard.check(self.name)


@contextlib.contextmanager
def stage(name: str, total: float | None = None, guard: BudgetGuard | None = None):
    """Wrap a pipeline stage: live bar + ETA, persisted timing, and an optional budget check."""
    state = _update(name, status="running", started=time.time(), fraction=0.0)
    progress = None
    try:
        if sys.stderr.isatty():
            from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TextColumn,
                                       TimeElapsedColumn, TimeRemainingColumn)

            progress = Progress(TextColumn(f"[bold]{name}"), BarColumn(), MofNCompleteColumn(),
                                TextColumn("{task.percentage:>5.1f}%"), TimeElapsedColumn(),
                                TextColumn("ETA"), TimeRemainingColumn())
            progress.start()
            task = progress.add_task(name, total=total)
            handle = StageHandle(name, total, progress, task, state, guard)
        else:
            handle = StageHandle(name, total, _Plain(name, total), None, state, guard)
    except ImportError:
        handle = StageHandle(name, total, _Plain(name, total), None, state, guard)
    started = time.time()
    try:
        yield handle
    except BaseException:
        _update(name, status="failed", seconds=time.time() - started)
        raise
    finally:
        if progress is not None:
            progress.stop()
    state = _update(name, status="done", seconds=time.time() - started, fraction=1.0)
    if guard is not None:
        guard.tick(name)
    summary = overall(state)
    print(f"[{name}] done in {fmt_duration(time.time() - started)} · pipeline {summary['fraction']:.0%}"
          f" · ETA for everything left {fmt_duration(summary['eta_seconds'])}", file=sys.stderr, flush=True)


def status() -> str:
    state = _load()
    summary = overall(state)
    guard = BudgetGuard()
    fraction = summary["fraction"]
    bar = "█" * int(fraction * 40) + "░" * (40 - int(fraction * 40))
    lambda_left = sum(m for n, where, m in PIPELINE
                      if where == "lambda" and state["stages"].get(n, {}).get("status") != "done") / 60
    projected = guard.spent() + lambda_left * summary["speed_factor"] * guard.rate_usd_per_hour
    lines = [
        f"Pipeline [{bar}] {fraction:5.1%}   ETA {fmt_duration(summary['eta_seconds'])}"
        f"   (speed factor {summary['speed_factor']:.2f})",
        f"Budget   spent ${guard.spent():.2f} (GPU ${guard.gpu_usd():.2f}, API ${guard.api_usd():.2f})"
        f" · projected ${projected:.2f} · stop ${guard.stop_usd:.0f} · cap ${guard.budget_usd:.0f}",
        "",
        f"{'stage':<22}{'where':<8}{'status':<10}{'took / est':>16}",
    ]
    for name, where, minutes in PIPELINE:
        info = state["stages"].get(name, {})
        took = fmt_duration(info.get("seconds")) if info.get("seconds") else "-"
        lines.append(f"{name:<22}{where:<8}{info.get('status', 'queued'):<10}{took + ' / ' + fmt_duration(minutes * 60):>16}")
    return "\n".join(lines)


def _bar(fraction: float, width: int = 30) -> str:
    filled = int(max(0.0, min(1.0, fraction)) * width)
    return "█" * filled + "░" * (width - filled)


def live_text(now: float | None = None) -> str:
    """Running stages with live ETA, the remaining Mac queue (outputs/queue.json), Lambda plan and budget."""
    now = now or time.time()
    state = _load()["stages"]
    queue_path = out_dir() / "queue.json"
    queue = json.loads(queue_path.read_text()) if queue_path.exists() else {"created": now, "steps": []}
    lines = [f"OA-RLAIF live progress · {time.strftime('%H:%M:%S', time.localtime(now))} · refreshes every 5 s (Ctrl-C to exit)", ""]
    lines.append("RUNNING NOW")
    etas = []
    for name, info in state.items():
        if info.get("status") != "running":
            continue
        fraction, elapsed = info.get("fraction") or 0.0, now - info.get("started", now)
        eta = elapsed * (1 - fraction) / fraction if fraction > 0.005 else None
        etas.append(eta or 0.0)
        lines.append(f"  {name:<22}[{_bar(fraction)}] {fraction:6.1%}  elapsed {fmt_duration(elapsed):>8}"
                     f"  left {fmt_duration(eta) if eta is not None else 'estimating':>9}")
    if not etas:
        lines.append("  (nothing running)")
    lines += ["", "MAC QUEUE (in order)"]
    queued_minutes = 0.0
    for step in queue["steps"]:
        info = state.get(step.get("stage", ""), {})
        finished = info.get("started", 0) + (info.get("seconds") or 0)
        after = step.get("after", queue["created"])
        fresh = max(info.get("started", 0), finished) >= after  # started, or finished, after this step was queued
        status = info.get("status") if (fresh or info.get("status") == "running") else "queued"
        status = {"done": "done", "running": "running", "failed": "FAILED"}.get(status, "queued")
        if status == "queued":
            queued_minutes += step["minutes"]
        mark = {"done": "✔", "running": "▶", "FAILED": "✖", "queued": "·"}[status]
        note = "  (needs your OK)" if step.get("needs_ok") else ""
        lines.append(f"  {mark} {step['label']:<50}{status:<9}~{step['minutes']:.0f} min{note}")
    mac_left = (max(etas) if etas else 0.0) + queued_minutes * 60
    def finished_after(info: dict, step: dict) -> bool:
        return (info.get("status") == "done"
                and info.get("started", 0) + (info.get("seconds") or 0) >= step.get("after", queue["created"]))
    done = sum(1 for st in queue["steps"] if finished_after(state.get(st.get("stage", ""), {}), st))
    total = max(1, len(queue["steps"]))
    lines += ["", f"MAC BUILD  [{_bar(done / total)}] {done}/{total} queue steps   ≈ {fmt_duration(mac_left)} left"]
    lambda_minutes = sum(m for n, where, m in PIPELINE if where == "lambda" and state.get(n, {}).get("status") != "done")
    guard = BudgetGuard()
    lines.append(f"LAMBDA     [{_bar(0.0)}] not started            ≈ {fmt_duration(lambda_minutes * 60)} on 4×B200 "
                 f"(≈ ${lambda_minutes / 60 * guard.rate_usd_per_hour:.0f})")
    lines.append(f"BUDGET     spent ${guard.spent():.2f} · stop ${guard.stop_usd:.0f} · cap ${guard.budget_usd:.0f}")
    return "\n".join(lines)


def live(interval: float = 5.0) -> None:
    try:
        while True:
            sys.stdout.write("\033[2J\033[H" + live_text() + "\n")
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        print(status())
    elif len(sys.argv) > 1 and sys.argv[1] == "live":
        live()
    else:
        print("usage: python -m oa_rlaif.progress status|live")

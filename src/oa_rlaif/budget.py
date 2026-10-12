"""Budget guard: GPU wall-clock and API spend against the $500 cap, with a hard stop at $450.

GPU time is billed only on machines that set OA_BILLABLE=1 (the cost calculator does this), so data
preparation on the Mac costs nothing. The price and the machine's billed time come from oa_rlaif.cost (the price
of THIS instance, never a default; the machine's own start time, across restarts). This ledger adds the per-process
GPU records and API spend; every run, retry and stage adds to the same total.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from .paths import out_dir


class BudgetExceeded(RuntimeError):
    pass


def _float(name: str, default: float) -> float:
    value = os.environ.get(name, "")
    return float(value) if value.strip() else default


@dataclass
class BudgetGuard:
    rate_usd_per_hour: float | None = None   # None: the recorded price of this machine (oa_rlaif.cost)
    budget_usd: float = field(default_factory=lambda: _float("OA_BUDGET_USD", 500.0))
    stop_usd: float = field(default_factory=lambda: _float("OA_BUDGET_STOP_USD", 450.0))
    billable: bool = field(default_factory=lambda: os.environ.get("OA_BILLABLE") == "1")
    ledger_path: Path = field(default_factory=lambda: out_dir() / "budget_ledger.json")

    def __post_init__(self):
        from .cost import Meter

        self.ledger_path = Path(self.ledger_path)
        self._meter = Meter(self.ledger_path.parent / "cost_ledger.json")
        if self.rate_usd_per_hour is None:  # only the price recorded for this machine; never a value from .env
            self.rate_usd_per_hour = self._meter.price or 0.0
        self.price_known = self.rate_usd_per_hour > 0
        self._ledger = self._read()
        self._run_start = time.time()
        self._run_key = f"run-{int(self._run_start)}-{os.getpid()}"

    def _read(self) -> dict:
        if self.ledger_path.exists():
            return json.loads(self.ledger_path.read_text())
        return {"gpu_runs": {}, "api": {}}

    def _write(self) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.ledger_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._ledger, indent=2))
        temporary.replace(self.ledger_path)

    # ---- recording -------------------------------------------------------------------------
    def tick(self, stage: str = "") -> None:
        """Record this run's elapsed GPU time (idempotent; call as often as you like)."""
        if not self.billable:
            return
        self._meter.observe()
        seconds = time.time() - self._run_start
        self._ledger["gpu_runs"][self._run_key] = {"stage": stage, "seconds": seconds,
                                                   "rate_usd_per_hour": self.rate_usd_per_hour}
        self._write()

    def add_api(self, service: str, usd: float, tokens_in: int = 0, tokens_out: int = 0) -> None:
        entry = self._ledger["api"].setdefault(service, {"usd": 0.0, "tokens_in": 0, "tokens_out": 0, "calls": 0})
        entry["usd"] += usd
        entry["tokens_in"] += tokens_in
        entry["tokens_out"] += tokens_out
        entry["calls"] += 1
        self._write()

    # ---- reading ---------------------------------------------------------------------------
    def gpu_usd(self) -> float:
        """GPU spend: the larger of the per-process records and the machine's billed time (oa_rlaif.cost.Meter)."""
        ledger = sum(r["seconds"] / 3600 * r["rate_usd_per_hour"] for r in self._ledger["gpu_runs"].values())
        return max(ledger, self._meter.spent() if self.billable else 0.0)

    def api_usd(self) -> float:
        return sum(v["usd"] for v in self._ledger["api"].values())

    def spent(self) -> float:
        return self.gpu_usd() + self.api_usd()

    def remaining(self) -> float:
        return self.budget_usd - self.spent()

    def projected(self, remaining_hours: float) -> float:
        rate = self.rate_usd_per_hour if self.billable else 0.0
        return self.spent() + remaining_hours * rate

    def check(self, stage: str = "") -> None:
        if self.billable and not self.price_known:
            raise BudgetExceeded("no price recorded for this machine: run `python -m oa_rlaif.cost price` first")
        self.tick(stage)
        if self.spent() >= self.stop_usd:
            raise BudgetExceeded(
                f"Spent ${self.spent():.2f} >= stop ${self.stop_usd:.2f} of ${self.budget_usd:.2f} budget"
            )

    def summary(self) -> dict:
        return {
            "spent_usd": round(self.spent(), 2),
            "gpu_usd": round(self.gpu_usd(), 2),
            "api_usd": round(self.api_usd(), 2),
            "remaining_usd": round(self.remaining(), 2),
            "stop_usd": self.stop_usd,
            "budget_usd": self.budget_usd,
            "billable": self.billable,
        }


def trainer_callback(guard: BudgetGuard, stage: str):
    """HF/TRL TrainerCallback: saves a checkpoint and stops training cleanly at the stop threshold."""
    from transformers import TrainerCallback

    class BudgetCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            guard.tick(stage)
            if guard.spent() >= guard.stop_usd:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                logs.update({f"budget/{k}": v for k, v in guard.summary().items() if isinstance(v, float)})

    return BudgetCallback()

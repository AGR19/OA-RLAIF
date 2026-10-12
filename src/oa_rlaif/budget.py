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

from . import jsonfile
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
        self._main = jsonfile.is_main_process()

    def _read(self) -> dict:
        return jsonfile.read(self.ledger_path) or {"gpu_runs": {}, "api": {}}

    # ---- recording -------------------------------------------------------------------------
    # Several processes share this ledger (one per GPU, plus scripts): every change is a locked read-modify-write.
    def tick(self, stage: str = "") -> None:
        """Note that the machine is up and record this run's time. Only the main process of a multi-GPU launch
        records: it is one machine, however many GPU processes run on it."""
        if not self.billable or not self._main:
            return
        self._meter.observe()
        entry = {"stage": stage, "seconds": time.time() - self._run_start, "rate_usd_per_hour": self.rate_usd_per_hour}

        def change(data):
            data.setdefault("api", {})
            data.setdefault("gpu_runs", {})[self._run_key] = entry
        self._ledger = jsonfile.update(self.ledger_path, change, default=lambda: {"gpu_runs": {}, "api": {}})

    def add_api(self, service: str, usd: float, tokens_in: int = 0, tokens_out: int = 0) -> None:
        """API spend from any process (each GPU process calls the judge): added under the lock, never lost."""
        def change(data):
            data.setdefault("gpu_runs", {})
            entry = data.setdefault("api", {}).setdefault(service, {"usd": 0.0, "tokens_in": 0, "tokens_out": 0,
                                                                    "calls": 0})
            entry["usd"] += usd
            entry["tokens_in"] += tokens_in
            entry["tokens_out"] += tokens_out
            entry["calls"] += 1
        self._ledger = jsonfile.update(self.ledger_path, change, default=lambda: {"gpu_runs": {}, "api": {}})

    # ---- reading ---------------------------------------------------------------------------
    def gpu_usd(self, now: float | None = None) -> float:
        """The machine's billed time x its price (oa_rlaif.cost.Meter) once a price is recorded: one machine, not one
        per GPU process. Without a recorded price: the run records (written by the main process only)."""
        if not self.billable:
            return 0.0
        metered = self._meter.spent(now)
        if self._meter.price:
            return metered
        self._ledger = self._read()
        return sum(r["seconds"] / 3600 * r["rate_usd_per_hour"] for r in self._ledger["gpu_runs"].values())

    def api_usd(self) -> float:
        self._ledger = self._read()
        return sum(v["usd"] for v in self._ledger["api"].values())

    def spent(self, now: float | None = None) -> float:
        """With the same `now`, every process gets the same number (the planner passes OA_PLAN_TIME)."""
        return self.gpu_usd(now) + self.api_usd()

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


def any_process(flag: bool) -> bool:
    """True if any GPU process says True. Every process must stop at the same step: one that stops alone leaves
    the others waiting forever in the next collective operation."""
    try:
        import torch
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            value = torch.tensor([1.0 if flag else 0.0], device=device)
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            return bool(value.item())
    except Exception:  # noqa: BLE001 (no process group: a single process decides alone)
        pass
    return flag


def trainer_callback(guard: BudgetGuard, stage: str):
    """HF/TRL TrainerCallback: saves a checkpoint and stops training cleanly at the stop threshold, in every GPU
    process at the same step."""
    from transformers import TrainerCallback

    class BudgetCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            try:
                guard.tick(stage)
                over = guard.spent() >= guard.stop_usd
            except Exception as error:  # noqa: BLE001 (a ledger problem must not desynchronise the processes)
                print(f"[budget] check failed this step ({type(error).__name__}: {error}); continuing")
                over = False
            if any_process(over):
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                logs.update({f"budget/{k}": v for k, v in guard.summary().items() if isinstance(v, float)})

    return BudgetCallback()

"""Several training processes (one per GPU) and the dashboard update the same ledgers at once.

Each worker plays one GPU process: it ticks the budget guard (which observes the machine meter), records API spend
and writes the step plan, many times. No write may fail, no API dollar may be lost, the spend must count the
machine once (not once per process), and the files must stay valid JSON. (A shared temporary file name once
crashed a 3-GPU OPSD dry run on Vast.ai with FileNotFoundError.)
"""

import json
import multiprocessing as mp
import time

WORKERS, ROUNDS = 4, 60


def worker(out_dir: str, rank: int, start: float, errors) -> None:
    import os

    os.environ.update(OA_OUT_DIR=out_dir, OA_BILLABLE="1", LOCAL_RANK=str(rank))
    from oa_rlaif import cost
    from oa_rlaif.budget import BudgetGuard
    from oa_rlaif.train import common

    cost.machine_start = lambda: start
    try:
        guard = BudgetGuard()
        for i in range(ROUNDS):
            guard.tick("sdft")
            guard.add_api("judge", 0.01)
            common.write_plan("sdft", {"stage": "sdft", "steps": i, "rank": rank})
            common.record_dry_run(**{f"rank{rank}": i})
            cost.Meter().observe()
    except Exception as error:  # noqa: BLE001
        errors.put(f"rank {rank}: {type(error).__name__}: {error}")


def test_parallel_writers_share_the_ledgers(tmp_path, monkeypatch):
    from oa_rlaif import cost

    start = time.time() - 3600  # the machine has been up for 1 h
    monkeypatch.setattr(cost, "machine_start", lambda: start)
    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("OA_BILLABLE", "1")
    cost.Meter(tmp_path / "cost_ledger.json").set_price(10.0, "test", [("NVIDIA B200", 179)] * 4)
    ctx = mp.get_context("spawn")
    errors = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(str(tmp_path), r, start, errors)) for r in range(WORKERS)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    problems = []
    while not errors.empty():
        problems.append(errors.get())
    assert not problems, problems
    assert all(p.exitcode == 0 for p in procs)
    for name in ("cost_ledger.json", "budget_ledger.json", "step_plan.json", "dry_run.json"):
        json.loads((tmp_path / name).read_text())  # still valid JSON
    ledger = json.loads((tmp_path / "budget_ledger.json").read_text())
    assert abs(ledger["api"]["judge"]["usd"] - WORKERS * ROUNDS * 0.01) < 1e-6  # no API spend lost
    from oa_rlaif.budget import BudgetGuard

    spent = BudgetGuard().gpu_usd()
    assert 9.9 < spent < 10.5  # 1 h of the machine at $10/h: counted once, not once per GPU process


def test_update_edits_in_place_and_ignores_return_values(tmp_path):
    from oa_rlaif import jsonfile

    path = tmp_path / "plan.json"
    jsonfile.update(path, lambda d: d.__setitem__("sdft", {"steps": 3}))
    jsonfile.update(path, lambda d: d.setdefault("budget", {"usd": 1}))   # setdefault returns the value
    assert jsonfile.read(path) == {"sdft": {"steps": 3}, "budget": {"usd": 1}}

from __future__ import annotations

import pytest

from oa_rlaif import progress
from oa_rlaif.budget import BudgetExceeded, BudgetGuard


def test_budget_accounting_and_stop(tmp_path):
    guard = BudgetGuard(rate_usd_per_hour=36.0, budget_usd=500, stop_usd=450, billable=True,
                        ledger_path=tmp_path / "ledger.json")
    guard._run_start -= 3600  # pretend one hour has elapsed
    guard.tick("sdft")
    assert guard.gpu_usd() == pytest.approx(36.0, rel=1e-3)
    guard.add_api("judge", 4.0, tokens_in=1000)
    assert guard.spent() == pytest.approx(40.0, rel=1e-3)
    assert BudgetGuard(billable=True, ledger_path=tmp_path / "ledger.json").spent() == pytest.approx(40.0, rel=1e-3)
    guard._run_start -= 3600 * 12
    with pytest.raises(BudgetExceeded):
        guard.check("grpo")


def test_non_billable_machines_cost_nothing(tmp_path):
    guard = BudgetGuard(rate_usd_per_hour=27.16, billable=False, ledger_path=tmp_path / "l.json")
    guard._run_start -= 10 * 3600
    guard.tick("prepare_data")
    assert guard.spent() == 0.0


def test_stage_records_progress_and_eta(tmp_path, monkeypatch):
    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    with progress.stage("prepare_data", total=4) as handle:
        for _ in range(4):
            handle.advance()
    state = progress._load()
    assert state["stages"]["prepare_data"]["status"] == "done"
    summary = progress.overall(state)
    assert 0 < summary["fraction"] < 1 and summary["eta_seconds"] > 0
    assert "Pipeline [" in progress.status()

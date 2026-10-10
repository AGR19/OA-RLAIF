"""Training steps are planned from the measured dry-run speed so each stage fits its time slot."""

import json


def test_plan_fits_time_slot_and_caps_at_full_epochs(tmp_path, monkeypatch):
    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    from oa_rlaif.train.common import plan_steps

    cfg = {"sdft": {"max_steps": -1, "time_budget_hours": 6.0}, "grpo": {"max_steps": -1, "time_budget_hours": 6.0},
           "dry_run": {"sdft_seconds_per_step": 12.0, "grpo_seconds_per_step": 30.0}}
    # 328,944 rows, 96 rows/step, 2 epochs = 6,853 steps; 6 h at 12 s/step with a 10% margin fits 1,620
    assert plan_steps(cfg, "sdft", 328_944, 96, 2) == 1620
    # 12,000 prompts, 6 prompts/step, 1 epoch = 2,000 steps; 6 h at 30 s/step fits 648
    assert plan_steps(cfg, "grpo", 12_000, 6, 1) == 648
    # a fast machine never runs more than the full epochs
    cfg["dry_run"]["grpo_seconds_per_step"] = 1.0
    assert plan_steps(cfg, "grpo", 12_000, 6, 1) == 2000
    plan = json.loads((tmp_path / "step_plan.json").read_text())
    assert plan["sdft"]["steps"] == 1620 and plan["grpo"]["full_steps"] == 2000


def test_explicit_max_steps_wins_and_no_timing_runs_full(tmp_path, monkeypatch):
    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    from oa_rlaif.train.common import plan_steps

    assert plan_steps({"sdft": {"max_steps": 50}}, "sdft", 1000, 10, 2) == 50
    assert plan_steps({"sdft": {"max_steps": -1}}, "sdft", 1000, 10, 2) == 200


def _billable(monkeypatch, tmp_path, rate, hours_ago):
    import time

    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("OA_BILLABLE", "1")
    monkeypatch.setenv("OA_GPU_USD_PER_HOUR", str(rate))
    monkeypatch.delenv("OA_SLOT_SCALE", raising=False)
    monkeypatch.delenv("OA_WALLCLOCK_BUDGET", raising=False)
    (tmp_path / "stages").mkdir(parents=True, exist_ok=True)
    (tmp_path / "stages" / "run.start").write_text(str(time.time() - hours_ago * 3600))


SLOTS = {"sdft": {"max_steps": -1, "time_budget_hours": 4.0}, "grpo": {"max_steps": -1, "time_budget_hours": 4.0},
         "sft": {"time_budget_hours": 2.0}, "budget_plan": {"fixed_hours": 4.5, "min_slot_scale": 0.05},
         "dry_run": {"sdft_seconds_per_step": 10.0}}


def test_budget_counts_machine_wallclock(tmp_path, monkeypatch):
    """A node bills from the moment the run starts (setup, downloads, gaps), not only while a trainer runs."""
    from oa_rlaif.budget import BudgetGuard

    _billable(monkeypatch, tmp_path, 68.80, hours_ago=2.0)
    assert abs(BudgetGuard().spent() - 137.6) < 0.5
    monkeypatch.setenv("OA_WALLCLOCK_BUDGET", "0")
    assert BudgetGuard().spent() == 0.0


def test_slots_full_at_lambda_price(tmp_path, monkeypatch):
    from oa_rlaif.train.common import plan_steps, slot_scale

    _billable(monkeypatch, tmp_path, 27.16, hours_ago=1.5)  # ~$41 used: 15 h left, 4.5 fixed -> all 10 slot hours fit
    assert slot_scale(dict(SLOTS)) == 1.0
    assert plan_steps(dict(SLOTS), "sdft", 1_000_000, 10, 2) == int(4.0 * 3600 * 0.9 / 10.0)


def test_slots_shrink_at_coreweave_price_and_stay_fixed(tmp_path, monkeypatch):
    from oa_rlaif.train.common import plan_steps, slot_scale

    _billable(monkeypatch, tmp_path, 68.80, hours_ago=1.5)  # $103 used: 5.04 h left - 4.5 fixed = 0.54 of 10 h
    scale = slot_scale(dict(SLOTS))
    assert 0.05 < scale < 0.06
    plan = json.loads((tmp_path / "step_plan.json").read_text())["budget"]
    assert plan["usd_per_hour"] == 68.80 and plan["slot_scale"] == round(scale, 4)
    # later stages reuse the stored scale even though more money has been spent by then
    _billable(monkeypatch, tmp_path, 68.80, hours_ago=3.0)
    assert slot_scale(dict(SLOTS)) == round(scale, 4)
    assert plan_steps(dict(SLOTS), "sdft", 1_000_000, 10, 2) == int(4.0 * round(scale, 4) * 3600 * 0.9 / 10.0)


def test_budget_too_small_stops_with_reason(tmp_path, monkeypatch):
    import pytest

    from oa_rlaif.train.common import slot_scale

    _billable(monkeypatch, tmp_path, 120.0, hours_ago=1.5)  # $180 used: 2.25 h left < 4.5 fixed
    with pytest.raises(SystemExit, match="budget too small"):
        slot_scale(dict(SLOTS))
    monkeypatch.setenv("OA_SLOT_SCALE", "0.2")
    assert slot_scale(dict(SLOTS)) == 0.2

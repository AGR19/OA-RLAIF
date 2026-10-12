"""Training steps from the measured dry-run speed (money-based planning is tested in test_cost.py)."""

import json


def test_plan_fits_time_slot_and_caps_at_full_epochs(tmp_path, monkeypatch):
    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    from oa_rlaif.train.common import plan_steps

    monkeypatch.delenv("OA_BILLABLE", raising=False)   # an unbilled machine: no money limit, full epochs
    cfg = {"sdft": {"max_steps": -1}, "grpo": {"max_steps": -1},
           "dry_run": {"sdft_seconds_per_step": 12.0, "grpo_seconds_per_step": 30.0}}
    # 328,944 rows, 96 rows/step, 2 epochs = 6,853 steps
    assert plan_steps(cfg, "sdft", 328_944, 96, 2) == 6853
    assert plan_steps(cfg, "grpo", 12_000, 6, 1) == 2000
    plan = json.loads((tmp_path / "step_plan.json").read_text())
    assert plan["sdft"]["steps"] == 6853 and plan["grpo"]["full_steps"] == 2000


def test_explicit_max_steps_wins_and_no_timing_runs_full(tmp_path, monkeypatch):
    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    from oa_rlaif.train.common import plan_steps

    assert plan_steps({"sdft": {"max_steps": 50}}, "sdft", 1000, 10, 2) == 50
    assert plan_steps({"sdft": {"max_steps": -1}}, "sdft", 1000, 10, 2) == 200


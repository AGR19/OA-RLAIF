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

"""The cost calculator: the real price of this machine (provider API, else checked against its GPUs), the machine's
billed time across restarts, and the plan in dollars."""

import json

import pytest

from oa_rlaif import cost

B200x4 = [("NVIDIA B200", 179)] * 4
A100_40 = [("NVIDIA A100-SXM4-40GB", 40)]


def test_implausible_prices_are_caught():
    assert not cost.plausible(27.16, A100_40)[0]          # the Lambda 4xB200 price on a 1xA100-40GB machine
    assert cost.plausible(1.29, A100_40)[0]
    assert cost.plausible(27.16, B200x4)[0]                # $6.79 per B200-hour
    assert not cost.plausible(2.0, B200x4)[0]              # a typo: far too cheap for 4 B200s
    assert cost.plausible(20.0, [("NVIDIA H100 80GB HBM3", 79)] * 8)[0]


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("OA_ENV_FILE", str(tmp_path / ".env"))
    clock = {"start": 1_000_000.0, "now": 1_000_000.0}
    monkeypatch.setattr(cost, "machine_start", lambda: clock["start"])
    monkeypatch.setattr(cost.time, "time", lambda: clock["now"])   # one fake clock for the whole test
    return tmp_path, clock


def test_meter_bills_from_machine_start_across_restarts(ledger):
    tmp_path, clock = ledger
    meter = cost.Meter(tmp_path / "cost_ledger.json")
    clock["now"] = 1_000_000.0 + 7200
    assert meter.spent() == 0.0                                    # no price yet: nothing is recorded
    meter.set_price(27.16, "test", B200x4)
    assert meter.spent() == pytest.approx(54.32)                   # 2 h since the machine started, not since now
    clock["now"] = 1_000_000.0 + 3 * 3600
    meter.observe()                                                # last seen 3 h after the first start
    clock["start"] = clock["now"] = 1_000_000.0 + 5 * 3600         # restart after 2 h off
    clock["now"] += 3600
    later = cost.Meter(tmp_path / "cost_ledger.json")
    later.observe()
    # 3 h in the first segment + 1 h in the new one; the 2 h the machine was off are not billed
    assert later.spent() == pytest.approx(4 * 27.16)


def test_vast_price_from_the_instance_api(monkeypatch):
    monkeypatch.setenv("CONTAINER_ID", "123")
    monkeypatch.setenv("CONTAINER_API_KEY", "k")
    seen = {}

    def fake(url, headers, timeout=15):
        seen["url"], seen["auth"] = url, headers["Authorization"]
        return {"instances": {"id": 123, "dph_total": 21.5, "num_gpus": 4}}
    monkeypatch.setattr(cost, "_get_json", fake)
    price, source = cost.vast_price()
    assert price == 21.5 and "instance 123" in source
    assert seen["url"].endswith("/instances/123/") and seen["auth"] == "Bearer k"


def test_lambda_price_matches_this_machine_by_ip(monkeypatch):
    monkeypatch.setattr(cost, "_env", lambda name: "secret" if name == "LAMBDA_API_KEY" else "")
    monkeypatch.setattr(cost, "local_ips", lambda: {"10.0.0.7"})
    data = {"data": [
        {"status": "active", "ip": "1.2.3.4", "private_ip": "10.0.0.9",
         "instance_type": {"name": "gpu_1x_a100", "price_cents_per_hour": 129}},
        {"status": "active", "ip": "5.6.7.8", "private_ip": "10.0.0.7",
         "instance_type": {"name": "gpu_4x_b200", "price_cents_per_hour": 2716}}]}
    monkeypatch.setattr(cost, "_get_json", lambda url, headers, timeout=15: data)
    price, source = cost.lambda_price()
    assert price == 27.16 and "gpu_4x_b200" in source and "matched by IP" in source
    monkeypatch.setattr(cost, "local_ips", lambda: {"192.168.1.1"})   # two instances, neither is this machine
    assert cost.lambda_price() is None


def test_price_cli_refuses_an_implausible_price_unless_confirmed(ledger, monkeypatch):
    tmp_path, _ = ledger
    monkeypatch.setattr(cost, "gpus", lambda: A100_40)
    monkeypatch.setattr(cost, "lambda_price", lambda: None)
    monkeypatch.setattr(cost, "SHELL_PRICE", "27.16")
    monkeypatch.delenv("OA_PRICE_CONFIRMED", raising=False)
    with pytest.raises(SystemExit):
        cost.main(["price", "--platform", "lambda", "--yes"])
    assert not (tmp_path / "cost_ledger.json").exists()
    monkeypatch.setattr(cost, "SHELL_PRICE", "1.29")
    assert cost.main(["price", "--platform", "lambda", "--yes"]) == 0
    recorded = json.loads((tmp_path / "cost_ledger.json").read_text())["price"]
    assert recorded["usd_per_hour"] == 1.29 and recorded["source"] == "OA_GPU_USD_PER_HOUR (shell)"
    assert "OA_GPU_USD_PER_HOUR=1.29" in (tmp_path / ".env").read_text()
    # the next run reuses the recorded price for the same GPUs, without asking
    monkeypatch.setattr(cost, "SHELL_PRICE", "")
    assert cost.main(["price", "--platform", "lambda", "--yes"]) == 0
    # the same ledger on a machine with other GPUs: not reused; not interactive, so it stops and asks for a price
    monkeypatch.setattr(cost, "gpus", lambda: B200x4)
    with pytest.raises(SystemExit, match="no price"):
        cost.main(["price", "--platform", "lambda", "--yes"])


def test_stale_env_price_is_not_trusted(ledger, monkeypatch):
    """A value that only sits in .env (for example the old $27.16 default) is never used as the price."""
    tmp_path, _ = ledger
    (tmp_path / ".env").write_text("OA_GPU_USD_PER_HOUR=27.16\n")
    monkeypatch.setattr(cost, "gpus", lambda: A100_40)
    monkeypatch.setattr(cost, "lambda_price", lambda: None)
    monkeypatch.setattr(cost, "SHELL_PRICE", "")
    with pytest.raises(SystemExit, match="no price"):
        cost.main(["price", "--platform", "lambda", "--yes"])


def test_dollar_plan_shares_and_caps():
    plan = cost.make_plan(price=27.16, spent=40.7, stop=450)   # Lambda 4xB200 at the end of the dry run
    assert plan["training_usd"] == pytest.approx((409.3 - 4.3 * 27.16) * 0.95, abs=0.05)
    a = plan["allocated_usd"]
    assert a["sdft"] == a["grpo"] == pytest.approx(2 * a["sft"], abs=0.02)
    assert cost.stage_dollars(plan, "sdft", 27.16, 61.1, 450) == a["sdft"]
    # OPSD overspent by $30: GRPO gets less, so the evaluations, export and SFT still fit under the stop
    spent_before_grpo = 61.1 + a["sdft"] + 30 + 0.75 * 27.16
    grpo = cost.stage_dollars(plan, "grpo", 27.16, spent_before_grpo, 450)
    assert grpo < a["grpo"]
    assert spent_before_grpo + grpo + 2.8 * 27.16 + a["sft"] <= 450 + 1e-6


def test_plan_steps_from_money(tmp_path, monkeypatch):
    import time

    from oa_rlaif.train.common import plan_steps

    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("OA_BILLABLE", "1")
    monkeypatch.setattr(cost, "machine_start", lambda: time.time() - 1.5 * 3600)   # setup + dry run done
    cost.Meter(tmp_path / "cost_ledger.json").set_price(27.16, "test", B200x4)
    cfg = {"sdft": {"max_steps": -1}, "grpo": {"max_steps": -1}, "dry_run": {"sdft_seconds_per_step": 12.0}}
    steps = plan_steps(cfg, "sdft", 328_944, 96, 2)
    plan = json.loads((tmp_path / "step_plan.json").read_text())
    usd_per_step = 27.16 * 12.0 / 3600
    assert steps == int(plan["sdft"]["usd_allowed"] * 0.9 / usd_per_step)
    assert plan["budget"]["usd_per_hour"] == 27.16 and plan["sdft"]["forecast_usd"] <= plan["sdft"]["usd_allowed"]
    # a fast machine never runs more than the full epochs
    cfg["dry_run"]["sdft_seconds_per_step"] = 0.01
    assert plan_steps(cfg, "sdft", 1000, 10, 2) == 200


def test_plan_stops_when_the_budget_is_too_small(tmp_path, monkeypatch):
    import time

    from oa_rlaif.train.common import plan_steps

    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("OA_BILLABLE", "1")
    monkeypatch.setattr(cost, "machine_start", lambda: time.time() - 1.5 * 3600)
    cost.Meter(tmp_path / "cost_ledger.json").set_price(120.0, "test", B200x4)   # $180 used, 4.3 h fixed = $516
    with pytest.raises(SystemExit, match="budget too small"):
        plan_steps({"sdft": {"max_steps": -1}, "dry_run": {"sdft_seconds_per_step": 12.0}}, "sdft", 1000, 10, 2)

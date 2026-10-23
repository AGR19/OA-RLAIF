"""The decision-model run, the three arms, resumable, within the money cap.

    python -m dm_harness.run            # shows the plan and what the cap buys, asks once, then runs to the end
    python -m dm_harness.run --yes      # no question
    python -m dm_harness.run --plan     # only the plan

  arm A  "decider"        pplx-decider alone, every GPU (dm_harness.base_run)
  arm B  "jah-agent"      JevAgentHarness + pplx-decider + All merged + mini-SWE-agent, with the NONE/invalid guard;
                          thresholds fixed before the run (routing 0.75, guard 0.5) (dm_harness.harness_run)
  arm C  "jah-fit-agent"  the same agent with the thresholds fitted on a calibration sample (dm_harness.calibrate:
                          shadow runs -> equal-mass bins -> tau* per decision point -> frozen), plus the fitted cascade;
                          then every class, the sample included (dm_harness.harness_run --tag jah-fit-agent)
Arms B and C share one stack (the decision model on DM_DECIDER_GPUS, vLLM on DM_GEMMA_GPUS, the proxy); the proxy is
restarted with the fitted routing tau before arm C.

Every class with a reference partner and the complete SNOMED CT -> ICD-11 map (FULL_EVAL_SCOPE=reference, 299,961 per
arm; FULL_EVAL_TASKS narrows it), the same episodes as the full evaluation (full_eval.build_episodes, built in the
background while the GPUs work). A unit (arm, task) runs only if its projected cost fits the money left (DM_BUDGET_USD,
default $450, setup included); outputs/full_eval/STOP stops the workers between shards when the spend reaches the
cap. Each finished unit is scored at once (full_eval.score), then dm_harness.report writes the tables.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

from full_eval.schedule import Budget, decisions, results_dir, root, selected_tasks, task_order

from . import stack
from .jah_config import write as write_jah_config

ARMS = {"decider": "dm_harness.base_run", "jah-agent": "dm_harness.harness_run",
        "jah-fit-agent": "dm_harness.harness_run"}
STACK = ("jah-agent", "jah-fit-agent")  # the arms that run through the proxy
# classes/s on 4 x B200 until a unit of the arm is measured; arm C asks Gemma less often (the fitted cascade), so its
# assumed rate is arm B's
DEFAULT_RATES = {"decider": 60.0, "jah-agent": 25.0, "jah-fit-agent": 25.0}
LOAD_S = {"decider": 600.0, "jah-agent": 1200.0, "jah-fit-agent": 0.0}  # servers up (download excluded), once per stack
CAL_EPISODES_PER_CLASS = 2.2  # strong + fast shadow runs, and a redo for each NONE / invalid answer
PY = sys.executable


def selected_arms() -> list[str]:
    arms = [a.strip() for a in os.environ.get("DM_ARMS", "decider,jah-agent,jah-fit-agent").split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"DM_ARMS: unknown arms {unknown}; choose from {list(ARMS)}")
    return arms


def speed_path():
    return root() / "dm_speed.json"


def rate(arm: str) -> tuple[float, str]:
    data = json.loads(speed_path().read_text()) if speed_path().exists() else {}
    n, s = data.get(arm, {}).get("classes", 0), data.get(arm, {}).get("seconds", 0.0)
    if n >= 500 and s > 0:
        return n / s, "measured"
    return float(os.environ.get(f"DM_RATE_{arm.upper().replace('-', '_')}", DEFAULT_RATES[arm])), "assumed"


def record(arm: str, n: int, seconds: float) -> None:
    data = json.loads(speed_path().read_text()) if speed_path().exists() else {}
    cur = data.setdefault(arm, {"classes": 0, "seconds": 0.0})
    cur["classes"] += n
    cur["seconds"] += seconds
    speed_path().write_text(json.dumps(data, indent=1))


def done(arm: str, slug: str) -> bool:
    return (results_dir(arm, slug) / "scores.json").exists()


def plan(budget: Budget) -> list[dict]:
    left, rows = budget.left(), []
    for arm in selected_arms():
        r, basis = rate(arm)
        first = True
        if arm == "jah-fit-agent" and not (root() / "calibration" / "thresholds.json").exists():
            n = int(os.environ.get("DM_CAL_N", "3000"))
            seconds = n * CAL_EPISODES_PER_CLASS / rate("jah-agent")[0] + 120  # + the proxy's restart
            usd = budget.dollars(seconds)
            fits = usd <= left
            left -= usd if fits else 0
            rows.append({"arm": "calibrate", "task": f"sample of {n:,}", "classes": n, "exact": False,
                         "rate": round(rate("jah-agent")[0], 1), "rate_basis": "agent", "seconds": round(seconds),
                         "usd": round(usd, 2), "fits": fits})
        for t in task_order(selected_tasks()):
            if done(arm, t.slug):
                continue
            n, exact = decisions(t.slug)
            seconds = n / r + (LOAD_S[arm] if first else 0)
            usd = budget.dollars(seconds)
            fits = usd <= left
            if fits:
                left -= usd
                first = False
            rows.append({"arm": arm, "task": t.slug, "classes": n, "exact": exact, "rate": round(r, 1),
                         "rate_basis": basis, "seconds": round(seconds), "usd": round(usd, 2), "fits": fits})
    return rows


def show(rows: list[dict], budget: Budget) -> None:
    print(f"\nMoney: cap ${budget.cap:.0f}, spent ${budget.spent():.2f}, price ${budget.price:.2f}/h. "
          f"Rates: " + ", ".join(f"{a} {rate(a)[0]:g}/s ({rate(a)[1]})" for a in selected_arms()))
    print(f"{'arm':<10} {'task':<26} {'classes':>9} {'hours':>6} {'$':>7}  fits")
    for p in rows:
        print(f"{p['arm']:<10} {p['task']:<26} {p['classes']:>9,}{'' if p['exact'] else '~'} "
              f"{p['seconds'] / 3600:>6.2f} {p['usd']:>7.2f}  {'yes' if p['fits'] else 'NO (money)'}")
    total = sum(p["usd"] for p in rows if p["fits"])
    print(f"total projected: ${total:.2f} + spent ${budget.spent():.2f} of ${budget.cap:.0f}")


def watchdog(budget: Budget, stop: threading.Event) -> None:
    """Writes outputs/full_eval/STOP when the spend reaches the cap; the workers stop between shards."""
    while not stop.wait(20):
        if budget.spent() >= budget.cap - budget.reserve / 2:
            (root() / "STOP").write_text(f"spend reached ${budget.spent():.2f} of ${budget.cap:.0f}")
            print("\n[money] the cap is reached: the workers stop after their current shard", flush=True)
            return


def run_units(arm: str, budget: Budget, statuses: dict, scorers: list) -> bool:
    """Every unit of one arm whose projected cost fits, on servers that are already up."""
    for p in [p for p in plan(budget) if p["arm"] == arm]:
        if not p["fits"]:
            statuses[(arm, p["task"])] = {"status": "not run (money)", "usd": p["usd"]}
            continue
        if (root() / "STOP").exists():
            return False
        print(f"\n[run] {arm} · {p['task']}: {p['classes']:,} classes", flush=True)
        started = time.time()
        extra = ["--tag", arm] if arm in STACK else []
        code = subprocess.run([PY, "-m", ARMS[arm], "--tasks", p["task"], *extra]).returncode
        seconds = time.time() - started
        if code == 3:
            statuses[(arm, p["task"])] = {"status": "stopped (money); continues on the next run"}
            return False
        if code != 0:
            statuses[(arm, p["task"])] = {"status": f"worker failed (exit {code})"}
            continue
        n, _ = decisions(p["task"])
        record(arm, n, seconds)
        statuses[(arm, p["task"])] = {"status": "scoring", "seconds": round(seconds),
                                      "usd": round(budget.dollars(seconds), 2)}
        log = open(root() / "logs" / f"score_{arm}_{p['task']}.log", "a")
        scorers.append((subprocess.Popen([PY, "-m", "full_eval.score", arm, p["task"]], stdout=log,
                                         stderr=subprocess.STDOUT), arm, p["task"]))
    return True


def run_decider(budget: Budget, statuses: dict, scorers: list) -> bool:
    if not [p for p in plan(budget) if p["arm"] == "decider"]:
        return True
    proc = stack.start_decider(stack.gpus())
    try:
        return run_units("decider", budget, statuses, scorers)
    finally:
        stack.stop(proc)


def run_stack(budget: Budget, statuses: dict, scorers: list) -> bool:
    """Arms B and C on one stack; the calibration and the fit come between them, then the proxy restarts with the
    fitted routing tau."""
    arms = [a for a in selected_arms() if a in STACK]
    if not any(p["arm"] in arms or p["arm"] == "calibrate" for p in plan(budget)):
        return True
    if not stack.fake():
        from full_eval.models import ready

        while not ready("all_merged"):
            print("[run] waiting for the All-merged model (outputs/full_eval/logs/models.log)", flush=True)
            time.sleep(30)
    dec, gem = stack.split_gpus()
    procs = [stack.start_decider(dec), stack.start_vllm(gem)]
    jah = stack.start_jah(write_jah_config(root() / "jah_config.yaml", root() / "jah_logs"))
    try:
        if "jah-agent" in arms and not run_units("jah-agent", budget, statuses, scorers):
            return False
        if "jah-fit-agent" not in arms:
            return True
        thresholds = root() / "calibration" / "thresholds.json"
        if not thresholds.exists():
            cal = next((p for p in plan(budget) if p["arm"] == "calibrate"), None)
            if cal is not None and not cal["fits"]:
                statuses[("calibrate", "sample")] = {"status": "not run (money)"}
                return False
            print("\n[run] calibration: shadow runs on the sample, then the fit", flush=True)
            if subprocess.run([PY, "-m", "dm_harness.calibrate"]).returncode != 0 or not thresholds.exists():
                statuses[("calibrate", "sample")] = {"status": "calibration failed (see logs)"}
                return False
        tau_r = json.loads(thresholds.read_text())["routing"]["tau"]
        stack.stop(jah)
        jah = stack.start_jah(write_jah_config(root() / "jah_config.yaml", root() / "jah_logs", fitted_tau=tau_r))
        return run_units("jah-fit-agent", budget, statuses, scorers)
    finally:
        stack.stop(jah, *reversed(procs))


def save(statuses: dict, budget: Budget) -> None:
    rows = [{"model": arm, "task": t.slug, "decisions": decisions(t.slug)[0],
             **statuses.get((arm, t.slug), {}), "status": "done" if done(arm, t.slug) else
             statuses.get((arm, t.slug), {}).get("status", "not run")}
            for arm in selected_arms() for t in task_order(selected_tasks())]
    (root() / "plan.json").write_text(json.dumps({"budget_usd": budget.cap, "spent_usd": round(budget.spent(), 2),
                                                  "price_usd_per_hour": budget.price, "units": rows}, indent=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dm_harness.run")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args(argv)
    (root() / "logs").mkdir(parents=True, exist_ok=True)
    if stack.fake():
        os.environ["FULL_EVAL_FAKE"] = "lexical"  # the money guard's $1/h stand-in price
    budget = Budget(float(os.environ.get("DM_BUDGET_USD", "450")))
    if (root() / "STOP").exists():
        (root() / "STOP").unlink()
    rows = plan(budget)
    show(rows, budget)
    if args.plan:
        return 0
    if not args.yes and sys.stdin.isatty():
        if input("\nStart? [y/N] ").strip().lower() not in ("y", "yes"):
            print("[run] not started.")
            return 1
    helpers = [subprocess.Popen([PY, "-m", "full_eval.build_episodes"],
                                stdout=open(root() / "logs" / "episodes.log", "a"), stderr=subprocess.STDOUT)]
    if any(a in STACK for a in selected_arms()) and not stack.fake():  # arms B and C need the All-merged weights
        helpers.append(subprocess.Popen([PY, "-m", "full_eval.models", "all_merged"],
                                        stdout=open(root() / "logs" / "models.log", "a"), stderr=subprocess.STDOUT))
    stop = threading.Event()
    threading.Thread(target=watchdog, args=(budget, stop), daemon=True).start()
    statuses: dict = {}
    scorers: list = []
    try:
        if "decider" not in selected_arms() or run_decider(budget, statuses, scorers):
            save(statuses, budget)
            run_stack(budget, statuses, scorers)
        save(statuses, budget)
    finally:
        stop.set()
        for proc, arm, slug in scorers:
            proc.wait()
            if proc.returncode != 0:
                statuses[(arm, slug)] = {**statuses.get((arm, slug), {}), "status": "scoring failed (see logs)"}
        for h in helpers:
            if h.poll() is None:
                h.terminate()
        save(statuses, budget)
    subprocess.run([PY, "-m", "dm_harness.report"], check=False)
    print(f"\nspent ${budget.spent():.2f} of ${budget.cap:.0f}; report: {root() / 'dm_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

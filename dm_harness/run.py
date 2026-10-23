"""The decision-model run, both arms, resumable, within the money cap.

    python -m dm_harness.run            # shows the plan and what the cap buys, asks once, then runs to the end
    python -m dm_harness.run --yes      # no question
    python -m dm_harness.run --plan     # only the plan

  arm A  "decider"    pplx-decider alone, every GPU (dm_harness.base_run)
  arm B  "jah-agent"  JevAgentHarness + pplx-decider + All merged + mini-SWE-agent, with the NONE/invalid guard:
                      the decision model on DM_DECIDER_GPUS, vLLM on DM_GEMMA_GPUS (dm_harness.harness_run)

Every class with a reference partner and the complete SNOMED CT -> ICD-11 map (FULL_EVAL_SCOPE=reference, 299,961 per
arm; FULL_EVAL_TASKS narrows it), the same episodes as the full evaluation (full_eval.build_episodes, built in the
background while the GPUs work). A unit (arm, task) runs only if its projected cost fits the money left (DM_BUDGET_USD,
default $250, setup included); outputs/full_eval/STOP stops the workers between shards when the spend reaches the
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

ARMS = {"decider": "dm_harness.base_run", "jah-agent": "dm_harness.harness_run"}
DEFAULT_RATES = {"decider": 60.0, "jah-agent": 25.0}  # classes/s on 4 x B200 until a unit of the arm is measured
LOAD_S = {"decider": 600.0, "jah-agent": 1200.0}       # servers up (download excluded), once per arm
PY = sys.executable


def selected_arms() -> list[str]:
    arms = [a.strip() for a in os.environ.get("DM_ARMS", "decider,jah-agent").split(",") if a.strip()]
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


def run_arm(arm: str, budget: Budget, statuses: dict, scorers: list) -> bool:
    todo = [p for p in plan(budget) if p["arm"] == arm]
    if not todo:
        return True
    procs = []
    try:
        if arm == "decider":
            procs.append(stack.start_decider(stack.gpus()))
        else:
            if not stack.fake():
                from full_eval.models import ready

                while not ready("all_merged"):
                    print("[run] waiting for the All-merged model (outputs/full_eval/logs/models.log)", flush=True)
                    time.sleep(30)
            dec, gem = stack.split_gpus()
            procs.append(stack.start_decider(dec))
            procs.append(stack.start_vllm(gem))
            config = write_jah_config(root() / "jah_config.yaml", root() / "jah_logs")
            procs.append(stack.start_jah(config))
        for p in todo:
            if not p["fits"]:
                statuses[(arm, p["task"])] = {"status": "not run (money)", "usd": p["usd"]}
                continue
            if (root() / "STOP").exists():
                return False
            print(f"\n[run] {arm} · {p['task']}: {p['classes']:,} classes", flush=True)
            started = time.time()
            code = subprocess.run([PY, "-m", ARMS[arm], "--tasks", p["task"]]).returncode
            seconds = time.time() - started
            if code == 3:
                statuses[(arm, p["task"])] = {"status": "stopped (money); continues on the next run"}
                return False
            if code != 0:
                statuses[(arm, p["task"])] = {"status": f"worker failed (exit {code})"}
                continue
            n, _ = decisions(p["task"])
            record(arm, n, seconds)
            statuses[(arm, p["task"])] = {"status": "scoring", "seconds": round(seconds), "usd": round(budget.dollars(seconds), 2)}
            log = open(root() / "logs" / f"score_{arm}_{p['task']}.log", "a")
            scorers.append((subprocess.Popen([PY, "-m", "full_eval.score", arm, p["task"]], stdout=log,
                                             stderr=subprocess.STDOUT), arm, p["task"]))
        return True
    finally:
        stack.stop(*reversed(procs))


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
    budget = Budget(float(os.environ.get("DM_BUDGET_USD", "250")))
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
    if "jah-agent" in selected_arms() and not stack.fake():
        helpers.append(subprocess.Popen([PY, "-m", "full_eval.models", "all_merged"],
                                        stdout=open(root() / "logs" / "models.log", "a"), stderr=subprocess.STDOUT))
    stop = threading.Event()
    threading.Thread(target=watchdog, args=(budget, stop), daemon=True).start()
    statuses: dict = {}
    scorers: list = []
    try:
        for arm in selected_arms():
            if not run_arm(arm, budget, statuses, scorers):
                break
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

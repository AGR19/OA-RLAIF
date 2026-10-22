"""The full-alignment run: models, episodes, GPU workers, money guard, scoring and the report, resumable.

    python -m full_eval.run            # shows the plan and what $100 buys, asks once, then runs to the end
    python -m full_eval.run --yes      # no question
    python -m full_eval.run --plan     # only the plan

In the background: the models are rebuilt in run order (full_eval.models) and the episodes are built in task order
(full_eval.build_episodes), while the GPUs already work on what is ready. Each block of units of one method runs on
all GPUs: one vLLM per GPU, offline for the one-decision methods (full_eval.infer), `vllm serve` behind the
mini-SWE-agent for the agent method (full_eval.agent_run). A unit starts only if its projected cost fits the money
left, and outputs/full_eval/STOP stops the workers between shards when the spend reaches the cap. Scoring runs on the CPU while the GPUs go on. A unit that
was stopped is reported as not run; the next run continues its shards.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from .schedule import Budget, Speed, decisions, label, method, project, replies_complete, root, unit_done, units

PY = sys.executable


def gpu_count() -> int:
    if os.environ.get("FULL_EVAL_GPUS"):
        return int(os.environ["FULL_EVAL_GPUS"])
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
        return max(1, sum(1 for line in out.splitlines() if line.startswith("GPU")))
    except (OSError, subprocess.SubprocessError):
        return 1


def background(name: str, args: list[str]) -> subprocess.Popen:
    log = open(root() / "logs" / f"{name}.log", "a")
    return subprocess.Popen([PY, "-m", *args], stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())


def model_ready(tag: str) -> bool:
    from .models import ready

    return os.environ.get("FULL_EVAL_FAKE") == "lexical" or ready(method(tag).model)


def show_plan(plan: list[dict], budget: Budget) -> None:
    print(f"\nMoney: cap ${budget.cap:.0f}, spent so far ${budget.spent():.2f}, reserve ${budget.reserve:.0f} for scoring "
          f"and the upload; price ${budget.price:.2f}/h. Speed: {plan[0]['rate'] if plan else '?'} decisions/s "
          f"({plan[0]['rate_basis'] if plan else ''}).")
    print(f"{'method':<23} {'task':<26} {'decisions':>10} {'/s':>6} {'hours':>6} {'$':>7}  fits")
    for p in plan:
        print(f"{p['model']:<23} {p['task']:<26} {p['decisions']:>10,}{'' if p['exact'] else '~'} {p['rate']:>6g} "
              f"{p['seconds'] / 3600:>6.2f} {p['usd']:>7.2f}  {'yes' if p['fits'] else 'NO: not run unless the measured speed is higher'}")


def save_plan(statuses: dict, budget: Budget) -> None:
    rows = []
    for model, slug in units():
        n, _ = decisions(slug)
        rows.append({"model": model, "task": slug, "decisions": n, **statuses.get((model, slug), {}),
                     "status": "done" if unit_done(model, slug) else statuses.get((model, slug), {}).get("status", "not run")})
    (root() / "plan.json").write_text(json.dumps({"budget_usd": budget.cap, "spent_usd": round(budget.spent(), 2),
                                                  "price_usd_per_hour": budget.price, "units": rows}, indent=1))


def progress_line(model: str, block: list[str], budget: Budget, started: float) -> str:
    done = total = 0
    for slug in block:
        manifest = root() / "episodes" / slug / "manifest.json"
        if not (manifest.parent / "READY").exists():
            continue
        shards = json.loads(manifest.read_text())["shards"]
        folder = root() / "replies" / model / slug
        total += sum(shards.values())
        done += sum(n for name, n in shards.items() if (folder / name.replace(".jsonl.gz", ".replies.jsonl.gz")).exists())
    rate = done / max(1.0, time.time() - started)
    return (f"[{time.strftime('%H:%M:%S')}] {label(model)}: {done:,}/{total:,} decisions "
            f"({rate:.0f}/s incl. load) · spent ${budget.spent():.2f} of ${budget.cap:.0f}")


def run_block(model: str, block: list[str], budget: Budget, speed: Speed, ngpu: int) -> bool:
    """All GPUs on one method for the given tasks. Returns False when the money guard stopped it."""
    while not model_ready(model):
        print(f"[{time.strftime('%H:%M:%S')}] waiting for model {model} (full_eval/logs/models.log)", flush=True)
        time.sleep(30)
    for path in (root() / "progress").glob(f"{model}.*.json"):  # fresh load times for this block
        state = json.loads(path.read_text())
        state.pop("load_seconds", None)
        path.write_text(json.dumps(state))
    workers = []
    for rank in range(ngpu):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(rank)}
        worker = "full_eval.agent_run" if method(model).agent else "full_eval.infer"
        args = [PY, "-m", worker, "--tag", model, "--tasks", ",".join(block), "--rank", str(rank), "--world", str(ngpu)]
        if rank == 0 and model in ("grpo", "sdft"):
            args.append("--verify-first")  # weight check of the re-merged models, thinking off only
        log = open(root() / "logs" / f"infer_{model}_{rank}.log", "a")
        workers.append(subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env))
    started = time.time()
    stopped = False
    while any(w.poll() is None for w in workers):
        time.sleep(20)
        if budget.spent() >= budget.cap - budget.reserve / 2 and not stopped:
            (root() / "STOP").write_text(f"spend reached ${budget.spent():.2f} of ${budget.cap:.0f}")
            stopped = True
            print("[money] the cap is reached: the workers stop after their current shard", flush=True)
        print(progress_line(model, block, budget, started), flush=True)
    failed = [w.returncode for w in workers if w.returncode not in (0, 3)]
    if failed:
        raise SystemExit(f"a GPU worker failed (exit {failed}); see outputs/full_eval/logs/infer_{model}_*.log")
    loads = [json.loads(p.read_text()).get("load_seconds") for p in (root() / "progress").glob(f"{model}.*.json")]
    loads = [x for x in loads if x]
    if loads:
        speed.record_load(max(loads))
    return not stopped and not (root() / "STOP").exists()


def main(argv: list[str] | None = None) -> int:
    from .score import unit_timing

    parser = argparse.ArgumentParser(prog="full_eval.run")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--budget", type=float, default=None)
    args = parser.parse_args(argv)
    (root() / "logs").mkdir(parents=True, exist_ok=True)
    budget, speed = Budget(args.budget), Speed()
    if (root() / "STOP").exists():
        (root() / "STOP").unlink()  # a new run decides afresh against the cap
    todo = [u for u in units() if not unit_done(*u)]
    plan = project(todo, budget, speed)
    show_plan(plan, budget)
    if args.plan:
        return 0
    if not args.yes and sys.stdin.isatty():
        if input("\nStart? Units marked NO run only if the measured speed allows them. [y/N] ").strip().lower() not in ("y", "yes"):
            return 1
    helpers = []
    if os.environ.get("FULL_EVAL_FAKE") != "lexical":
        order = list(dict.fromkeys(method(m).model for m, _ in todo))
        helpers.append(background("models", ["full_eval.models", *order]))
    helpers.append(background("episodes", ["full_eval.build_episodes"]))
    ngpu = gpu_count()
    statuses: dict = {}
    scoring: list[tuple[subprocess.Popen, str, str]] = []
    loaded = None
    while True:
        failed = {p.parent.name for p in (root() / "episodes").glob("*/FAILED")}
        for u in units():
            if u[1] in failed:
                statuses[u] = {"status": "episodes failed (see logs/episodes.log)"}
        todo = [u for u in units() if not unit_done(*u) and not replies_complete(*u) and u[1] not in failed]
        if not todo:
            break
        plan = project(todo, budget, speed, loaded)
        fit = [p for p in plan if p["fits"]]
        for p in plan:
            if not p["fits"]:
                statuses[(p["model"], p["task"])] = {"status": "not run (money)", "usd": p["usd"], "seconds": p["seconds"]}
        if not fit:
            break
        model = fit[0]["model"]
        block = []
        for p in fit:
            if p["model"] != model:
                break
            block.append(p["task"])
            statuses[(model, p["task"])] = {"status": "running", "usd": p["usd"], "seconds": p["seconds"]}
        save_plan(statuses, budget)
        print(f"\n[block] {label(model)} on {ngpu} GPU(s): {', '.join(block)}", flush=True)
        finished = run_block(model, block, budget, speed, ngpu)
        loaded = model
        for slug in block:
            if replies_complete(model, slug):
                timing = unit_timing(model, slug)
                n, _ = decisions(slug)
                if timing["runtime_s"]:
                    speed.record_unit(f"{model}/{slug}", n, timing["runtime_s"])
                statuses[(model, slug)] = {**statuses[(model, slug)], "status": "scoring"}
                log = open(root() / "logs" / f"score_{model}_{slug}.log", "a")
                scoring.append((subprocess.Popen([PY, "-m", "full_eval.score", model, slug], stdout=log,
                                                 stderr=subprocess.STDOUT), model, slug))
            else:
                statuses[(model, slug)] = {**statuses[(model, slug)], "status": "stopped (money); continues on the next run"}
        save_plan(statuses, budget)
        if not finished:
            break
    for proc, model, slug in scoring:
        proc.wait()
        if proc.returncode != 0:
            statuses[(model, slug)] = {**statuses.get((model, slug), {}), "status": "scoring failed (see logs)"}
    for h in helpers:
        if h.poll() is None:
            h.terminate()
    save_plan(statuses, budget)
    subprocess.run([PY, "-m", "full_eval.report"], check=False)
    print(f"\nspent ${budget.spent():.2f} of ${budget.cap:.0f}; report: {root() / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

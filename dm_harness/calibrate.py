"""Arm C's thresholds, fitted on a calibration sample, then frozen: the paper's procedure (Zhu et al. 2026, §2.1 and
App. A, "Deployed thresholds"): shadow-mode decisions -> equal-mass confidence bins of at least 20 -> tau* = the lowest
confidence from which every higher bin has a non-negative estimated advantage -> frozen for the run. No threshold if no
bin qualifies (the LLM keeps that decision).

    python -m dm_harness.calibrate            # sample, shadow runs, fit -> outputs/full_eval/calibration/
    python -m dm_harness.calibrate --fit      # only the fit, from the shadow runs already recorded

**The sample.** DM_CAL_N classes (default 3,000), seeded (DM_CAL_SEED, 13), stratified by task: proportional to the
task's scored classes, at least 20 per task (or all of a smaller task). Only classes the Top-k evaluation scores.
The sample is used to choose the thresholds; arm C then aligns and scores **every** class (299,961), the sample included,
and the report also gives arm C's scores without the sample.

**Shadow runs** (the decision model answers, never acts), per sample class:
  strong   the agent on `jah-shadow`: every step on the strong tier (thinking on); the proxy asks pplx-decider
           "routine or hard?" and logs p(hard) (routing: shadow)
  fast     the agent on `jah-fast` (thinking off)
  redo     each NONE or invalid answer of either run, redone on `jah-strong` with the hint of pplx-decider's best
           candidate, whatever its probability (so every bin has the outcome)
  decider  pplx-decider's own answer and confidence (arm A's reply; asked here if arm A has not run)
Labels: the class's reference partners (dm_harness.labels); an answer is right when its first target is a partner.

**The three decision points** (V = $1 per wrong class, the plan's pre-registered value; dC = the measured GPU cost of
one agent step):
  routing  (TRIGGER) the strong tier when p(hard) >= tau_r. Gain of firing = right(strong) - right(fast) - dC/V.
  guard    (GUARD)   redo a NONE / invalid answer when p(best candidate) >= tau_g. Gain = right(redo) - right(kept) - dC/V.
  cascade  (CASCADE) pplx-decider answers alone when its confidence >= tau_c (Eq. 3: g_d - q_L >= -dC/V), with q_L the
                     LLM policy at the fitted tau_r and tau_g. Advantage = right(decider) - right(policy) + dC/V.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from full_eval.build_episodes import read_shard
from full_eval.schedule import root, selected_tasks, task_order

from . import labels
from .harness import VALID, HarnessMatcher, decide
from .stack import fake

V_USD = 1.0
MIN_BIN = 20
MAX_BINS = 10


def cal_dir() -> Path:
    d = root() / "calibration"
    d.mkdir(parents=True, exist_ok=True)
    return d


def answer(rec: dict) -> list[str]:
    return list(rec.get("targets") or [])[:1] if rec.get("decision") == "match" else []


# ---- the sample -------------------------------------------------------------------------------------------------------
def scored_episodes(task) -> list[dict]:
    """The task's episodes whose class has partners (the classes the Top-k evaluation scores)."""
    part = labels.partners(task)
    out = []
    for path in sorted((root() / "episodes" / task.slug).glob("*.jsonl.gz")):
        out += [e for e in read_shard(path) if e["source_iri"] in part.get(e["pair_id"], {})]
    return out


def draw(n: int, seed: int) -> dict[str, list[dict]]:
    pools = {t.slug: scored_episodes(t) for t in task_order(selected_tasks())}
    total = sum(len(p) for p in pools.values()) or 1
    sample = {}
    for slug, pool in pools.items():
        want = min(len(pool), max(MIN_BIN, round(n * len(pool) / total)))
        rng = random.Random(f"{seed}:{slug}")
        sample[slug] = rng.sample(pool, want) if want < len(pool) else list(pool)
    return sample


# ---- the shadow runs --------------------------------------------------------------------------------------------------
def decider_rows(slug: str) -> dict[str, dict]:
    out = {}
    for path in sorted((root() / "replies" / "decider" / slug).glob("*.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[r["id"]] = r
    return out


def shadow(sample: dict[str, list[dict]], matcher: HarnessMatcher, workers: int) -> Path:
    path = cal_dir() / "shadow.jsonl"
    done = {json.loads(line)["id"] for line in path.read_text().splitlines()} if path.exists() else set()

    def one(item):
        slug, e, a = item
        run = lambda model, sid, note="": matcher.run_episode(  # noqa: E731
            e, model_name=model, model_kwargs=matcher.session(f"{e['id']}-cal-{sid}"), note=note)
        try:
            check = decide(e, matcher.decider_url)
        except Exception as error:  # noqa: BLE001
            check = {"best_candidate": None, "p_best_candidate": 0.0, "pick": "NONE", "p": 0.0,
                     "error": f"{type(error).__name__}: {error}"[:200]}
        rec = {"id": e["id"], "task": slug, "pair_id": e["pair_id"], "source": e["source_iri"],
               "decider": {"answer": answer(a) if a else ([] if check["pick"] == "NONE" else [check["pick"]]),
                           "confidence": (a or {}).get("confidence", check.get("p", 0.0)),
                           "best_candidate": check["best_candidate"], "p_best": check["p_best_candidate"]}}
        for tier, model in (("strong", "jah-shadow"), ("fast", "jah-fast")):
            r = run(model, tier)
            rec[tier] = {"decision": r["decision"], "answer": answer(r), "seconds": r["seconds"]}
            if r["decision"] != "match":  # the guard's opportunity: its redo, with the hint whatever p is
                from .harness import hint

                note = hint(e, check["best_candidate"]) if check["best_candidate"] else ""
                redo = run("jah-strong", f"redo-{tier}", note)
                rec[f"redo_{tier}"] = {"decision": redo["decision"], "answer": answer(redo), "hinted": bool(note)}
        return rec

    todo = []
    for slug, episodes in sample.items():
        rows = decider_rows(slug)
        todo += [(slug, e, rows.get(e["id"])) for e in episodes if e["id"] not in done]
    started = time.time()
    with ThreadPoolExecutor(workers) as pool, path.open("a") as out:
        for i, rec in enumerate(pool.map(one, todo), 1):
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if i % 100 == 0 or i == len(todo):
                print(f"[calibrate] shadow {i:,}/{len(todo):,} ({i / max(1e-9, time.time() - started):.1f}/s)",
                      flush=True)
    return path


def p_hard() -> dict[str, float]:
    """p(hard) of each sample class's strong shadow run, from the proxy's step log (session 'jah-shadow/<id>-cal-strong')."""
    out = {}
    log = root() / "jah_logs" / "steps.jsonl"
    if log.exists():
        for line in log.open():
            r = json.loads(line)
            sid = str(r.get("session", ""))
            if sid.startswith("jah-shadow/") and sid.endswith("-cal-strong") and r.get("p_hard") is not None:
                out[sid.split("/", 1)[1].removesuffix("-cal-strong")] = float(r["p_hard"])
    return out


# ---- the fit ----------------------------------------------------------------------------------------------------------
def fit_threshold(scores: list[float], gains: list[float], extra: list[dict] | None = None) -> tuple[float | None, list]:
    """The paper's rule. Equal-mass bins of at least MIN_BIN decisions (at most MAX_BINS bins), sorted by the decision
    model's score; tau = the lowest score from which every higher bin has a non-negative mean gain; None when the top
    bin is negative (never act). A bin edge never splits equal scores, so 'score >= tau' acts on exactly the bins at and
    above tau; the sign is read from the unrounded mean (a cost of a fraction of a cent still counts)."""
    items = sorted(zip(scores, gains, extra or [{}] * len(scores)), key=lambda x: x[0])
    n = len(items)
    if n < MIN_BIN:
        return None, []
    nb = max(1, min(MAX_BINS, n // MIN_BIN))
    edges = [0]
    for b in range(1, nb):
        e = round(b * n / nb)
        while 0 < e < n and items[e][0] == items[e - 1][0]:
            e += 1
        if e - edges[-1] >= MIN_BIN and n - e >= MIN_BIN:
            edges.append(e)
    edges.append(n)
    bins, means = [], []
    for lo, hi in zip(edges, edges[1:]):
        part = items[lo:hi]
        means.append(statistics.fmean(g for _, g, _ in part))
        row = {"from": round(part[0][0], 4), "to": round(part[-1][0], 4), "n": len(part), "gain": round(means[-1], 6)}
        for key in (part[0][2] or {}):
            row[key] = round(statistics.fmean(x[2][key] for x in part), 4)
        bins.append(row)
    tau = None
    for (lo, _), g in zip(reversed(list(zip(edges, edges[1:]))), reversed(means)):
        if g < 0:
            break
        tau = items[lo][0]
    for b, lo in zip(bins, edges):
        b["acts"] = tau is not None and items[lo][0] >= tau
    return tau, bins


def fit(dc_usd: float) -> dict:
    rows = [json.loads(line) for line in (cal_dir() / "shadow.jsonl").read_text().splitlines()]
    hard = p_hard()
    part = {}
    for t in task_order(selected_tasks()):
        for pid, p in labels.partners(t).items():
            part[pid] = p
    right = lambda r, ans: bool(labels.correct(ans, part.get(r["pair_id"], {}).get(r["source"])))  # noqa: E731
    k = dc_usd / V_USD

    # routing (TRIGGER): fire the strong tier when p(hard) >= tau_r
    route = [(hard[r["id"]], right(r, r["strong"]["answer"]) - right(r, r["fast"]["answer"]) - k,
              {"right_strong": right(r, r["strong"]["answer"]), "right_fast": right(r, r["fast"]["answer"])})
             for r in rows if r["id"] in hard]
    tau_r, bins_r = fit_threshold(*zip(*route)) if route else (None, [])

    # guard (GUARD): redo a NONE / invalid answer when p(best candidate) >= tau_g
    guard = [(r["decider"]["p_best"], right(r, r[f"redo_{tier}"]["answer"]) - right(r, r[tier]["answer"]) - k,
              {"right_redo": right(r, r[f"redo_{tier}"]["answer"])})
             for r in rows for tier in ("strong", "fast") if f"redo_{tier}" in r]
    tau_g, bins_g = fit_threshold(*zip(*guard)) if guard else (None, [])

    # the LLM policy at the fitted tau_r and tau_g (q_L of the cascade)
    def policy(r: dict) -> list[str]:
        tier = "strong" if tau_r is not None and hard.get(r["id"], 1.0) >= tau_r else "fast"
        out = r[tier]
        redo = r.get(f"redo_{tier}")
        if redo and (out["decision"] not in VALID or (tau_g is not None and r["decider"]["p_best"] >= tau_g)):
            if redo["decision"] in VALID:
                return redo["answer"]
            return r["decider"]["answer"]  # still invalid: the decision model's pick
        return out["answer"]

    # cascade (CASCADE, Eq. 3): pplx-decider answers alone when its confidence >= tau_c
    cascade = [(float(r["decider"]["confidence"] or 0.0),
                right(r, r["decider"]["answer"]) - right(r, policy(r)) + k,
                {"g_d": right(r, r["decider"]["answer"]), "q_L": right(r, policy(r))}) for r in rows]
    tau_c, bins_c = fit_threshold(*zip(*cascade)) if cascade else (None, [])

    out = {"procedure": "shadow mode -> equal-mass bins (>= 20) -> lowest confidence from which every higher bin has a "
                        "non-negative advantage (Zhu et al. 2026, App. A) -> frozen",
           "V_usd": V_USD, "dC_usd": dc_usd, "sample_classes": len(rows),
           "seed": int(os.environ.get("DM_CAL_SEED", "13")),
           "routing": {"tau": tau_r, "decisions": len(route), "bins": bins_r,
                       "acts": "strong tier (thinking on) when p(hard) >= tau; never when tau is null"},
           "guard": {"tau": tau_g, "decisions": len(guard), "bins": bins_g,
                     "acts": "redo NONE / invalid on the strong tier with the hint when p(best candidate) >= tau"},
           "cascade": {"tau": tau_c, "decisions": len(cascade), "bins": bins_c,
                       "acts": "pplx-decider answers alone when its confidence >= tau; never when tau is null"}}
    (cal_dir() / "thresholds.json").write_text(json.dumps(out, indent=1))
    (cal_dir() / "sample_ids.json").write_text(json.dumps(sorted(r["id"] for r in rows)))
    print(f"[calibrate] fitted and frozen: routing tau {tau_r}, guard tau {tau_g}, cascade tau {tau_c} "
          f"({len(rows):,} sample classes)", flush=True)
    return out


def step_cost_usd() -> float:
    """dC: the GPU cost of one agent step, from arm B's measured speed and this machine's price."""
    from full_eval.schedule import Budget

    speed = root() / "dm_speed.json"
    rate = 25.0
    if speed.exists():
        s = json.loads(speed.read_text()).get("jah-agent") or {}
        if s.get("classes", 0) >= 500 and s.get("seconds"):
            rate = s["classes"] / s["seconds"]
    return Budget().price / 3600 / rate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dm_harness.calibrate")
    ap.add_argument("--fit", action="store_true", help="only the fit (the shadow runs are recorded)")
    args = ap.parse_args(argv)
    if not args.fit:
        n, seed = int(os.environ.get("DM_CAL_N", "3000")), int(os.environ.get("DM_CAL_SEED", "13"))
        sample = draw(n, seed)
        print(f"[calibrate] sample: {sum(len(v) for v in sample.values()):,} classes "
              f"({', '.join(f'{k} {len(v)}' for k, v in sample.items())})", flush=True)
        workers = int(os.environ.get("DM_AGENT_WORKERS", "8" if fake() else "256"))
        sandbox = os.environ.get("OA_AGENT_ENV") or ("local" if fake() else "user")
        matcher = HarnessMatcher(root() / "agent_tasks" / "calibration", workers, sandbox, save_trajectory=False,
                                 keep_task_dirs=False)
        shadow(sample, matcher, workers)
    fit(step_cost_usd())
    return 0


if __name__ == "__main__":
    sys.exit(main())

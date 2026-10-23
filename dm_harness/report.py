"""Tables of the decision-model run, beside the All-merged rows of the full evaluation.

    python -m dm_harness.report      -> outputs/full_eval/dm_report.md, dm_report.json, dm_report.tex, figures/

Rows (the user's scope: only All merged and the two new arms):
  All merged (one decision)               results/<task>/5-all-merged/                  } the first run, read from
  All merged + mini-SWE-agent, thinking on results/<task>/6-all-merged-agent-thinking/  } oa-rlaif-full-eval-results
  pplx-decider-v1.2 alone (arm A)         results/<task>/7-pplx-decider/
  JevAgentHarness, fixed tau (arm B)      results/<task>/8-jah-all-merged-agent/
  JevAgentHarness, fitted tau (arm C)     results/<task>/9-jah-all-merged-agent-fitted/
A unit that did not run shows "–"; nothing is estimated.

Focus   SNOMED CT -> ICD-11 and Cybersecurity: P, R, F at k = 1, 3, 5.
1, 2, 3 the full evaluation's Top-k tables (F@1 per dataset; P, R, F at k per dataset; per pair with the ceiling).
4       what each method did (answers, NONE, invalid, tokens, the agent's steps, the thinking, runtime).
5       the decision model's actions per arm: arm C's cascade share, the NONE/invalid guard and what came of it.
6       routing: the share of model calls the decision model sent to thinking on, p(hard), decision latency (the
        proxy's steps.jsonl; no prompt text is logged).
7       the complete SNOMED CT -> ICD-11 map.
8       sensitivity without a rerun: Top-1 F if pplx-decider answered a class alone whenever its probability was at
        least tau, else arm B's answer (the paper's CASCADE; fixed taus, read from the logged probabilities).
10      arm C's thresholds, fitted on the calibration sample (dm_harness.calibrate), with each decision point's bins.
11      arm C's Top-1 F on every class and without the calibration sample.
12      the units and the money.
"""

from __future__ import annotations

import gzip
import json
import statistics
import sys
from collections import Counter, defaultdict

from full_eval import report as R
from full_eval.schedule import label, root
from full_eval.tasks import COLUMNS, tasks
from oa_rlaif.eval.metrics import TOP_K, TOP_K_NAME
from oa_rlaif.eval.report import render

ROWS = ["all_merged", "all_merged-agent-think", "decider", "jah-agent", "jah-fit-agent"]
FOCUS = ["SNOMED→ICD-11", "Cybersecurity"]
TAUS = (0.5, 0.75, 0.9, 0.95, 0.99)


def rows() -> list[str]:
    return list(ROWS)


def focus_table(results: dict, column: str) -> list:
    out = []
    for _ in (column,):
        for tag in rows():
            per = results.get(tag, {})
            values = [R.topk_value(per, column, k, m) for k in TOP_K for m in ("Precision", "Recall", "F-measure")]
            if all(v is None for v in values):
                continue
            out.append((f"{column} · {label(tag)}", values))
    return out


HARNESS = {"jah-agent": "B (fixed τ)", "jah-fit-agent": "C (fitted τ)"}


def read_rows(tag: str, slug: str) -> dict[str, dict]:
    out = {}
    for path in sorted((root() / "replies" / tag / slug).glob("*.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[r["id"]] = r
    return out


def guard_table(results: dict) -> list:
    out = []
    for tag, arm in HARNESS.items():
        for t in tasks():
            s = results.get(tag, {}).get(t.slug)
            if not s:
                continue
            b = s["behaviour"]
            g = b.get("guard") or {}
            out.append((f"{arm} · {t.slug}", [b["classes"], (b.get("cascade") or 0) * 100, sum(g.values()),
                                              g.get("kept_none", 0), g.get("redo_match", 0), g.get("redo_none", 0),
                                              g.get("decider_pick", 0), g.get("decider_error", 0), b["none"] * 100,
                                              b["invalid"] * 100]))
    return out


def routing_table() -> list:
    """Per arm and task: model calls through the proxy, the share sent to the strong tier (thinking on) and why,
    p(hard), the decision call's median latency. Sessions are '<model name>/<episode id><suffix>': arm B none, arm C
    '-fit', the guard's redo '-redo', the calibration's shadow runs '-cal-...' (left out here)."""
    log = root() / "jah_logs" / "steps.jsonl"
    if not log.exists():
        return []
    task_of = {}
    for tag in HARNESS:
        for path in (root() / "replies" / tag).glob("*/*.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    task_of[json.loads(line)["id"]] = path.parent.name
    per = defaultdict(lambda: {"calls": 0, "strong": 0, "reasons": Counter(), "p": [], "lat": []})
    with log.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "cache" or "tier" not in r:
                continue
            sid = str(r.get("session", "")).split("/", 1)[-1]
            if "-cal" in sid:
                continue
            arm = "jah-fit-agent" if "-fit" in sid else "jah-agent"
            slug = task_of.get(sid.split("-", 1)[0])
            if slug is None:
                continue
            d = per[(arm, slug)]
            d["calls"] += 1
            d["strong"] += r.get("tier") == "strong"
            d["reasons"][str(r.get("reason"))] += 1
            if r.get("p_hard") is not None:
                d["p"].append(float(r["p_hard"]))
            lat = (r.get("decision") or {}).get("latency_s")
            if lat is not None:
                d["lat"].append(float(lat))
    out = []
    for tag, arm in HARNESS.items():
        for t in tasks():
            d = per.get((tag, t.slug))
            if not d:
                continue
            n = max(1, d["calls"])
            out.append((f"{arm} · {t.slug}", [d["calls"], d["strong"] / n * 100, d["reasons"].get("hard", 0) / n * 100,
                                              d["reasons"].get("routine", 0) / n * 100,
                                              d["reasons"].get("pinned", 0) / n * 100,
                                              d["reasons"].get("decision_error", 0) / n * 100,
                                              statistics.fmean(d["p"]) if d["p"] else None,
                                              statistics.median(d["lat"]) if d["lat"] else None]))
    return out


def f1_by_column(choose, exclude: set | None = None) -> dict[str, float | None]:
    """Top-1 F ×100 per dataset column of the answers `choose(task, episode)` gives ([] = nothing found; None = no
    row), over every scored class except `exclude`, aggregated exactly as the full evaluation's tables
    (full_eval.report.topk_value: the mean over a column's pairs; "–" while one of its tasks has not run)."""
    from full_eval.build_episodes import read_shard
    from full_eval.tasks import zero_shot_pairs
    from oa_rlaif.eval.metrics import AlignmentEvaluator

    from .labels import partners

    zero, per = zero_shot_pairs(), {}
    for t in tasks():
        if not (root() / "episodes" / t.slug / "READY").exists():
            continue
        per_pair: dict = defaultdict(dict)
        left_out: dict = defaultdict(set)  # the excluded classes leave the denominator too
        for path in sorted((root() / "episodes" / t.slug).glob("*.jsonl.gz")):
            for e in read_shard(path):
                if exclude and e["id"] in exclude:
                    left_out[e["pair_id"]].add(e["source_iri"])
                    continue
                ans = choose(t, e)
                if ans is not None:
                    per_pair[e["pair_id"]][e["source_iri"]] = ans
        if not per_pair:
            continue
        part = partners(t)
        per[t.slug] = {"pairs": {}}
        for pid, rk in per_pair.items():
            keep = {k: v for k, v in part.get(pid, {}).items() if k not in left_out[pid]}
            per[t.slug]["pairs"][pid] = {"zero_shot": pid in zero,
                                         "topk": AlignmentEvaluator().topk(rk, keep, ks=(1,)) if keep else {}}
    return {c: R.topk_value(per, c) for c in COLUMNS}


def first(r: dict | None) -> list[str] | None:
    if r is None:
        return None
    return list(r.get("targets") or [])[:1] if r.get("decision") == "match" else []


def cascade_table() -> list:
    """Top-1 F per dataset column if pplx-decider's pick (arm A) replaced arm B's answer whenever its probability was
    >= tau, at fixed taus (no rerun; read from the logged probabilities)."""
    rows_a: dict = {}
    rows_b: dict = {}
    for t in tasks():
        rows_a.update(read_rows("decider", t.slug))
        rows_b.update(read_rows("jah-agent", t.slug))
    if not rows_a or not rows_b:
        return []
    out = []
    for tau in TAUS:
        def choose(t, e, tau=tau):
            a, b = rows_a.get(e["id"]), rows_b.get(e["id"])
            if b is None:
                return None
            use_a = a is not None and a.get("decision") in ("match", "NONE") and (a.get("confidence") or 0) >= tau
            return first(a if use_a else b)

        taken = sum(1 for i, b in rows_b.items() if rows_a.get(i) and rows_a[i].get("decision") in ("match", "NONE")
                    and (rows_a[i].get("confidence") or 0) >= tau)
        cells = f1_by_column(choose)
        out.append((f"τ = {tau:g}", [taken / max(1, len(rows_b)) * 100] + [cells.get(c) for c in COLUMNS]))
    return out


def thresholds_table() -> tuple[list, dict]:
    """Arm C's fitted thresholds (dm_harness.calibrate): one row per decision point, then its bins."""
    path = root() / "calibration" / "thresholds.json"
    if not path.exists():
        return [], {}
    th = json.loads(path.read_text())
    out = []
    for point, mode in (("routing", "TRIGGER: thinking on when p(hard) ≥ τ"),
                        ("guard", "GUARD: redo NONE/invalid when p(best) ≥ τ"),
                        ("cascade", "CASCADE: pplx-decider alone when confidence ≥ τ")):
        d = th[point]
        out.append((f"{point} · {mode}", ["–" if d["tau"] is None else f"{d['tau']:.3f}", d["decisions"], "", "", ""]))
        for i, b in enumerate(d["bins"], 1):
            extra = ", ".join(f"{k} {v:.2f}" for k, v in b.items() if k not in ("from", "to", "n", "gain", "acts"))
            out.append((f"  bin {i}", [f"{b['from']:.3f}–{b['to']:.3f}", b["n"], f"{b['gain']:+.3g}",
                                       "acts" if b.get("acts") else "", extra]))
    return out, th


def sample_table() -> list:
    """Arm C's Top-1 F on every class and without the calibration sample (the sample chose the thresholds)."""
    path = root() / "calibration" / "sample_ids.json"
    rows_c: dict = {}
    for t in tasks():
        rows_c.update(read_rows("jah-fit-agent", t.slug))
    if not rows_c or not path.exists():
        return []
    sample = set(json.loads(path.read_text()))
    choose = lambda t, e: first(rows_c.get(e["id"]))  # noqa: E731
    full, rest = f1_by_column(choose), f1_by_column(choose, exclude=sample)
    return [("C (fitted τ) · every class", [len(rows_c)] + [full.get(c) for c in COLUMNS]),
            ("C (fitted τ) · without the calibration sample", [len(rows_c) - len(sample & set(rows_c))]
             + [rest.get(c) for c in COLUMNS])]


def behaviour_rows(results: dict, price: float | None) -> list:
    """Table 4's rows. The first run's rows keep their own runtime (4 x B300), but their cost is not this machine's
    price times that runtime, so it is left out ("–")."""
    first_run = {label(tag) for tag in ("all_merged", "all_merged-agent-think")}
    return [(name, values[:-1] + [None] if name in first_run else values)
            for name, values in R.table_behaviour(results, price)]


def build() -> dict:
    R.row_tags = rows  # the full evaluation's table functions, with this report's five rows
    results = R.collect()
    th_rows, _ = thresholds_table()
    plan = json.loads((root() / "plan.json").read_text()) if (root() / "plan.json").exists() else {"units": []}
    price = plan.get("price_usd_per_hour")
    header = COLUMNS + ["Average"]
    pct = "{:.1f}"
    tables = {
        "t0": render(f"SNOMED CT → ICD-11 (scored on the strong ICD-10-composed links, silver): {TOP_K_NAME}, "
                     "Precision@k, Recall@k, F-measure@k ×100.",
                     [f"{m[0]}@{k}" for k in TOP_K for m in ("Precision", "Recall", "F-measure")],
                     focus_table(results, "SNOMED→ICD-11"), pct),
        "t00": render(f"Cybersecurity (IoTSec → UCO): {TOP_K_NAME}, Precision@k, Recall@k, F-measure@k ×100.",
                      [f"{m[0]}@{k}" for k in TOP_K for m in ("Precision", "Recall", "F-measure")],
                      focus_table(results, "Cybersecurity"), pct),
        "t1": render(f"Table 1. {TOP_K_NAME}: F-measure@1 ×100 per dataset (mean over pairs).", header,
                     R.table5(results), pct),
        "t2": render(f"Table 2. {TOP_K_NAME} per dataset: Precision@k, Recall@k and F-measure@k ×100.",
                     [f"{m[0]}@{k}" for k in TOP_K for m in ("Precision", "Recall", "F-measure")] + ["Classes scored"],
                     R.table6(results), [pct] * 9 + ["{:,}"], higher=[True] * 9 + [None]),
        "t3": render(f"Table 3. {TOP_K_NAME} per pair (MultiFarm, DH and DH-ML: the mean over their pairs); Ceiling = "
                     "share of scored classes whose partner was among the candidates.",
                     ["Classes scored", "Ceiling"] + [f"{m[0]}@{k}" for k in TOP_K for m in ("Precision", "Recall",
                                                                                              "F-measure")],
                     R.table_pairs(results), ["{:,}", "{:.3f}"] + ["{:.3f}"] * 9, higher=[None, None] + [True] * 9),
        "t4": render("Table 4. What each method did (every task it ran, weighted by classes). Runtime and $ of the "
                     "first-run rows are from their 4 × B300 run; the new arms ran on 4 × B200.",
                     ["Classes", "Match %", "NONE %", "Invalid %", "Prompt tokens/class", "Completion tokens/class",
                      "Model calls/class", "Extra commands/class", "Format errors/class", "Seconds/class",
                      "Thinking characters/class", "Answer inside thinking %", "Cut at token limit %", "Runtime h",
                      "Cost $"], behaviour_rows(results, price),
                     ["{:,}", pct, pct, pct, "{:,.0f}", "{:,.0f}", "{:.2f}", "{:.2f}", "{:.2f}", "{:.1f}", "{:,.0f}",
                      pct, pct, "{:.2f}", "{:.2f}"], higher=[None] * 15),
        "t5": render("Table 5. The decision model's actions per arm and task (classes). Cascade % = classes "
                     "pplx-decider answered alone (arm C, fitted τ). The guard acts on a NONE or invalid submission: kept "
                     "NONE = best candidate below τ (B: 0.5; C: fitted); redo = Gemma again on the strong tier (with the "
                     "hint above τ); decider's pick = the redo was invalid too.",
                     ["Classes", "Cascade %", "Guard acted", "Kept NONE", "Redo → match", "Redo → NONE",
                      "Decider's pick", "Decider error", "NONE % (final)", "Invalid % (final)"], guard_table(results),
                     ["{:,}", pct] + ["{:,}"] * 6 + [pct, pct], higher=[None] * 10),
        "t6": render("Table 6. Routing by pplx-decider per arm (the proxy's step log): model calls, share on the strong "
                     "tier (thinking on), why (hard: p(hard) ≥ τ, B 0.75, C fitted; routine; pinned: a fixed tier, i.e. the "
                     "guard's redo, or every arm-C step when the fit gave routing no τ (thinking off) or τ = 0 (thinking "
                     "on); decision error → strong), mean p(hard), median decision latency.",
                     ["Model calls", "Strong %", "Hard %", "Routine %", "Pinned %", "Decision error %",
                      "Mean p(hard)", "Decision latency s"], routing_table(),
                     ["{:,}", pct, pct, pct, pct, pct, "{:.3f}", "{:.3f}"], higher=[None] * 8),
        "t7": R.snomed_table(results, 7),
        "t8": render("Table 8. Without a rerun: Top-1 F ×100 if pplx-decider answered a class alone whenever its "
                     "probability was ≥ τ, and arm B's answer stood otherwise (the paper's CASCADE). Share = classes "
                     "pplx-decider would answer. Fixed τ values, not fitted.", ["Share %"] + COLUMNS,
                     cascade_table(), pct, higher=[None] + [True] * len(COLUMNS)),
        "t10": render("Table 10. Arm C's thresholds, fitted on the calibration sample and frozen (the paper's rule: "
                      "equal-mass bins of ≥ 20 decisions; τ = the lowest score from which every higher bin has a "
                      "non-negative gain; “–” = no bin qualified, the LLM keeps the decision). Gain per bin: routing "
                      "right(strong) − right(fast) − ΔC/V; guard right(redo) − right(kept) − ΔC/V; cascade g_d − q_L + "
                      "ΔC/V.", ["τ / bin range", "Decisions", "Gain", "Acts", "Rates in the bin"], th_rows,
                      ["{}", "{:,}", "{}", "{}", "{}"], higher=[None] * 5),
        "t11": render("Table 11. Arm C, Top-1 F ×100 per dataset on every class and without the calibration sample "
                      "(the classes that chose the thresholds).", ["Classes"] + COLUMNS, sample_table(),
                      ["{:,}"] + [pct] * len(COLUMNS), higher=[None] + [True] * len(COLUMNS)),
    }
    units = R.units_table(plan, price, 12)
    charts = R.figures(results, split_thinking=False)
    notes = ["_Every class with a partner in the reference is scored (no held-out view), and the complete SNOMED CT → "
             "ICD-11 map is aligned. The first two rows are the first full run's, read from "
             "gonibeed/oa-rlaif-full-eval-results (unchanged). Arm A: pplx-decider-v1.2 alone, one candidate question "
             "per class (its pick first, then its probability order). Arm B: JevAgentHarness routes each model call of "
             "All merged + mini-SWE-agent to thinking on (p(hard) ≥ 0.75) or off, and the NONE/invalid guard "
             "(τ 0.5) re-checks NONE and invalid submissions; both thresholds were fixed before the run. Arm C: the same "
             "stack with thresholds fitted on a stratified calibration sample by the paper's rule (Table 10) and then "
             "frozen, plus the CASCADE (pplx-decider answers alone above its fitted τ). Arm C still aligns every class; "
             "Table 11 also shows it without the sample._",
             "_“–” = not run or not measured. Nothing is estimated._"]
    md = ["# Decision model: pplx-decider and JevAgentHarness", *notes]
    md += ["## SNOMED CT → ICD-11 and Cybersecurity", tables["t0"][0], tables["t00"][0]]
    md += [f"## {TOP_K_NAME}", tables["t1"][0], tables["t2"][0], tables["t3"][0]]
    if charts:
        md.append("Figures: " + ", ".join(f"`{c}`" for c in charts))
    md += ["## What the decision model did", tables["t4"][0], tables["t5"][0], tables["t6"][0]]
    md += ["## The complete SNOMED CT → ICD-11 map", tables["t7"][0]]
    md += ["## Arm C: the fitted thresholds", tables["t10"][0], tables["t11"][0]]
    md += ["## Sensitivity (no rerun)", tables["t8"][0]]
    md += ["## Units and cost", units]
    out = root()
    (out / "dm_report.md").write_text("\n\n".join(md) + "\n")
    (out / "dm_report.tex").write_text("\n\n".join(tables[k][1] for k in sorted(tables, key=lambda k: (int(k[1:]), k)))
                                       + "\n")
    report = {"rows": {tag: label(tag) for tag in rows()}, "tables": {k: v[0] for k, v in tables.items()},
              "figures": charts}
    (out / "dm_report.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def main() -> int:
    build()
    print((root() / "dm_report.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())

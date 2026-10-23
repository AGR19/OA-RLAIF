"""Tables of the decision-model run, beside the All-merged rows of the full evaluation.

    python -m dm_harness.report      -> outputs/full_eval/dm_report.md, dm_report.json, dm_report.tex, figures/

Rows (the user's scope: only All merged and the two new arms):
  All merged (one decision)               results/<task>/5-all-merged/                  } the first run, read from
  All merged + mini-SWE-agent, thinking on results/<task>/6-all-merged-agent-thinking/  } oa-rlaif-full-eval-results
  pplx-decider-v1.2 alone (arm A)         results/<task>/7-pplx-decider/
  JevAgentHarness (arm B)                 results/<task>/8-jah-all-merged-agent/
A unit that did not run shows "–"; nothing is estimated.

Focus   SNOMED CT -> ICD-11 and Cybersecurity: P, R, F at k = 1, 3, 5.
1, 2, 3 the full evaluation's Top-k tables (F@1 per dataset; P, R, F at k per dataset; per pair with the ceiling).
4       what each method did (answers, NONE, invalid, tokens, the agent's steps, the thinking, runtime).
5       the NONE/invalid guard: how often it acted per dataset and what came of it.
6       routing: the share of model calls the decision model sent to thinking on, p(hard), decision latency (the
        proxy's steps.jsonl; no prompt text is logged).
7       the complete SNOMED CT -> ICD-11 map.
8       sensitivity without a rerun: Top-1 F if pplx-decider answered a class alone whenever its probability was at
        least tau, else arm B's answer (the paper's CASCADE; fixed taus, read from the logged probabilities).
9       the units and the money.
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

ROWS = ["all_merged", "all_merged-agent-think", "decider", "jah-agent"]
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


def guard_table(results: dict) -> list:
    out = []
    for t in tasks():
        s = results.get("jah-agent", {}).get(t.slug)
        if not s:
            continue
        b = s["behaviour"]
        g = b.get("guard") or {}
        n = b["classes"]
        acted = sum(g.values())
        out.append((t.slug, [n, acted, g.get("kept_none", 0), g.get("redo_match", 0), g.get("redo_none", 0),
                             g.get("decider_pick", 0), g.get("decider_error", 0), b["none"] * 100, b["invalid"] * 100]))
    return out


def routing_table() -> list:
    """Per task: model calls through the proxy, the share sent to the strong tier (thinking on) and why, p(hard), the
    decision call's median latency. Sessions are '<model name>/<episode id>' (the guard's redo: '<id>-redo')."""
    log = root() / "jah_logs" / "steps.jsonl"
    if not log.exists():
        return []
    task_of = {}
    for path in (root() / "replies" / "jah-agent").glob("*/*.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                task_of[json.loads(line)["id"]] = path.parent.name
    per = defaultdict(lambda: {"calls": 0, "strong": 0, "reasons": Counter(), "p": [], "lat": []})
    with log.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "cache" or "tier" not in r:
                continue
            slug = task_of.get(str(r.get("session", "")).split("/", 1)[-1].removesuffix("-redo"))
            if slug is None:
                continue
            d = per[slug]
            d["calls"] += 1
            d["strong"] += r.get("tier") == "strong"
            d["reasons"][str(r.get("reason"))] += 1
            if r.get("p_hard") is not None:
                d["p"].append(float(r["p_hard"]))
            lat = (r.get("decision") or {}).get("latency_s")
            if lat is not None:
                d["lat"].append(float(lat))
    out = []
    for t in tasks():
        d = per.get(t.slug)
        if not d:
            continue
        n = max(1, d["calls"])
        out.append((t.slug, [d["calls"], d["strong"] / n * 100, d["reasons"].get("hard", 0) / n * 100,
                             d["reasons"].get("routine", 0) / n * 100,
                             (d["reasons"].get("pinned", 0)) / n * 100,
                             d["reasons"].get("decision_error", 0) / n * 100,
                             statistics.fmean(d["p"]) if d["p"] else None,
                             statistics.median(d["lat"]) if d["lat"] else None]))
    return out


def cascade_table() -> list:
    """Top-1 F per dataset column if pplx-decider's pick (arm A) replaced arm B's answer whenever its probability was
    >= tau. Uses the scored decisions of both arms (decisions.tsv.gz) and the arm A replies' probabilities."""
    from full_eval.score import PairScorer
    from full_eval.tasks import zero_shot_pairs
    from oa_rlaif.eval.metrics import AlignmentEvaluator

    zero = zero_shot_pairs()

    def column_of(t, pair_id: str) -> str:
        """The report column of a pair, as the full evaluation's tables place it (zero-shot pairs inside family tasks
        go to Zero-shot; Cybersecurity and Biodiv 2026 keep their own columns)."""
        return "Zero-shot" if pair_id in zero and t.family not in R.OWN_COLUMN else t.column

    def read_rows(tag: str, slug: str) -> dict[str, dict]:
        out = {}
        for path in sorted((root() / "replies" / tag / slug).glob("*.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    out[r["id"]] = r
        return out

    def partners_of(t, slug: str) -> dict[str, dict]:
        """pair id -> source -> reference partners (the Top-k evaluation's classes)."""
        if t.family == "icd10":
            gold = json.loads((root() / "episodes" / slug / "gold.json").read_text())
            return {t.pair_ids[0]: {k: set(v["targets"]) for k, v in gold.items() if v["targets"]}}
        if t.family == "snomed_icd11":
            from oa_rlaif.io import read_jsonl
            from oa_rlaif.paths import data_dir

            strong = defaultdict(set)
            for link in read_jsonl(data_dir() / "icd" / "triangulated.jsonl"):
                if link["leg_a"]["strong"] and link["leg_b"].startswith("strong"):
                    strong[link["snomed"]].add(link["icd11"])
            return {t.pair_ids[0]: dict(strong)}
        return {s.pair_id: PairScorer(s, oaei=False).partners() for s in t.specs}

    from full_eval.build_episodes import read_shard

    by_column: dict[str, dict[float, list]] = defaultdict(lambda: defaultdict(list))
    share: dict[float, list] = defaultdict(list)
    for t in tasks():
        a, b = read_rows("decider", t.slug), read_rows("jah-agent", t.slug)
        if not a or not b:
            continue
        episodes = {}
        for path in sorted((root() / "episodes" / t.slug).glob("*.jsonl.gz")):
            for e in read_shard(path):
                episodes[e["id"]] = e
        partners = partners_of(t, t.slug)
        for tau in TAUS:
            per_pair = defaultdict(dict)
            taken = 0
            for eid, rb in b.items():
                e, ra = episodes.get(eid), a.get(eid)
                if e is None:
                    continue
                use_a = ra is not None and ra.get("decision") in ("match", "NONE") and (ra.get("confidence") or 0) >= tau
                r = ra if use_a else rb
                taken += use_a
                answer = (r.get("targets") or [])[:1] if r.get("decision") == "match" else []
                per_pair[e["pair_id"]][e["source_iri"]] = answer
            share[tau].append((taken, len(b)))
            for pid, rk in per_pair.items():
                if partners.get(pid):
                    by_column[column_of(t, pid)][tau].append(
                        AlignmentEvaluator().topk(rk, partners[pid], ks=(1,))["k"][1]["F-measure"])
    out = []
    for tau in TAUS:
        taken = sum(x for x, _ in share[tau])
        total = sum(n for _, n in share[tau]) or 1
        cells = [statistics.fmean(by_column[c][tau]) * 100 if by_column[c][tau] else None for c in COLUMNS]
        out.append((f"τ = {tau:g}", [taken / total * 100] + cells))
    return out


def behaviour_rows(results: dict, price: float | None) -> list:
    """Table 4's rows. The first run's rows keep their own runtime (4 x B300), but their cost is not this machine's
    price times that runtime, so it is left out ("–")."""
    first_run = {label(tag) for tag in ("all_merged", "all_merged-agent-think")}
    return [(name, values[:-1] + [None] if name in first_run else values)
            for name, values in R.table_behaviour(results, price)]


def build() -> dict:
    R.row_tags = rows  # the full evaluation's table functions, with this report's four rows
    results = R.collect()
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
        "t5": render("Table 5. The NONE/invalid guard of arm B per task (classes): it acts on a NONE or an invalid "
                     "submission; kept NONE = pplx-decider's best candidate below tau 0.5; redo = Gemma again on the "
                     "strong tier (with the hint when p ≥ 0.5); decider's pick = the redo was invalid too.",
                     ["Classes", "Guard acted", "Kept NONE", "Redo → match", "Redo → NONE", "Decider's pick",
                      "Decider error", "NONE % (final)", "Invalid % (final)"], guard_table(results),
                     ["{:,}"] * 7 + [pct, pct], higher=[None] * 9),
        "t6": render("Table 6. Routing by pplx-decider in arm B (the proxy's step log): model calls, share on the strong "
                     "tier (thinking on), why (hard: p(hard) ≥ 0.75; routine; pinned: the guard's redo on jah-strong; "
                     "decision error → strong), mean p(hard), median decision latency.",
                     ["Model calls", "Strong %", "Hard %", "Routine %", "Pinned (redo) %", "Decision error %",
                      "Mean p(hard)", "Decision latency s"], routing_table(),
                     ["{:,}", pct, pct, pct, pct, pct, "{:.3f}", "{:.3f}"], higher=[None] * 8),
        "t7": R.snomed_table(results, 7),
        "t8": render("Table 8. Without a rerun: Top-1 F ×100 if pplx-decider answered a class alone whenever its "
                     "probability was ≥ τ, and arm B's answer stood otherwise (the paper's CASCADE). Share = classes "
                     "pplx-decider would answer. Fixed τ values, not fitted.", ["Share %"] + COLUMNS,
                     cascade_table(), pct, higher=[None] + [True] * len(COLUMNS)),
    }
    units = R.units_table(plan, price, 9)
    charts = R.figures(results, split_thinking=False)
    notes = ["_Every class with a partner in the reference is scored (no held-out view), and the complete SNOMED CT → "
             "ICD-11 map is aligned. The first two rows are the first full run's, read from "
             "gonibeed/oa-rlaif-full-eval-results (unchanged). Arm A: pplx-decider-v1.2 alone, one candidate question "
             "per class (its pick first, then its probability order). Arm B: JevAgentHarness routes each model call of "
             "All merged + mini-SWE-agent to thinking on (p(hard) ≥ 0.75) or off, and the NONE/invalid guard "
             "(τ 0.5) re-checks NONE and invalid submissions. Both thresholds were fixed before the run._",
             "_“–” = not run or not measured. Nothing is estimated._"]
    md = ["# Decision model: pplx-decider and JevAgentHarness", *notes]
    md += ["## SNOMED CT → ICD-11 and Cybersecurity", tables["t0"][0], tables["t00"][0]]
    md += [f"## {TOP_K_NAME}", tables["t1"][0], tables["t2"][0], tables["t3"][0]]
    if charts:
        md.append("Figures: " + ", ".join(f"`{c}`" for c in charts))
    md += ["## What the decision model did", tables["t4"][0], tables["t5"][0], tables["t6"][0]]
    md += ["## The complete SNOMED CT → ICD-11 map", tables["t7"][0]]
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

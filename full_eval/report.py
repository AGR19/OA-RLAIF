"""Tables of the full-alignment evaluation: full alignments only (every class aligned, whole references, no held-out
view).

    python -m full_eval.report        -> outputs/full_eval/report.md, report.tex, report.json, figures/

Rows: the six methods (base, SFT, OPSD, OPSD -> GRPO, All merged, All merged + mini-SWE-agent), each with thinking off
and on. A unit that did not run shows "–"; nothing is estimated or filled in.
Part 1  OAEI scores of the whole alignment.     Table 1 F-measure per dataset, ICD-10 -> ICD-11 accuracy, SNOMED CT ->
        ICD-11 agreement; Table 2 the OAEI rows per pair; Table 3 Anatomy beside the published systems; Table 4 ICD-10
        -> ICD-11 in detail.
Part 2  The Top-k evaluation (k = 1, 3, 5).     Table 5 F-measure@1 per dataset; Table 6 Precision, Recall and
        F-measure at k = 1, 3, 5 per dataset; figures/topk_<task>_<thinking>.svg.
Part 3  What the mini-SWE-agent adds to All merged.   Table 7 the differences; Table 8 what the agent did and cost.
Part 4  Thinking on against thinking off.        Table 9 the differences; Table 10 what the thinking did.
Table 11 the complete SNOMED CT -> ICD-11 map; Table 12 the units (method x task): what ran, or why not, and the cost.
"""

from __future__ import annotations

import json
import statistics
import sys

from oa_rlaif.eval.metrics import TOP_K, TOP_K_NAME
from oa_rlaif.eval.report import render

from .schedule import AGENT, REPORT_ORDER, label, method, root
from .tasks import COLUMNS, tasks, zero_shot_pairs

THINKING = (False, True)
PUBLISHED_ANATOMY = [  # whole alignments of systems not trained on the reference
    # OAEI 2026 Anatomy track, preliminary results (oaei.ontologymatching.org/2026/results/anatomy)
    ("DRAMA (OAEI 2026)", 0.945, 0.951, 0.957, 0.902), ("Matcha (OAEI 2026)", 0.951, 0.941, 0.931, 0.820),
    ("LogMapBioLLM (OAEI 2026)", 0.936, 0.932, 0.929, 0.813), ("Agent-OM (OAEI 2026)", 0.979, 0.924, 0.875, 0.674),
    ("LogMap (OAEI 2026)", 0.917, 0.881, 0.848, 0.602), ("StringEquiv baseline", 0.997, 0.766, 0.622, 0.000),
    ("DRAL-OA (OAEI 2025, official)", 0.83, 0.828, 0.827, None)]


def tag_of(base: str, think: bool) -> str:
    return base + ("-think" if think else "")


def row_tags() -> list[str]:
    return [tag_of(b, t) for t in THINKING for b in REPORT_ORDER]


def collect() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for path in (root() / "results").glob("*/*/scores.json"):
        s = json.loads(path.read_text())
        out.setdefault(s["model"], {})[s["task"]] = s
    return out


def mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


# ---- one cell per (method, dataset column) ------------------------------------------------------------------------------
def column_tasks(column: str) -> list[str]:
    """The tasks a column needs. Zero-shot: the never-trained pairs inside the family tasks and the zero-shot task
    (Hydrography cree -> SWO); IoTSec -> UCO, also never trained on, has its own Cybersecurity column."""
    if column == "Zero-shot":
        zero = zero_shot_pairs()
        return [t.slug for t in tasks() if t.family != "cyber" and (t.column == "Zero-shot"
                                                                     or any(s.pair_id in zero for s in t.specs))]
    return [t.slug for t in tasks() if t.column == column]


def column_pairs(per: dict, column: str) -> list[dict] | None:
    """The pair results that make a column, or None while one of its tasks has not run."""
    slugs = column_tasks(column)
    if not slugs or any(s not in per for s in slugs):
        return None
    if column == "Zero-shot":
        return [p for s in slugs for p in per[s]["pairs"].values() if p.get("zero_shot")]
    if column == "Cybersecurity":
        return [p for s in slugs for p in per[s]["pairs"].values()]
    return [p for s in slugs for p in per[s]["pairs"].values() if not p.get("zero_shot")]


def topk_cell(p: dict, k: int, measure: str) -> float | None:
    t = p.get("topk") or {}
    if not t.get("classes_scored"):
        return None
    return t["k"][str(k) if str(k) in t["k"] else k][measure]


def oaei_value(per: dict, column: str) -> float | None:
    if column == "ICD-10→11":
        s = per.get("icd__who-icd10-to-icd11")
        return s["accuracy"] * 100 if s else None
    if column == "SNOMED→ICD-11":
        s = per.get("icd__snomed-to-icd11")
        return s["silver"]["strong_links"]["agreement"] * 100 if s else None
    pairs = column_pairs(per, column)
    value = mean([p["oaei"]["F-Measure"] for p in pairs]) if pairs else None
    return None if value is None else value * 100


def topk_value(per: dict, column: str, k: int = 1, measure: str = "F-measure") -> float | None:
    if column == "ICD-10→11":
        pairs = list(per["icd__who-icd10-to-icd11"]["pairs"].values()) if "icd__who-icd10-to-icd11" in per else None
    elif column == "SNOMED→ICD-11":
        pairs = list(per["icd__snomed-to-icd11"]["pairs"].values()) if "icd__snomed-to-icd11" in per else None
    else:
        pairs = column_pairs(per, column)
    value = mean([topk_cell(p, k, measure) for p in pairs]) if pairs else None
    return None if value is None else value * 100


def with_average(values: list) -> list:
    present = [v for v in values if v is not None]
    return values + [statistics.fmean(present) if present and len(present) == len(values) else None]


# ---- Part 1: OAEI ----------------------------------------------------------------------------------------------------
def table1(results: dict) -> list:
    return [(label(tag), with_average([oaei_value(results.get(tag, {}), c) for c in COLUMNS])) for tag in row_tags()]


def table2(results: dict) -> list:
    rows = []
    for t in tasks():
        if t.family in ("icd10", "snomed_icd11"):
            continue
        for tag in row_tags():
            s = results.get(tag, {}).get(t.slug)
            if not s:
                continue
            if len(s["pairs"]) > 3:  # MultiFarm (1,125 pairs), DH-ML: pooled; every pair is in scores.json
                pooled = [p["oaei"] for p in s["pairs"].values()]
                tp, size, ref = (sum(p[k] for p in pooled) for k in ("true_positive", "Size", "reference"))
                p_, r_ = (tp / size if size else 0.0), (tp / ref if ref else 0.0)
                rows.append((f"{t.slug} (pooled over {len(pooled)} pairs) · {label(tag)}",
                             [s.get("runtime_s"), size, p_, 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0, r_,
                              mean([p["Recall+"] for p in pooled]), "N/A"]))
                continue
            for pair_id, p in sorted(s["pairs"].items()):
                f = p["oaei"]
                rows.append((f"{pair_id} · {label(tag)}", [s.get("runtime_s"), f["Size"], f["Precision"], f["F-Measure"],
                                                         f["Recall"], f["Recall+"], "N/A"]))
    return rows


def table3(results: dict) -> list:
    rows = []
    for tag in row_tags():
        s = results.get(tag, {}).get("anatomy__mouse-human")
        if s:
            f = next(iter(s["pairs"].values()))["oaei"]
            rows.append((label(tag), [f["Precision"], f["F-Measure"], f["Recall"], f["Recall+"]]))
    rows += [(name, [p, f, r, rp]) for name, p, f, r, rp in PUBLISHED_ANATOMY]
    return rows


def table4(results: dict) -> list:
    rows = []
    for tag in row_tags():
        s = results.get(tag, {}).get("icd__who-icd10-to-icd11")
        if s:
            p = next(iter(s["pairs"].values()))
            f = p["oaei"]
            rows.append((label(tag), [p["codes"], p["accuracy"] * 100, p["retrieval_ceiling"] * 100, f["Precision"],
                                      f["Recall"], f["F-Measure"], topk_cell(p, 1, "F-measure")]))
    return rows


# ---- Part 2: Top-k ---------------------------------------------------------------------------------------------------
def table5(results: dict) -> list:
    return [(label(tag), with_average([topk_value(results.get(tag, {}), c) for c in COLUMNS])) for tag in row_tags()]


def table6(results: dict) -> list:
    rows = []
    for column in COLUMNS:
        for tag in row_tags():
            per = results.get(tag, {})
            values = [topk_value(per, column, k, m) for k in TOP_K for m in ("Precision", "Recall", "F-measure")]
            if all(v is None for v in values):
                continue
            if column in ("ICD-10→11", "SNOMED→ICD-11"):
                slug = "icd__who-icd10-to-icd11" if column == "ICD-10→11" else "icd__snomed-to-icd11"
                pairs = list(per[slug]["pairs"].values())
            else:
                pairs = column_pairs(per, column) or []
            scored = sum((p.get("topk") or {}).get("classes_scored", 0) for p in pairs)
            rows.append((f"{column} · {label(tag)}", values + [scored]))
    return rows


def figures(results: dict) -> list[str]:
    """One chart per dataset column and thinking setting: Precision, Recall and F-measure vs. Top-k, one line per
    method."""
    from oa_rlaif.eval.topk_chart import write_topk_chart

    written = []
    for column in COLUMNS:
        for think in THINKING:
            series = {}
            for base in REPORT_ORDER:
                per = results.get(tag_of(base, think), {})
                cells = {k: {m: topk_value(per, column, k, m) for m in ("Precision", "Recall", "F-measure")} for k in TOP_K}
                if all(v is not None for c in cells.values() for v in c.values()):
                    series[label(base)] = {k: {m: v / 100 for m, v in c.items()} for k, c in cells.items()}
            if series:
                slug = column.lower().replace("→", "-to-").replace(" ", "-")
                name = f"figures/topk_{slug}_thinking-{'on' if think else 'off'}.svg"
                write_topk_chart(root() / name, series, f"{column}, full alignment, thinking {'on' if think else 'off'}: "
                                                        f"{TOP_K_NAME}")
                written.append(name)
    return written


# ---- Part 3: the agent -----------------------------------------------------------------------------------------------
def table7(results: dict) -> list:
    rows = []
    for name, value in (("OAEI F-measure", oaei_value), ("Top-k F-measure@1", topk_value)):
        for think in THINKING:
            agent, single = results.get(tag_of(AGENT, think), {}), results.get(tag_of("all_merged", think), {})
            diffs = []
            for c in COLUMNS:
                a, b = value(agent, c), value(single, c)
                diffs.append(None if a is None or b is None else a - b)
            rows.append((f"{name}, thinking {'on' if think else 'off'}", with_average(diffs)))
    return rows


def table8(results: dict, price: float | None) -> list:
    rows = []
    for t in tasks():
        for think in THINKING:
            for base in ("all_merged", AGENT):
                s = results.get(tag_of(base, think), {}).get(t.slug)
                if not s:
                    continue
                b = s["behaviour"]
                agent = base == AGENT
                runtime = s.get("runtime_s")
                rows.append((f"{t.slug} · {label(tag_of(base, think))}",
                             [b["classes"], b.get("model_calls_per_class") if agent else 1.0,
                              b.get("extra_commands_per_class") if agent else 0.0,
                              (b.get("classes_with_extra_commands") or 0) * 100 if agent else 0.0,
                              b.get("format_errors_per_class") if agent else None, b["invalid"] * 100,
                              b["prompt_tokens_per_class"], b["completion_tokens_per_class"], runtime,
                              runtime * price / 3600 if runtime is not None and price else None]))
    return rows


# ---- Part 4: thinking ------------------------------------------------------------------------------------------------
def table9(results: dict) -> list:
    rows = []
    for name, value in (("OAEI F", oaei_value), ("Top-1 F", topk_value)):
        for base in REPORT_ORDER:
            on, off = results.get(tag_of(base, True), {}), results.get(tag_of(base, False), {})
            diffs = []
            for c in COLUMNS:
                a, b = value(on, c), value(off, c)
                diffs.append(None if a is None or b is None else a - b)
            rows.append((f"{label(base)} · {name}", with_average(diffs)))
    return rows


def pooled(results: dict, tag: str, key: str, scale: float = 1.0) -> float | None:
    """A behaviour share or per-class mean over every task the method ran, weighted by classes."""
    units = [s["behaviour"] for s in results.get(tag, {}).values() if s["behaviour"].get(key) is not None]
    total = sum(b["classes"] for b in units)
    return sum(b[key] * b["classes"] for b in units) / total * scale if total else None


def table10(results: dict) -> list:
    rows = []
    for base in REPORT_ORDER:
        on, off = tag_of(base, True), tag_of(base, False)
        if on not in results and off not in results:
            continue
        hours = lambda tag: sum(s.get("runtime_s") or 0 for s in results.get(tag, {}).values()) / 3600 or None  # noqa: E731
        rows.append((label(base), [sum(s["behaviour"]["classes"] for s in results.get(on, {}).values()) or None,
                                   pooled(results, on, "answer_in_thinking", 100), pooled(results, on, "no_thinking", 100),
                                   pooled(results, on, "truncated", 100), pooled(results, on, "thinking_chars_per_class"),
                                   pooled(results, off, "completion_tokens_per_class"),
                                   pooled(results, on, "completion_tokens_per_class"),
                                   pooled(results, off, "invalid", 100), pooled(results, on, "invalid", 100),
                                   hours(off), hours(on)]))
    return rows


# ---- SNOMED map and units ----------------------------------------------------------------------------------------------
def table11(results: dict) -> list:
    rows = []
    for tag in row_tags():
        s = results.get(tag, {}).get("icd__snomed-to-icd11")
        if not s:
            continue
        a = s["silver"]
        rows.append((label(tag), [a["concepts"], a["mapped"] * 100, a["none"] * 100, a["invalid"] * 100,
                                  a["strong_links"]["agreement"] * 100, a["strong_links"]["retrieval_ceiling"] * 100,
                                  a["any_composed_link"]["agreement"] * 100, topk_cell(a, 1, "F-measure")]))
    return rows


def plain_table(title: str, header: list[str], rows: list[list]) -> str:
    lines = [f"**{title}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join("–" if v is None else str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def build() -> dict:
    results = collect()
    plan = json.loads((root() / "plan.json").read_text()) if (root() / "plan.json").exists() else {"units": []}
    price = plan.get("price_usd_per_hour")
    verify = {p.stem.split("_", 1)[1]: json.loads(p.read_text()) for p in root().glob("verify_*.json")}
    header = COLUMNS + ["Average"]
    pct = "{:.1f}"
    signed = "{:+.1f}"
    tables = {
        "t1": render("Table 1. Full alignment, OAEI scores: F-measure ×100 of the whole alignment against the whole "
                     "reference (mean over pairs); ICD-10→11: accuracy ×100 over every WHO code; SNOMED→ICD-11: agreement "
                     "×100 with the strong ICD-10-composed links (silver, no reference exists).", header,
                     table1(results), pct),
        "t2": render("Table 2. OAEI rows per pair (MultiFarm and DH-ML pooled). Size = mappings in alignment.rdf; Runtime "
                     "= inference seconds on all GPUs; Coherent not computed. Bio-ML: public references only, so "
                     "precision is a lower bound.",
                     ["Runtime (s)", "Size", "Precision", "F-Measure", "Recall", "Recall+", "Coherent"], table2(results),
                     ["{}", "{:,}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.3f}", "{}"],
                     higher=[None, None, True, True, True, True, None]),
        "t3": render("Table 3. Anatomy (mouse → human), whole alignment, beside the published systems. The published "
                     "systems were not trained on the reference; our models were trained on part of it.",
                     ["Precision", "F-Measure", "Recall", "Recall+"], table3(results), "{:.3f}"),
        "t4": render("Table 4. ICD-10 → ICD-11 against the WHO tables, every code. Accuracy as in the earlier Table 1; "
                     "retrieval ceiling = codes whose answer is among the candidates.",
                     ["Codes", "Accuracy %", "Retrieval ceiling %", "Precision", "Recall", "F-Measure", "Top-1 F"],
                     table4(results), ["{:,}", pct, pct, "{:.3f}", "{:.3f}", "{:.3f}", "{:.3f}"],
                     higher=[None, True, None, True, True, True, True]),
        "t5": render(f"Table 5. {TOP_K_NAME}, full alignment: F-measure@1 ×100 per dataset (mean over pairs; only the "
                     "classes that have a partner in the reference are scored; SNOMED→ICD-11 against the strong composed "
                     "links, silver).", header, table5(results), pct),
        "t6": render(f"Table 6. {TOP_K_NAME} per dataset: Precision@k, Recall@k and F-measure@k ×100.",
                     [f"{m[0]}@{k}" for k in TOP_K for m in ("Precision", "Recall", "F-measure")] + ["Classes scored"],
                     table6(results), [pct] * 9 + ["{:,}"], higher=[True] * 9 + [None]),
        "t7": render("Table 7. What the mini-SWE-agent adds: All merged + mini-SWE-agent minus All merged (same "
                     "episodes, same thinking setting), ×100 points.", header, table7(results), signed,
                     higher=[None] * len(header)),
        "t8": render("Table 8. What the agent did and cost, per task (All merged answers in one call). Extra commands = "
                     "commands the model chose beyond the scripted om profile / om show / om candidates and its submit.",
                     ["Classes", "Model calls/class", "Extra commands/class", "With extra commands %",
                      "Format errors/class", "Invalid or no submission %", "Prompt tokens/class",
                      "Completion tokens/class", "Runtime (s)", "Cost $"], table8(results, price),
                     ["{:,}", "{:.2f}", "{:.2f}", pct, "{:.2f}", pct, "{:,.0f}", "{:,.0f}", "{:,}", "{:.2f}"],
                     higher=[None] * 10),
        "t9": render("Table 9. Thinking on minus thinking off, ×100 points (the models were trained with thinking off).",
                     header, table9(results), signed, higher=[None] * len(header)),
        "t10": render("Table 10. What the thinking did (every task the method ran, weighted by classes).",
                      ["Classes (on)", "Answer inside thinking %", "No thinking text %", "Cut at token limit %",
                       "Thinking characters/class", "Completion tokens/class (off)", "Completion tokens/class (on)",
                       "Invalid % (off)", "Invalid % (on)", "Runtime h (off)", "Runtime h (on)"], table10(results),
                      ["{:,}", pct, pct, pct, "{:,.0f}", "{:,.0f}", "{:,.0f}", pct, pct, "{:.2f}", "{:.2f}"],
                      higher=[None] * 11),
        "t11": render("Table 11. The complete SNOMED CT → ICD-11 map (every active concept in ICD's scope). Agreement = "
                      "the model's ICD-11 category is one of the concept's ICD-10-composed links (silver, not gold); "
                      "retrieval misses count as failures.",
                      ["Concepts", "Mapped %", "NONE %", "Invalid %", "Agree (strong) %", "Retrieval ceiling %",
                       "Agree (any link) %", "Top-1 F (strong)"], table11(results),
                      ["{:,}", pct, pct, pct, pct, pct, pct, "{:.3f}"], higher=[None, None, None, False, True, None, True, True]),
    }
    units = [[u["model"], u["task"], f"{u['decisions']:,}", u.get("status", ""), u.get("seconds", ""), u.get("usd", "")]
             for u in plan.get("units", [])]
    t12 = plain_table(f"Table 12. Units (method × task): budget ${plan.get('budget_usd', '?')}, spent "
                      f"${plan.get('spent_usd', '?')} at ${price or '?'}/h.",
                      ["Method", "Task", "Decisions", "Status", "Seconds", "Projected $"], units)
    charts = figures(results)
    notes = ["_Full alignments only: every class of every source ontology is aligned, and every score uses the whole "
             "reference. There is no held-out view. The models were trained on part of these references (the 80/20 "
             "split of the earlier run), which OAEI rules do not allow for an official entry; the zero-shot pairs and "
             "IoTSec → UCO were never trained on._",
             "_“–” = not run or not measured. Nothing is estimated. Coherence (OAEI's Pellet check) was not computed. "
             "IoTSec → UCO: the reference has only 'related' links, so all its links count. Bio-ML: its test references "
             "are kept by the organisers; the alignment.rdf files are submission-ready._",
             "_The agent is never given a reference: its task folder holds the public episode only, and the references "
             "are read from the raw folders for scoring only._"]
    if verify:
        notes.append("_Weight check of the rebuilt models (not an evaluation): same decision as the earlier run's "
                     "replies: " + "; ".join(f"{k}: {v['same_decision']} of {v['checked']}" if v.get("checked")
                                             else f"{k}: not done ({v.get('note', 'no replies to compare')})"
                                             for k, v in sorted(verify.items())) + "._")
    parts = [("## Part 1. Full alignment: OAEI scores", ["t1", "t2", "t3", "t4"]),
             (f"## Part 2. Full alignment: {TOP_K_NAME}", ["t5", "t6"]),
             ("## Part 3. What the mini-SWE-agent adds to All merged", ["t7", "t8"]),
             ("## Part 4. Thinking on against thinking off", ["t9", "t10"]),
             ("## The complete SNOMED CT → ICD-11 map", ["t11"])]
    md = ["# Full-alignment evaluation", *notes]
    for title, keys in parts:
        md.append(title)
        md += [tables[k][0] for k in keys]
        if keys == ["t5", "t6"] and charts:
            md.append("Figures (Precision, Recall and F-measure vs. Top-k): " + ", ".join(f"`{c}`" for c in charts))
    md += ["## Units and cost", t12]
    out = root()
    (out / "report.md").write_text("\n\n".join(md) + "\n")
    (out / "report.tex").write_text("\n\n".join(tables[k][1] for k in sorted(tables, key=lambda k: int(k[1:]))) + "\n")
    report = {"view": "full alignment", "tables": {k: v[0] for k, v in tables.items()}, "figures": charts,
              "methods": {tag: {"label": label(tag), "thinking": method(tag).think, "agent": method(tag).agent}
                          for tag in row_tags()}, "weight_check": verify}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def main() -> int:
    build()
    print((root() / "report.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())

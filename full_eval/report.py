"""Tables of the full-alignment evaluation, in the layout of the earlier report (outputs/eval/report.md).

    python -m full_eval.report        -> outputs/full_eval/report.md, report.tex, report.json

Main scoring (the user's choice): DRAL-OA clf.py's regime, top-1 over the classes that have a reference partner.
Table 1   Precision@1 x100 (= top-1 hit rate) per dataset family (mean over pairs; zero-shot pairs only in the
          Zero-shot column), ICD-10 -> ICD-11 and SNOMED CT -> ICD-11 in the same regime.
Table 1b  the same over held-out classes only (training links left out): the numbers free of training data.
Table 2   OAEI layout (Runtime, Size, Precision@1, F, Recall@1, Recall+@1, classes scored) per pair and model.
Table 3   OAEI's set-based leaderboard measure (whole alignment.rdf vs whole reference), F x100, with the variants.
Table 4   set-based OAEI rows per pair; then the OAEI 2026 Anatomy rows for reference.
Table 5   the complete SNOMED CT -> ICD-11 map per model. Table 6: what ran, or why not.
A unit that did not run shows "–"; nothing is estimated or filled in.
"""

from __future__ import annotations

import json
import sys

from oa_rlaif.eval.report import render

from .schedule import LABELS, REPORT_ORDER, root
from .tasks import COLUMNS, tasks
from .variants import VARIANTS

PUBLISHED_ANATOMY = [  # OAEI 2026 Anatomy track, preliminary results (oaei.ontologymatching.org/2026/results/anatomy)
    ("DRAMA (OAEI 2026)", 0.945, 0.951, 0.957, 0.902), ("Matcha (OAEI 2026)", 0.951, 0.941, 0.931, 0.820),
    ("LogMapBioLLM (OAEI 2026)", 0.936, 0.932, 0.929, 0.813), ("Agent-OM (OAEI 2026)", 0.979, 0.924, 0.875, 0.674),
    ("LogMap (OAEI 2026)", 0.917, 0.881, 0.848, 0.602), ("StringEquiv baseline", 0.997, 0.766, 0.622, 0.000)]


def collect() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for path in (root() / "results").glob("*/*/scores.json"):
        s = json.loads(path.read_text())
        out.setdefault(s["model"], {})[s["task"]] = s
    return out


def family_f(s: dict | None, view: str) -> float | None:
    if not s or not s.get("aggregate", {}).get(view):
        return None
    return s["aggregate"][view]["macro"]["F-Measure"] * 100


def zero_shot_f(per_task: dict[str, dict], view: str) -> float | None:
    rows = [p[view]["F-Measure"] for s in per_task.values() if s.get("family") not in ("icd10", "snomed_icd11")
            for p in s.get("pairs", {}).values() if p.get("zero_shot")]
    return sum(rows) / len(rows) * 100 if rows else None


def bioml_f(per_task: dict[str, dict], view: str) -> tuple[float | None, int]:
    rows = [family_f(s, view) for slug, s in per_task.items() if slug.startswith("bioml__")]
    rows = [r for r in rows if r is not None]
    return (sum(rows) / len(rows) if rows else None), len(rows)


def row_models(results: dict) -> list[str]:
    """The five models (a row even when not run), then the variants that were built."""
    return REPORT_ORDER + [v for v in VARIANTS if v in results]


def top1_cell(s: dict | None, view: str) -> float | None:
    agg = (s or {}).get("top1", {}).get(view) or {}
    if "macro" in agg:
        return agg["macro"]["Precision"] * 100
    return agg["Precision"] * 100 if "Precision" in agg else None


def top1_zero_shot(per_task: dict[str, dict]) -> float | None:
    rows = [p["top1"]["full"]["Precision"] for s in per_task.values() for p in s.get("pairs", {}).values()
            if isinstance(p, dict) and p.get("zero_shot") and "top1" in p]
    return sum(rows) / len(rows) * 100 if rows else None


def table_top1(results: dict, view: str) -> list[tuple[str, list]]:
    rows = []
    for model in REPORT_ORDER:
        per = results.get(model, {})
        bio = [top1_cell(s, view) for slug, s in per.items() if slug.startswith("bioml__")]
        bio = [b for b in bio if b is not None]
        values = [top1_cell(per.get("anatomy__mouse-human"), view), sum(bio) / 3 if len(bio) == 3 else None,
                  top1_cell(per.get("multifarm"), view), top1_cell(per.get("dh"), view),
                  top1_cell(per.get("dh_multiling"), view), top1_cell(per.get("biodiv"), view),
                  top1_cell(per.get("icd__who-icd10-to-icd11"), view), top1_cell(per.get("icd__snomed-to-icd11"), view),
                  top1_zero_shot(per)]
        present = [v for v in values if v is not None]
        rows.append((LABELS[model], values + [sum(present) / len(present) if len(present) == len(values) else None]))
    return rows


def table_top1_pairs(results: dict) -> list[tuple[str, list]]:
    rows = []
    for t in tasks():
        for model in REPORT_ORDER:
            s = results.get(model, {}).get(t.slug)
            if not s:
                continue
            if t.family in ("icd10", "snomed_icd11"):
                f = s["top1"]["full"]
                rows.append((f"{t.pair_ids[0]} · {LABELS[model]}", [s.get("runtime_s"), f["Size"], f["Precision"],
                             f["F-Measure"], f["Recall"], f["Recall+"], f["classes_scored"]]))
            elif len(s["pairs"]) > 3:
                a = s["top1"]["full"]
                rows.append((f"{t.slug} (mean over {a['pairs']} pairs) · {LABELS[model]}",
                             [s.get("runtime_s"), sum(p["top1"]["full"]["Size"] for p in s["pairs"].values()),
                              a["macro"]["Precision"], a["macro"]["F-Measure"], a["macro"]["Recall"], None, a["classes_scored"]]))
            else:
                for pair_id, p in sorted(s["pairs"].items()):
                    f = p["top1"]["full"]
                    rows.append((f"{pair_id} · {LABELS[model]}", [s.get("runtime_s"), f["Size"], f["Precision"],
                                 f["F-Measure"], f["Recall"], f["Recall+"], f["classes_scored"]]))
    return rows


def table1(results: dict, view: str) -> list[tuple[str, list]]:
    rows = []
    for model in row_models(results):
        per = results.get(model, {})
        bio, n_bio = bioml_f(per, view)
        icd10 = per.get("icd__who-icd10-to-icd11", {}).get("aggregate", {}).get(view, {}).get("accuracy")
        snomed = per.get("icd__snomed-to-icd11", {}).get("aggregate", {})
        snomed = snomed.get("strong_links" if view == "full" else "strong_links_heldout", {}).get("agreement")
        values = [family_f(per.get("anatomy__mouse-human"), view), bio if n_bio == 3 else None,
                  family_f(per.get("multifarm"), view), family_f(per.get("dh"), view),
                  family_f(per.get("dh_multiling"), view), family_f(per.get("biodiv"), view),
                  None if icd10 is None else icd10 * 100, None if snomed is None else snomed * 100,
                  zero_shot_f(per, view)]
        present = [v for v in values if v is not None]
        rows.append((LABELS[model], values + [sum(present) / len(present) if len(present) == len(values) else None]))
    return rows


def table2(results: dict) -> list[tuple[str, list]]:
    rows = []
    for t in tasks():
        if t.family in ("icd10", "snomed_icd11"):
            continue
        for model in row_models(results):
            s = results.get(model, {}).get(t.slug)
            if not s:
                continue
            for pair_id, p in sorted(s["pairs"].items()):
                if len(s["pairs"]) > 3:  # MultiFarm (1,125 pairs), DH-ML: the pooled row; pairs are in scores.json
                    continue
                f = p["full"]
                rows.append((f"{pair_id} · {LABELS[model]}", [s.get("runtime_s"), f["Size"], f["Precision"],
                                                             f["F-Measure"], f["Recall"], f["Recall+"], "N/A"]))
            if len(s["pairs"]) > 3:
                a = s["aggregate"]["full"]
                rows.append((f"{t.slug} (micro over {a['pairs']} pairs) · {LABELS[model]}",
                             [s.get("runtime_s"), sum(p["full"]["Size"] for p in s["pairs"].values()),
                              a["micro"]["Precision"], a["micro"]["F-Measure"], a["micro"]["Recall"],
                              a["macro_recall_plus"], "N/A"]))
    return rows


def table3(results: dict) -> list[tuple[str, list]]:
    rows = []
    for model in REPORT_ORDER:
        s = results.get(model, {}).get("icd__snomed-to-icd11")
        if not s:
            continue
        a = s["aggregate"]
        rows.append((LABELS[model], [a["concepts"], a["mapped"] * 100, a["none"] * 100, a["invalid"] * 100,
                                     a["strong_links"]["agreement"] * 100, a["strong_links"]["retrieval_ceiling"] * 100,
                                     a["strong_links_heldout"]["agreement"] * 100, a["any_composed_link"]["agreement"] * 100]))
    return rows


def plain_table(title: str, header: list[str], rows: list[list]) -> str:
    lines = [f"**{title}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join("–" if v is None else str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def build() -> dict:
    results = collect()
    plan = json.loads((root() / "plan.json").read_text()) if (root() / "plan.json").exists() else {"units": []}
    verify = {p.stem.split("_", 1)[1]: json.loads(p.read_text()) for p in root().glob("verify_*.json")}
    header1 = COLUMNS + ["Average"]
    regime = ("DRAL-OA clf.py regime: only classes that have a partner in the reference are scored; the model's own top-1 "
              "choice is a hit if it is a partner")
    t1_md, t1_tex = render(f"Table 1. Precision@1 ×100 (= top-1 hit rate) per dataset, mean over pairs. {regime}. All "
                           "reference classes, including the links the models were trained on.", header1,
                           table_top1(results, "full"), "{:.1f}")
    t1b_md, t1b_tex = render("Table 1b. The same over held-out classes only (classes whose links were not in training; ICD: "
                             "held-out ICD-10 categories). Free of training data.", header1, table_top1(results, "heldout"),
                             "{:.1f}")
    t2_md, t2_tex = render("Table 2. OAEI layout, clf.py top-1 regime. Size = scored classes the model answered; Recall+@1 = "
                           "hit rate on classes whose reference links are non-trivial (names differ).",
                           ["Runtime (s)", "Size", "Precision@1", "F-Measure", "Recall@1", "Recall+@1", "Classes scored"],
                           table_top1_pairs(results), ["{}", "{:,}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.3f}", "{:,}"],
                           higher=[None, None, True, True, True, True, None])
    t3_md, t3_tex = render("Table 3. OAEI's set-based leaderboard measure: the whole alignment.rdf (every class, matches ≥ 0.5, "
                           "1:1) against the whole reference, F-measure ×100. Classes without a counterpart count here: a "
                           "match proposed for one is wrong. Includes the two matcher variants.", header1,
                           table1(results, "full"), "{:.1f}")
    t4_md, t4_tex = render("Table 4. Set-based OAEI rows per pair (MultiFarm and DH-ML pooled). Runtime = inference seconds "
                           "on 4 GPUs. Coherent not computed. Bio-ML: public references only, so precision is a lower bound.",
                           ["Runtime (s)", "Size", "Precision", "F-Measure", "Recall", "Recall+", "Coherent"],
                           table2(results), ["{}", "{}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.3f}", "{}"],
                           higher=[None, None, True, True, True, True, None])
    t5_md, t5_tex = render("Table 5. Complete SNOMED CT→ICD-11 map (every active concept in ICD's scope). Agreement = the "
                           "model's ICD-11 category is one of the concept's ICD-10-composed links (a silver standard, not "
                           "gold); retrieval misses count as failures.",
                           ["Concepts", "Mapped %", "NONE %", "Invalid %", "Agree (strong) %", "Retrieval ceiling %",
                            "Agree held-out %", "Agree (any link) %"], table3(results),
                           ["{:,}", "{:.1f}", "{:.1f}", "{:.1f}", "{:.1f}", "{:.1f}", "{:.1f}", "{:.1f}"],
                           higher=[None, None, None, False, True, None, True, True])
    anatomy_rows = [[name, p, f, r, rp] for name, p, f, r, rp in PUBLISHED_ANATOMY]
    t_pub = plain_table("For reference: OAEI 2026 Anatomy (whole alignment, systems not trained on the reference).",
                        ["Matcher", "Precision", "F-Measure", "Recall", "Recall+"], anatomy_rows)
    units = [[u["model"], u["task"], f"{u['decisions']:,}", u.get("status", ""), u.get("seconds", ""),
              u.get("usd", "")] for u in plan.get("units", [])]
    t6 = plain_table(f"Table 6. Units (model × task): budget ${plan.get('budget_usd', '?')}, spent "
                     f"${plan.get('spent_usd', '?')} at ${plan.get('price_usd_per_hour', '?')}/h.",
                     ["Model", "Task", "Decisions", "Status", "Seconds", "Projected $"], units)
    notes = ["_“–” = not run or not measured. Nothing is estimated. Coherence (OAEI's Pellet check) was not computed._",
             "_Every alignment covers every class of its source ontology (vocabulary and deprecated entities left out; "
             "counts in each scores.json). Bio-ML's test references are kept by the organisers: its alignment.rdf files "
             "are submission-ready._"]
    if verify:
        notes.append("_Rebuilt models checked against the earlier run's held-out Anatomy predictions: " + "; ".join(
            f"{k}: same decision {v.get('same_decision', '?')} of {v.get('checked', 0)}" for k, v in sorted(verify.items())) + "._")
    md = "\n\n".join([t1_md, t1b_md, t2_md, t3_md, t4_md, t_pub, t5_md, t6] + notes) + "\n"
    out = root()
    (out / "report.md").write_text(md)
    (out / "report.tex").write_text("\n\n".join([t1_tex, t1b_tex, t2_tex, t3_tex, t4_tex, t5_tex]) + "\n")
    report = {"table1_top1_full": table_top1(results, "full"), "table1b_top1_heldout": table_top1(results, "heldout"),
              "table2_top1_pairs": table_top1_pairs(results), "table3_set_based": table1(results, "full"),
              "table4_set_based_pairs": table2(results), "table5_snomed_icd11": table3(results), "verify": verify}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def main() -> int:
    build()
    print((root() / "report.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())

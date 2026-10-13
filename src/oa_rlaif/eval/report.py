"""Results tables for the paper, built from the evaluation outputs of the Lambda run (no GPU needed).

    python -m oa_rlaif.eval.report          # -> outputs/eval/report.md, report.tex, report.json

Table 1 (main): one row per training stage, one column per evaluation set, best value per column in bold:
    Gemma 4 E4B (base) | + SFT (compulsory comparison) | + OPSD | + OPSD -> GRPO (RLAIF)
    held-out decision F1 per family, ICD bridge accuracy, zero-shot pairs, closed-book knowledge (token F1), average
Table 2: the same models as mini-SWE-agents in the Docker sandbox (200 Anatomy episodes).
Table 3: the whole Anatomy alignment in OAEI form (Runtime, Size, P, F, R, Recall+), with the lexical baseline and
         DRAL-OA's official OAEI 2025 result (P 0.83, R 0.827, F 0.828; CEUR Vol-4144) for reference.
A value that was not measured is shown as "–", never filled in.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from ..paths import out_dir

# Stage 1 is OPSD: the frozen teacher sees the ground-truth answer (internal tag "sdft" = TRL's SDFTTrainer class).
ROWS = [("base", "Gemma 4 E4B (base)"), ("sft", "+ SFT"), ("sdft", "+ OPSD"), ("grpo", "+ OPSD → GRPO (RLAIF)")]
# whole-Anatomy alignments in OAEI form: output folder under full_align/ -> row label
FULL_ALIGN = [("anatomy", "OA-RLAIF E4B (GRPO)")]
# The research variant (scripts/all_merged.sh) never enters the main report or the results repo: it gets its own
# comparison report (outputs/all_merged_report/), uploaded only to its own repo (data/push_all_merged.py).
ALL_MERGED_ROW = ("all_merged", "All merged (OPSD → GRPO + SFT)")
ALL_MERGED_FULL_ALIGN = ("anatomy_all_merged", "All merged (OPSD → GRPO + SFT)")
DECISIONS = [("heldout_anatomy", "Anatomy", "f1"), ("heldout_bioml", "Bio-ML", "f1"),
             ("heldout_multifarm", "MultiFarm", "f1"), ("heldout_dh", "DH", "f1"),
             ("heldout_dh_multiling", "DH-ML", "f1"), ("heldout_biodiv", "Biodiv", "f1"),
             ("heldout_icd:who-icd10-to-icd11", "ICD-10→11", "accuracy"),
             ("heldout_icd:snomed-to-icd11", "SNOMED→ICD-11", "bridge_recovery"), ("zero_shot", "Zero-shot", "f1")]
KNOWLEDGE = [("closed_book_knowledge_qa", "Datasource QA"), ("closed_book_fact", "Onto. facts")]
DRAL_OA_OFFICIAL = {"Matcher": "DRAL-OA (OAEI 2025, official)", "Runtime": None, "Size": None,
                    "Precision": 0.83, "F-Measure": 0.828, "Recall": 0.827, "Recall+": None}


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def main_table(base: Path, rows_spec=None) -> tuple[list[str], list[tuple[str, list]]]:
    header = [label for _, label, _ in DECISIONS] + [label for _, label in KNOWLEDGE] + ["Average"]
    rows = []
    for tag, label in rows_spec or ROWS:
        metrics = _load(base / tag / "metrics.json")
        if metrics is None:
            continue
        values = [((metrics.get("decisions") or {}).get(key) or {}).get(field) for key, _, field in DECISIONS]
        values += [((metrics.get("closed_book") or {}).get(key) or {}).get("token_f1") for key, _ in KNOWLEDGE]
        measured = [v for v in values if v is not None]
        values.append(statistics.mean(measured) if measured else None)
        rows.append((label, [None if v is None else 100 * v for v in values]))
    return header, rows


def agent_table(root: Path, rows_spec=None) -> tuple[list[str], list[tuple[str, list]]]:
    rows = []
    for tag, label in rows_spec or ROWS:
        s = _load(root / "agent_runs" / tag / "summary.json")
        if s:
            rows.append((label, [100 * s["accuracy"], 100 * s["submitted"] / max(1, s["episodes"]), s["episodes"]]))
    return ["Correct (%)", "Submitted (%)", "Episodes"], rows


def sandbox_note(root: Path, rows_spec=None) -> str:
    """How the agents were isolated (Docker, or the user sandbox on machines without Docker), from summary.json."""
    seen = {(s.get("sandbox", "docker"), s.get("network", "none")) for tag, _ in rows_spec or ROWS
            if (s := _load(root / "agent_runs" / tag / "summary.json"))}
    return "; ".join(f"sandbox {box}, network {net}" for box, net in sorted(seen))


def oaei_table(root: Path, full_align=None) -> tuple[list[str], list[tuple[str, list]]]:
    header = ["Runtime (s)", "Size", "Precision", "F-Measure", "Recall", "Recall+"]
    rows = []
    for folder, label in full_align or FULL_ALIGN:
        ours = _load(root / "full_align" / folder / "summary.json")
        if ours and ours.get("scores"):
            s = ours["scores"]
            rows.append((label, [ours.get("runtime_s"), s.get("Size"), s.get("Precision"),
                                 s.get("F-Measure"), s.get("Recall"), s.get("Recall+")]))
    lexical = _load(root / "align" / "anatomy_lexical_baseline" / "report.json")
    if lexical and lexical.get("scores"):
        s = lexical["scores"]
        rows.append(("Lexical top-1 baseline", [lexical.get("runtime_s"), s.get("Size"), s.get("Precision"),
                                                s.get("F-Measure"), s.get("Recall"), s.get("Recall+")]))
    d = DRAL_OA_OFFICIAL
    rows.append((d["Matcher"], [d["Runtime"], d["Size"], d["Precision"], d["F-Measure"], d["Recall"], d["Recall+"]]))
    return header, rows


def _best(rows: list[tuple[str, list]], higher_is_better: list[bool | None]) -> list[float | None]:
    """Best value per column; None (no bold) for count columns (higher=None), single values and all-equal columns."""
    best = []
    for j, higher in enumerate(higher_is_better):
        col = [r[1][j] for r in rows if r[1][j] is not None]
        if higher is None or len(col) < 2 or max(col) == min(col):
            best.append(None)
        else:
            best.append(max(col) if higher else min(col))
    return best


def _cell(v, best, fmt, bold) -> str:
    if v is None:
        return "–"
    text = fmt.format(v)
    return bold(text) if best is not None and v == best else text


def render(title: str, header: list[str], rows, fmt: str | list[str], higher=None) -> tuple[str, str]:
    higher = higher or [True] * len(header)
    fmts = fmt if isinstance(fmt, list) else [fmt] * len(header)
    best = _best(rows, higher)
    md = [f"**{title}**", "", "| Method | " + " | ".join(header) + " |", "|---|" + "---:|" * len(header)]
    tex = [r"\begin{tabular}{l" + "c" * len(header) + "}", r"\hline",
           "Method & " + " & ".join(header) + r" \\", r"\hline"]
    for label, values in rows:
        md.append(f"| {label} | " + " | ".join(_cell(v, b, f, lambda t: f"**{t}**") for v, b, f in zip(values, best, fmts))
                  + " |")
        tex.append(label.replace("→", r"$\rightarrow$") + " & "
                   + " & ".join(_cell(v, b, f, lambda t: r"\textbf{" + t + "}") for v, b, f in zip(values, best, fmts))
                   + r" \\")
    tex += [r"\hline", r"\end{tabular}"]
    return "\n".join(md), "\n".join(tex)


def build(root: Path | None = None, all_merged: bool = False) -> dict:
    """The main report (outputs/eval/report.*), or with all_merged=True the research variant's own comparison report
    (outputs/all_merged_report/report.*: the main rows plus "All merged")."""
    root = root or out_dir()
    rows_spec = ROWS + ([ALL_MERGED_ROW] if all_merged else [])
    full_align = FULL_ALIGN + ([ALL_MERGED_FULL_ALIGN] if all_merged else [])
    h1, r1 = main_table(root / "eval", rows_spec)
    h2, r2 = agent_table(root, rows_spec)
    h3, r3 = oaei_table(root, full_align)
    parts = [render("Table 1. Held-out and zero-shot decisions (F1 ×100), ICD-10→11 accuracy ×100, SNOMED→ICD-11 "
                    "bridge recovery ×100 (held-out ICD-10 categories; retrieval misses count as failures) and "
                    "closed-book knowledge (token F1 ×100).", h1, r1, "{:.1f}"),
             render(f"Table 2. The same models as mini-SWE-agents (Anatomy; {sandbox_note(root, rows_spec) or 'sandbox not run'}).", h2, r2,
                    ["{:.1f}", "{:.1f}", "{:.0f}"], higher=[True, True, None]),
             render("Table 3. Whole Anatomy alignment in OAEI form.", h3, r3, ["{:.0f}", "{:.0f}"] + ["{:.3f}"] * 4,
                    higher=[False, None, True, True, True, True])]
    md = "\n\n".join(p[0] for p in parts) + "\n\n_“–” = not measured. DRAL-OA row: official OAEI 2025 results (CEUR Vol-4144)._\n"
    tex = "\n\n".join(p[1] for p in parts) + "\n"
    out = root / ("all_merged_report" if all_merged else "eval")
    out.mkdir(parents=True, exist_ok=True)
    # gather the agent and whole-alignment outputs next to the report, so the private results repo holds everything
    import shutil
    for tag, _ in rows_spec:
        summary = root / "agent_runs" / tag / "summary.json"
        if summary.exists():
            shutil.copyfile(summary, out / f"agent_{tag}.json")
    for folder, _ in full_align:
        full = root / "full_align" / folder
        if full.exists():
            shutil.copytree(full, out / f"full_align_{folder}", dirs_exist_ok=True)
    (out / "report.md").write_text(md)
    (out / "report.tex").write_text(tex)
    data = {"table1": {"header": h1, "rows": r1}, "table2": {"header": h2, "rows": r2}, "table3": {"header": h3, "rows": r3}}
    (out / "report.json").write_text(json.dumps(data, indent=1))
    return data


def main() -> int:
    import sys

    all_merged = "--all-merged" in sys.argv[1:]
    build(all_merged=all_merged)
    print((out_dir() / ("all_merged_report" if all_merged else "eval") / "report.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

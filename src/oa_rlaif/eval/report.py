"""Results tables for the paper, built from the evaluation outputs of the Lambda run (no GPU needed).

    python -m oa_rlaif.eval.report          # -> outputs/eval/report.md, report.tex, report.json

Table 1 (main): one row per training stage, one column per evaluation set, best value per column in bold:
    Gemma 4 E4B (base) | + SFT (optional arm) | + SDFT (OPSD) | + SDFT -> GRPO (RLAIF)
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

ROWS = [("base", "Gemma 4 E4B (base)"), ("sft", "+ SFT"), ("sdft", "+ SDFT (OPSD)"), ("grpo", "+ SDFT → GRPO (RLAIF)")]
DECISIONS = [("heldout_anatomy", "Anatomy", "f1"), ("heldout_bioml", "Bio-ML", "f1"),
             ("heldout_multifarm", "MultiFarm", "f1"), ("heldout_dh", "DH", "f1"),
             ("heldout_dh_multiling", "DH-ML", "f1"), ("heldout_biodiv", "Biodiv", "f1"),
             ("heldout_icd", "ICD bridge", "accuracy"), ("zero_shot", "Zero-shot", "f1")]
KNOWLEDGE = [("closed_book_knowledge_qa", "Datasource QA"), ("closed_book_fact", "Onto. facts")]
DRAL_OA_OFFICIAL = {"Matcher": "DRAL-OA (OAEI 2025, official)", "Runtime": None, "Size": None,
                    "Precision": 0.83, "F-Measure": 0.828, "Recall": 0.827, "Recall+": None}


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def main_table(base: Path) -> tuple[list[str], list[tuple[str, list]]]:
    header = [label for _, label, _ in DECISIONS] + [label for _, label in KNOWLEDGE] + ["Average"]
    rows = []
    for tag, label in ROWS:
        metrics = _load(base / tag / "metrics.json")
        if metrics is None:
            continue
        values = [((metrics.get("decisions") or {}).get(key) or {}).get(field) for key, _, field in DECISIONS]
        values += [((metrics.get("closed_book") or {}).get(key) or {}).get("token_f1") for key, _ in KNOWLEDGE]
        measured = [v for v in values if v is not None]
        values.append(statistics.mean(measured) if measured else None)
        rows.append((label, [None if v is None else 100 * v for v in values]))
    return header, rows


def agent_table(root: Path) -> tuple[list[str], list[tuple[str, list]]]:
    rows = []
    for tag, label in ROWS:
        s = _load(root / "agent_runs" / tag / "summary.json")
        if s:
            rows.append((label, [100 * s["accuracy"], 100 * s["submitted"] / max(1, s["episodes"]), s["episodes"]]))
    return ["Correct (%)", "Submitted (%)", "Episodes"], rows


def oaei_table(root: Path) -> tuple[list[str], list[tuple[str, list]]]:
    header = ["Runtime (s)", "Size", "Precision", "F-Measure", "Recall", "Recall+"]
    rows = []
    ours = _load(root / "full_align" / "anatomy" / "summary.json")
    if ours and ours.get("scores"):
        s = ours["scores"]
        rows.append(("OA-RLAIF E4B (GRPO)", [ours.get("runtime_s"), s.get("Size"), s.get("Precision"),
                                              s.get("F-Measure"), s.get("Recall"), s.get("Recall+")]))
    lexical = _load(root / "align" / "anatomy_lexical_baseline" / "report.json")
    if lexical and lexical.get("scores"):
        s = lexical["scores"]
        rows.append(("Lexical top-1 baseline", [lexical.get("runtime_s"), s.get("Size"), s.get("Precision"),
                                                s.get("F-Measure"), s.get("Recall"), s.get("Recall+")]))
    d = DRAL_OA_OFFICIAL
    rows.append((d["Matcher"], [d["Runtime"], d["Size"], d["Precision"], d["F-Measure"], d["Recall"], d["Recall+"]]))
    return header, rows


def _best(rows: list[tuple[str, list]], higher_is_better: list[bool]) -> list[float | None]:
    best = []
    for j, higher in enumerate(higher_is_better):
        col = [r[1][j] for r in rows if r[1][j] is not None]
        best.append((max(col) if higher else min(col)) if len(col) > 1 else None)
    return best


def _cell(v, best, fmt, bold) -> str:
    if v is None:
        return "–"
    text = fmt.format(v)
    return bold(text) if best is not None and v == best else text


def render(title: str, header: list[str], rows, fmt: str, higher=None) -> tuple[str, str]:
    higher = higher or [True] * len(header)
    best = _best(rows, higher)
    md = [f"**{title}**", "", "| Method | " + " | ".join(header) + " |", "|---|" + "---:|" * len(header)]
    tex = [r"\begin{tabular}{l" + "c" * len(header) + "}", r"\hline",
           "Method & " + " & ".join(header) + r" \\", r"\hline"]
    for label, values in rows:
        md.append(f"| {label} | " + " | ".join(_cell(v, b, fmt, lambda t: f"**{t}**") for v, b in zip(values, best)) + " |")
        tex.append(label.replace("→", r"$\rightarrow$") + " & "
                   + " & ".join(_cell(v, b, fmt, lambda t: r"\textbf{" + t + "}") for v, b in zip(values, best)) + r" \\")
    tex += [r"\hline", r"\end{tabular}"]
    return "\n".join(md), "\n".join(tex)


def build(root: Path | None = None) -> dict:
    root = root or out_dir()
    h1, r1 = main_table(root / "eval")
    h2, r2 = agent_table(root)
    h3, r3 = oaei_table(root)
    parts = [render("Table 1. Held-out and zero-shot decisions (F1 ×100; ICD bridge: accuracy ×100) and closed-book "
                    "knowledge (token F1 ×100).", h1, r1, "{:.1f}"),
             render("Table 2. The same models as mini-SWE-agents (Docker sandbox, Anatomy).", h2, r2, "{:.1f}"),
             render("Table 3. Whole Anatomy alignment in OAEI form.", h3, r3, "{:.3f}",
                    higher=[False, True, True, True, True, True])]
    md = "\n\n".join(p[0] for p in parts) + "\n\n_“–” = not measured. DRAL-OA row: official OAEI 2025 results (CEUR Vol-4144)._\n"
    tex = "\n\n".join(p[1] for p in parts) + "\n"
    out = root / "eval"
    out.mkdir(parents=True, exist_ok=True)
    # gather the agent and whole-alignment outputs next to the report, so the private results repo holds everything
    import shutil
    for tag, _ in ROWS:
        summary = root / "agent_runs" / tag / "summary.json"
        if summary.exists():
            shutil.copyfile(summary, out / f"agent_{tag}.json")
    full = root / "full_align" / "anatomy"
    if full.exists():
        shutil.copytree(full, out / "full_align_anatomy", dirs_exist_ok=True)
    (out / "report.md").write_text(md)
    (out / "report.tex").write_text(tex)
    data = {"table1": {"header": h1, "rows": r1}, "table2": {"header": h2, "rows": r2}, "table3": {"header": h3, "rows": r3}}
    (out / "report.json").write_text(json.dumps(data, indent=1))
    return data


def main() -> int:
    build()
    print((out_dir() / "eval" / "report.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

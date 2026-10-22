#!/usr/bin/env python3
"""om-score: the agent's scoring skill. Top-k evaluation (k = 1, 3, 5): Precision@k, Recall@k, F-measure@k (DRAL-OA's
clf.py precision_at_k / recall_at_k).

    ./om-score output/s2                               reference from the run's run.json
    ./om-score output/s1 output/s2 --reference ontologies/reference.rdf     several runs, one chart

For each run folder: the source classes that have a partner in the reference are scored; each class's ranked answers
are the model's own choice first, then its other candidates in retrieval order (from tasks/*/episode.json); NONE =
nothing found. Per class c, with F_c its top k and R_c its partners:
    precision@k = |F_c ∩ R_c| / min(k, |R_c|)        recall@k = |F_c ∩ R_c| / |R_c|
averaged over the scored classes; F-measure@k = 2PR / (P + R). The reference is read from each run's source side.

Prints the table and writes scores/<name>/ (topk.json, results.md, topk.svg: Precision, Recall and F-measure vs.
Top-k). It never writes into the run folders.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_matching_ai as tm  # noqa: E402  (sets the OA-RLAIF import path)

tm.own_folders()


def run_scores(folder: Path, reference: Path | None) -> tuple[dict, dict]:
    from oa_rlaif.ontology.parser import OntologyParser

    run = json.loads((folder / "run.json").read_text()) if (folder / "run.json").exists() else {}
    summary = json.loads((folder / "summary.json").read_text()) if (folder / "summary.json").exists() else {}
    source_path = Path(run.get("source") or summary["source"])
    target_path = Path(run.get("target") or summary["target"])
    reference = reference or (Path(run["reference"]) if run.get("reference") else None)
    if reference is None or not reference.exists():
        raise SystemExit(f"{folder}: no reference (give --reference FILE)")
    records = [json.loads(line) for line in (folder / "decisions.jsonl").read_text().splitlines() if line.strip()]
    latest = {}
    for r in records:  # a rerun appends: the last decided record per class counts
        if r["decision"] in ("match", "NONE") or r["source"] not in latest:
            latest[r["source"]] = r
    candidates = {}
    for source_iri in latest:
        task = folder / "tasks" / re.sub(r"[^A-Za-z0-9_.-]+", "_", source_iri)[-120:] / "episode.json"
        if task.exists():
            candidates[source_iri] = [c["iri"] for c in json.loads(task.read_text())["candidates"]]
    source, target = OntologyParser().parse(source_path), OntologyParser().parse(target_path)
    topk = tm.score(list(latest.values()), candidates, reference, source, target)
    info = {"run": str(folder), "source": source_path.name, "target": target_path.name, "reference": str(reference),
            "model": summary.get("model", "?"), "classes_decided": len(latest)}
    return topk, info


def main(argv: list[str] | None = None) -> int:
    from rich.console import Console
    from rich.table import Table

    from oa_rlaif.eval.topk_chart import write_topk_chart

    parser = argparse.ArgumentParser(prog="om-score", description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run folders (output/<name>)")
    parser.add_argument("--reference", type=Path, default=None, help="default: the reference in each run's run.json")
    parser.add_argument("--out", type=Path, default=None, help="default: scores/<run name(s)>")
    args = parser.parse_args(argv)
    console = Console(highlight=False)
    results = {}
    for folder in args.runs:
        folder = folder if folder.is_absolute() else (Path.cwd() / folder)
        topk, info = run_scores(folder.resolve(), args.reference.resolve() if args.reference else None)
        label = f"{folder.name} ({info['source'].rsplit('.', 1)[0]} → {info['target'].rsplit('.', 1)[0]})"
        results[label] = (topk, info)
        table = Table(title=f"{topk['method']}: {label}, {topk['classes_scored']:,} reference "
                            f"classes scored", title_justify="left")
        for column in ("Top-k", "Precision", "Recall", "F-measure", "Classes with a hit"):
            table.add_column(column, justify="left" if column == "Top-k" else "right")
        for k, v in topk["k"].items():
            table.add_row(f"Top-{k}", f"{v['Precision']:.3f}", f"{v['Recall']:.3f}", f"{v['F-measure']:.3f}",
                          f"{v['classes_with_a_hit']:,}")
        console.print(table)
    out = args.out or HERE / "scores" / "+".join(f.name for f in args.runs)
    out.mkdir(parents=True, exist_ok=True)
    (out / "topk.json").write_text(json.dumps({label: {"scores": t, **i} for label, (t, i) in results.items()}, indent=1))
    (out / "results.md").write_text("\n".join(tm.results_table(t, f"{label}, {i['model']}") for label, (t, i) in results.items()))
    chart = write_topk_chart(out / "topk.svg", {label: t for label, (t, _) in results.items()},
                             "Top-k evaluation (k = 1, 3, 5)")
    console.print(f"[dim]wrote {out / 'topk.json'}, {out / 'results.md'}, {chart}[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

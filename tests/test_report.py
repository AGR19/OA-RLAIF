"""The results report: rows in training order, best value bold, unmeasured values shown as '–' (synthetic inputs)."""

import json


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_report_tables(tmp_path):
    from oa_rlaif.eval.report import build

    def metrics(f1, qa):
        return {"decisions": {"heldout_anatomy": {"f1": f1}, "heldout_icd:snomed-to-icd11": {"bridge_recovery": f1 / 2}},
                "closed_book": {"closed_book_knowledge_qa": {"token_f1": qa}}}
    write(tmp_path / "eval/base/metrics.json", metrics(0.50, 0.20))
    write(tmp_path / "eval/sdft/metrics.json", metrics(0.60, 0.40))
    write(tmp_path / "eval/grpo/metrics.json", metrics(0.70, 0.35))
    write(tmp_path / "agent_runs/grpo/summary.json", {"episodes": 200, "accuracy": 0.8, "submitted": 196, "tag": "grpo"})
    write(tmp_path / "full_align/anatomy/summary.json",
          {"runtime_s": 300, "scores": {"Size": 1500, "Precision": 0.9, "F-Measure": 0.85, "Recall": 0.8, "Recall+": 0.5}})
    data = build(tmp_path)
    labels = [r[0] for r in data["table1"]["rows"]]
    assert labels == ["Gemma 4 E4B (base)", "+ OPSD", "+ OPSD → GRPO (RLAIF)"]  # SFT row absent: not run
    assert "SNOMED→ICD-11" in data["table1"]["header"] and "ICD-10→11" in data["table1"]["header"]
    md = (tmp_path / "eval/report.md").read_text()
    assert "**70.0**" in md and "**40.0**" in md            # best Anatomy F1 (GRPO) and best QA (SDFT) in bold
    assert "–" in md                                          # unmeasured columns are never filled in
    assert "DRAL-OA (OAEI 2025, official)" in md and "0.828" in md
    assert (tmp_path / "eval/agent_grpo.json").exists() and (tmp_path / "eval/full_align_anatomy/summary.json").exists()
    assert r"\textbf{70.0}" in (tmp_path / "eval/report.tex").read_text()


def test_report_bold_rules(tmp_path):
    """No bold for ties across all rows or for count columns (Episodes, Size); counts print as integers."""
    from oa_rlaif.eval.report import build

    for tag in ("base", "grpo"):
        write(tmp_path / f"eval/{tag}/metrics.json", {"decisions": {"heldout_anatomy": {"f1": 0.0}}})
        write(tmp_path / f"agent_runs/{tag}/summary.json", {"episodes": 2, "accuracy": 0.0, "submitted": 2})
    write(tmp_path / "full_align/anatomy/summary.json",
          {"runtime_s": 19, "scores": {"Size": 1589, "Precision": 0.7, "F-Measure": 0.7, "Recall": 0.7, "Recall+": 0.5}})
    build(tmp_path)
    md = (tmp_path / "eval/report.md").read_text()
    assert "**0.0**" not in md and "**100.0**" not in md and "**2**" not in md
    assert "| 1589 |" in md and "| 19 |" in md and "| 2 |" in md


def test_all_merged_has_its_own_report(tmp_path):
    """The research variant never enters the main report; it gets its own comparison report in its own folder."""
    from oa_rlaif.eval.report import build

    write(tmp_path / "eval/grpo/metrics.json", {"decisions": {"heldout_anatomy": {"f1": 0.7}}})
    write(tmp_path / "eval/all_merged/metrics.json", {"decisions": {"heldout_anatomy": {"f1": 0.72}}})
    write(tmp_path / "agent_runs/all_merged/summary.json", {"episodes": 2, "accuracy": 0.5, "submitted": 2})
    for folder, f in (("anatomy", 0.85), ("anatomy_all_merged", 0.86)):
        write(tmp_path / f"full_align/{folder}/summary.json",
              {"runtime_s": 10, "scores": {"Size": 1, "Precision": f, "F-Measure": f, "Recall": f, "Recall+": 0.5}})
    label = "All merged (OPSD → GRPO + SFT)"
    main = build(tmp_path)
    assert all(label not in [r[0] for r in main[t]["rows"]] for t in ("table1", "table2", "table3"))
    assert "all_merged" not in " ".join(p.name for p in (tmp_path / "eval").iterdir() if p.is_file())
    assert not (tmp_path / "eval/full_align_anatomy_all_merged").exists()
    own = build(tmp_path, all_merged=True)
    assert [r[0] for r in own["table1"]["rows"]][-1] == label and label in [r[0] for r in own["table2"]["rows"]]
    assert [r[0] for r in own["table3"]["rows"]][:2] == ["OA-RLAIF E4B (GRPO)", label]
    assert (tmp_path / "all_merged_report/report.md").exists()
    assert (tmp_path / "all_merged_report/full_align_anatomy_all_merged/summary.json").exists()

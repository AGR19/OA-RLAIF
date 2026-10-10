"""The results report: rows in training order, best value bold, unmeasured values shown as '–' (synthetic inputs)."""

import json


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_report_tables(tmp_path):
    from oa_rlaif.eval.report import build

    def metrics(f1, qa):
        return {"decisions": {"heldout_anatomy": {"f1": f1}, "heldout_icd": {"accuracy": f1 / 2}},
                "closed_book": {"closed_book_knowledge_qa": {"token_f1": qa}}}
    write(tmp_path / "eval/base/metrics.json", metrics(0.50, 0.20))
    write(tmp_path / "eval/sdft/metrics.json", metrics(0.60, 0.40))
    write(tmp_path / "eval/grpo/metrics.json", metrics(0.70, 0.35))
    write(tmp_path / "agent_runs/grpo/summary.json", {"episodes": 200, "accuracy": 0.8, "submitted": 196, "tag": "grpo"})
    write(tmp_path / "full_align/anatomy/summary.json",
          {"runtime_s": 300, "scores": {"Size": 1500, "Precision": 0.9, "F-Measure": 0.85, "Recall": 0.8, "Recall+": 0.5}})
    data = build(tmp_path)
    labels = [r[0] for r in data["table1"]["rows"]]
    assert labels == ["Gemma 4 E4B (base)", "+ SDFT (OPSD)", "+ SDFT → GRPO (RLAIF)"]  # SFT row absent: not run
    md = (tmp_path / "eval/report.md").read_text()
    assert "**70.0**" in md and "**40.0**" in md            # best Anatomy F1 (GRPO) and best QA (SDFT) in bold
    assert "–" in md                                          # unmeasured columns are never filled in
    assert "DRAL-OA (OAEI 2025, official)" in md and "0.828" in md
    assert (tmp_path / "eval/agent_grpo.json").exists() and (tmp_path / "eval/full_align_anatomy/summary.json").exists()
    assert r"\textbf{70.0}" in (tmp_path / "eval/report.tex").read_text()

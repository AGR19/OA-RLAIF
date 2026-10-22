"""The separate SFT push: adapter + evaluation + logs in one private repo, the merged model in another, our own
model cards (the trainer's README is never uploaded), and nothing pushed before step 13 has finished."""

import json

import pytest


def make_outputs(root, evaluated=True):
    final = root / "checkpoints" / "sft" / "final"
    final.mkdir(parents=True)
    (final / "adapter_model.safetensors").write_bytes(b"w" * 10)
    (final / "adapter_config.json").write_text("{}")
    (final / "README.md").write_text("---\nbase_model: outputs/x\n---\n")
    merged = root / "merged" / "sft"
    merged.mkdir(parents=True)
    (merged / "model.safetensors").write_bytes(b"m" * 10)
    (merged / "config.json").write_text("{}")
    (root / "eval" / "sft").mkdir(parents=True)
    (root / "eval" / "sft" / "metrics.json").write_text("{}")
    (root / "agent_runs" / "sft").mkdir(parents=True)
    (root / "agent_runs" / "sft" / "summary.json").write_text("{}")
    (root / "logs").mkdir()
    (root / "logs" / "stage_sft.log").write_text("log")
    (root / "step_plan.json").write_text(json.dumps({"sft": {"steps": 7}}))
    (root / "stages").mkdir()
    (root / "stages" / "sft.done").touch()
    if evaluated:
        (root / "stages" / "eval_sft.done").touch()


def test_push_sft_uploads_everything_privately(tmp_path, monkeypatch):
    import huggingface_hub

    from oa_rlaif.data import push_sft

    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    make_outputs(tmp_path)
    calls = []

    class FakeApi:
        def create_repo(self, repo, **kw): calls.append(("create", repo, kw.get("private")))
        def repo_info(self, repo, **kw): return type("Info", (), {"private": True})()
        def create_commit(self, **kw): calls.append(("commit", kw["repo_id"], [op.path_in_repo for op in kw["operations"]]))
        def upload_large_folder(self, **kw): calls.append(("large", kw["repo_id"], kw["ignore_patterns"], kw["private"]))
        def upload_file(self, **kw): calls.append(("card", kw["repo_id"], kw["path_or_fileobj"].decode()))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(push_sft, "namespace", lambda: "me")
    assert push_sft.main([]) == 0
    assert ("create", "me/oa-rlaif-e4b-sft-weights", True) in calls and ("create", "me/oa-rlaif-e4b-sft-model-safetensors", True) in calls
    commit = next(c for c in calls if c[0] == "commit")
    assert commit[1] == "me/oa-rlaif-e4b-sft-weights"
    paths = set(commit[2])
    assert {"adapter_model.safetensors", "adapter_config.json", "eval/metrics.json", "eval/agent/summary.json",
            "logs/stage_sft.log", "plan.json", "README.md"} <= paths
    large = next(c for c in calls if c[0] == "large")
    assert large[1] == "me/oa-rlaif-e4b-sft-model-safetensors" and "README.md" in large[2] and large[3] is True
    merged_card = next(c[2] for c in calls if c[0] == "card")
    assert "base_model: google/gemma-4-E4B-it" in merged_card and "library_name: transformers" in merged_card


def test_push_sft_waits_for_step_13(tmp_path, monkeypatch):
    from oa_rlaif.data import push_sft

    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    make_outputs(tmp_path)
    (tmp_path / "stages" / "sft.done").unlink()
    with pytest.raises(SystemExit, match="step 13"):
        push_sft.main(["--dry-run"])


def test_push_all_merged(tmp_path, monkeypatch):
    import huggingface_hub

    from oa_rlaif.data import push_all_merged

    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    with pytest.raises(SystemExit, match="all_merged.sh"):
        push_all_merged.main(["--dry-run"])
    m = tmp_path / "merged" / "all_merged"
    m.mkdir(parents=True)
    (m / "config.json").write_text("{}")
    (tmp_path / "eval" / "all_merged").mkdir(parents=True)
    (tmp_path / "eval" / "all_merged" / "metrics.json").write_text("{}")
    (tmp_path / "all_merged_report").mkdir()
    (tmp_path / "all_merged_report" / "report.md").write_text("tables")
    calls = []

    class FakeApi:
        def create_repo(self, repo, **kw): calls.append(("create", repo, kw.get("private")))
        def repo_info(self, repo, **kw): return type("Info", (), {"private": True})()
        def upload_large_folder(self, **kw): calls.append(("large", kw["repo_id"], kw["private"]))
        def create_commit(self, **kw): calls.append(("commit", [op.path_in_repo for op in kw["operations"]]))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(push_all_merged, "namespace", lambda: "me")
    assert push_all_merged.main([]) == 0
    assert ("create", "me/oa-rlaif-e4b-all-combined-model-safetensors", True) in calls and ("large", "me/oa-rlaif-e4b-all-combined-model-safetensors", True) in calls
    assert {"eval/metrics.json", "report/report.md", "README.md"} <= set(next(c[1] for c in calls if c[0] == "commit"))

"""Model push: the GGUF goes first, adapters carry our own model card (the trainers' README names a local folder as
base_model, which the Hub rejects), and nothing is uploaded to a public repo."""

import re


def test_model_push_order_and_cards(tmp_path, monkeypatch):
    import huggingface_hub

    from oa_rlaif.data import hf_push

    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    for folder, _ in hf_push.MODELS.values():
        d = tmp_path / folder.split("/", 1)[1]
        d.mkdir(parents=True)
        (d / "README.md").write_text("---\nbase_model: outputs/merged/sdft\n---\n")  # what the trainer writes
    calls = []

    class FakeApi:
        def create_repo(self, repo, **kw): calls.append(("create", repo))
        def repo_info(self, repo, **kw): return type("Info", (), {"private": True})()
        def upload_folder(self, **kw): calls.append(("folder", kw["repo_id"], kw.get("ignore_patterns")))
        def upload_file(self, **kw): calls.append(("card", kw["repo_id"], kw["path_or_fileobj"].decode()))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(hf_push, "namespace", lambda: "me")
    hf_push.push_models()
    folders = [c for c in calls if c[0] == "folder"]
    assert folders[0][1] == "me/oa-rlaif-e4b-opsd-grpo-model-gguf"                         # the deliverable first
    assert all(c[2] == ["README.md"] for c in folders)                    # trainer READMEs never uploaded
    cards = {c[1]: c[2] for c in calls if c[0] == "card"}
    assert set(cards) == {f"me/{n}" for n in hf_push.MODELS}
    for text in cards.values():
        base = re.search(r"^base_model: (.+)$", text, re.M).group(1)
        assert base == "google/gemma-4-E4B-it" and "outputs/" not in base
    assert "trained on base + OPSD, not on the plain base. Merge OPSD first" in cards["me/oa-rlaif-e4b-grpo-weights"]
    assert all("pipeline_tag: text-generation" in text for text in cards.values())


def test_results_repo_never_gets_all_merged(tmp_path, monkeypatch, capsys):
    import huggingface_hub
    from fnmatch import fnmatch

    from oa_rlaif.data import hf_push

    monkeypatch.setenv("OA_OUT_DIR", str(tmp_path))
    ev = tmp_path / "eval"
    for rel in ("report.md", "grpo/metrics.json", "agent_grpo.json", "all_merged/metrics.json",
                "agent_all_merged.json", "full_align_anatomy_all_merged/summary.json"):
        (ev / rel).parent.mkdir(parents=True, exist_ok=True)
        (ev / rel).write_text("{}")
    hf_push.push_models(hf_push.RESULTS, dry_run=True)
    assert "would push 3 files" in capsys.readouterr().out     # report, grpo metrics, grpo agent: no all-merged
    calls = []

    class FakeApi:
        def create_repo(self, repo, **kw): pass
        def repo_info(self, repo, **kw): return type("Info", (), {"private": True})()
        def upload_folder(self, **kw): calls.append(kw["ignore_patterns"])

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(hf_push, "namespace", lambda: "me")
    hf_push.push_models(hf_push.RESULTS)
    ignore = calls[0]
    files = [p.relative_to(ev).as_posix() for p in ev.rglob("*") if p.is_file()]
    uploaded = [f for f in files if not any(fnmatch(f, pat) for pat in ignore)]
    assert sorted(uploaded) == ["agent_grpo.json", "grpo/metrics.json", "report.md"]

"""Push the research variant "all merged" (base E4B + OPSD + GRPO + SFT) to a private repo, with its evaluation.

    python -m oa_rlaif.data.push_all_merged            # scripts/all_merged.sh --push calls this
    python -m oa_rlaif.data.push_all_merged --dry-run

{ns}/oa-rlaif-e4b-all-combined-model-safetensors   the merged weights (outputs/merged/all_merged; resumable upload), then one commit with
                               eval/ (held-out, closed-book, ICD), eval/agent/ (agent runs),
                               eval/full_align/ (whole-Anatomy alignment, OAEI form), report/ (its own comparison
                               tables: base, SFT, OPSD, GRPO and All merged) and the model card.
This repo is the only place on the Hub with all-merged material: the results repo (oa-rlaif-heldout-eval-results) and the main
report never include it.
Private (checked after creation); our own card. Not the released model: the GGUF stays base + OPSD + GRPO.
"""

from __future__ import annotations

import argparse
import sys

from ..paths import out_dir
from .hf_push import namespace
from .push_sft import private_repo

REPO = "oa-rlaif-e4b-all-combined-model-safetensors"

def card() -> str:
    from .hf_cards import card as hf_card

    return hf_card(REPO)


def eval_files(out):
    files = []
    for src, dest in ((out / "eval" / "all_merged", "eval"), (out / "agent_runs" / "all_merged", "eval/agent"),
                      (out / "full_align" / "anatomy_all_merged", "eval/full_align"),
                      (out / "all_merged_report", "report")):
        if src.exists():
            files += [(p, f"{dest}/{p.relative_to(src).as_posix()}") for p in sorted(src.rglob("*")) if p.is_file()]
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="push_all_merged")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    out = out_dir()
    folder = out / "merged" / "all_merged"
    if not (folder / "config.json").exists():
        raise SystemExit("outputs/merged/all_merged is missing: run scripts/all_merged.sh first")
    weights = [p for p in folder.rglob("*") if p.is_file() and ".cache" not in p.parts]
    files = eval_files(out)
    ns = "<namespace>" if args.dry_run else namespace()
    print(f"{ns}/{REPO}: model {sum(p.stat().st_size for p in weights) / 1e9:,.2f} GB in {len(weights)} files; "
          f"evaluation {len(files)} files" + ("" if files else " (none yet: run the evaluation first)"))
    if args.dry_run:
        return 0
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    repo = f"{ns}/{REPO}"
    private_repo(api, repo)
    api.upload_large_folder(repo_id=repo, folder_path=str(folder), repo_type="model", private=True,
                            ignore_patterns=["README.md", "*.tmp", ".cache/**"])
    operations = [CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(src)) for src, dest in files]
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card().encode()))
    api.create_commit(repo_id=repo, repo_type="model", operations=operations,
                      commit_message="All merged: evaluation and model card")
    print(f"pushed {repo} (private)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

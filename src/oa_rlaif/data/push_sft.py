"""Push everything of the SFT comparison arm to private Hugging Face repos (separate from the main pipeline).

    python -m oa_rlaif.data.push_sft             # after step 13 (sft); run again after step 14 to add the evaluation
    python -m oa_rlaif.data.push_sft --dry-run   # list what would be uploaded, upload nothing

{ns}/oa-rlaif-e4b-sft           the LoRA adapter (outputs/checkpoints/sft/final), one commit with
                                eval/        its held-out and closed-book results (outputs/eval/sft)
                                eval/agent/  its mini-SWE-agent runs (outputs/agent_runs/sft)
                                logs/        step logs and TensorBoard training curves
                                plan.json    its step plan and dry-run timing
{ns}/oa-rlaif-e4b-sft-merged    the merged model (outputs/merged/sft), resumable upload of the large weights

Both repos are private (checked after creation) and carry our own model card: the trainer's README.md is never
uploaded (its metadata can name a local folder, which the Hub rejects). SFT starts from the plain base model, so the
adapter applies to google/gemma-4-E4B-it directly. The SFT arm is the comparison row of the study; it is not part of
the released GGUF.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..paths import out_dir
from .hf_push import BASE_MODEL, namespace

ADAPTER_REPO, MERGED_REPO = "oa-rlaif-e4b-sft", "oa-rlaif-e4b-sft-merged"
LOGS = ["stage_sft.log", "stage_eval_sft.log", "eval_sft.log", "agent_eval_sft.log", "vllm_agent_sft.log",
        "dry_run_sft.log"]
CARDS = {
    ADAPTER_REPO: ("peft", "SFT comparison arm of OA-RLAIF: a LoRA adapter for `google/gemma-4-E4B-it`, trained with "
                           "plain supervised fine-tuning on the same rows, steps, learning rate and batch as Stage 1 "
                           "(OPSD). Apply it to the base model. `eval/` holds its held-out, closed-book and agent "
                           "results; `logs/` its training and evaluation logs; `plan.json` its step plan."),
    MERGED_REPO: ("transformers", "SFT comparison arm of OA-RLAIF: `google/gemma-4-E4B-it` with the SFT LoRA adapter "
                                  "(`oa-rlaif-e4b-sft`) merged into the weights. Load it like the base model."),
}


def card(name: str) -> str:
    library, text = CARDS[name]
    return (f"---\nlicense: other\nlicense_name: mixed-private\nbase_model: {BASE_MODEL}\nlibrary_name: {library}\n"
            f"tags: [ontology-matching, sft, oa-rlaif, comparison]\n---\n# {name} (private)\n\n{text}\n\n"
            "Part of OA-RLAIF (github.com/AGR19/OA-RLAIF, private).\n")


def adapter_files(out: Path) -> list[tuple[Path, str]]:
    """(local file, path in the repo) for the adapter repo: weights, evaluation, agent runs, logs."""
    files: list[tuple[Path, str]] = []
    final = out / "checkpoints" / "sft" / "final"
    files += [(p, p.relative_to(final).as_posix()) for p in sorted(final.rglob("*"))
              if p.is_file() and p.name != "README.md"]
    for src, dest in ((out / "eval" / "sft", "eval"), (out / "agent_runs" / "sft", "eval/agent"),
                      (out / "checkpoints" / "sft" / "runs", "logs/tensorboard")):
        if src.exists():
            files += [(p, f"{dest}/{p.relative_to(src).as_posix()}") for p in sorted(src.rglob("*")) if p.is_file()]
    files += [(out / "logs" / name, f"logs/{name}") for name in LOGS if (out / "logs" / name).is_file()]
    return files


def plan_json(out: Path) -> bytes:
    plan = json.loads((out / "step_plan.json").read_text()).get("sft") if (out / "step_plan.json").exists() else None
    timing = json.loads((out / "dry_run.json").read_text()) if (out / "dry_run.json").exists() else {}
    return json.dumps({"step_plan": plan, "sft_seconds_per_step": timing.get("sft_seconds_per_step")}, indent=1).encode()


def private_repo(api, repo: str) -> None:
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="model").private:
        raise SystemExit(f"{repo} is not private; refusing to upload")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="push_sft")
    parser.add_argument("--dry-run", action="store_true", help="list what would be uploaded; upload nothing")
    parser.add_argument("--force", action="store_true", help="push even if step 13 (sft) has not finished")
    args = parser.parse_args(argv)
    out = out_dir()
    stages = out / "stages"
    adapter, merged = out / "checkpoints" / "sft" / "final", out / "merged" / "sft"
    if not (stages / "sft.done").exists() and not args.force:
        raise SystemExit("step 13 (sft) has not finished (outputs/stages/sft.done is missing); run this after it, "
                         "or pass --force to push what exists")
    missing = [str(p) for p in (adapter, merged) if not p.is_dir()]
    if missing:
        raise SystemExit(f"nothing to push yet: {missing} missing")
    evaluated = (stages / "eval_sft.done").exists()

    files = adapter_files(out)
    merged_files = [p for p in sorted(merged.rglob("*")) if p.is_file() and p.name != "README.md"
                    and ".cache" not in p.parts]
    ns = "<namespace>" if args.dry_run else namespace()
    print(f"{ns}/{ADAPTER_REPO}: {len(files) + 2} files, {sum(p.stat().st_size for p, _ in files) / 1e6:,.1f} MB "
          f"(adapter, eval, agent runs, logs, plan.json, README.md)")
    print(f"{ns}/{MERGED_REPO}: {len(merged_files) + 1} files, "
          f"{sum(p.stat().st_size for p in merged_files) / 1e9:,.2f} GB (merged model, README.md)")
    if not evaluated:
        print("note: step 14 (eval_sft) has not finished; run this script again afterwards to add the evaluation")
    if args.dry_run:
        for _, dest in files[:40]:
            print("   ", dest)
        return 0

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    repo = f"{ns}/{ADAPTER_REPO}"
    private_repo(api, repo)
    operations = [CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(src)) for src, dest in files]
    operations += [CommitOperationAdd(path_in_repo="plan.json", path_or_fileobj=plan_json(out)),
                   CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card(ADAPTER_REPO).encode())]
    api.create_commit(repo_id=repo, repo_type="model", operations=operations,
                      commit_message="SFT adapter, evaluation and logs" + ("" if evaluated else " (before eval_sft)"))
    print(f"pushed {repo} (private)")

    repo = f"{ns}/{MERGED_REPO}"
    private_repo(api, repo)
    # resumable: an interrupted upload continues where it stopped when the script runs again
    api.upload_large_folder(repo_id=repo, folder_path=str(merged), repo_type="model", private=True,
                            ignore_patterns=["README.md", "*.tmp", ".cache/**"])
    api.upload_file(path_or_fileobj=card(MERGED_REPO).encode(), path_in_repo="README.md", repo_id=repo,
                    repo_type="model")
    print(f"pushed {repo} (private)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

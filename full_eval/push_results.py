"""Upload every output of the full evaluation to its own private dataset. Resumable.

    python -m full_eval.push_results [--dry-run] [--episodes]

  first run                      -> <you>/oa-rlaif-full-eval-results
  a rerun (FULL_EVAL_RERUN=tags) -> <you>/oa-rlaif-fixed-full-eval-results: the first run's files (copied by
                                    full_eval.pull_inputs) with the rerun methods' outputs and the report replaced. A
                                    rerun never writes to the first run's repo (target_repo refuses).
  FULL_EVAL_RESULTS_REPO overrides the repo name.

  report.md / .tex / .json, figures/ (Top-k charts), plan.json, speed.json, verify_*.json (weight check)
  results/<task>/<method folder>/  scores.json, decisions.tsv.gz (every class run: decision, target, label, confidence,
                            THOUGHT), pairs/<pair>/alignment.rdf (OAEI) and alignment.sssom.tsv, one file per method
                            and pair; for SNOMED CT -> ICD-11 the complete map (icd11_snomed_complete.sssom.tsv,
                            snomed_to_icd11.tsv.gz)
  replies/<method>/<task>/  the model's output for every class (with thinking on: the thinking text; agent: its
                            commands and every model turn)
  episodes/*/manifest.json, gold.json; logs/
  --episodes also uploads the prompts (episode shards; large, and rebuilt identically by full_eval.build_episodes).
Nothing goes into the existing repos. Private (checked after creation): SNOMED CT-derived content is inside.
"""

from __future__ import annotations

import argparse
import os
import sys

from .schedule import root

REPO = "oa-rlaif-full-eval-results"  # the first run (2026-10-22); never changed by a rerun
FIXED_REPO = "oa-rlaif-fixed-full-eval-results"  # a rerun: the first run with the rerun methods replaced


def rerun_tags() -> list[str]:
    return [t.strip() for t in os.environ.get("FULL_EVAL_RERUN", "").split(",") if t.strip()]


def target_repo() -> str:
    """This run's results repo: FULL_EVAL_RESULTS_REPO, else FIXED_REPO for a rerun, else REPO. A rerun never
    writes to REPO."""
    repo = os.environ.get("FULL_EVAL_RESULTS_REPO") or (FIXED_REPO if rerun_tags() else REPO)
    if rerun_tags() and repo == REPO:
        raise SystemExit(f"a rerun must not change {REPO} (the first run's results); choose another "
                         "FULL_EVAL_RESULTS_REPO")
    return repo


FIXED_CARD = """---
license: other
license_name: mixed-private
tags: [ontology-matching, oa-rlaif, top-k-evaluation, private]
---
# oa-rlaif-fixed-full-eval-results (private)

The full evaluation of github.com/AGR19/OA-RLAIF (`full_eval/`) with the reply-length fix. It starts from
`oa-rlaif-full-eval-results` (the first run, 2026-10-22, which stays unchanged): every file of the methods that were
not run again is copied from there as it was. The methods run again ({rerun}) replace their files, because the first
run cut their one-call replies at 256 new tokens before the answer; they now have up to 1,024. `report.md`, the
figures and `plan.json` are rebuilt from all six methods. Scored with the Top-k evaluation (k = 1, 3, 5) on every
source class that has a partner in the reference (no held-out view); the complete SNOMED CT -> ICD-11 map of ICD's
scope. Results: `results/<dataset>/<method>/`, one alignment file per method and pair. Start with `report.md`.
**Licensing:** contains SNOMED CT-derived mappings (affiliate licence) and UMLS-derived Bio-ML references. Private
research use only; never make this repo public.
"""
CARD = """---
license: other
license_name: mixed-private
tags: [ontology-matching, oa-rlaif, top-k-evaluation, private]
---
# oa-rlaif-full-eval-results (private)

Full evaluation of the OA-RLAIF models (github.com/AGR19/OA-RLAIF, `full_eval/`): base Gemma 4 E4B, + SFT, + OPSD,
+ OPSD -> GRPO and All merged with thinking off, and All merged + mini-SWE-agent with thinking on. Scored with the
Top-k evaluation (k = 1, 3, 5; DRAL-OA's clf.py formula) on every source class that has a partner in the reference (no
held-out view), and the complete SNOMED CT -> ICD-11 map of ICD's scope. Results are sorted by dataset, then
method; one alignment file per method and pair:
`results/<task>/<method folder>/pairs/<pair>/alignment.rdf` (e.g. `results/anatomy__mouse-human/4-opsd-grpo/`) (OAEI format) and `alignment.sssom.tsv`. Start with `report.md`.
**Licensing:** contains SNOMED CT-derived mappings (affiliate licence) and UMLS-derived Bio-ML references. Private
research use only; never make this repo public.
"""
PATTERNS = ["report.*", "figures/**", "plan.json", "speed.json", "verify_*.json", "results/**", "replies/**",
            "episodes/*/manifest.json", "episodes/*/gold.json", "logs/**", "progress/**"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="full_eval.push_results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--episodes", action="store_true")
    args = parser.parse_args(argv)
    patterns = PATTERNS + (["episodes/**"] if args.episodes else [])
    from fnmatch import fnmatch

    files = [p for p in root().rglob("*") if p.is_file()
             and any(fnmatch(p.relative_to(root()).as_posix(), pat) for pat in patterns)]
    print(f"{len(files):,} files, {sum(p.stat().st_size for p in files) / 1e9:.2f} GB from {root()}")
    if args.dry_run:
        return 0
    from huggingface_hub import HfApi

    from oa_rlaif.data.hf_push import namespace

    api = HfApi()
    repo = f"{namespace()}/{target_repo()}"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:
        raise SystemExit(f"{repo} is not private; refusing to upload")
    api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=str(root()), private=True,
                            allow_patterns=patterns)
    card = FIXED_CARD.format(rerun=", ".join(rerun_tags())) if rerun_tags() else CARD
    api.upload_file(repo_id=repo, repo_type="dataset", path_or_fileobj=card.encode(), path_in_repo="README.md")
    print(f"pushed {repo} (private)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

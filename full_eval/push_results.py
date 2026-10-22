"""Upload every output of the full evaluation to its own private dataset <you>/oa-rlaif-full-eval-results. Resumable.

    python -m full_eval.push_results [--dry-run] [--episodes]

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
import sys

from .schedule import root

REPO = "oa-rlaif-full-eval-results"
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
    repo = f"{namespace()}/{REPO}"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:
        raise SystemExit(f"{repo} is not private; refusing to upload")
    api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=str(root()), private=True,
                            allow_patterns=patterns)
    api.upload_file(repo_id=repo, repo_type="dataset", path_or_fileobj=CARD.encode(), path_in_repo="README.md")
    print(f"pushed {repo} (private)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

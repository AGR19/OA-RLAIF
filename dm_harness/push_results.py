"""Upload the decision-model run to its own private dataset <you>/oa-rlaif-dm-harness-results. Resumable.

    python -m dm_harness.push_results [--dry-run]

  dm_report.md / .tex / .json, figures/, plan.json, dm_speed.json, jah_config.yaml
  results/<task>/7-pplx-decider/, results/<task>/8-jah-all-merged-agent/   scores, decisions, alignment files
  replies/decider/, replies/jah-agent/                                     every class's answer (and the guard's record)
  jah_logs/                                    the proxy's step and decision logs (no prompt text), logs/, progress/
The comparison rows (5-all-merged, 6-all-merged-agent-thinking) are the first run's and stay only in
oa-rlaif-full-eval-results: they are not uploaded, and nothing is ever written to that repo from here.
"""

from __future__ import annotations

import argparse
import sys

from full_eval.push_results import REPO as FIRST_RUN
from full_eval.schedule import root

REPO = "oa-rlaif-dm-harness-results"
PATTERNS = ["dm_report.*", "figures/*", "plan.json", "dm_speed.json", "jah_config.yaml", "results/*/7-pplx-decider/*",
            "results/*/8-jah-all-merged-agent/*", "replies/decider/*", "replies/jah-agent/*", "jah_logs/*", "logs/*",
            "progress/*", "episodes/*/manifest.json"]
CARD = """---
license: other
license_name: mixed-private
tags: [ontology-matching, oa-rlaif, decision-model, top-k-evaluation, private]
---
# oa-rlaif-dm-harness-results (private)

A decision model controls the OA-RLAIF All-merged Gemma 4 E4B agent (github.com/AGR19/OA-RLAIF, branch
`dm_harness`, `dm_harness/`). Two arms on every source class with a partner in the reference (no held-out view) and
the complete SNOMED CT -> ICD-11 map, scored with the Top-k evaluation (k = 1, 3, 5):

- **7-pplx-decider**: perplexity-ai/pplx-decider-v1.2 alone, one candidate question per class.
- **8-jah-all-merged-agent**: JevAgentHarness (arXiv:2610.21412) routes each model call of All merged +
  mini-SWE-agent to thinking on (pplx-decider's p(hard) >= 0.75) or off; a NONE or invalid submission is re-checked by
  pplx-decider (tau 0.5) and redone on the strong tier with its hint. Thresholds fixed before the run.

The comparison rows (All merged, All merged + mini-SWE-agent) are read from `oa-rlaif-full-eval-results`, which this
run never changes. Start with `dm_report.md`. **Licensing:** contains SNOMED CT-derived mappings (affiliate licence)
and UMLS-derived Bio-ML references. Private research use only; never make this repo public.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dm_harness.push_results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if REPO == FIRST_RUN:
        raise SystemExit(f"refusing to write to {FIRST_RUN}")
    from fnmatch import fnmatch

    files = [p for p in root().rglob("*") if p.is_file()
             and any(fnmatch(p.relative_to(root()).as_posix(), pat) for pat in PATTERNS)]
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
                            allow_patterns=PATTERNS)
    api.upload_file(repo_id=repo, repo_type="dataset", path_or_fileobj=CARD.encode(), path_in_repo="README.md")
    print(f"pushed {repo} (private)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

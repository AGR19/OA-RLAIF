"""The comparison rows, read only: the All-merged units of the first full run, from <you>/oa-rlaif-full-eval-results.

    python -m dm_harness.pull_reference

Downloads results/<task>/5-all-merged/scores.json and results/<task>/6-all-merged-agent-thinking/scores.json into
outputs/full_eval/results/, where dm_harness.report reads them next to the two new arms. Nothing is written to that
repo, and dm_harness.push_results never uploads these files (they stay the first run's).
"""

from __future__ import annotations

import sys

from full_eval.push_results import REPO as FIRST_RUN
from full_eval.schedule import root

REFERENCE_ROWS = ("5-all-merged", "6-all-merged-agent-thinking")
PATTERNS = [f"results/*/{folder}/scores.json" for folder in REFERENCE_ROWS]


def main() -> int:
    from huggingface_hub import snapshot_download

    from oa_rlaif.data.hf_push import namespace

    repo = f"{namespace()}/{FIRST_RUN}"
    print(f"[reference] {repo} (read only): {', '.join(REFERENCE_ROWS)}", flush=True)
    snapshot_download(repo, repo_type="dataset", local_dir=str(root()), allow_patterns=PATTERNS, max_workers=8)
    n = sum(len(list((root() / "results").glob(f"*/{f}/scores.json"))) for f in REFERENCE_ROWS)
    print(f"[reference] {n} units (expected 26: 2 rows x 13 tasks)", flush=True)
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())

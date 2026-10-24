"""GPU box: download only what the full evaluation reads, from the private Hugging Face repos.

    python -m full_eval.pull_inputs

  <you>/oa-rlaif-raw-data (dataset)        -> data/full_eval_inputs/  (raw ontology folders at its root, bio-ml/,
                                             profiles/); not downloaded: Complex-OM except hydrography (5.9 GB, not
                                             evaluated), the SNOMED RF2 Full release (2.4 GB; the Snapshot is used),
                                             ICD-10-CM
  <you>/oa-rlaif-icd-bridge-data           -> data/icd/triangulated.jsonl (the ICD-10-composed SNOMED -> ICD-11 links)
  <you>/oa-rlaif-grpo-data                 -> data/eval/heldout_anatomy.jsonl       } only for the weight check of
  <you>/oa-rlaif-heldout-eval-results      -> outputs/full_eval/earlier/<model>/   } the re-merged OPSD and GRPO
                                             models (the earlier run's replies); never scored or reported
  <you>/oa-rlaif-full-eval-results         -> outputs/full_eval/, only with FULL_EVAL_RERUN=<tags>: a copy of the
                                             first run without what the listed methods produced and without the
                                             files rebuilt at the end (rerun_ignore). The other methods' units count
                                             as done, so only the listed methods run again; the report has every
                                             row, and full_eval.push_results uploads the whole set to
                                             <you>/oa-rlaif-fixed-full-eval-results. The first run's repo is only read
It also writes OA_RAW_DIR and OA_BIOML_DIR to .env and puts the IoTSec -> UCO pair and the profile texts where the
code expects them.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from oa_rlaif.paths import ROOT, data_dir, out_dir

RAW_PATTERNS = [
    "anatomy-dataset-2/*", "DH-benchmark-main/**", "DH-benchmark-multiling-main/**", "dataset-2015-testing/**",
    "biodiv/*", "complex-OM-benchmark-main/hydrography/**", "ICD-10 to SNOWMEDCT/icd11.owl",
    "ICD-10 to SNOWMEDCT/snomed-ct-us-*.owl", "ICD-10 to SNOWMEDCT/ICD 10 to 11 Mapings/**",
    "ICD-10 to SNOWMEDCT/SnomedCT_*/Snapshot/**", "ICD-10 to SNOWMEDCT/SnomedCT_*/*.json",
    "cybersecurity/**", "biodiv-2026/**", "bio-ml/**", "external/cyber/**", "profiles/**", "ontology_profiles.json",
]


def set_env(values: dict[str, str]) -> None:
    path = ROOT / ".env"
    lines = path.read_text().splitlines() if path.exists() else []
    lines = [line for line in lines if line.split("=", 1)[0] not in values] + [f"{k}={v}" for k, v in values.items()]
    path.write_text("\n".join(lines) + "\n")


def rerun_ignore(rerun: list[str]) -> list[str]:
    """What a rerun does not copy from the first run: everything the rerun methods produced (they run again; their old
    progress files would also mix the two runs' timings), the files rebuilt at the end, and this box's own logs."""
    from .schedule import METHODS, folder_name

    unknown = [t for t in rerun if t not in METHODS]
    if unknown:
        raise SystemExit(f"FULL_EVAL_RERUN: unknown method tags {unknown}; choose from {METHODS}")
    out = ["report.*", "figures/*", "plan.json", "episodes/*", "README.md", ".gitattributes", "logs/setup.log",
           "logs/inputs.log", "logs/models.log", "logs/episodes.log", "logs/push_results.log"]
    for t in rerun:
        out += [f"results/*/{folder_name(t)}/*", f"replies/{t}/*", f"progress/{t}.*", f"logs/infer_{t}_*",
                f"logs/score_{t}_*", f"verify_{t}.json"]
    return out


def main() -> int:
    from huggingface_hub import hf_hub_download, snapshot_download

    from oa_rlaif.data.hf_push import namespace

    ns = namespace()
    inputs = data_dir() / "full_eval_inputs"
    print(f"[inputs] {ns}/oa-rlaif-raw-data -> {inputs}", flush=True)
    snapshot_download(f"{ns}/oa-rlaif-raw-data", repo_type="dataset", local_dir=str(inputs), allow_patterns=RAW_PATTERNS,
                      max_workers=16)
    set_env({"OA_RAW_DIR": str(inputs.relative_to(ROOT)), "OA_BIOML_DIR": str((inputs / "bio-ml").relative_to(ROOT))})
    cyber = data_dir() / "external" / "cyber"
    if (inputs / "external" / "cyber").exists():
        shutil.copytree(inputs / "external" / "cyber", cyber, dirs_exist_ok=True)
    shutil.copy2(inputs / "ontology_profiles.json", data_dir() / "ontology_profiles.json")
    print(f"[inputs] {ns}/oa-rlaif-grpo-data: eval/heldout_anatomy.jsonl (weight check only)", flush=True)
    snapshot_download(f"{ns}/oa-rlaif-grpo-data", repo_type="dataset", local_dir=str(data_dir()),
                      allow_patterns=["eval/heldout_anatomy.jsonl"])
    print(f"[inputs] {ns}/oa-rlaif-icd-bridge-data: icd/triangulated.jsonl", flush=True)
    snapshot_download(f"{ns}/oa-rlaif-icd-bridge-data", repo_type="dataset", local_dir=str(data_dir()),
                      allow_patterns=["icd/triangulated.jsonl", "icd/triangulation_stats.json"])
    earlier = out_dir() / "full_eval" / "earlier"
    for tag in ("base", "sft", "sdft", "grpo"):
        try:
            path = hf_hub_download(f"{ns}/oa-rlaif-heldout-eval-results", f"{tag}/heldout_anatomy.predictions.jsonl",
                                   repo_type="dataset", local_dir=str(earlier))
            print(f"[inputs] earlier predictions: {Path(path).relative_to(earlier)}", flush=True)
        except Exception as error:  # noqa: BLE001 (the check is then reported as not done)
            print(f"[inputs] no earlier predictions for {tag}: {type(error).__name__}", flush=True)
    rerun = [t.strip() for t in os.environ.get("FULL_EVAL_RERUN", "").split(",") if t.strip()]
    if rerun:
        from .push_results import REPO, target_repo
        from .schedule import root

        print(f"[inputs] {ns}/{REPO} (read only): the first run without {', '.join(rerun)}; the results go to "
              f"{ns}/{target_repo()}", flush=True)
        snapshot_download(f"{ns}/{REPO}", repo_type="dataset", local_dir=str(root()), ignore_patterns=rerun_ignore(rerun),
                          max_workers=16)
        kept = len(list((root() / "results").glob("*/*/scores.json")))
        print(f"[inputs] {kept} finished units copied; running again: {', '.join(rerun)}", flush=True)
    missing = [p for p in ("icd/triangulated.jsonl",)
               if not (data_dir() / p).exists()]
    if missing:
        raise SystemExit(f"[inputs] missing after download: {missing}")
    print("[inputs] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

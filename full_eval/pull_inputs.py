"""GPU box: download only what the full evaluation reads, from the private Hugging Face repos.

    python -m full_eval.pull_inputs

  <you>/oa-rlaif-raw (dataset)          -> data/full_eval_inputs/  (raw ontology folders at its root, bio-ml/, profiles/)
       not downloaded: Complex-OM except hydrography (5.9 GB, not evaluated), the SNOMED RF2 Full release (2.4 GB; the
       Snapshot is used), ICD-10-CM
  <you>/oa-rlaif-grpo (dataset)         -> data/eval/heldout_anatomy.jsonl, data/splits/ (the 80/20 split)
  <you>/oa-rlaif-icd-triangulation      -> data/icd/triangulated.jsonl (the ICD-10-composed SNOMED -> ICD-11 links)
  <you>/oa-rlaif-results                -> outputs/full_eval/earlier/<model>/heldout_anatomy.predictions.jsonl
                                           (the earlier predictions the rebuilt models are checked against)
It also writes OA_RAW_DIR and OA_BIOML_DIR to .env and puts the IoTSec -> UCO pair and the profile texts where the
code expects them.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from oa_rlaif.paths import ROOT, data_dir, out_dir

RAW_PATTERNS = [
    "anatomy-dataset-2/*", "DH-benchmark-main/**", "DH-benchmark-multiling-main/**", "dataset-2015-testing/**",
    "biodiv/*", "complex-OM-benchmark-main/hydrography/**", "ICD-10 to SNOWMEDCT/icd11.owl",
    "ICD-10 to SNOWMEDCT/snomed-ct-us-*.owl", "ICD-10 to SNOWMEDCT/ICD 10 to 11 Mapings/**",
    "ICD-10 to SNOWMEDCT/SnomedCT_*/Snapshot/**", "ICD-10 to SNOWMEDCT/SnomedCT_*/*.json",
    "bio-ml/**", "external/cyber/**", "profiles/**", "ontology_profiles.json",
]


def set_env(values: dict[str, str]) -> None:
    path = ROOT / ".env"
    lines = path.read_text().splitlines() if path.exists() else []
    lines = [line for line in lines if line.split("=", 1)[0] not in values] + [f"{k}={v}" for k, v in values.items()]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    from huggingface_hub import hf_hub_download, snapshot_download

    from oa_rlaif.data.hf_push import namespace

    ns = namespace()
    inputs = data_dir() / "full_eval_inputs"
    print(f"[inputs] {ns}/oa-rlaif-raw -> {inputs}", flush=True)
    snapshot_download(f"{ns}/oa-rlaif-raw", repo_type="dataset", local_dir=str(inputs), allow_patterns=RAW_PATTERNS,
                      max_workers=16)
    set_env({"OA_RAW_DIR": str(inputs.relative_to(ROOT)), "OA_BIOML_DIR": str((inputs / "bio-ml").relative_to(ROOT))})
    cyber = data_dir() / "external" / "cyber"
    if (inputs / "external" / "cyber").exists():
        shutil.copytree(inputs / "external" / "cyber", cyber, dirs_exist_ok=True)
    shutil.copy2(inputs / "ontology_profiles.json", data_dir() / "ontology_profiles.json")
    print(f"[inputs] {ns}/oa-rlaif-grpo: eval/heldout_anatomy.jsonl, splits/", flush=True)
    snapshot_download(f"{ns}/oa-rlaif-grpo", repo_type="dataset", local_dir=str(data_dir()),
                      allow_patterns=["eval/heldout_anatomy.jsonl", "splits/*"])
    print(f"[inputs] {ns}/oa-rlaif-icd-triangulation: icd/triangulated.jsonl", flush=True)
    snapshot_download(f"{ns}/oa-rlaif-icd-triangulation", repo_type="dataset", local_dir=str(data_dir()),
                      allow_patterns=["icd/triangulated.jsonl", "icd/triangulation_stats.json"])
    earlier = out_dir() / "full_eval" / "earlier"
    for tag in ("base", "sft", "sdft", "grpo"):
        try:
            path = hf_hub_download(f"{ns}/oa-rlaif-results", f"{tag}/heldout_anatomy.predictions.jsonl",
                                   repo_type="dataset", local_dir=str(earlier))
            print(f"[inputs] earlier predictions: {Path(path).relative_to(earlier)}", flush=True)
        except Exception as error:  # noqa: BLE001 (the check is then reported as not done)
            print(f"[inputs] no earlier predictions for {tag}: {type(error).__name__}", flush=True)
    missing = [p for p in ("splits/components.tsv", "icd/triangulated.jsonl", "eval/heldout_anatomy.jsonl")
               if not (data_dir() / p).exists()]
    if missing:
        raise SystemExit(f"[inputs] missing after download: {missing}")
    print("[inputs] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

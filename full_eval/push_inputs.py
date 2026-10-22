"""Mac -> private HF dataset <you>/oa-rlaif-raw: everything the full evaluation reads on the GPU box. Resumable.

    PYTHONPATH=src:. .venv/bin/python full_eval/push_inputs.py [--dry-run]

  (repo root)       the supplied 'Raw Ontology Folders', without PDFs (the DRAL-OA paper and the textbook); with
                    --skip-raw only the folder added later, cybersecurity/ (IoTSec -> UCO)
  bio-ml/           the Bio-ML 2026 download (references, candidate lists, NCIT/DOID/FMA)
  external/cyber/   IoTSec -> UCO (zero-shot pair, built by oa_rlaif.data.cyber)
  ontology_profiles.json   the modelling-profile texts every prompt carries (as in training)
  profiles/<key>/   the parsed-ontology cache for every ontology the evaluation reads (parsing NCIT and SNOMED CT again
                    would cost GPU-box time); profiles/index.json maps raw-relative path -> cache folder and size

Private, checked after creation: SNOMED CT (affiliate licence) and UMLS-derived Bio-ML references are inside.
"""

from __future__ import annotations

import argparse
import json
import sys

from oa_rlaif.data.profiles import cache_key, is_cached
from oa_rlaif.paths import bioml_dir, data_dir, raw_dir, relative_to_raw

REPO = "oa-rlaif-raw"
IGNORE = ["*.pdf", "**/*.pdf", ".DS_Store", "**/.DS_Store", "**/__pycache__/**"]
CARD = """---
license: other
license_name: mixed-private
tags: [ontology-matching, oa-rlaif, private]
---
# oa-rlaif-raw (private)

Inputs of the OA-RLAIF full-alignment evaluation (github.com/AGR19/OA-RLAIF, `full_eval/`): the supplied raw ontology
folders (without PDFs), the Bio-ML 2026 download, the IoTSec -> UCO pair, the modelling profiles and the parsed-ontology
cache. **Licensing:** contains SNOMED CT (affiliate licence) and UMLS-derived Bio-ML references. Private research use
only; never make this repo public.
"""


def profile_index(task_list) -> dict[str, dict]:
    from full_eval.tasks import ontology_paths

    index, missing = {}, []
    for path in ontology_paths(task_list):
        if not is_cached(path):
            missing.append(relative_to_raw(path))
            continue
        index[relative_to_raw(path)] = {"key": cache_key(path), "size": path.stat().st_size}
    if missing:
        raise SystemExit(f"not parsed yet (run python -m oa_rlaif.data.profiles first): {missing}")
    return index


def main(argv: list[str] | None = None) -> int:
    from full_eval.tasks import tasks

    parser = argparse.ArgumentParser(prog="push_inputs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-raw", action="store_true", help="only the extras (raw already uploaded)")
    args = parser.parse_args(argv)
    raw, bioml, cyber = raw_dir(), bioml_dir(), data_dir() / "external" / "cyber"
    index = profile_index(tasks())
    raw_files = [p for p in raw.rglob("*") if p.is_file() and p.suffix.lower() != ".pdf" and p.name != ".DS_Store"]
    profile_bytes = sum(f.stat().st_size for entry in index.values() for f in (data_dir() / "profiles" / entry["key"]).rglob("*") if f.is_file())
    print(f"raw: {len(raw_files):,} files, {sum(p.stat().st_size for p in raw_files) / 1e9:.2f} GB (PDFs left out)")
    print(f"bio-ml: {sum(p.stat().st_size for p in bioml.rglob('*') if p.is_file()) / 1e9:.2f} GB; "
          f"profiles: {len(index)} ontologies, {profile_bytes / 1e9:.2f} GB")
    if args.dry_run:
        return 0
    from huggingface_hub import HfApi

    from oa_rlaif.data.hf_push import namespace

    api = HfApi()
    repo = f"{namespace()}/{REPO}"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:
        raise SystemExit(f"{repo} is not private; refusing to upload")
    if not args.skip_raw:
        api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=str(raw), private=True,
                                ignore_patterns=IGNORE)
    elif (raw / "cybersecurity").exists():  # added to the raw folders after the first upload
        api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(raw / "cybersecurity"),
                          path_in_repo="cybersecurity", ignore_patterns=IGNORE, commit_message="cybersecurity: IoTSec -> UCO")
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(bioml), path_in_repo="bio-ml",
                      ignore_patterns=IGNORE, commit_message="Bio-ML 2026 download")
    if cyber.exists():
        api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(cyber), path_in_repo="external/cyber",
                          ignore_patterns=IGNORE, commit_message="IoTSec -> UCO")
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(data_dir() / "profiles"), path_in_repo="profiles",
                      allow_patterns=[f"{e['key']}/*" for e in index.values()], commit_message="parsed-ontology cache")
    api.upload_file(repo_id=repo, repo_type="dataset", path_or_fileobj=json.dumps(index, indent=1).encode(),
                    path_in_repo="profiles/index.json")
    api.upload_file(repo_id=repo, repo_type="dataset", path_or_fileobj=str(data_dir() / "ontology_profiles.json"),
                    path_in_repo="ontology_profiles.json")
    api.upload_file(repo_id=repo, repo_type="dataset", path_or_fileobj=CARD.encode(), path_in_repo="README.md")
    print(f"pushed {repo} (private)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

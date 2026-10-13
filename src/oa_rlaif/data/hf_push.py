"""Push verified datasets (and later the adapters) to PRIVATE Hugging Face repos.

    python -m oa_rlaif.data.hf_push --datasets          # after `python -m oa_rlaif.data.verify` passes
    python -m oa_rlaif.data.hf_push --models            # on Lambda after training
    python -m oa_rlaif.data.hf_push --pull              # on Lambda: download the datasets into data/

Requires `hf auth login` by the user and HF_NAMESPACE in .env. Refuses to push unless verification passed.
Every repo is created private; licensed content (Springer datasource, SNOMED CT/UMLS) must stay private.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from ..paths import ROOT, data_dir, out_dir

DATASETS = {
    "oa-rlaif-datasource": ["knowledge"],
    "oa-rlaif-sdft": ["sdft", "facts", "complex"],
    "oa-rlaif-grpo": ["grpo", "eval", "episodes", "splits", "baselines", "raw", "external"],  # raw = Anatomy OWL for full mapping
    "oa-rlaif-dral-reward": ["dral"],
    "oa-rlaif-icd-triangulation": ["icd"],
}
# repo -> (folder, files to upload). The GGUF repo holds the final model: base E4B + SDFT + GRPO (DRAL-OA + A4B judge)
# merged and quantized; the f16 intermediate is not uploaded.
# The GGUF goes first: it is the deliverable, so a problem with a later repo cannot keep it off the Hub.
MODELS = {
    "oa-rlaif-e4b-gguf": ("outputs/release/oa-rlaif-e4b", ["*Q4_K_M.gguf", "Modelfile"]),
    "oa-rlaif-e4b-sdft": ("outputs/checkpoints/sdft/final", None),
    "oa-rlaif-e4b-grpo": ("outputs/checkpoints/grpo/final", None),
}
BASE_MODEL = "google/gemma-4-E4B-it"
# Our own model cards. The trainers write a README.md into each adapter folder whose metadata names the local folder
# the adapter was trained on (GRPO: base_model "outputs/merged/sdft"); the Hub rejects that, so it is not uploaded.
MODEL_CARDS = {
    "oa-rlaif-e4b-gguf": ("gguf", "The OA-RLAIF model: Gemma 4 E4B + Stage 1 OPSD + Stage 2 GRPO (DRAL-OA reward + A4B "
                                  "judge, i.e. RLAIF), merged and quantized to GGUF Q4_K_M.\n\n"
                                  "Run with Ollama: `ollama create oa-rlaif-e4b -f Modelfile`, then `ollama run oa-rlaif-e4b`."),
    "oa-rlaif-e4b-sdft": ("peft", "Stage 1 (OPSD, on-policy self-distillation) LoRA adapter for `google/gemma-4-E4B-it`. "
                                  "Apply it to the base model."),
    "oa-rlaif-e4b-grpo": ("peft", "Stage 2 (GRPO with the DRAL-OA reward and the A4B judge, i.e. RLAIF) LoRA adapter. "
                                  "It was trained on top of the OPSD model, not the plain base: first merge the "
                                  "`oa-rlaif-e4b-sdft` adapter into `google/gemma-4-E4B-it`, then apply this adapter. "
                                  "(`adapter_config.json` names the local folder of that merged model.)"),
}


def model_card(name: str) -> str:
    kind, text = MODEL_CARDS[name]
    library = "library_name: peft\n" if kind == "peft" else ""
    return (f"---\nlicense: other\nlicense_name: mixed-private\nbase_model: {BASE_MODEL}\n{library}"
            f"tags: [ontology-matching, opsd, grpo, rlaif, oa-rlaif{', gguf' if kind == 'gguf' else ''}]\n---\n"
            f"# {name} (private)\n\n{text}\n\nPart of OA-RLAIF (github.com/AGR19/OA-RLAIF, private).\n")
RESULTS = {"oa-rlaif-results": ("outputs/eval", None)}  # dataset repo, pushed last (after the SFT comparison)
# the research variant "all merged" has its own repo (data/push_all_merged.py); it never goes into the results repo
RESULTS_IGNORE = ["all_merged/*", "all_merged/**", "*all_merged*"]

CARD = """---
license: other
license_name: mixed-private
tags: [ontology-matching, sdft, grpo, oa-rlaif]
---
# {name} (private)

Part of OA-RLAIF (github.com/AGR19/OA-RLAIF, private). Built by `python -m oa_rlaif.data.build` and checked by
`python -m oa_rlaif.data.verify` (see docs/DATA_CARD.md in the code repo).

**Licensing.** Contains or derives from licensed sources: the OM datasource (Springer © 2007), SNOMED CT
(affiliate licence), UMLS-derived Bio-ML references. Private research use only; do not make this repo public.
"""


def namespace() -> str:
    ns = os.environ.get("HF_NAMESPACE", "").strip()
    if not ns:
        from huggingface_hub import HfApi
        try:
            ns = HfApi().whoami()["name"]
        except Exception:
            raise SystemExit("Run `hf auth login` (and optionally set HF_NAMESPACE in .env) first") from None
    return ns


def push_datasets() -> None:
    from huggingface_hub import HfApi

    report = json.loads((out_dir() / "coverage" / "verification.json").read_text())
    if not report.get("ok"):
        raise SystemExit("verification has not passed; run python -m oa_rlaif.data.verify")
    api, ns = HfApi(), namespace()
    for name, folders in DATASETS.items():
        repo = f"{ns}/{name}"
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        if not api.repo_info(repo, repo_type="dataset").private:
            raise SystemExit(f"{repo} is not private; refusing to upload")
        api.upload_file(path_or_fileobj=CARD.format(name=name).encode(), path_in_repo="README.md", repo_id=repo, repo_type="dataset")
        for folder in folders:
            path = data_dir() / folder
            if path.exists():
                api.upload_folder(folder_path=str(path), path_in_repo=folder, repo_id=repo, repo_type="dataset")
        api.upload_file(path_or_fileobj=str(ROOT / "docs" / "DATA_CARD.md"), path_in_repo="DATA_CARD.md", repo_id=repo, repo_type="dataset")
        print(f"pushed {repo} (private)")


def push_models(repos: dict | None = None, dry_run: bool = False) -> None:
    from fnmatch import fnmatch

    from huggingface_hub import HfApi

    api, ns = (None, "<namespace>") if dry_run else (HfApi(), namespace())
    missing = []
    for name, (folder, patterns) in (repos or MODELS).items():
        path = out_dir() / folder.split("/", 1)[1] if folder.startswith("outputs/") else ROOT / folder
        if not path.exists():
            missing.append(folder)
            print(f"skip {name}: {folder} missing", file=sys.stderr)
            continue
        ignore = ["README.md"] if name in MODEL_CARDS else RESULTS_IGNORE if name in RESULTS else None
        if dry_run:
            files = [p for p in path.rglob("*") if p.is_file() and (not patterns or any(fnmatch(p.name, pat) for pat in patterns))
                     and not (ignore and any(fnmatch(p.relative_to(path).as_posix(), pat) for pat in ignore))]
            print(f"[dry-run] would push {len(files)} files ({sum(f.stat().st_size for f in files) / 1e6:.1f} MB) "
                  f"to {ns}/{name} (private)")
            continue
        repo = f"{ns}/{name}"
        repo_type = "dataset" if name == "oa-rlaif-results" else "model"
        api.create_repo(repo, repo_type=repo_type, private=True, exist_ok=True)
        if not api.repo_info(repo, repo_type=repo_type).private:
            raise SystemExit(f"{repo} is not private; refusing to upload")
        api.upload_folder(folder_path=str(path), repo_id=repo, repo_type=repo_type, allow_patterns=patterns,
                          ignore_patterns=ignore)
        if name in MODEL_CARDS:
            api.upload_file(path_or_fileobj=model_card(name).encode(), path_in_repo="README.md", repo_id=repo,
                            repo_type=repo_type)
        print(f"pushed {repo} (private)")
    if missing:
        raise SystemExit(f"not pushed (missing): {missing}")


def pull_datasets() -> None:
    from huggingface_hub import snapshot_download

    ns = namespace()
    for name in DATASETS:
        snapshot_download(f"{ns}/{name}", repo_type="dataset", local_dir=str(data_dir()))
        print(f"pulled {ns}/{name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", action="store_true")
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--results", action="store_true", help="push outputs/eval to the private results dataset")
    parser.add_argument("--dry-run", action="store_true", help="list what --models/--results would upload")
    args = parser.parse_args()
    if args.datasets:
        push_datasets()
    if args.models:
        push_models(dry_run=args.dry_run)
    if args.pull:
        pull_datasets()
    if args.results:
        push_models(RESULTS, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

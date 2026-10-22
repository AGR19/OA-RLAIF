"""The five models of the earlier evaluation, rebuilt on the GPU box from the private Hugging Face repos.

    python -m full_eval.models grpo sft all_merged sdft base      # in this order (the run's order)

  base        google/gemma-4-E4B-it
  sdft        base + the OPSD LoRA adapter <you>/oa-rlaif-e4b-opsd-weights, merged (the same merge as scripts/40_sdft.sh)
  grpo        the OPSD model + the GRPO adapter <you>/oa-rlaif-e4b-grpo-weights, merged (GRPO was trained on the merged OPSD model)
  sft         <you>/oa-rlaif-e4b-sft-model-safetensors (pushed by scripts/push_sft.sh)
  all_merged  <you>/oa-rlaif-e4b-all-combined-model-safetensors (weights only; its eval/ and report/ folders are not downloaded)

Merged models go to outputs/full_eval_models/<tag>/ with a READY file; a second call returns at once.
The two rebuilt merges are checked against the earlier run's own predictions (full_eval.infer --verify).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from oa_rlaif.paths import out_dir

BASE = "google/gemma-4-E4B-it"
WEIGHTS = ["*.json", "*.safetensors", "*.jinja", "tokenizer*", "*.model"]
SOURCES = {
    "base": ("hub", BASE, None),
    "sdft": ("merge", "oa-rlaif-e4b-opsd-weights", "base"),
    "grpo": ("merge", "oa-rlaif-e4b-grpo-weights", "sdft"),
    "sft": ("hub", "oa-rlaif-e4b-sft-model-safetensors", None),
    "all_merged": ("hub", "oa-rlaif-e4b-all-combined-model-safetensors", None),
}


def models_dir() -> Path:
    return out_dir() / "full_eval_models"


def model_path(tag: str) -> Path:
    return models_dir() / tag


def ready(tag: str) -> bool:
    return (model_path(tag) / "READY").exists()


def prepare(tag: str) -> Path:
    from huggingface_hub import snapshot_download

    from oa_rlaif.data.hf_push import namespace

    out = model_path(tag)
    if ready(tag):
        return out
    kind, name, base = SOURCES[tag]
    repo = name if "/" in name else f"{namespace()}/{name}"
    if kind == "hub":
        snapshot_download(repo, local_dir=str(out), allow_patterns=WEIGHTS)
    else:
        from oa_rlaif.export.release import merge

        base_path = prepare(base)
        adapter = models_dir() / "adapters" / tag
        snapshot_download(repo, local_dir=str(adapter), allow_patterns=["adapter_*", "*.json"])
        partial = out.with_name(tag + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        merge(adapter, partial, base=str(base_path))
        shutil.rmtree(out, ignore_errors=True)
        partial.rename(out)
    (out / "READY").write_text(repo)
    return out


def main(argv: list[str] | None = None) -> int:
    for tag in (argv if argv is not None else sys.argv[1:]) or list(SOURCES):
        print(f"[models] {tag}: {prepare(tag)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

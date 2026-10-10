"""Shared training utilities: config, model/LoRA construction, datasets, callbacks (progress, budget, samples)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ..budget import BudgetGuard, trainer_callback
from ..io import read_yaml
from ..paths import ROOT, data_dir, out_dir
from ..progress import _update, fmt_duration


def load_config() -> dict:
    cfg = read_yaml(ROOT / "configs" / "train.yaml")
    dry = out_dir() / "dry_run.json"
    if dry.exists():
        cfg["dry_run"] = json.loads(dry.read_text())
    return cfg


def offline_after_download() -> None:
    """Everything runs locally once weights are cached (only the A4B judge API goes out)."""
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    if os.environ.get("OA_ALLOW_HUB_DOWNLOAD") != "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_model(cfg: dict, path: str | None = None):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = path or cfg["model"]["id"]
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=getattr(torch, cfg["model"]["dtype"]), attn_implementation=cfg["model"]["attn_implementation"])
    processor = AutoProcessor.from_pretrained(cfg["model"]["id"])
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def lora_config(cfg: dict):
    from peft import LoraConfig

    return LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
                      target_modules=cfg["lora"]["target_regex"], bias="none", task_type="CAUSAL_LM")


def load_jsonl_dataset(path: Path):
    from datasets import load_dataset

    return load_dataset("json", data_files=str(path), split="train")


def progress_callback(stage: str, guard: BudgetGuard, dump_dir: Path | None = None, dump_every: int = 0):
    """Live ETA + $ meter in the logs and outputs/progress.json; optional completion dumps for manual review."""
    from transformers import TrainerCallback

    class ProgressCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            self.start = time.time()
            _update(stage, status="running", started=self.start, fraction=0.0)

        def on_step_end(self, args, state, control, **kwargs):
            if state.max_steps and state.global_step % max(1, args.logging_steps) == 0:
                fraction = state.global_step / state.max_steps
                elapsed = time.time() - self.start
                eta = elapsed / fraction - elapsed if fraction else None
                _update(stage, fraction=fraction)
                print(f"[{stage}] step {state.global_step}/{state.max_steps} ({fraction:.1%}) "
                      f"elapsed {fmt_duration(elapsed)} ETA {fmt_duration(eta)} | spent ${guard.spent():.2f} "
                      f"of ${guard.budget_usd:.0f} (stop ${guard.stop_usd:.0f})", flush=True)

        def on_train_end(self, args, state, control, **kwargs):
            _update(stage, status="done", seconds=time.time() - self.start, fraction=1.0)

    return [ProgressCallback(), trainer_callback(guard, stage)]


def sdft_paths() -> tuple[Path, Path]:
    return data_dir() / "sdft" / "train.jsonl", data_dir() / "sdft" / "test.jsonl"


def grpo_paths() -> tuple[Path, Path]:
    return data_dir() / "grpo" / "train.jsonl", data_dir() / "grpo" / "test.jsonl"

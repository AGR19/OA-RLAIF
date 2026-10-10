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


SMOKE = os.environ.get("OA_SMOKE") == "1"  # Mac end-to-end test: tiny model, CPU, no vLLM (scripts/smoke_mac.sh)


def load_config() -> dict:
    cfg = read_yaml(ROOT / "configs" / "train.yaml")
    if os.environ.get("OA_MODEL_ID"):
        cfg["model"]["id"] = os.environ["OA_MODEL_ID"]
    if SMOKE:
        from ..dev.cpu_kernels import install
        install()  # TRL's SFT loss kernel needs Triton (Linux only); the Mac uses a plain-PyTorch stand-in
        cfg["smoke"] = True
        cfg["model"]["dtype"] = "float32"
        cfg["sdft"].update(per_device_batch=2, grad_accum=1, max_completion_length=24, logging_steps=1, save_steps=10_000)
        cfg["grpo"].update(per_device_batch=4, grad_accum=1, num_generations=2, max_completion_length=32,
                           logging_steps=1, save_steps=10_000, sample_dump_every=1)
    dry = out_dir() / "dry_run.json"
    if dry.exists():
        cfg["dry_run"] = json.loads(dry.read_text())
    return cfg


def plan_steps(cfg: dict, stage: str, prompts: int, prompts_per_step: float, epochs: float) -> int:
    """Optimizer steps for a stage: the configured max_steps, else min(full epochs, steps that fit the time slot).

    The time slot (`time_budget_hours`) uses the seconds per step measured by scripts/35_dry_run.sh on this machine,
    with a 10% margin for evaluation, checkpointing and merging. The plan is written to outputs/step_plan.json.
    """
    import math

    s = cfg[stage]
    if s.get("max_steps", -1) and s["max_steps"] > 0:
        return int(s["max_steps"])
    if cfg.get("smoke"):
        return int(os.environ.get("OA_SMOKE_STEPS", "3"))
    full = math.ceil(prompts / max(1e-9, prompts_per_step) * epochs)
    measured = (cfg.get("dry_run") or {}).get(f"{stage}_seconds_per_step")
    if not measured:
        print(f"[plan] {stage}: no dry-run timing; running the full {full} steps (budget guard still applies)")
        return full
    hours_slot = s.get("time_budget_hours", 6.0) * slot_scale(cfg)
    fits = int(hours_slot * 3600 * 0.9 / measured)
    steps = max(1, min(full, fits))
    plan = {"stage": stage, "prompts": prompts, "prompts_per_step": prompts_per_step, "epochs": epochs,
            "full_steps": full, "seconds_per_step": round(measured, 2), "hours_slot": round(hours_slot, 3),
            "steps": steps, "epochs_covered": round(steps * prompts_per_step / max(1, prompts), 3)}
    path = out_dir() / "step_plan.json"
    plans = json.loads(path.read_text()) if path.exists() else {}
    plans[stage] = plan
    path.write_text(json.dumps(plans, indent=1))
    print(f"[plan] {stage}: {steps} of {full} steps ({plan['epochs_covered']} epochs) in ~{steps * measured / 3600:.1f} h")
    return steps


def slot_scale(cfg: dict) -> float:
    """Share of the training time slots (OPSD + GRPO + SFT) that the rest of the budget pays for; 1.0 = full slots.

    Decided once, at the end of the dry run, from the real hourly price (OA_GPU_USD_PER_HOUR):
        hours left = (stop - spent) / price;  slot_scale = min(1, (hours left - fixed hours) / sum of slots)
    The fixed hours are the stages that do not train (evaluations, export, uploads; `budget_plan` in train.yaml).
    At Lambda's 4xB200 price the full slots fit (1.0). On a pricier node the slots shrink so the run still ends
    with the GGUF, the evaluations and the SFT row inside the budget. Stored in outputs/step_plan.json["budget"],
    so every later stage uses the same plan. OA_SLOT_SCALE overrides it.
    """
    if os.environ.get("OA_SLOT_SCALE"):
        return float(os.environ["OA_SLOT_SCALE"])
    path = out_dir() / "step_plan.json"
    plans = json.loads(path.read_text()) if path.exists() else {}
    if "budget" in plans:
        return float(plans["budget"]["slot_scale"])
    guard = BudgetGuard()
    if not guard.billable:
        return 1.0
    b = cfg.get("budget_plan", {})
    slots = sum(cfg[k]["time_budget_hours"] for k in ("sdft", "grpo", "sft"))
    hours_left = (guard.stop_usd - guard.spent()) / guard.rate_usd_per_hour
    scale = max(0.0, min(1.0, (hours_left - b.get("fixed_hours", 4.5)) / slots))
    plans["budget"] = {"usd_per_hour": guard.rate_usd_per_hour, "spent_usd": round(guard.spent(), 2),
                       "stop_usd": guard.stop_usd, "hours_left": round(hours_left, 2),
                       "fixed_hours": b.get("fixed_hours", 4.5), "slot_hours_full": slots,
                       "slot_scale": round(scale, 4), "training_hours": round(scale * slots, 2)}
    if scale < b.get("min_slot_scale", 0.05):
        raise SystemExit(
            f"[plan] budget too small: at ${guard.rate_usd_per_hour:.2f}/h, ${guard.stop_usd - guard.spent():.0f} left buys "
            f"{hours_left:.1f} h, and the evaluations, export and uploads need {b.get('fixed_hours', 4.5)} h. "
            f"Training would get {scale * slots * 60:.0f} min. Check OA_GPU_USD_PER_HOUR, use fewer GPUs or a cheaper "
            f"node, or set OA_SLOT_SCALE to force a scale.")
    path.write_text(json.dumps(plans, indent=1))
    print(f"[plan] budget: ${guard.rate_usd_per_hour:.2f}/h, {hours_left:.1f} h left -> training slots x{scale:.2f} "
          f"({scale * slots:.1f} of {slots:.0f} h)")
    return scale


def record_dry_run(**values) -> None:
    """Merge throughput measurements into outputs/dry_run.json (SDFT and GRPO write separately)."""
    path = out_dir() / "dry_run.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(values)
    path.write_text(json.dumps(data, indent=1))


def out_path(relative: str) -> Path:
    """Config paths under outputs/ follow OA_OUT_DIR (the smoke test writes to outputs/smoke)."""
    return out_dir() / relative.split("/", 1)[1] if relative.startswith("outputs/") else ROOT / relative


def runtime_kwargs(cfg: dict, generation: bool = True) -> dict:
    """GPU box: bf16, gradient checkpointing and the vLLM rollout server. Smoke test: plain CPU settings."""
    if cfg.get("smoke"):
        return {"bf16": False, "gradient_checkpointing": False, **({"use_vllm": False} if generation else {})}
    v = cfg["vllm"]
    vllm = {"use_vllm": True, "vllm_mode": "server", "vllm_server_host": v["server_host"], "vllm_server_port": v["server_port"]}
    return {"bf16": True, "gradient_checkpointing": True, **(vllm if generation else {})}


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
    try:
        processor = AutoProcessor.from_pretrained(cfg["model"]["id"])
        tokenizer = getattr(processor, "tokenizer", processor)
    except Exception:  # a text-only checkpoint (the smoke-test model) has no image/audio processor configs
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def lora_config(cfg: dict):
    from peft import LoraConfig

    return LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
                      target_modules=cfg["lora"]["target_regex"], bias="none", task_type="CAUSAL_LM")


def load_jsonl_dataset(path: Path):
    from datasets import load_dataset

    data = load_dataset("json", data_files=str(path), split="train")
    if SMOKE:  # a small, mixed sample of the real rows
        data = data.shuffle(seed=5).select(range(min(len(data), int(os.environ.get("OA_SMOKE_ROWS", "24")))))
    return data


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

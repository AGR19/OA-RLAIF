"""Stage 1: on-policy self-distillation (SDFT/OPSD) with TRL's SDFTTrainer — knowledge and skill into the weights.

    accelerate launch --num_processes 3 -m oa_rlaif.train.sdft            # GPUs 0-2; vLLM server on GPU 3
    python -m oa_rlaif.train.sdft --mode sft                              # SFT comparison row (off-policy)
    python -m oa_rlaif.train.sdft --dry-run --max-steps 20                # throughput measurement

The student generates from `prompt` only; the frozen teacher (initial weights) re-scores the student's tokens
with `privileged_context` appended (passage, entity card, gold decision with evidence, ICD bridge rows).
"""

from __future__ import annotations

import argparse
import json
import os
import time

from ..budget import BudgetGuard
from ..paths import ROOT, out_dir
from .common import (load_config, load_jsonl_dataset, load_model, lora_config, offline_after_download, out_path,
                     plan_steps, progress_callback, record_dry_run, runtime_kwargs, sdft_paths, steps_for_money,
                     write_plan)


def sdft_args(cfg: dict, max_steps: int):
    from trl.experimental.sdft import SDFTConfig

    s, v = cfg["sdft"], cfg["vllm"]
    return SDFTConfig(
        output_dir=str(out_path(s["output_dir"])), num_train_epochs=s["epochs"], max_steps=max_steps,
        learning_rate=s["learning_rate"], warmup_steps=s["warmup_ratio"], lr_scheduler_type=s["lr_scheduler"],  # transformers 5: a float < 1 is a ratio
        per_device_train_batch_size=s["per_device_batch"], gradient_accumulation_steps=s["grad_accum"],
        logging_steps=s["logging_steps"], save_steps=s["save_steps"],
        save_total_limit=3, report_to=["tensorboard"], max_prompt_length=s["max_prompt_length"],
        max_completion_length=s["max_completion_length"], num_generations=s["num_generations"],
        temperature=s["temperature"], chat_template_kwargs=cfg["model"]["chat_template_kwargs"],
        **runtime_kwargs(cfg),
        teacher_model_kind=s["teacher_model_kind"], distillation_mode=s["distillation_mode"],
        distillation_topk=s["distillation_topk"], distillation_alpha=s["distillation_alpha"],
        teacher_prompt_template=s["teacher_prompt_template"])


def plan_sft_steps(cfg: dict, train) -> int:
    """SFT runs as many steps as OPSD (same rows, same batch: a fair comparison), capped by what its money buys."""
    plans = json.loads((out_dir() / "step_plan.json").read_text()) if (out_dir() / "step_plan.json").exists() else {}
    s = cfg["sdft"]
    rows_per_step = s["per_device_batch"] * s["grad_accum"] * max(1, int(os.environ.get("WORLD_SIZE", "1")))
    opsd = plans.get("sdft", {}).get("steps") or plan_steps(cfg, "sdft", len(train), rows_per_step, s["epochs"])
    measured = (cfg.get("dry_run") or {}).get("sft_seconds_per_step")
    if not measured:
        write_plan("sft", {"stage": "sft", "steps": opsd, "opsd_steps": opsd})
        return opsd
    steps, plan = steps_for_money("sft", measured, opsd)
    plan["opsd_steps"] = opsd
    write_plan("sft", plan)
    print(f"[plan] sft: {steps} steps (OPSD runs {opsd})")
    return steps


def run_sft(cfg: dict, train, max_steps: int, guard: BudgetGuard, dry_run: bool = False):
    """Off-policy comparison row: cross-entropy on the teacher's demonstration (the gold completion)."""
    from trl import SFTConfig, SFTTrainer

    def to_messages(row):
        demo = row["privileged_context"].split("Example of a correct reply:\n", 1)[-1]
        if "Reference answer:" in row["privileged_context"]:
            demo = row["privileged_context"].rsplit("Reference answer:", 1)[-1].strip()
        # conversational prompt/completion: the loss covers the completion only, without chat-template markers
        return {"prompt": row["prompt"], "completion": [{"role": "assistant", "content": demo}]}

    data = train.map(to_messages, remove_columns=train.column_names)
    model, tokenizer = load_model(cfg)
    s = cfg["sdft"]
    # Same optimisation as OPSD (fair comparison): learning rate, cosine schedule, warm-up, batch. The sequence limit is
    # OPSD's prompt + completion limits; TRL's default (1,024 tokens, cut at the end) would drop the answer of most
    # decision rows (their prompts are 1,700-2,800 tokens). keep_end: if a row is ever longer, the prompt start goes.
    args = SFTConfig(output_dir=str(out_path("outputs/checkpoints/sft")), num_train_epochs=s["epochs"], max_steps=max_steps,
                     learning_rate=s["learning_rate"], warmup_steps=s["warmup_ratio"], lr_scheduler_type=s["lr_scheduler"],
                     per_device_train_batch_size=s["per_device_batch"], gradient_accumulation_steps=s["grad_accum"],
                     max_length=s["max_prompt_length"] + s["max_completion_length"], truncation_mode="keep_end",
                     **runtime_kwargs(cfg, generation=False), completion_only_loss=True,
                     logging_steps=s["logging_steps"], save_steps=s["save_steps"], save_total_limit=3,
                     report_to=["tensorboard"])
    trainer = SFTTrainer(model=model, args=args, train_dataset=data, processing_class=tokenizer,
                         peft_config=lora_config(cfg), callbacks=progress_callback("sft", guard))
    started = time.time()
    trainer.train()
    if dry_run:
        record_dry_run(sft_seconds_per_step=(time.time() - started) / max(1, trainer.state.global_step))
        return
    trainer.save_model(str(out_path("outputs/checkpoints/sft/final")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sdft", "sft"], default="sdft")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    offline_after_download()
    cfg = load_config()
    guard = BudgetGuard()
    guard.check("sdft")
    train_path, _ = sdft_paths()
    train = load_jsonl_dataset(train_path)
    s = cfg["sdft"]
    rows_per_step = s["per_device_batch"] * s["grad_accum"] * max(1, int(os.environ.get("WORLD_SIZE", "1")))
    max_steps = (args.max_steps if args.max_steps is not None
                 else (0 if args.mode == "sft" else plan_steps(cfg, "sdft", len(train), rows_per_step, s["epochs"])))
    if args.mode == "sft":
        steps = args.max_steps if args.max_steps is not None else plan_sft_steps(cfg, train)
        run_sft(cfg, train, steps, guard, dry_run=args.dry_run)
        return 0
    from trl.experimental.sdft import SDFTTrainer

    model, tokenizer = load_model(cfg)
    trainer = SDFTTrainer(model=model, args=sdft_args(cfg, max_steps), train_dataset=train,
                          processing_class=tokenizer, peft_config=lora_config(cfg),
                          callbacks=progress_callback("dry_run" if args.dry_run else "sdft", guard))
    started = time.time()
    trainer.train()
    if args.dry_run:
        steps = trainer.state.global_step
        seconds = time.time() - started
        rows_per_step = cfg["sdft"]["per_device_batch"] * cfg["sdft"]["grad_accum"] * trainer.accelerator.num_processes
        record_dry_run(sdft_seconds_per_step=seconds / max(1, steps), sdft_rows_per_step=rows_per_step,
                       sdft_train_rows=len(train))
        return 0
    trainer.save_model(str(out_path(cfg["sdft"]["output_dir"]) / "final"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

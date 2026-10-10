"""Stage 2: GRPO with R = 0.8 * R_task + 0.2 * J/10 (gold match + DRAL-OA Eq. 1, A4B judge), TRL GRPOTrainer.

    accelerate launch --num_processes 3 -m oa_rlaif.train.grpo           # GPUs 0-2; vLLM server on GPU 3
    python -m oa_rlaif.train.grpo --no-judge --max-steps 10              # dry run without API spend

Starts from the SDFT adapter (merged into the base weights for the vLLM server). Every reward term is logged,
completions are dumped every N steps for manual reading, and a monitor flags reward hacking: rising NONE rate,
invalid-format rate, length drift, and judge scores that diverge from correctness.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from ..budget import BudgetGuard
from ..paths import ROOT, data_dir, out_dir
from ..reward.combined import RECENT, make_reward_functions
from ..reward.dral_reward import DralScorer
from .common import grpo_paths, load_config, load_jsonl_dataset, load_model, lora_config, offline_after_download, progress_callback


def hacking_monitor(dump_dir: Path, every: int):
    from transformers import TrainerCallback

    class Monitor(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not RECENT or logs is None:
                return
            window = list(RECENT)[-1024:]
            valid = [r for r in window if r[2]]
            logs["monitor/invalid_rate"] = 1 - len(valid) / len(window)
            labeled = [r for r in valid if r[3] is not None]
            if labeled:
                logs["monitor/gold_accuracy"] = statistics.fmean(r[3] for r in labeled)
            drals = [r[4] for r in valid if r[4] is not None]
            if drals:
                logs["monitor/dral_mean"] = statistics.fmean(drals)
            kinds = Counter(r[1] for r in window)
            logs.update({f"monitor/share_{k}": v / len(window) for k, v in kinds.items()})

        def on_step_end(self, args, state, control, **kwargs):
            if every and state.global_step % every == 0 and RECENT:
                dump_dir.mkdir(parents=True, exist_ok=True)
                with (dump_dir / f"step_{state.global_step:06d}.jsonl").open("w") as f:
                    for record in list(RECENT)[-64:]:
                        f.write(json.dumps(dict(zip(["episode", "kind", "valid", "gold", "dral", "task", "judge"], record))) + "\n")

    return Monitor()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-judge", action="store_true", help="skip the A4B API (dry runs)")
    parser.add_argument("--init", default=None, help="merged SDFT checkpoint directory")
    args = parser.parse_args()
    offline_after_download()
    cfg = load_config()
    g, v = cfg["grpo"], cfg["vllm"]
    guard = BudgetGuard()
    guard.check("grpo")
    from trl import GRPOConfig, GRPOTrainer

    train_path, _ = grpo_paths()
    train = load_jsonl_dataset(train_path)
    scorer = DralScorer(sorted((data_dir() / "dral").glob("*.npz")), temperature=g["dral_temperature"])
    judge = None
    if not args.no_judge:
        from ..reward.judge import Judge
        judge = Judge(guard=guard, fraction=g.get("judge_fraction", 1.0))
        calls = round(len(train) * g["num_generations"] * judge.fraction)
        print(json.dumps({"judge_model": judge.model, "judge_fraction": judge.fraction, "expected_judge_calls": calls,
                          "judge_rpm_per_process": judge.rpm}))
    reward_funcs, weights = make_reward_functions(scorer=scorer, judge=judge)
    config = GRPOConfig(
        output_dir=str(ROOT / g["output_dir"]), learning_rate=g["learning_rate"],
        per_device_train_batch_size=g["per_device_batch"], gradient_accumulation_steps=g["grad_accum"],
        max_steps=args.max_steps if args.max_steps is not None else g["max_steps"], num_train_epochs=1,
        bf16=True, gradient_checkpointing=True, logging_steps=g["logging_steps"], save_steps=g["save_steps"],
        save_total_limit=3, report_to=["tensorboard"], num_generations=g["num_generations"],
        temperature=g["temperature"], top_p=g["top_p"], max_completion_length=g["max_completion_length"],
        loss_type=g["loss_type"], beta=g["beta"], scale_rewards="none" if not g["scale_rewards"] else "group",
        mask_truncated_completions=g["mask_truncated_completions"], reward_weights=weights,
        chat_template_kwargs=cfg["model"]["chat_template_kwargs"], log_completions=True, num_completions_to_print=2,
        use_vllm=True, vllm_mode="server", vllm_server_host=v["server_host"], vllm_server_port=v["server_port"])
    model, tokenizer = load_model(cfg, args.init or str(ROOT / g["init_from"]))
    callbacks = progress_callback("grpo", guard) + [hacking_monitor(out_dir() / "grpo_samples", g["sample_dump_every"])]
    trainer = GRPOTrainer(model=model, reward_funcs=reward_funcs, args=config, train_dataset=train,
                          processing_class=tokenizer, peft_config=lora_config(cfg), callbacks=callbacks)
    trainer.train()
    trainer.save_model(str(ROOT / g["output_dir"] / "final"))
    if judge is not None:
        print(json.dumps({"judge_fallbacks": judge.fallbacks, "spent": guard.summary()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

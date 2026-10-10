#!/usr/bin/env bash
# LAMBDA: measure throughput on this machine before the real run; writes outputs/dry_run.json.
#   1. 20 OPSD steps (policy vLLM server on GPU 3)
#   2. 5 SFT steps (comparison row; no vLLM)
#   3. reward check (gold replies -> DRAL-OA + judge), then 10 GRPO steps WITH the local A4B judge
#      (policy + judge share GPU 3), so the judge cost is in the timing
# 40_sdft.sh and 60_grpo.sh then plan their step counts from these timings (train.common.plan_steps):
# steps = min(full epochs, what fits time_budget_hours in configs/train.yaml).
. "$(dirname "$0")/_common.sh"
budget_gate dry_run
MODEL="$($PY -c "from oa_rlaif.train.common import load_config; print(load_config()['model']['id'])")"
rm -f outputs/dry_run.json outputs/step_plan.json

vllm_up "$MODEL"
CUDA_VISIBLE_DEVICES=0,1,2 .venv/bin/accelerate launch --num_processes 3 -m oa_rlaif.train.sdft --dry-run --max-steps 20 \
  2>&1 | tee outputs/logs/dry_run_sdft.log
vllm_down

CUDA_VISIBLE_DEVICES=0,1,2 .venv/bin/accelerate launch --num_processes 3 -m oa_rlaif.train.sdft --mode sft --dry-run \
  --max-steps 5 2>&1 | tee outputs/logs/dry_run_sft.log

VLLM_GPU_UTIL=0.40 vllm_up "$MODEL"; trap 'judge_down; vllm_down' EXIT
judge_up
# gold replies through the GRPO rewards: proves the local judge answers and parses before GPU hours are spent
$PY -m oa_rlaif.dev.reward_check --n 8 2>&1 | tee outputs/logs/dry_run_reward_check.log
CUDA_VISIBLE_DEVICES=0,1,2 .venv/bin/accelerate launch --num_processes 3 -m oa_rlaif.train.grpo --dry-run --max-steps 10 \
  --init "$MODEL" 2>&1 | tee outputs/logs/dry_run_grpo.log

cat outputs/dry_run.json
$PY - <<'PY'
import json
from oa_rlaif.train.common import load_config, plan_steps
cfg = load_config()
d = cfg["dry_run"]
for stage, rows, per_step, epochs in (("sdft", d["sdft_train_rows"], d["sdft_rows_per_step"], cfg["sdft"]["epochs"]),
                                      ("grpo", d["grpo_train_prompts"], d["grpo_prompts_per_step"], 1)):
    plan_steps(cfg, stage, rows, per_step, epochs)
from oa_rlaif.train.sdft import plan_sft_steps
from oa_rlaif.train.common import load_jsonl_dataset, sdft_paths
plan_sft_steps(load_config(), load_jsonl_dataset(sdft_paths()[0]))
print(open("outputs/step_plan.json").read())
PY
$PY -m oa_rlaif.progress status

#!/usr/bin/env bash
# MAC end-to-end smoke test of the Lambda pipeline code (run -> train -> progress -> agent -> evaluation).
#
# Runs the SAME Python entry points as scripts/lambda_all.sh, in the same order, with:
#   - a tiny random Gemma 4 (same architecture as E4B, 26M params; built from the cached config, nothing downloaded)
#   - CPU instead of vLLM / 4 GPUs (TRL generates with the model itself)
#   - a stand-in OpenAI-compatible server for the A4B judge and the agent's model (scripts/dev/stub_openai.py)
#   - the real data, real reward (DRAL-OA layers), real Docker sandbox, real mini-SWE-agent loop, real report
#   - every output under outputs/smoke/ (nothing else is touched), the same live dashboard
# Not covered here (GPU-box only, checked by check_env / gguf_check / dry_run on Lambda): vLLM servers,
# multi-GPU accelerate, CUDA/B200, GGUF conversion of the merged model, TRL's Triton loss kernel (the Mac uses the
# plain-PyTorch stand-in in src/oa_rlaif/dev/cpu_kernels.py).
#
#   scripts/smoke_mac.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export OA_SMOKE=1 OA_OUT_DIR=outputs/smoke OA_DASH_PLAN=smoke OA_SMOKE_STEPS="${OA_SMOKE_STEPS:-3}"
export PYTHONPATH=src:third_party/mini-swe-agent/src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export JUDGE_API_BASE=http://127.0.0.1:8099/v1 JUDGE_API_KEY=stub JUDGE_MODEL=stub JUDGE_RPM=0 JUDGE_MAX_TOKENS=64
export JUDGE_EXTRA_BODY='{}' MSWEA_SILENT_STARTUP=1 LITELLM_LOCAL_MODEL_COST_MAP=True
OUT="$OA_OUT_DIR"; PY=.venv/bin/python; TINY="$OUT/tiny-gemma4"; export OA_MODEL_ID="$TINY"
rm -rf "$OUT"; mkdir -p "$OUT/stages" "$OUT/logs"; date +%s > "$OUT/stages/run.start"

python3 scripts/dev/stub_openai.py 8099 > "$OUT/logs/stub.log" 2>&1 & STUB=$!
trap 'kill $STUB 2>/dev/null || true' EXIT
for _ in $(seq 1 20); do curl -sf http://127.0.0.1:8099/health >/dev/null && break; sleep 0.3; done

stage() {
  local name="$1"; shift
  date +%s > "$OUT/stages/$name.start"
  "$@" > "$OUT/logs/stage_$name.log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do python3 scripts/dashboard.py --clear || true; sleep 2; done
  if wait "$pid"; then touch "$OUT/stages/$name.done"; python3 scripts/dashboard.py --clear || true
  else touch "$OUT/stages/$name.failed"; python3 scripts/dashboard.py --clear || true
       echo; echo "SMOKE STEP '$name' FAILED:"; tail -n 40 "$OUT/logs/stage_$name.log"; exit 1; fi
}
ev()    { $PY -m oa_rlaif.eval.run_eval --model "$1" --tag "$2" --cap 3; }
agent() { $PY -m oa_rlaif.eval.agent_eval --episodes data/eval/heldout_anatomy.jsonl --n 2 --tag "$1" --env docker \
            --api-base http://127.0.0.1:8099/v1; }
merge() { $PY -m oa_rlaif.export.release merge --adapter "$OUT/checkpoints/$1/final" --out "$OUT/merged/$1" ${2:+--base "$2"}; }
export -f ev agent merge; export PY OUT TINY

stage tiny_model $PY -m oa_rlaif.dev.tiny_model "$TINY"
stage dry_run    bash -c '$PY -m oa_rlaif.train.sdft --dry-run --max-steps 2 && $PY -m oa_rlaif.train.sdft --mode sft --dry-run --max-steps 2 && $PY -m oa_rlaif.train.grpo --dry-run --max-steps 2 --init "$TINY" && cat "$OUT/dry_run.json"'
stage baseline   bash -c 'ev "$TINY" base && agent base'
stage sdft       bash -c '$PY -m oa_rlaif.train.sdft && merge sdft'
stage eval_sdft  bash -c 'ev "$OUT/merged/sdft" sdft && agent sdft'
stage grpo       bash -c '$PY -m oa_rlaif.dev.reward_check --n 4 && $PY -m oa_rlaif.train.grpo --init "$OUT/merged/sdft" && merge grpo "$OUT/merged/sdft"'
stage eval_grpo  bash -c 'ev "$OUT/merged/grpo" grpo && A=data/raw/anatomy-dataset-2 && $PY -m oa_rlaif.eval.full_align --model "$OUT/merged/grpo" --name anatomy --source $A/mouse.owl --target $A/human.owl --reference $A/reference.rdf --limit 12 && agent grpo'
# GGUF conversion needs the pinned llama.cpp build (Lambda: gguf_check + export). Here: the real Modelfile next to a
# placeholder GGUF in the release layout, so the model-push dry run below checks the GGUF repo's file selection too.
stage export     bash -c 'R="$OUT/release/oa-rlaif-e4b"; ls "$OUT/merged/grpo" && mkdir -p "$R" \
                   && echo "smoke placeholder; the real GGUF is built on Lambda" > "$R/oa-rlaif-e4b-Q4_K_M.gguf" \
                   && $PY -c "import sys; from pathlib import Path; from oa_rlaif.export.release import write_modelfile; write_modelfile(Path(sys.argv[1]), \"oa-rlaif-e4b-Q4_K_M.gguf\")" "$R" \
                   && cat "$R/Modelfile"'
stage push       $PY -m oa_rlaif.data.hf_push --models --dry-run
stage sft        bash -c '$PY -m oa_rlaif.train.sdft --mode sft && merge sft'
stage eval_sft   bash -c 'ev "$OUT/merged/sft" sft && agent sft'
# research variant (scripts/all_merged.sh on the GPU box): SFT adapter added onto the GRPO model, then evaluated
stage all_merged bash -c '$PY -m oa_rlaif.export.release merge --adapter "$OUT/checkpoints/sft/final" --base "$OUT/merged/grpo" --out "$OUT/merged/all_merged" && ev "$OUT/merged/all_merged" all_merged && A=data/raw/anatomy-dataset-2 && $PY -m oa_rlaif.eval.full_align --model "$OUT/merged/all_merged" --name anatomy_all_merged --source $A/mouse.owl --target $A/human.owl --reference $A/reference.rdf --limit 12 && agent all_merged && $PY -m oa_rlaif.eval.report --all-merged && $PY -m oa_rlaif.data.push_all_merged --dry-run'
stage report     bash -c 'A=data/raw/anatomy-dataset-2; $PY -m oa_rlaif.eval.align --backend lexical --model lexical --source $A/mouse.owl --target $A/human.owl --reference $A/reference.rdf --format tsv --no-coherence --out "$OUT/align/anatomy_lexical_baseline" && $PY -m oa_rlaif.eval.report'
stage results    $PY -m oa_rlaif.data.hf_push --results --dry-run

echo; echo "SMOKE TEST PASSED: every stage ran on the real code paths. Report: $OUT/eval/report.md"; echo
cat "$OUT/eval/report.md"

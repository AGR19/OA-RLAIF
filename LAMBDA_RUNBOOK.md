# Lambda runbook (4×B200, $500 total, stop at $450)

On CoreWeave, use `COREWEAVE_RUNBOOK.md` and `scripts/coreweave_all.sh` instead (same stages; Docker install, price prompt, any GPU count).

Everything runs on the instance. The GRPO judge (Gemma 4 26B-A4B) is served by a second vLLM server on GPU 3; question generation already ran on the Mac through AI Studio. Each command prints a progress bar with an ETA. `scripts/status.sh` shows the whole pipeline and the $ meter at any time.

## 0. Before renting (on the Mac — free)
```bash
scripts/10_prepare_data.sh          # coverage, full build, render
scripts/15_datasource.sh              # A4B questions via AI Studio (needs JUDGE_API_* in .env)
scripts/20_fit_dral.sh              # DRAL-OA mapping layers -> data/dral/*.npz
scripts/80_verify_and_push.sh       # tests + verification gate (add --hf to push private datasets after your OK)
```

## 1. Launch and copy
1. In the Lambda console, launch 4×B200 and attach a **persistent filesystem** (outputs survive instance stops).
2. Copy the project over, using either:
   - `scripts/sync_to_lambda.sh ubuntu@<ip>` (rsync from the Mac), or
   - `python -m oa_rlaif.data.hf_push --pull` on the instance (private HF datasets).
3. On the instance: `scripts/cloud_setup.sh lambda`. This is the only download step: system packages, Docker Engine (installed on the host; nothing is assumed), wheels from the lock, the Gemma 4 E4B and A4B weights, the sandbox image and llama.cpp.
4. Edit `.env`: `HF_NAMESPACE` (gonibeed). The `JUDGE_*` settings are not needed on Lambda, because `60_grpo.sh` points the judge at the local server. `OA_BILLABLE=1` is set by the setup script.
5. `scripts/00_check_env.sh` must show 4 GPUs at sm_100, the SDFTTrainer import, the Docker image, data present and both weight snapshots cached (E4B and the 26B-A4B judge).

## 2. Run (inside tmux; every stage checks the budget first)
| Step | Command | GPUs | Est. |
|---|---|---|---|
| GGUF check (base model) | `scripts/72_gguf_check.sh` | — | 15 m |
| Dry run (sets step counts, checks the judge) | `scripts/35_dry_run.sh` | 0–3 | 40 m |
| Baseline (base E4B) | `scripts/30_baseline.sh` | 3 (vLLM) | 45 m |
| Stage 1 OPSD | `scripts/40_sdft.sh` | 0–2 train, 3 vLLM | ≤ 4 h |
| Eval OPSD | `scripts/50_eval.sh sdft` | 0–3 | 45 m |
| Stage 2 GRPO | `scripts/60_grpo.sh` | 0–2 train, 3 vLLM + judge | ≤ 4 h |
| Eval GRPO + full mappings | `scripts/50_eval.sh grpo --full` | 0–3 | 70 m |
| Export (merge, GGUF, Ollama) | `scripts/70_export.sh` | 0 | 30 m |
| Push GGUF + adapters (private) | `python -m oa_rlaif.data.hf_push --models` | — | 15 m |
| SFT comparison (compulsory) | `scripts/40_sdft.sh --mode sft` | 0–2 | ≤ 2 h |
| Eval SFT | `scripts/50_eval.sh sft` | 0–3 | 45 m |
| Results tables + push | `python -m oa_rlaif.eval.report`, then `hf_push --results` | — | 7 m |

**Budget.** In dollars (`src/oa_rlaif/cost.py`): the price of this instance (Lambda API with `LAMBDA_API_KEY`, else asked and checked against the GPUs), spend = the machine's billed time from its start. At 4×B200 ($27.16/h) training gets ≈ $278 (OPSD $111, GRPO $111, SFT $56). At $450 the trainers save a checkpoint and stop. `python -m oa_rlaif.cost status` shows the numbers.

**Watching for cheating** during GRPO (step 14 of the original plan):
- `outputs/grpo_samples/step_*.jsonl`;
- TensorBoard `monitor/*` (invalid rate, gold accuracy, NONE share, DRAL mean);
- `judge_fallbacks` in the log.

**Optional: keep the SFT comparison model.** The one-command run uploads only its scores. To also keep its weights, run `scripts/push_sft.sh` after step 13 (and again after step 14 to add its evaluation). It creates two private repos: `<you>/oa-rlaif-e4b-sft-weights` (LoRA adapter + evaluation + logs) and `<you>/oa-rlaif-e4b-sft-model-safetensors` (merged model, about 16 GB, resumable upload). `scripts/push_sft.sh --dry-run` lists what would be uploaded.

**Optional research variant: all merged.** After DONE, `scripts/all_merged.sh` adds the SFT adapter onto the GRPO model (base + OPSD + GRPO + SFT; the SFT weight change is summed in) and runs the same evaluations as for GRPO. Its own comparison tables (base, SFT, OPSD, GRPO and All merged) go to `outputs/all_merged_report/`; the main report is unchanged. `--push` uploads everything all-merged to its own private repo `<you>/oa-rlaif-e4b-all-combined-model-safetensors` (weights, evaluation, comparison report); nothing goes into the existing repos. `--no-full` skips the whole-Anatomy alignment. It takes about 1–1.5 h of machine time, and the released GGUF does not change.

## 3. Results
- `outputs/eval/<tag>/metrics.json` for base, sdft and grpo;
- `outputs/agent_runs/<tag>/summary.json` (mini-SWE-agent runs);
- `outputs/full_align/*/alignment.{sssom.tsv,rdf}`.

Copy `outputs/` back to the persistent filesystem before terminating.

## One command (start here)
**Instance:** 4×B200 (the plan), or at least 4 GPUs with 80 GB or more each (H100, H200, A100 80GB). The command checks this first and stops before installing anything on a smaller instance (for example 1×A100 40GB).

The command installs everything on the host itself: system packages, Docker Engine, the Python requirements from the lock, the weights and llama.cpp. It asks once for the **price of the instance** (or reads it from the Lambda API if `.env` has `LAMBDA_API_KEY`) and refuses a price that does not fit the GPUs.

Before you start:
1. Create a GitHub token (read access to `AGR19/OA-RLAIF`) at github.com/settings/tokens.
2. Create a Hugging Face token **with write access** at huggingface.co/settings/tokens. Write access is needed for the final GGUF push.

On a fresh Lambda 4×B200 instance:

```bash
tmux new -s run                       # keeps the run alive if your SSH connection drops (reattach: tmux attach -t run)
git clone https://github.com/AGR19/OA-RLAIF.git && cd OA-RLAIF    # username AGR19, password = the GitHub token
scripts/lambda_all.sh                 # asks once for the Hugging Face token, then runs everything
```

The terminal shows a live dashboard: the total progress bar, the current step (e.g. "STEP 7/16 sdft") with its own bar, elapsed time, ETA, GPU spend so far and the latest log line. Each step's full output is in `outputs/logs/stage_<step>.log`.

| # | Step | What happens | Est. |
|---|---|---|---|
| 1 | setup | packages, Gemma 4 E4B + 26B-A4B judge weights, Docker sandbox, llama.cpp | 25 m |
| 2 | data | download the private datasets | 5 m |
| 3 | check_env | 4 GPUs, libraries, data, Docker agent test. **Stops here if anything is wrong** | 5 m |
| 4 | gguf_check | convert the base model to GGUF (proves export works before training) | 15 m |
| 5 | dry_run | time OPSD, SFT and GRPO (with the judge); check the judge; plan step counts from the money left | 40 m |
| 6 | baseline | base E4B: held-out sets + agent episodes | 45 m |
| 7 | sdft | Stage 1 OPSD (on-policy self-distillation; the teacher sees the gold answer), then merge | ≤ 4 h |
| 8 | eval_sdft | OPSD model: held-out sets + agent episodes | 45 m |
| 9 | grpo | Stage 2 GRPO: DRAL-OA reward + local A4B judge (RLAIF), then merge | ≤ 4 h |
| 10 | eval_grpo | held-out, zero-shot, ICD-11↔SNOMED, full Anatomy alignment (OAEI row), agent episodes | 70 m |
| 11 | export | GGUF Q4_K_M + Ollama Modelfile (base + OPSD + GRPO in one file) | 30 m |
| 12 | push | private HF: `oa-rlaif-e4b-opsd-grpo-model-gguf`, `-sdft`, `-grpo` (the GGUF is safe from here) | 15 m |
| 13 | sft | **compulsory** SFT comparison: same rows and same step count as OPSD, then merge | ≤ 2 h |
| 14 | eval_sft | SFT model: held-out sets + agent episodes | 45 m |
| 15 | report | results tables: base / SFT / OPSD / OPSD→GRPO, ICD-11↔SNOMED, agents, OAEI row | 2 m |
| 16 | results | private HF dataset `oa-rlaif-heldout-eval-results` (tables, metrics, ICD-11↔SNOMED SSSOM) | 5 m |

At 4×B200 the total is about 16 h and stays under the $450 stop. Steps 3–5 are the safety checks: a problem shows within the first hour. The plan is in dollars: after the dry run, the money left minus the fixed stages (less a 5% reserve) is shared OPSD 40%, GRPO 40%, SFT 20%, and each stage runs the steps its share pays for (`outputs/step_plan.json`), re-checked at the start of each stage.

The SFT arm is part of every run (it is not optional). It runs after the GGUF push, so a late problem cannot lose the main model.

**If a step fails**, the dashboard marks it ✖ and prints the end of its log. Fix the cause, then run `scripts/lambda_all.sh` again; finished steps are skipped.

**When it prints DONE**, your GGUF is at `huggingface.co/<you>/oa-rlaif-e4b-opsd-grpo-model-gguf` (private). **Terminate the instance** in the Lambda console; billing continues until you do.

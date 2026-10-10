# Lambda runbook (4×B200, $500 total, stop at $450)

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
3. On the instance: `scripts/lambda_setup.sh`. This is the only download step: wheels from the lock, Gemma 4 E4B weights, the Docker base image and llama.cpp.
4. Edit `.env`: `HF_NAMESPACE` (gonibeed). The `JUDGE_*` settings are not needed on Lambda, because `60_grpo.sh` points the judge at the local server. `OA_BILLABLE=1` is set by the setup script.
5. `scripts/00_check_env.sh` must show 4 GPUs at sm_100, the SDFTTrainer import, the Docker image, data present and both weight snapshots cached (E4B and the 26B-A4B judge).

## 2. Run (inside tmux; every stage checks the budget first)
| Step | Command | GPUs | Est. |
|---|---|---|---|
| Dry run (sets step counts) | `scripts/35_dry_run.sh` | 0–3 | 30 m |
| Baseline (base E4B) | `scripts/30_baseline.sh` | 3 (vLLM) | 45 m |
| Stage 1 SDFT/OPSD | `scripts/40_sdft.sh` | 0–2 train, 3 vLLM | ~6 h |
| Eval SDFT | `scripts/50_eval.sh sdft` | 0–3 | 45 m |
| Stage 2 GRPO | `scripts/60_grpo.sh` | 0–2 train, 3 vLLM | ~6 h |
| Eval GRPO + full mappings | `scripts/50_eval.sh grpo --full` | 0–3 | 75 m |
| Export (merge, GGUF, Ollama) | `scripts/70_export.sh` | 0 | 30 m |
| Push adapters (private) | `scripts/80_verify_and_push.sh --models` | — | 10 m |

**Budget.** At ~$27.16/h, the plan is ≈16 h ≈ $440 plus API. `outputs/budget_ledger.json` accumulates every run. At $450 the trainers save a checkpoint and stop.

**Watching for cheating** during GRPO (step 14 of the original plan):
- `outputs/grpo_samples/step_*.jsonl`;
- TensorBoard `monitor/*` (invalid rate, gold accuracy, NONE share, DRAL mean);
- `judge_fallbacks` in the log.

## 3. Results
- `outputs/eval/<tag>/metrics.json` for base, sdft and grpo;
- `outputs/agent_runs/<tag>/summary.json` (mini-SWE-agent runs);
- `outputs/full_align/*/alignment.{sssom.tsv,rdf}`.

Copy `outputs/` back to the persistent filesystem before terminating.

## One command (start here)
Before you start:
1. Create a GitHub token (read access to `AGR19/OA-RLAIF`) at github.com/settings/tokens.
2. Create a Hugging Face token **with write access** at huggingface.co/settings/tokens. Write access is needed for the final GGUF push.

On a fresh Lambda 4×B200 instance:

```bash
tmux new -s run                       # keeps the run alive if your SSH connection drops (reattach: tmux attach -t run)
git clone https://github.com/AGR19/OA-RLAIF.git && cd OA-RLAIF    # username AGR19, password = the GitHub token
scripts/lambda_all.sh                 # asks once for the Hugging Face token, then runs everything
```

The terminal shows a live dashboard: the total progress bar, the current step (e.g. "STEP 7/12 sdft") with its own bar, elapsed time, ETA, GPU spend so far and the latest log line. Each step's full output is in `outputs/logs/stage_<step>.log`.

| # | Step | What happens | Est. |
|---|---|---|---|
| 1 | setup | packages, Gemma 4 E4B + 26B-A4B judge weights, Docker sandbox, llama.cpp | 25 m |
| 2 | data | download the private datasets | 5 m |
| 3 | check_env | 4 GPUs, libraries, data, Docker agent test. **Stops here if anything is wrong** | 5 m |
| 4 | gguf_check | convert the base model to GGUF (proves export works before training) | 15 m |
| 5 | dry_run | time SDFT and GRPO with the judge; plan step counts that fit the budget | 35 m |
| 6 | baseline | base E4B: held-out sets + agent episodes | 45 m |
| 7 | sdft | Stage 1 on-policy self-distillation, then merge | ~5.5 h |
| 8 | eval_sdft | SDFT model: held-out sets + agent episodes | 1 h |
| 9 | grpo | Stage 2 GRPO: DRAL-OA reward + local A4B judge (RLAIF), then merge | ~5.5 h |
| 10 | eval_grpo | held-out, zero-shot, full Anatomy alignment (OAEI row), agent episodes | 1.5 h |
| 11 | export | GGUF Q4_K_M + Ollama Modelfile (base + SDFT + GRPO in one file) | 30 m |
| 12 | push | private HF: `oa-rlaif-e4b-gguf`, `-sdft`, `-grpo`, dataset `oa-rlaif-results` | 15 m |

The total is about 17 h and at most $450 (the budget guard stops every step there). Steps 3–5 are the safety checks: a problem shows within the first hour (about $30).

**If a step fails**, the dashboard marks it ✖ and prints the end of its log. Fix the cause, then run `scripts/lambda_all.sh` again; finished steps are skipped.

**When it prints DONE**, your GGUF is at `huggingface.co/<you>/oa-rlaif-e4b-gguf` (private). **Terminate the instance** in the Lambda console; billing continues until you do.

Optional: `RUN_SFT_ARM=1 scripts/lambda_all.sh` adds the plain-SFT comparison model (about 3 more GPU hours, ~$80). This can exceed the $450 stop.

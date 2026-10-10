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
On a fresh Lambda 4×B200 instance:

```bash
git clone https://github.com/AGR19/OA-RLAIF.git && cd OA-RLAIF   # private repo: sign in with a GitHub token
scripts/lambda_all.sh       # 1st run: installs everything, then asks for the Hugging Face login
.venv/bin/hf auth login     # the account that owns the private datasets (gonibeed)
scripts/lambda_all.sh       # 2nd run: data -> checks -> GGUF check -> dry run -> train -> eval -> GGUF -> push
```

`scripts/lambda_all.sh` runs these stages in order:
1. setup: uv, pinned wheels, Gemma 4 E4B and 26B-A4B weights, Docker sandbox, llama.cpp.
2. data: pull the private HF datasets.
3. check_env: 4 GPUs, libraries, data present, the Docker agent test. **Fails hard** if anything is missing.
4. gguf_check: converts the *base* model to GGUF, so an export problem shows before 12 h of training.
5. dry_run.
6. baseline.
7. SDFT, then merge.
8. eval.
9. GRPO with the local A4B judge, then merge.
10. eval + full Anatomy alignment.
11. export: GGUF Q4_K_M + Modelfile.
12. push: private HF `oa-rlaif-e4b-gguf`, `-sdft`, `-grpo`, and the dataset `oa-rlaif-results`.

Resume after a failure:
- Each stage writes `outputs/stages/<name>.done`.
- After a failure, fix the cause and run the same command again; finished stages are skipped.
- To redo a stage, delete its `.done` file.

The budget guard stops every billable stage at $450. Watch from a second shell with `scripts/status.sh --live`.

**When it prints DONE, terminate the instance in the Lambda console.** Billing continues until you do.

The GGUF is the final model: base E4B + SDFT + GRPO (DRAL-OA reward + A4B judge, the RLAIF part), merged and quantized.

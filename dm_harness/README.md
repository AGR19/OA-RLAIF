# Decision model: pplx-decider controls the All-merged agent through JevAgentHarness (`dm_harness/`)

Branch `dm_harness`. A decision model, [perplexity-ai/pplx-decider-v1.2](https://huggingface.co/perplexity-ai/pplx-decider-v1.2)
(Apache-2.0, 52 GB, Qwen3.8-27B + a readout head), answers small typed questions about the All-merged Gemma 4 E4B agent
(the full evaluation's "All merged + mini-SWE-agent"). It answers them through
[JevAgentHarness](https://github.com/JevAgentHarness/JevAgentHarness) (MIT, vendored in `third_party/JevAgentHarness`,
paper: Zhu et al. 2026, arXiv:2610.21412), a local OpenAI-compatible proxy between the agent and the model.

**Scope:**
- **Classes:** every source class with a partner in the reference, and the complete SNOMED CT → ICD-11 map. That is
  299,961 classes per arm, with no held-out view.
- **Score:** the Top-k evaluation (k = 1, 3, 5).
- **Comparison:** the All-merged rows of `gonibeed/oa-rlaif-full-eval-results`, which is read only.
- **Results:** a new private dataset, `gonibeed/oa-rlaif-dm-harness-results`.

## The two arms

| Arm | Tag (folder) | What answers each class |
|---|---|---|
| A | `decider` (`7-pplx-decider`) | pplx-decider alone. One `choice` question: the state is the source card; the options are the candidates (C1, C2, … with name, synonyms, parents) and NONE. Its pick is the answer; its probabilities order Top-k ranks 2–5. |
| B | `jah-agent` (`8-jah-all-merged-agent`) | The console's mini-SWE-agent (`om_agent/`) with model `jah-auto`, through the proxy, plus the NONE/invalid guard (below). |

**Arm B, step by step:**
1. **The agent explores.** It runs the scripted `om profile`, `om show` and `om candidates` steps, as in the full
   evaluation.
2. **The proxy routes the model call.** Before it, the proxy asks pplx-decider: "is the next step `routine` or
   `hard`?"
   - p(hard) ≥ **0.75** → the strong tier: All merged, **thinking on**.
   - Otherwise → the fast tier: the same weights, **thinking off**.

   One vLLM server serves both tiers; each tier's `extra` sets `chat_template_kwargs`. Any decision-model error sends
   the step to the strong tier.
3. **The NONE/invalid guard checks the submission.** The proxy's own finish check needs native tool calls, so this
   repo's harness does it instead (the paper's GUARD mode, `dm_harness/harness.py`).
   - The submission is **NONE**, and pplx-decider's best candidate has p ≥ **0.5**: the class is redone on the strong
     tier (`jah-strong`), with a note naming that candidate. Gemma decides again.
   - The submission is **NONE**, and p < 0.5: NONE stays.
   - The submission is **invalid** (no valid submission): the class is redone on the strong tier, with the note if
     p ≥ 0.5. If the redo is still invalid, pplx-decider's pick is the answer.

**Thresholds:** both are **fixed before the run** (the repository's defaults); no reference answer chooses them. The
report also shows, from the logged probabilities, what pplx-decider answering alone above other τ values would have
given (the paper's CASCADE), without a rerun.

## Run (Vast.ai, 4 × B200)

1. Create an instance:
   - 4 × B200 (any GPU with 80 GB or more works);
   - **at least 250 GB disk**;
   - an Ubuntu + CUDA image, with SSH.
2. Start or attach tmux (Vast starts one for you), then:

```bash
cd /workspace 2>/dev/null || cd ~; git clone -b dm_harness https://github.com/AGR19/OA-RLAIF.git && cd OA-RLAIF
```

```bash
dm_harness/vast_dm_harness.sh
```

The script does these steps:
1. **Setup:** the price of this machine, the Python environment and the agent's user sandbox (as the full
   evaluation), and the proxy.
2. **Your Hugging Face token,** asked once.
3. **Inputs:** the raw ontologies, the decision model's own Python 3.12 environment and weights (52 GB), and the
   comparison rows (read only).
4. **The plan**, shown once with its cost.
5. **Arm A** on every GPU, then **arm B**: the decision model on GPUs 0–1, vLLM with All merged on GPUs 2–3, and the
   proxy.
6. **Scoring, the report** (`outputs/full_eval/dm_report.md`) **and the upload.**

Destroy the instance after `DONE`.

**Options:**
- `--plan` only shows the plan.
- `--yes` skips the question.
- `DM_BUDGET_USD=300` changes the cap (the default is $250).
- `DM_ARMS=decider` runs one arm only.
- `FULL_EVAL_TASKS=icd__snomed-to-icd11,cyber__iotsec-uco` runs only these tasks.
- `DM_DECIDER_GPUS=0,1,2 DM_GEMMA_GPUS=3` changes arm B's GPU split.

If anything stops the run, use the same command again: finished units are kept.

## Cost (the dry run measures the real speeds)

| Step | Time on 4 × B200 |
|---|---:|
| Setup and downloads (decider 52 GB, All merged 16 GB, inputs), building the episodes | ~0.8 h |
| Arm A: 299,961 decider questions (~60/s on 4 GPUs) | ~1.4 h |
| Arm B: 299,961 agent episodes (~25/s; the decision model on 2 GPUs, Gemma on 2) | ~3 h |
| Scoring, report and upload | ~0.2 h |
| **Total** | **~5.5 h** |

That is about $110 at $20/h, $165 at $30/h and $225 at $41/h. The plan prints the real price before it starts.

## The report (`dm_report.md`)

Rows: All merged (one decision) and All merged + mini-SWE-agent with thinking on, both from the first run; arm A; and
arm B.

| Section | What it shows |
|---|---|
| Focus | SNOMED CT → ICD-11 and Cybersecurity: P, R and F at k = 1, 3, 5 |
| Tables 1–3 | F@1 per dataset; P, R and F at k per dataset; per pair, with the retrieval ceiling |
| Table 4 | Answers, NONE, invalid, tokens, the agent's steps, the thinking, runtime |
| Table 5 | The guard per task: how often it acted, NONE kept, redo → match or NONE, the decider's pick |
| Table 6 | Routing: the share of model calls on thinking on, p(hard), decision latency (from the proxy's `steps.jsonl`; no prompt text is logged) |
| Table 7 | The complete SNOMED CT → ICD-11 map |
| Table 8 | pplx-decider answering alone above τ (0.5–0.99), with arm B's answer otherwise; no rerun |
| Table 9 | Units and money |

## Pieces

| File | What |
|---|---|
| `questions.py` | the candidate question (System-One format) and how its answer is read |
| `decider_server.py` | pplx-decider behind `POST /v1/systemone`, one worker per GPU, batched; `DM_FAKE=lexical` = a lexical stand-in |
| `stack.py` | starts and stops the decider, vLLM (data-parallel) and the proxy; the decider smoke test |
| `jah_config.py` | the proxy's configuration: tiers, router, τ |
| `base_run.py` | arm A's worker |
| `harness.py`, `harness_run.py` | arm B's agent (the guard) and its worker |
| `run.py` | both arms, the money cap, scoring, report |
| `report.py` | the tables |
| `pull_reference.py`, `push_results.py` | the comparison rows (read only) and the upload to the new repo |
| `setup_decider.sh`, `vast_dm_harness.sh` | the decision model's environment, and the one command |
| `fake_upstream.py` | Mac rehearsal only: a lexical stand-in for vLLM behind the proxy |

Mac rehearsal (no GPU; lexical stand-ins for pplx-decider and Gemma; the real proxy, agent, guard, scoring and report):

```bash
PYTHONPATH=src:.:third_party/JevAgentHarness/src OA_OUT_DIR=/tmp/dm DM_FAKE=lexical OA_AGENT_ENV=local DM_GPUS=0 FULL_EVAL_TASKS=anatomy__mouse-human,dh,cyber__iotsec-uco,complex_om__cree-swo .venv/bin/python -m dm_harness.run --yes
```

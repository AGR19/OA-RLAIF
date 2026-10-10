# OA-RLAIF — project instructions for agents

## Goal (measurable)

Train **Gemma 4 E4B** (LoRA) with **on-policy self-distillation** (OPSD, TRL `SDFTTrainer`), then **GRPO** (TRL `GRPOTrainer` + vLLM). The GRPO reward is the **DRAL-OA reward** (Eq. 1 InfoNCE over the mapping layer) plus an **A4B judge**. The trained model must produce **full ontology alignments** (SSSOM TSV + OAEI RDF) across every supplied domain. That includes **ICD-11↔SNOMED CT** via the WHO ICD-10 bridge.

**Success** is measured on the held-out 20% and on whole zero-shot ontology pairs. All of these must hold:
1. The GRPO model beats the base E4B and the OPSD checkpoint on P/R/F1 (OAEI-style).
2. On Anatomy, it beats DRAL-OA's official OAEI 2025 result: **F 0.828** (P 0.83, R 0.827).
3. It keeps closed-book knowledge from the datasource and ontology facts. Retention must be at least the OPSD checkpoint's, within the CI.
4. It recovers **≥ X%** of the held-out ICD bridge links. X is fixed after the baseline run and recorded in `MEMORY.md`.
5. Total spend is **≤ $500**: Lambda GPU time plus any API calls. Question generation uses the AI Studio free tier.

Only the held-out and zero-shot numbers are headline numbers. Report everything else as diagnostics. Never fabricate or round up a metric. If a value is missing, it stays missing.

## Hard rules

- **No licensed data in git.**
  - The datasource text is Springer © 2007, all rights reserved.
  - SNOMED CT / UMLS-derived rows fall under the SNOMED affiliate license.
  - Raw ontologies and any generated dataset also stay out.
  - These go only to **private** Hugging Face dataset repos, after verification, with the user's explicit OK in chat.
- **Budget:** $500 total. `train/budget.py` stops every run at **$450** (it checkpoints first). Do every data step on the Mac before renting GPUs. Always dry-run before a full run.
- **Portability:**
  - Relative or env-driven paths only. `tests/test_no_absolute_paths.py` enforces this.
  - Pinned dependencies in `requirements-train.lock`.
  - No machine-specific absolute paths (home directories) anywhere.
- **Everything runs locally except A4B question generation.** On the Mac, A4B through Google AI Studio writes the datasource questions. The model is `gemma-4-26b-a4b-it`, free tier, paced by `JUDGE_RPM`. This is the only remote model call.
  - The **GRPO judge is A4B served locally** on Lambda: `google/gemma-4-26B-A4B-it`, vLLM on GPU 3 next to the policy, thinking off. `judge_up` in `scripts/_common.sh` starts it.
  - Why (decided 2026-10-10): Gemma 4 is free-tier only on the Gemini API (paid tier "Not available"), so the quota cannot cover about 96K judge calls. API latency would also idle the 4×B200 during reward scoring.
  - Local: the policy (served by local vLLM; `localhost` URLs are not APIs), the sandbox (Docker with `--network none`), training, embeddings (FastText `fast.bin` copied over, not downloaded), node2vec and mini-swe-agent (vendored in `third_party/`, installed from source).
  - One-time setup downloads on the GPU box: Gemma 4 E4B and 26B-A4B (51.6 GB) weights from HF, the pinned wheels and the Docker base image. After setup, set `HF_HUB_OFFLINE=1` and `LITELLM_LOCAL_MODEL_COST_MAP=True`.
  - Publishing (GitHub, HF datasets) is a separate step that needs the user's OK.
- **No RAG at inference.** Knowledge goes into the weights through OPSD. Privileged context is used only by the frozen teacher during training.
- **Splits:** group-level 80/20. A source entity never appears in two splits. A concept is grouped across all languages, and SNOMED IRIs are grouped across Bio-ML pairs. Whole pairs are held out for zero-shot testing. The DRAL mapping layer is fit on train references only.
- **Gates before any push**, GitHub or HF:
  - `pytest` is green;
  - parse and reference verification passes (OWL API cross-parse, ELK, Beisswanger & Hahn checks);
  - `verify.py` passes (coverage, no leakage, schema);
  - the secret and licensed-data scan is clean.
- **Repos:** GitHub `AGR19/OA-RLAIF` is **private**. HF repos are **private**. Get the user's OK before creating or pushing anything new.
- Do not edit files outside `RLAIF/`. The one exception is the one-line parent `.gitignore` entry.

## Where things are

- Full plan: `docs/PLAN.md`. Decision log: `MEMORY.md`. Method figure: `docs/figures/`.
- Raw data on the Mac: `../OA Finetune Small Model/datasets/Raw Ontology Folders/`. Bio-ML: `../data/external/bio-ml/`. On Lambda, set `OA_RAW_DIR` and `OA_BIOML_DIR`.
- DRAL-OA original code (read-only reference): `~/Desktop/DRAL_OA/melt-master/examples/externalPythonMatcherWeb/oaei-resources/old bert 2/clf.py`.

## Run order

On the Mac: `10_prepare_data.sh` → `15_datasource.sh` → `20_fit_dral.sh` → `80_verify_and_push.sh --hf` (pushes the datasets to private HF).

On Lambda: `scripts/lambda_all.sh` (one resumable command; see `LAMBDA_RUNBOOK.md`). Stages:
1. setup, then pull the private HF datasets;
2. `00_check_env.sh`, then `72_gguf_check.sh`;
3. `35_dry_run.sh` (times OPSD, SFT and GRPO with the judge, then plans the steps);
4. `30_baseline.sh`;
5. `40_sdft.sh`, then `50_eval.sh sdft`;
6. `60_grpo.sh`, then `50_eval.sh grpo --full`;
7. `70_export.sh`, then the private model push;
8. compulsory SFT comparison: `40_sdft.sh --mode sft` (same rows and steps as OPSD), then `50_eval.sh sft`;
9. `oa_rlaif.eval.report` (tables), then the private results push.

Mac end-to-end check of the same code paths (tiny Gemma 4, CPU, stub judge, real Docker agent): `scripts/smoke_mac.sh`.

## Data coverage (all raw data is used)

Every one of the 2,573 files in `Raw Ontology Folders` has a role in `configs/coverage.yaml`. The roles are: ontology facts, gold reference, ICD bridge, hard negatives/baselines (system outputs, never gold), eval tooling, knowledge passages (datasource, papers, benchmark docs), and provenance. `data/verify.py` fails if any file is unassigned, or if a data-bearing file yields zero records. The table is in `docs/PLAN.md`.

## Modelling assumptions

Every decision prompt carries the source and target **modelling profiles** (`data/ontology_profiles.json`), built by `python -m oa_rlaif.data.ontology_profiles`. Never drop them from prompts: train, eval and harness must see the same format. Report the Coherent column (ELK) and the assumption-sensitive slices with every evaluation.

## Alignment outputs

`python -m oa_rlaif.eval.align --source A --target B --backend vllm|openai|lexical --format oaei|sssom|tsv [--explain] [--reference R]`
- With a reference it writes the OAEI row: Matcher | Runtime | Size | Precision | F-Measure | Recall | Recall+ | Coherent.
- Without a reference it never reports P/R/F; only Size, Runtime, Coherent and the silver proxy.
- `lexical` is a labelled baseline, never presented as the model.

## Parse and reference verification

`python -m oa_rlaif.data.parse_verify` re-parses every ontology with the OWL API (`tools/owl_verify`) and requires an exact declared-class match with our parser. ELK reports consistency. Every reference gets the Beisswanger & Hahn (2012) quality checks. Results are in `docs/PARSE_VERIFICATION.md`. Mismatches or invalid references fail `verify.py` unless `configs/parse_exceptions.yaml` documents the reason. Never "fix" a mismatch by editing gold data.

## Hugging Face push

Private repos only. `scripts/80_verify_and_push.sh --hf` creates and uploads them once the gates pass.
- Datasets: `{ns}/oa-rlaif-datasource`, `-sdft`, `-grpo`, `-dral-reward`, `-icd-triangulation`.
- Models (pushed by `lambda_all.sh` at the end):
  - `{ns}/oa-rlaif-e4b-gguf`: Q4_K_M + Modelfile; base + OPSD + GRPO merged.
  - `{ns}/oa-rlaif-e4b-sdft` and `-grpo`: LoRA adapters.
  - Dataset `{ns}/oa-rlaif-results`.

The user runs `hf auth login` and sets `HF_NAMESPACE`. Never handle the token, and get the user's OK in chat before each push.

## Thesis writing

The user's dissertation chapter "RLAIF" lives at `~/Downloads/adishesh-Thesis/` (`asudis` template; see the ★ Thesis section of `docs/PLAN.md`).
- All thesis prose follows **ASD-STE100** Simplified Technical English. The user marked this as important.
- Report only measured results. Cite only DOI-verified entries.

## Conventions

- Python ≥ 3.11 with the `src/` layout (package `oa_rlaif`).
- Configs are YAML in `configs/`.
- Data outputs are JSONL in `data/` (gitignored).
- Every run writes a `manifest.yaml`: config hash, git SHA, seed, cost, and the data split hash.

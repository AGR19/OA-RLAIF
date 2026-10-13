# Full-alignment evaluation (`full_eval/`)

The earlier evaluation (`outputs/eval/report.md`, Table 1) scored **held-out decisions**: 20% of the reference links,
each with a candidate list. This folder scores **whole alignments**: every class of every dataset is aligned by each
model, and the alignment is scored against the whole reference, the way OAEI scores its participants.

| Task | Source → target | Classes aligned (per model) |
|---|---|---:|
| Anatomy | mouse → human | 2,737 |
| Bio-ML 2026 (whole ontologies) | NCIT → DOID, SNOMED CT → FMA, SNOMED CT → NCIT | ≈ 994,000 |
| MultiFarm (2015 test set) | 1,125 cross-lingual pairs | ≈ 119,000 |
| DH, DH multilingual | 8 + 10 pairs | ≈ 23,000 |
| Biodiv | ENVO → SWEET, FLOPO → PTO (zero-shot) | ≈ 30,000 |
| Zero-shot | Hydrography cree → SWO, IoTSec → UCO (+ FLOPO-PTO, DH tadirah-unesco) | small |
| ICD-10 → ICD-11 | every WHO ICD-10 code in the WHO tables | ≈ 12,600 |
| **SNOMED CT → ICD-11 (complete map)** | every active SNOMED CT concept in ICD's scope (clinical finding, event, situation; `FULL_EVAL_SNOMED_SCOPE=all` for every concept) | ≈ 130,000 |

Models: base Gemma 4 E4B, + SFT, + OPSD, + OPSD → GRPO (the released model), All merged; rebuilt on the GPU box from
the private Hugging Face repos (OPSD and GRPO re-merged from their adapters and checked against the earlier run's
predictions).

## Run (Vast.ai, 4 × B200, $50 cap)

1. Create an instance: 4 × B200 (any GPU with ≥ 40 GB works), **≥ 250 GB disk**, Ubuntu + CUDA image, SSH. Prefer
   on-demand.
2. SSH in, `tmux new -s eval`, then (branch `full-eval`; a plain `git clone` once it is merged into main):

```bash
cd /workspace 2>/dev/null || cd ~; git clone -b full-eval https://github.com/AGR19/OA-RLAIF.git && cd OA-RLAIF
```

```bash
full_eval/vast_full_eval.sh
```

It reads this instance's price, installs the Python environment (no Docker, no judge), asks once for your Hugging Face
token, downloads the inputs from the private repo `<you>/oa-rlaif-raw` (pushed from the Mac by
`full_eval/push_inputs.py`), shows **the plan and what $50 buys**, asks once, and runs to the end. Then it uploads
everything to the private dataset `<you>/oa-rlaif-full-eval`. **Destroy the instance afterwards.**
`full_eval/vast_full_eval.sh --plan` only shows the plan. `FULL_EVAL_BUDGET_USD=80 full_eval/vast_full_eval.sh` raises
the cap. If anything stops it, run the same command again: finished work is kept.

## The money

The cap counts this machine's billed time from its start (setup and downloads included), priced from the Vast.ai API.
A unit (one model × one task) **runs to the end or does not start**, so every reported alignment is complete. Units run
in this order and start only if their projected cost fits:

1. GRPO on every small task
2. GRPO: the complete SNOMED CT → ICD-11 map
3. SFT, All merged, OPSD and base on every small task (the comparison table)
4. GRPO on the three whole-ontology Bio-ML pairs
5. the other models on SNOMED CT → ICD-11, then on Bio-ML

Projection = decisions ÷ measured decisions per second (re-measured after every unit) + model load, × price. If the
spend still reaches the cap, the GPU workers stop between shards; the unit is reported as not run and the next run
continues it. **What $50 buys:** at roughly 120–200 decisions/s and $12–16/h, about 1.5–2.5 million decisions after
setup: items 1–3 (every model on every task except Bio-ML) and part of item 4. All five models on everything (≈ 6.5
million decisions) would cost about $150–200. Faster but not the evaluated weights: `FULL_EVAL_QUANT=fp8`.

## How precision and recall are computed

**Main scoring: DRAL-OA's `clf.py` regime** (`precision_at_k` / `recall_at_k` at k = 1). Every reference has its
classes scored this way, on every dataset: only the source classes that have a partner in the reference are scored;
the model's own top-1 choice for each (its first submitted target; NONE = nothing found) is a hit if it is a partner.
Precision@1 = mean of hits / min(1, partners), Recall@1 = mean of hits / partners, per pair, then the mean over pairs
(`AlignmentEvaluator.top1`). ICD-10 → ICD-11 uses the WHO table as the reference, SNOMED CT → ICD-11 the
ICD-10-composed links. Tables 1, 1b and 2 of the report. Classes without a counterpart are not scored in this regime.

**Also reported: OAEI's set-based leaderboard measure** (Tables 3 and 4), the one the OAEI 2026 table uses, checked
against the published protocols on 2026-10-13. Here the whole alignment counts, so a match proposed for a class without
a counterpart is wrong:

- **OAEI** (Anatomy, MultiFarm, DH, Biodiv, Complex, IoTSec): P = |A∩R|/|A|, R = |A∩R|/|R|, F = 2PR/(P+R), with A the
  whole alignment and R the whole reference. Equivalences only (OAEI Anatomy and DH say so); vocabulary
  correspondences such as `oboInOwl#Synonym = oboInOwl#Synonym` are removed. **Recall+** = recall on non-trivial
  references (OAEI's `trivial.rdf` for Anatomy; normalised label equality elsewhere). DH reports per test case and the
  mean; MultiFarm and Bio-ML aggregate over tasks: we give the mean over pairs (macro) and pooled counts (micro).
- **Bio-ML 2026** (global matching): the same P/R/F against the repaired reference, macro-averaged over the three
  pairs. Its test part is kept by the organisers: recall here is over the public train and valid references, and
  precision is a lower bound. `alignment.rdf` is the submission file.
- **Coherence** (OAEI's Pellet check) is not computed.

Our alignment: every class gets C# or NONE (the trained prompt: k = 40 candidates for Anatomy, 20 elsewhere; both
modelling profiles; thinking off; temperature 0); matches with confidence ≥ 0.5 are kept; greedy 1:1 by confidence.

**Two views of every number.** *Full*: every reference link, as OAEI scores it. *Held-out*: the links in our training
split set to null (neither right nor wrong), as Bio-ML's semi-supervised setting does with its training mappings.
The models were trained on 80% of these references, which OAEI rules do not allow for an official entry ("Using
published OAEI tests (with or without reference alignments) for learning should be considered as prohibited"), so
**only the held-out view and the zero-shot pairs are free of training data.**

**SNOMED CT → ICD-11** has no gold: we report agreement with the links composed through WHO ICD-10 (the earlier
"bridge recovery", now over every concept) and the complete map as SSSOM (`ICD-11 category skos:narrowMatch SNOMED
concept`: the category classifies the concept; not clinical equivalence).

## Matcher variants

`full_eval/variants.py` adds two rows built from the models' own alignments (no GPU time): **GRPO + exact-label
anchors** (pairs with the same normalised label, OAEI's StringEquiv, as LogMap anchors and MILA's "identify" step) and
**GRPO ∩ All merged + exact-label anchors**. On the earlier run's whole-Anatomy alignments they scored F 0.904 and
0.908 against the full reference (GRPO alone 0.883). They were chosen after that measurement, so they are reported
next to the plain model rows, never instead of them, and with the held-out view too.

## Outputs (`outputs/full_eval/`, uploaded to `<you>/oa-rlaif-full-eval`)

- `report.md` / `.tex` / `.json`: Table 1 (full), Table 1b (held-out), Table 2 (OAEI rows per pair), the OAEI 2026
  Anatomy rows for reference, Table 3 (the SNOMED CT → ICD-11 map), Table 4 (every unit: done, or why not).
- `results/<model>/<task>/`: `scores.json`; `decisions.tsv.gz` (every class: decision, target IRI and label, relation,
  confidence, the model's THOUGHT); `pairs/<pair>/alignment.rdf` (OAEI format) and `alignment.sssom.tsv`. SNOMED CT →
  ICD-11: `icd11_snomed_complete.sssom.tsv` and `snomed_to_icd11.tsv.gz`.
- `replies/<model>/<task>/`: the raw model output for every class. `plan.json`, `speed.json`, `verify_*.json`, `logs/`.

## Pieces

| File | What |
|---|---|
| `vast_full_eval.sh`, `setup.sh`, `price.py` | the one command, the lean GPU-box setup, this machine's price |
| `push_inputs.py` (Mac), `pull_inputs.py` (box) | raw folders (without PDFs), Bio-ML, parsed-ontology cache → private HF → box |
| `tasks.py` | the tasks, what "every class" means (vocabulary and deprecated entities left out), references, splits |
| `build_episodes.py` | one prompt per class, as in training |
| `models.py` | the five models from the private repos |
| `infer.py` | one vLLM per GPU; resumable shards; the money guard's STOP |
| `schedule.py`, `run.py` | units, priority, money projection; the orchestrator |
| `score.py`, `variants.py`, `report.py` | scoring, variants, tables |
| `push_results.py` | every output to the private dataset |

Mac rehearsal (no GPU; a lexical matcher stands in for the model, `$1/h` stand-in price):

```bash
PYTHONPATH=src:. OA_OUT_DIR=/tmp/fe FULL_EVAL_FAKE=lexical FULL_EVAL_GPUS=1 FULL_EVAL_MODELS=grpo,all_merged FULL_EVAL_TASKS=anatomy__mouse-human,dh .venv/bin/python -m full_eval.run --yes
```

# Full-alignment evaluation (`full_eval/`)

The earlier evaluation (`outputs/eval/report.md`, Table 1) scored **held-out decisions**: 20% of the reference links,
each with a candidate list. This folder scores **full alignments only**: every class of every dataset is aligned by
each method, and every alignment is scored against the whole reference. There is no held-out view.

| Task | Source → target | Classes aligned (per method) |
|---|---|---:|
| Anatomy | mouse → human | 2,737 |
| Bio-ML 2026 (whole ontologies) | NCIT → DOID, SNOMED CT → FMA, SNOMED CT → NCIT | ≈ 994,000 |
| MultiFarm (2015 test set) | 1,125 cross-lingual pairs | ≈ 119,000 |
| DH, DH multilingual | 8 + 10 pairs | ≈ 23,000 |
| Biodiv | ENVO → SWEET, FLOPO → PTO (zero-shot) | ≈ 30,000 |
| Cybersecurity | IoTSec → UCO (raw folder `cybersecurity/`; never trained on) | small |
| Zero-shot | Hydrography cree → SWO (+ FLOPO → PTO, DH tadirah → UNESCO) | small |
| ICD-10 → ICD-11 | every WHO ICD-10 code in the WHO tables | ≈ 12,600 |
| **SNOMED CT → ICD-11 (complete map)** | every active SNOMED CT concept in ICD's scope (clinical finding, event, situation; `FULL_EVAL_SNOMED_SCOPE=all` for every concept) | ≈ 130,000 |

## Methods

Six methods, each run with **thinking off and thinking on** (the models were trained and evaluated with thinking off):

| Method | Tag | How it answers |
|---|---|---|
| Gemma 4 E4B (base) | `base` | one decision per class: the earlier evaluation's prompt (40 candidates for Anatomy, 20 elsewhere, both modelling profiles), C# or NONE, temperature 0 |
| + SFT | `sft` | the same |
| + OPSD | `sdft` | the same |
| + OPSD → GRPO (RLAIF) | `grpo` | the same |
| All merged (OPSD → GRPO + SFT) | `all_merged` | the same |
| All merged + mini-SWE-agent | `all_merged-agent` | the agent of the local console (`om_agent/`), one episode per class: scripted `om profile`, `om show <class>`, `om candidates <class> --k N`, then the model with the trained agent prompt until `om submit` (step limit 6) |

Thinking on = the tag + `-think` (e.g. `grpo-think`, `all_merged-agent-think`): the chat template's
`enable_thinking`, up to 4,096 new tokens. The thinking text is split off from the answer
(`oa_rlaif.eval.thinking.split`, the same function as in the console); when the model writes its answer inside the
thinking (it often does, having been trained with thinking off), the answer is taken from there, and the report counts
how often.

All six methods see the same episodes (the same candidates for each class). The models are rebuilt on the GPU box
from the private Hugging Face repos (OPSD and GRPO re-merged from their adapters; a weight check compares the re-merged
models with the earlier run's replies; it is not an evaluation and never enters the tables).

**The agent on Vast.ai** runs exactly the console's code (`om_agent/test_matching_ai.py`, `Matcher`), with vLLM
instead of Ollama (`vllm serve` per GPU, `full_eval/agent_run.py`). Its commands run in the **user sandbox**: the
unprivileged user `oa-agent`, a task folder outside the repo with the public episode only, the repo closed to it
(`scripts/sandbox_setup.sh`, self-tested by `setup.sh`). **The agent never asks for and never sees a reference**: the
references are read from the raw folders afterwards, by `score.py`, for scoring only.

## Run (Vast.ai, 4 × B200, $100 cap)

1. Create an instance: 4 × B200 (any GPU with ≥ 40 GB works), **≥ 250 GB disk**, Ubuntu + CUDA image, SSH. Prefer
   on-demand.
2. SSH in, `tmux new -s eval`, then (branch `full-eval`; a plain `git clone` once it is merged into main):

```bash
cd /workspace 2>/dev/null || cd ~; git clone -b full-eval https://github.com/AGR19/OA-RLAIF.git && cd OA-RLAIF
```

```bash
full_eval/vast_full_eval.sh
```

It reads this instance's price, installs the Python environment and the agent's user sandbox (no Docker, no judge),
asks once for your Hugging Face token, downloads the inputs from the private repo `<you>/oa-rlaif-raw` (pushed from the
Mac by `full_eval/push_inputs.py`), shows **the plan and what $100 buys**, asks once, and runs to the end. Then it
uploads everything to the private dataset `<you>/oa-rlaif-full-eval`. **Destroy the instance afterwards.**
`full_eval/vast_full_eval.sh --plan` only shows the plan. `FULL_EVAL_BUDGET_USD=150 full_eval/vast_full_eval.sh` raises
the cap; `FULL_EVAL_THINKING=off` runs thinking off only; `FULL_EVAL_METHODS=grpo,all_merged-agent` narrows the
methods. If anything stops it, run the same command again: finished work is kept.

## The money

The cap counts this machine's billed time from its start (setup and downloads included), priced from the Vast.ai API.
A unit (one method × one task) **runs to the end or does not start**, so every reported alignment is complete. Units run
in this order and start only if their projected cost fits:

1. thinking off, all six methods, every task except Bio-ML: the small tasks, then SNOMED CT → ICD-11
2. the same with thinking on
3. Bio-ML (three whole-ontology pairs): thinking off, then on

Projection = decisions ÷ measured decisions per second of that kind of method + load, × price. The four kinds (one
decision or agent, thinking off or on) are measured separately, after each finished unit; until then the plan uses
120/s (one decision), 20/s (one decision, thinking on), 25/s (agent) and 8/s (agent, thinking on), labelled
"assumed". If the spend still reaches the cap, the GPU workers stop between shards; the unit is reported as not run and
the next run continues it. All methods on everything (≈ 16 million decisions, most of it Bio-ML and the thinking-on
and agent runs) costs more than $100; the order above decides what is left out, and the plan shows it before the run.

## How precision and recall are computed (full alignments only)

Every score uses the whole reference; no reference link is set aside or treated as null. The models were trained on
part of these references (the 80/20 split of the earlier run), which OAEI rules do not allow for an official entry; the
zero-shot pairs and IoTSec → UCO were never trained on. The report says so.

**1. OAEI scores of the whole alignment** (the published protocol, as MELT computes it): P = |A∩R|/|A|,
R = |A∩R|/|R|, F = 2PR/(P+R), with A the method's whole alignment (matches with confidence ≥ 0.5, greedy 1:1 by
confidence) and R the whole reference. Equivalences only (OAEI Anatomy and DH say so); vocabulary correspondences such
as `oboInOwl#Synonym = oboInOwl#Synonym` are removed. IoTSec → UCO has only 'related' links, so all its links count.
**Recall+** = recall on non-trivial references (OAEI's `trivial.rdf` for Anatomy; normalised label equality elsewhere).
Several pairs: mean over pairs (macro) and pooled counts (micro). **Bio-ML 2026**: the organisers keep the test
references, so recall is over the public train and valid references and precision is a lower bound; `alignment.rdf` is
the submission file. Coherence (OAEI's Pellet check) is not computed.

**2. The Top-k evaluation (k = 1, 3, 5)**: Precision@k, Recall@k and F-measure@k of DRAL-OA's `clf.py`
(`precision_at_k` / `recall_at_k`; `AlignmentEvaluator.topk`). Only the source classes that have a partner in the
reference are scored. Each class's ranked answers are the method's own choice first, then its other candidates in
retrieval order; NONE or an invalid reply = nothing found. Per class c, with F_c its top k and R_c its partners:
Precision@k = |F_c ∩ R_c| / min(k, |R_c|), Recall@k = |F_c ∩ R_c| / |R_c|, averaged over the scored classes;
F-measure@k = 2PR / (P + R); then the mean over pairs. Every reference link counts, whatever its relation (as in
`clf.py`).

**ICD-10 → ICD-11**: against the WHO tables, every code: accuracy as in the earlier Table 1, P/R/F of the (code,
category) pairs, Top-k. **SNOMED CT → ICD-11** has no reference: agreement with the links composed through WHO ICD-10
(a silver standard, labelled so) and the complete map as SSSOM (`ICD-11 category skos:narrowMatch SNOMED concept`: the
category classifies the concept; not clinical equivalence).

## The report (`report.md`)

- **Part 1, OAEI scores**: Table 1 F-measure per dataset (+ ICD-10 → 11 accuracy, SNOMED → ICD-11 agreement), 12 rows
  (six methods × thinking off/on); Table 2 the OAEI rows per pair (Runtime, Size, P, F, R, Recall+, Coherent);
  Table 3 Anatomy beside the published OAEI 2026 systems and DRAL-OA's official OAEI 2025 result; Table 4 ICD-10 → 11.
- **Part 2, the Top-k evaluation**: Table 5 F-measure@1 per dataset; Table 6 Precision, Recall and F-measure at
  k = 1, 3, 5; `figures/topk_<dataset>_thinking-<off|on>.svg` (Precision, Recall and F-measure vs. Top-k).
- **Part 3, what the mini-SWE-agent adds to All merged**: Table 7 the differences (OAEI F and Top-1 F, thinking off
  and on); Table 8 what the agent did and cost (model calls, extra commands, format errors, no submission, tokens,
  runtime, $).
- **Part 4, thinking on against off**: Table 9 the differences per method; Table 10 what the thinking did (answer
  inside the thinking, no thinking text, cut at the token limit, thinking length, tokens, invalid replies, runtime).
- Table 11 the complete SNOMED CT → ICD-11 map; Table 12 every unit (method × task): done, or why not, and the money.

## Outputs (`outputs/full_eval/`, uploaded to `<you>/oa-rlaif-full-eval`)

- `report.md` / `.tex` / `.json`, `figures/`.
- `results/<method>/<task>/`: `scores.json` (OAEI scores, Top-k, behaviour); `decisions.tsv.gz` (every class:
  decision, target IRI and label, relation, confidence, the model's THOUGHT); `pairs/<pair>/alignment.rdf` (OAEI
  format) and `alignment.sssom.tsv`. SNOMED CT → ICD-11: `icd11_snomed_complete.sssom.tsv` and
  `snomed_to_icd11.tsv.gz`.
- `replies/<method>/<task>/`: the output for every class (thinking on: the thinking text; agent: its commands and
  every model turn). `plan.json`, `speed.json`, `verify_*.json` (weight check), `logs/`.

## Pieces

| File | What |
|---|---|
| `vast_full_eval.sh`, `setup.sh`, `price.py` | the one command, the lean GPU-box setup (with the agent's user sandbox), this machine's price |
| `push_inputs.py` (Mac), `pull_inputs.py` (box) | raw folders (without PDFs), Bio-ML, parsed-ontology cache → private HF → box |
| `tasks.py` | the tasks, what "every class" means (vocabulary and deprecated entities left out), references |
| `build_episodes.py` | one prompt per class, as in training |
| `models.py` | the five models from the private repos |
| `infer.py` | one-decision methods: one vLLM per GPU, thinking off or on; resumable shards; the money guard's STOP |
| `agent_run.py` | the agent method: `vllm serve` per GPU and `om_agent`'s Matcher over the same shards |
| `schedule.py`, `run.py` | methods, units, priority, money projection; the orchestrator |
| `score.py`, `report.py` | scoring (OAEI scores, Top-k, behaviour), tables and figures |
| `push_results.py` | every output to the private dataset |

Mac rehearsal (no GPU; a lexical matcher stands in for the model, also behind the agent; `$1/h` stand-in price):

```bash
PYTHONPATH=src:. OA_OUT_DIR=/tmp/fe FULL_EVAL_FAKE=lexical FULL_EVAL_GPUS=1 OA_AGENT_ENV=local FULL_EVAL_METHODS=grpo,all_merged,all_merged-agent,grpo-think,all_merged-agent-think FULL_EVAL_TASKS=cyber__iotsec-uco,complex_om__cree-swo .venv/bin/python -m full_eval.run --yes
```

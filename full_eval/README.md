# Full evaluation (`full_eval/`)

The earlier evaluation (`outputs/eval/report.md`, Table 1) scored **held-out decisions**: 20% of the reference links,
each with a candidate list. This folder has **no held-out view**: every score uses the whole reference.

**The default run** (decided 2026-10-22):

- **Score:** the **Top-k evaluation (k = 1, 3, 5)**, the formula of DRAL-OA's `clf.py`
  ([below](#the-top-k-evaluation-k--1-3-5)). No OAEI scores.
- **Classes:** every source class that has a partner in the reference, all of them. These are the classes that the
  formula scores. Each class is its own episode, so the scores are the same as when every class is aligned.
  - The model never sees the reference. The reference only selects the classes and, after the run, scores them.
  - Each class is matched against the whole target ontology: retrieval, then the model's choice.
- **SNOMED CT → ICD-11:** the complete map. Every concept in ICD's scope runs (141,182). The 111,511 concepts that
  have a strong composed link are scored.
- **Methods:**
  - five models with thinking off: base, + SFT, + OPSD, + OPSD → GRPO and All merged;
  - All merged + mini-SWE-agent with thinking on.
- **Money:** about 15 h and $363 on 4 × B300 at $24/h. The cap is $425.

`FULL_EVAL_SCOPE=all` aligns every class of every source ontology (1,438,731 per method) and adds the OAEI scores of the
whole alignment. `FULL_EVAL_METHODS=all` runs all 12 methods: the six, each with thinking off and on. That complete
evaluation costs about $4,400 (about 184 h).

| Task | Source → target | Pairs | Classes run (default) | Every class (`FULL_EVAL_SCOPE=all`) | Target entities | Reference links |
|---|---|---:|---:|---:|---:|---:|
| Anatomy | mouse (MA) → human (NCI) | 1 | 1,497 | 2,737 | 3,298 | 1,516 |
| Bio-ML | NCIT → DOID | 1 | 3,792 | 206,475 | 12,193 | 3,792 |
| Bio-ML | SNOMED CT → FMA | 1 | 3,776 | 390,809 | 104,888 | 3,780 (3,776) |
| Bio-ML | SNOMED CT → NCIT | 1 | 20,122 | 390,809 | 206,475 | 20,147 (20,123) |
| MultiFarm (2015 test set) | ontology in language 1 → ontology in language 2 | 1,125 | 35,037 | 119,081 | 74–182 | 35,394 (35,214) |
| DH | `source.rdf` → `target.rdf` (1 pair zero-shot) | 8 | 157 | 4,729 | 68–800 | 168 |
| DH multilingual | iDAI → PACTOLS, by language | 10 | 98 | 18,093 | 56–68 | 98 |
| Biodiv | ENVO → SWEET | 1 | 601 | 6,593 | 4,884 | 663 |
| Biodiv (zero-shot) | FLOPO → PTO | 1 | 227 | 23,236 | 1,503 | 244 (230) |
| Biodiv 2026 (zero-shot; OAEI 2026 track, raw folder `biodiv-2026/`, classes only) | ENVO → SWEET (2026), Fish → Zooplankton, Macroalgae → Macrozoobenthos, TAXREF-LD → NCBI Taxonomy (Animalia, Plantae, Fungi, Chromista, Protozoa, Bacteria) | 9 | 80,825 | 122,039 | 56–74,729 | 81,070 |
| Cybersecurity (zero-shot) | IoTSec → UCO | 1 | 35 | 251 | 206 | 69 |
| Zero-shot | hydrography Cree → SWO | 1 | 15 | 100 | 105 | 15 (+32 complex) |
| ICD-10 → ICD-11 | every WHO ICD-10 code (11,243 categories, 1,058 modified categories, 274 blocks, 22 chapters) → ICD-11 Foundation entity | 1 | 12,597 | 12,597 | 71,175 | 16,555 (WHO tables) |
| **SNOMED CT → ICD-11 (complete map)** | every active concept of Clinical finding (132,172), Situation with explicit context (5,605) and Event (3,402), plus the 3 roots: the scope of the official SNOMED CT → ICD-10 map (`FULL_EVAL_SNOMED_SCOPE=all` for all 386,110 active concepts) → ICD-11 Foundation entity | 1 | 141,182 (111,511 scored) | 141,182 | 71,175 | none (231,690 composed silver links) |
| **Total** | | 1,162 | **299,961** | 1,438,731 | | |

Notes on the counts:
- **Entities** are declared classes and properties, without vocabulary, placeholder and deprecated entities.
- **Classes run (default)** are the source classes that have a partner in the reference: links between the two
  ontologies, any relation, as `clf.py` counts them.
- **Deprecated reference classes:** 12 are not run and count as misses: 1 SNOMED CT concept in SNOMED CT → NCIT,
  5 ENVO classes in Biodiv and 6 in Biodiv 2026.
- **Reference links in parentheses** are the links whose two entities are in the two ontologies. The Top-k evaluation
  uses these; the OAEI scores (`FULL_EVAL_SCOPE=all`) use all.
- **SNOMED CT coverage**, counted from RF2 US 2026-03:
  - 136,507 concepts are in the official SNOMED CT → ICD-10 map (refset 447562003);
  - 111,591 of them have a usable ICD-10 target;
  - every one of those has a composed ICD-11 link, and 111,511 have a strong link.
- **Decisions:** the default run is 5 × 299,961 one-call decisions + 299,961 agent episodes, about 1.8 million. The
  complete evaluation is 12 × 1,438,731, about 17.3 million.

## Directions (checked 2026-10-22)

Every task aligns the source ontology to the target ontology in the direction of its official reference file
(`entity1` → `entity2`, or Bio-ML's `SrcEntity` → `TgtEntity`), the direction MELT and the OAEI organisers use. A
check over every pair (`orient` in `oa_rlaif.ontology.alignment_io`) found:

| Task | Source → target | Reference links as written | Notes |
|---|---|---:|---|
| Anatomy | mouse (MA) → human (NCI) | 1,516 of 1,516 | reference `entity1` = `mouse.owl#…`, `entity2` = `human.owl#…` |
| Bio-ML | NCIT → DOID, SNOMED CT → FMA, SNOMED CT → NCIT | 3,792; 3,776 (+4 unknown); 20,123 (+24 unknown) | `SrcEntity`/`TgtEntity`; SNOMED CT from our US 2026-09 OWL (Bio-ML pins 2026-03-01): 28 concepts not in it |
| MultiFarm | o1 in l1 → o2 in l2 (`cmt-confOf-fr-ar`: cmt in French → confOf in Arabic), 1,125 pairs | 35,127 (+87 flipped, +182 unknown) | `cmt-cmt-fr-ar.rdf` is written Arabic → French (flipped when read); 16 Arabic–French files name 182 entities that are in no ontology file (a defect of the 2015 test set) |
| DH, DH-ML | `source.rdf` → `target.rdf` (8 + 10 pairs) | all | |
| Biodiv | ENVO → SWEET, FLOPO → PTO | 666; 231 (+14 unknown) | 5 ENVO sources are obsolete classes (not aligned, so they count as misses); FLOPO → PTO cites 14 PATO and PO classes |
| Biodiv 2026 | `source.xml` → `target.xml` of each OAEI 2026 test case (ENVO → SWEET, Fish → Zooplankton, Macroalgae → Macrozoobenthos, TAXREF-LD → NCBI ×6) | all 81,071 | classes only (`parameters.xml`); 6 ENVO 2026 sources are obsolete classes; never trained on (zero-shot) |
| Cybersecurity | IoTSec → UCO | 69 of 69 | all links 'related'; IoTSec writes empty labels, so names come from the IRIs |
| Zero-shot | hydrography cree → SWO | 15 simple (+32 complex) | Cree names share no letters with SWO's English names: retrieval finds no partner (ceiling 0) |
| ICD-10 → ICD-11 | every WHO ICD-10 code → ICD-11 category | WHO `10To11` tables | |
| SNOMED CT → ICD-11 | every active concept in ICD's scope → ICD-11 category (`classified_under`) | no reference; silver links composed through ICD-10 | direction and relation follow the SNOMED International / WHO pilot (Fung et al. 2024, doi:10.1093/jamia/ocae143): SNOMED CT → ICD-11 is the statistical use case, where a specific concept goes to a broader category; only 32% of the pilot's concepts had an exact precoordinated match |

"Unknown" links name classes of neither ontology: no matcher can find them, and MELT counts them in the reference,
so the OAEI scores keep them (the Top-k evaluation uses the links between the two ontologies).

## Methods

Six methods exist. Each one can run with thinking off or on (the models were trained and evaluated with thinking off).
**The default run is the five models with thinking off and the agent with thinking on.** Choose others with
`FULL_EVAL_METHODS` (tags, or `all`) and `FULL_EVAL_THINKING` (`off`, `on` or `both`).

| Method | Tag | How it answers |
|---|---|---|
| Gemma 4 E4B (base) | `base` | one decision per class: the earlier evaluation's prompt (40 candidates for Anatomy, 20 elsewhere, both modelling profiles), C# or NONE, temperature 0 |
| + SFT | `sft` | the same |
| + OPSD | `sdft` | the same |
| + OPSD → GRPO (RLAIF) | `grpo` | the same |
| All merged (OPSD → GRPO + SFT) | `all_merged` | the same |
| All merged + mini-SWE-agent | `all_merged-agent` | the agent of the local console (`om_agent/`), one episode per class: scripted `om profile`, `om show <class>`, `om candidates <class> --k N`, then the model with the trained agent prompt until `om submit` (step limit 6) |

**Reply length.** One-call methods with thinking off may write up to **1,024** new tokens per reply
(`FULL_EVAL_MAX_TOKENS`). The first run (2026-10-22) used the earlier evaluation's 256. That limit cut base, OPSD and
OPSD → GRPO replies before their `om submit`: 63%, 27% and 10% of their answers became invalid, every one at exactly
256 tokens. SFT and All merged write short replies and were never cut. Every reply now records its finish reason, and
Table 4 shows the share cut at the limit.

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

## Run (Vast.ai, 4 × B300, $425 cap)

1. Create an instance:
   - 4 × B300 (any GPU with ≥ 40 GB works);
   - **≥ 250 GB disk**;
   - an Ubuntu + CUDA image, with SSH;
   - on-demand is better than interruptible.
2. SSH in and run `tmux new -s eval`. Then clone the `full-eval` branch (a plain `git clone` once it is merged into
   main):

```bash
cd /workspace 2>/dev/null || cd ~; git clone -b full-eval https://github.com/AGR19/OA-RLAIF.git && cd OA-RLAIF
```

```bash
full_eval/vast_full_eval.sh
```

The script does these steps, then runs to the end:
1. It reads this instance's price.
2. It installs the Python environment and the agent's user sandbox (no Docker, no judge).
3. It asks once for your Hugging Face token.
4. It downloads the inputs from the private repo `<you>/oa-rlaif-raw-data` (pushed from the Mac by
   `full_eval/push_inputs.py`).
5. It shows **the plan and what the cap buys**, and asks once.

Then it uploads everything to the private dataset `<you>/oa-rlaif-full-eval-results`. **Destroy the instance
afterwards.**

Options:
- `full_eval/vast_full_eval.sh --plan` only shows the plan.
- `FULL_EVAL_BUDGET_USD=500 full_eval/vast_full_eval.sh` changes the cap.
- `FULL_EVAL_RERUN=base,sdft,grpo full_eval/vast_full_eval.sh` runs only those methods again.
  - **The first run's repo is never changed.** `<you>/oa-rlaif-full-eval-results` is only read; uploading to it
    during a rerun is refused.
  - The box copies the first run without those methods' files. Only the three methods run again, and everything goes
    to a new private dataset, `<you>/oa-rlaif-fixed-full-eval-results`: the other methods' files exactly as they
    were, the rerun methods' new files, and a report with all six methods.
- `FULL_EVAL_SCOPE=all FULL_EVAL_METHODS=all FULL_EVAL_BUDGET_USD=4500 full_eval/vast_full_eval.sh` runs the complete
  evaluation (every class, 12 methods, OAEI scores too).

If anything stops it, run the same command again: finished work is kept. The script refuses episodes that were
built with the other `FULL_EVAL_SCOPE`. Use a new `OA_OUT_DIR` to change the scope.

## The money

The cap counts this machine's billed time from its start, setup and downloads included, priced from the Vast.ai API.
A unit (one method × one task) **runs to the end or does not start**, so every reported unit is complete.

**Order.** Units run in this order, and each one starts only if its projected cost fits:
1. The thinking-off methods, one after the other, each on every task: the small tasks, then SNOMED CT → ICD-11, then
   Bio-ML. Each model loads once.
2. The thinking-on methods, in the same order.

**Projection** = decisions ÷ the measured decisions per second of that kind of method, + load time, × price.
- The four kinds are measured separately, after each finished unit: one decision or agent, thinking off or on.
- Until a kind is measured, the plan uses an assumed rate, labelled "assumed": 120/s (one decision), 20/s (one
  decision, thinking on), 25/s (agent) and 8/s (agent, thinking on).
- If the spend still reaches the cap, the GPU workers stop between shards. The unit is reported as not run, and the
  next run continues it.

**The default run on 4 × B300 at about $24/h:**

| Task | Classes run | One model, thinking off | Agent, thinking on |
|---|---:|---:|---:|
| Anatomy | 1,497 | <1 min | 3 min |
| DH, DH multilingual, Biodiv | 1,083 | <1 min | 2 min |
| IoTSec → UCO, Cree → SWO | 50 | <1 min | <1 min |
| ICD-10 → ICD-11 | 12,597 | 2 min | 26 min |
| MultiFarm | 35,037 | 5 min | 1.2 h |
| Biodiv 2026 | 80,825 | 11 min | 2.8 h |
| SNOMED CT → ICD-11 (complete map) | 141,182 | 20 min | 4.9 h |
| Bio-ML | 27,690 | 4 min | 58 min |
| **Total** | **299,961** | **42 min per model** | **10.4 h** |

| Item | Time | Cost |
|---|---:|---:|
| Setup and downloads (4 models + 2 adapters to merge) | ~45 min | ~$18 |
| 5 models, thinking off (with loads) | 3.7 h | ~$88 |
| Agent, thinking on | 10.5 h | ~$251 |
| Scoring, report, upload | ~10 min | ~$4 |
| **Expected total** | **~15.1 h** | **~$363** |

- **Range:** about $300 at $20/h to about $422 at $28/h.
- **If the agent with thinking on runs at 5/s instead of 8/s:** add about $150.
- **With a $100 cap:** the five thinking-off models fit, but the agent reaches only the small tasks.

## How the scores are computed (no held-out view)

Every score uses the whole reference; no reference link is set aside or treated as null. The models were trained on
part of these references (the 80/20 split of the earlier run). The zero-shot pairs, IoTSec → UCO and the OAEI 2026
Biodiversity and Ecology test cases were never trained on. The report says so.

### The Top-k evaluation (k = 1, 3, 5)

This is the formula of DRAL-OA's evaluation code, `clf.py` (`precision_at_k`, `recall_at_k`), and
`AlignmentEvaluator.topk` in `src/oa_rlaif/eval/metrics.py`. For each source class that has partners in the
reference:

```
precision_at_k = (partners found in top k) / min(k, number of partners)
recall_at_k    = (partners found in top k) / number of partners
```

The reported Precision@k and Recall@k are the **means over those classes**:

```
Precision@k = 1/|C| Σ_c |F_c^k ∩ R_c| / min(k, |R_c|)
Recall@k    = 1/|C| Σ_c |F_c^k ∩ R_c| / |R_c|
F-measure@k = 2 · Precision@k · Recall@k / (Precision@k + Recall@k)
```

| In words | In the formula | In `clf.py` |
|---|---|---|
| partners found in top k | \|F_c^k ∩ R_c\| | `n_rel_and_rec_k` |
| number of partners | \|R_c\| | `len(trgs)` |
| min(k, number of partners) | min(k, \|R_c\|) | `min([k, len(trgs)])` |
| the mean over classes | 1/\|C\| Σ_c | `np.mean(precisions)`, `np.mean(recall)` |

The symbols in the formula:
- **C** is the source classes that have a partner in the reference. These are `clf.py`'s `link_labels`, the classes
  this run aligns.
- **R_c** is the partners of class c in the reference: every relation (as `clf.py` reads every reference row), links
  between the two ontologies.
- **F_c^k** is the method's top k answers for c: its own choice first, then its other candidates in retrieval order.
  NONE or an invalid reply counts as nothing found.

The table gives the mean over pairs for each dataset. A worked example is the Anatomy run of the local console
(`s2`): the two lines are computed for each of the 1,497 mouse classes with a partner, then averaged. At k = 1 this
gives Precision 0.941 and Recall 0.934.

**ICD-10 → ICD-11:** the reference is the WHO 10To11 tables, and every code is scored. **SNOMED CT → ICD-11** has no
reference:
- Top-k is computed against the strong links composed through WHO ICD-10. This is a silver standard, and the report
  labels it so.
- The report also gives the agreement with the strong composed links and with any composed link.
- The complete map is written as SSSOM (`ICD-11 category skos:narrowMatch SNOMED concept`). This means the category
  classifies the concept; it is not clinical equivalence.

### OAEI scores of the whole alignment (only with `FULL_EVAL_SCOPE=all`)

This is the published protocol, as MELT computes it: P = |A∩R|/|A|, R = |A∩R|/|R|, F = 2PR/(P+R).
- **A** is the method's whole alignment: matches with confidence ≥ 0.5, greedy 1:1 by confidence.
- **R** is the whole reference.
- **Equivalences only**, as OAEI Anatomy and DH say. Vocabulary correspondences such as
  `oboInOwl#Synonym = oboInOwl#Synonym` are removed.
- **IoTSec → UCO** has only 'related' links, so all its links count.
- **Links outside both ontologies** stay in R, as MELT reads the file: FLOPO → PTO cites 14 PATO and PO classes.
- **Recall+** is recall on the non-trivial references, OAEI's Anatomy measure (its `trivial.rdf`). No other track
  defines it, so it shows "–" there.
- **Several pairs:** the mean over pairs (macro) and the pooled counts (micro).
- **Bio-ML 2026:** the organisers keep the test references. So recall is over the public train and valid references,
  and precision is a lower bound.
- **Coherence** (OAEI's Pellet check) is not computed.

These scores need every class to be aligned, so the default run does not compute them.

## The report (`report.md`)

Default (`FULL_EVAL_SCOPE=reference`). The rows are the six methods of the run:
- **Part 1, the Top-k evaluation:**
  - Table 1: F-measure@1 per dataset;
  - Table 2: Precision, Recall and F-measure at k = 1, 3, 5 per dataset;
  - Table 3: per pair (MultiFarm, DH and DH-ML: the mean over their pairs), with the classes scored and the retrieval
    ceiling;
  - `figures/topk_<dataset>.svg`: Precision, Recall and F-measure vs. Top-k, one line per method.
- **Part 2, what the methods did.** Table 4 shows:
  - every method: match, NONE and invalid answers, and tokens;
  - the agent: model calls, extra commands, format errors and seconds per class;
  - the thinking: its length, answers written inside it, and replies cut at the token limit;
  - runtime and $.
- **Part 3:**
  - Table 5: the complete SNOMED CT → ICD-11 map;
  - Table 6: every unit (method × task), done or why not, and the money.

With `FULL_EVAL_SCOPE=all`, the report has the earlier 12 tables instead:
- the OAEI scores: Tables 1–4, with Anatomy beside the published OAEI 2026 systems;
- Top-k: Tables 5–6;
- what the agent adds: Tables 7–8;
- thinking on against off: Tables 9–10;
- the SNOMED map and the units: Tables 11–12.

## Outputs (`outputs/full_eval/`, uploaded file by file to `<you>/oa-rlaif-full-eval-results`)

- `report.md` / `.tex` / `.json`, and `figures/`.
- `results/<task>/<method folder>/`: one folder per dataset task, and one folder per method inside it (layout
  below):
  - `scores.json`: Top-k, the retrieval ceiling and the behaviour; the OAEI scores with `FULL_EVAL_SCOPE=all`.
  - `decisions.tsv.gz`: every class run, with its decision, target IRI and label, relation, confidence and the model's
    THOUGHT.
  - **One alignment per method and pair:** `pairs/<pair>/alignment.rdf` (OAEI format) and `alignment.sssom.tsv`,
    holding the matches with confidence ≥ 0.5, greedy 1:1. In the default run they hold the classes that ran (the
    reference classes), so they are not whole-ontology alignments and get no OAEI score.
  - ICD-10 → ICD-11: `alignment.rdf` of every WHO code.
  - SNOMED CT → ICD-11: the complete map, `icd11_snomed_complete.sssom.tsv` and `snomed_to_icd11.tsv.gz`.
- `replies/<method>/<task>/`: the output for every class. With thinking on it includes the thinking text; for the
  agent, its commands and every model turn.
- `plan.json`, `speed.json`, `verify_*.json` (the weight check) and `logs/`.

The results layout: one folder per dataset task, and one folder per method inside it, numbered in the order of the
tables:

```
results/
├── anatomy__mouse-human/
│   ├── 1-base/
│   │   ├── scores.json
│   │   ├── decisions.tsv.gz
│   │   └── pairs/anatomy__mouse-human/alignment.rdf, alignment.sssom.tsv
│   ├── 2-sft/
│   ├── 3-opsd/
│   ├── 4-opsd-grpo/
│   ├── 5-all-merged/
│   └── 6-all-merged-agent-thinking/
├── bioml__NCIT-DOID/            (and bioml__SNOMED-FMA/, bioml__SNOMED-NCIT/)
├── biodiv/                      pairs/biodiv__envo-sweet/, pairs/biodiv__flopo-pto/
├── biodiv2026/                  pairs/biodiv2026__<test case>/ (9)
├── complex_om__cree-swo/
├── cyber__iotsec-uco/
├── dh/                          pairs/dh__<pair>/ (8)
├── dh_multiling/                pairs/dh_multiling__<pair>/ (10)
├── icd__who-icd10-to-icd11/     <method>/alignment.rdf (every WHO code)
├── icd__snomed-to-icd11/        <method>/icd11_snomed_complete.sssom.tsv, snomed_to_icd11.tsv.gz (the complete map)
└── multifarm/                   pairs/multifarm__<pair>/ (1,125)
```

| Method | Tag (`FULL_EVAL_METHODS`) | Results folder |
|---|---|---|
| Gemma 4 E4B (base) | `base` | `1-base` |
| + SFT | `sft` | `2-sft` |
| + OPSD | `sdft` | `3-opsd` |
| + OPSD → GRPO | `grpo` | `4-opsd-grpo` |
| All merged | `all_merged` | `5-all-merged` |
| All merged + mini-SWE-agent | `all_merged-agent` | `6-all-merged-agent` |
| any of these, thinking on | `<tag>-think` | `<folder>-thinking` (e.g. `6-all-merged-agent-thinking`) |

## Pieces

| File | What |
|---|---|
| `vast_full_eval.sh`, `setup.sh`, `price.py` | the one command, the lean GPU-box setup (with the agent's user sandbox), this machine's price |
| `push_inputs.py` (Mac), `pull_inputs.py` (box) | raw folders (without PDFs), Bio-ML, parsed-ontology cache → private HF → box |
| `tasks.py` | the tasks, which classes run (`FULL_EVAL_SCOPE`; `reference_classes`), references |
| `build_episodes.py` | one prompt per class, as in training |
| `models.py` | the five models from the private repos |
| `infer.py` | one-decision methods: one vLLM per GPU, thinking off or on; resumable shards; the money guard's STOP |
| `agent_run.py` | the agent method: `vllm serve` per GPU and `om_agent`'s Matcher over the same shards |
| `schedule.py`, `run.py` | methods, units, priority, money projection; the orchestrator |
| `score.py`, `report.py` | scoring (Top-k, behaviour, alignment files; OAEI scores with `FULL_EVAL_SCOPE=all`), tables and figures |
| `push_results.py` | every output to the private dataset |

Mac rehearsal: no GPU; a lexical matcher stands in for the model, also behind the agent; the stand-in price is `$1/h`.

```bash
PYTHONPATH=src:. OA_OUT_DIR=/tmp/fe FULL_EVAL_FAKE=lexical FULL_EVAL_GPUS=1 OA_AGENT_ENV=local FULL_EVAL_METHODS=grpo,all_merged-agent-think FULL_EVAL_TASKS=anatomy__mouse-human,dh,cyber__iotsec-uco,complex_om__cree-swo .venv/bin/python -m full_eval.run --yes
```

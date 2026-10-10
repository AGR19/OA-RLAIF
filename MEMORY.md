# OA-RLAIF — decision log

Entries are dated and newest last. Each one records what was decided, why, and the evidence behind it. Do not rewrite history; add a new entry when a decision changes.

## 2026-10-10 — project reset to RLAIF

- **Previous path discontinued.** The Inkling-Small / Tinker SFT pilots in the parent repo `oa-sft-sm-1` are discontinued.
  - The only measured result there is a lexical q-gram baseline on the Anatomy test split: P 1.0, R 0.159, F 0.274, Hits@1 0.774.
  - `docs/PROJECT_HISTORY_AND_CURRENT_PLAN.md` in the parent repo (no SFT, Qwen3.5-4B) is **superseded** by this plan.
- **Models.**
  - Policy: **Gemma 4 E4B-it** (LoRA).
  - Judge and question generator: **Gemma 4 26B-A4B-it via an OpenAI-compatible API**. The user chose an API rather than self-hosting.
  - Non-thinking chat template.
- **Hardware.** Lambda **4×B200**, about 2 epochs, at roughly $6.79/GPU-hr (third-party trackers, Aug 2026; confirm at lambda.ai/pricing). That is about $27/hr.
  - Layout: vLLM generation on GPU 3, trainer on GPUs 0–2.
  - Lambda on-demand ships with Docker and nvidia-container-toolkit preinstalled.
- **Budget: $500 total**, with a hard stop at $450. All data prep runs on the Mac first.
- **Trainer: TRL + PEFT, not Unsloth.**
  - Unsloth's docs say Gemma 4 GRPO needs `fast_inference=False`, which means no vLLM, and Unsloth is single-GPU oriented.
  - Unsloth is kept only as a GGUF-export fallback.
- **Continued pretraining and plain SFT are replaced by on-policy self-distillation.** The user asked for a 2026 method that puts knowledge in the weights with no RAG. The method is SDFT/OPSD:
  - **OPSD**: arXiv 2601.18734, Zhao et al., 2026, 412 Semantic Scholar citations on 2026-10-10.
  - **SDFT**: arXiv 2601.19897, Shenfeld et al., 2026, 237 citations.
  - Foundation: **GKD**, ICLR 2024, 1,034 citations.
  - Implementation: TRL `trl.experimental.sdft.SDFTTrainer` with a `privileged_context` field. Teacher = frozen base E4B with the context.
  - SFT stays only as an **ablation arm**.
  - Ordering rationale: *OPD Before RL* (arXiv 2610.02781).
  - Risks noted: privileged-context distillation can hurt *thinking* models (we use non-thinking), and dense self-distillation may forget more than GRPO. Retention is measured.
- **Data: rebuild from raw** inside `RLAIF/` for every family in `Raw Ontology Folders` plus Bio-ML 2026 (`../data/external/bio-ml`).
  - The old SQLite views in `Finetuning Dataset/` (41 GB) are used only to cross-check counts.
  - Why: they have Tinker-era prompts, absolute provenance paths, test-heavy splits (1,426/149/11,136), and `complete_tasks.sqlite` has an un-checkpointed WAL.
- **Datasource.** Euzenat & Shvaiko, *Ontology Matching*, Springer 2007, 332 pp. A4B writes questions per passage, and the passage is the SDFT privileged context. 20% of passages are held out for a closed-book test.
  - The text is never committed to git. It goes only to private HF.
- **GRPO reward.** The user's goal is "100% correct, domain-agnostic, new and full mappings". The recommendation adopted:
  - `R = 0.8·R_task + 0.2·judge/10`.
  - Labeled episodes: `R_task = 0.5·1[pick ∈ gold] + 0.5·r_DRAL`.
  - Silver ICD episodes: weight 0.5. Unlabeled episodes: DRAL term only, weight 0.3, at most 10% of prompts.
  - Bad format or a hallucinated IRI scores 0.
  - `r_DRAL` is Eq. 1 from the DRAL-OA papers: `softmax_T(sim(z_s, z_pick))` over the prompt's candidates, with `sim = 0.5·cos(mapping-layer) + 0.5·cos(label)` (wv = wr = 0.5). For NONE the term is `1 − max_k p_k`. T is not given in the paper; default 0.1, configurable.
  - 100% correctness is neither reachable nor measurable, because reference alignments are incomplete or noisy (Bio-ML had to repair its references). The target is to maximize precision with calibrated abstention.
- **DRAL-OA sources.**
  - Zaeifi, Mosallanezhad, Bansal, AIxDKE 2024, DOI 10.1109/AIxDKE63520.2024.00012. Reports internal F 0.908.
  - Gonibeed Ravishankar, Zaefi, Bansal, OM@ISWC 2025, CEUR Vol-4144, `om2025-oaei-paper6`. Official OAEI 2025 Anatomy: **P 0.83 / R 0.827 / F 0.828 / Recall+ 0.56, size 1509**, top-1. **This is the baseline to beat.**
  - Local code `~/Desktop/DRAL_OA/.../old bert 2/clf.py`, the same as github.com/AGR19/DRAL-OA, contains the mapping layer: node2vec dim 100 / walk 30 / 20 walks / window 20, then MLP 256-tanh-128 and a sigmoid classifier, BCE, 50/50 positives/negatives.
  - It does **not** contain the REINFORCE agent, the ±σ actions or the Eq. 1 code; those are reimplemented from the paper.
  - clf.py's own evaluation limits candidates to reference targets and scores on its training references, so it is optimistic. That evaluation is not reused.
- **ICD-11↔SNOMED triangulation** (user request). Composition: SNOMED → WHO ICD-10 (RF2 refset **447562003**, 153,799 active rows in US 2026-03) → ICD-11 (WHO 2026-Jan tables).
  - Default relation: `classified_under` (skos:narrowMatch).
  - Promoted to silver exactMatch only when both legs are strong and 1:1 and the label similarity is ≥ 0.85 (difflib). Corrected 2026-10-10 to match the code; DRAL is not part of this rule.
  - Conditional map rules are kept as low-confidence evidence. ICD-10-CM refset 6011000124106 (285,108 rows) is auxiliary evidence only.
  - Validated by a held-out bridge test. Output: SSSOM + OAEI RDF.
- **Harness.** mini-SWE-agent v2 (bash-only, `mswea_bash_command` blocks, completion with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, Docker environment) is used for baseline, eval and deployment. GRPO trains on the final-decision turn in the same format.
- **Visibility.** GitHub `AGR19/OA-RLAIF` is **private**. HF dataset repos are **private**; the namespace is to be given by the user at push time, after `hf auth login`.
- **Target venues.** NeurIPS, ICLR, ICML, AAAI.
  - Required: ablations (base / SFT / SDFT / SDFT→GRPO; reward terms; with/without triangulation; teacher mode), 3 seeds where affordable, bootstrap CIs, compute and $ reporting, the NeurIPS checklist, and an anonymized code mirror for review.
- **First deliverable.** A publication-quality method figure (TikZ → PDF/PNG/SVG) in `docs/figures/`.

## 2026-10-10 — scope edits (later the same day)
- At the user's request, the "Paper requirements" section (ablation grid, seeds/CIs, reproducibility checklist, anonymized mirror) is **removed** from `docs/PLAN.md` and `CLAUDE.md`. It is no longer a deliverable. The SFT baseline remains only as a comparison row in eval and in the figure.
- **The HF push is now an explicit step**, run twice:
  1. on the Mac, after verification and before Lambda, so Lambda pulls the data from HF;
  2. on Lambda, after training, for the adapters.
  - Repos: `oa-rlaif-{datasource,sdft,grpo,dral-reward,icd-triangulation}` and `oa-rlaif-e4b-{sdft,grpo}`, all private.
- **All raw data must be used.** There is a per-file coverage manifest (`configs/coverage.yaml`) over all 2,573 files, enforced by `verify.py`.
  - System outputs (DH results 2024/2025, Complex-OM GED-and-class-eval, Bio-ML results, `biollm_result.tsv`) serve as hard negatives and baselines, never gold.
  - SNOMED SimpleMaps (CTV3, ICD-O) become extra mapping facts.
  - Benchmark docs and both papers become SDFT knowledge passages.
- **Work order changed:** the user asked to start from step 3, the literature review. The step 1 figure exists as a compiled draft in `docs/figures/build/`; its layout fixes are pending. Step 2, the repo review, is deferred.

## 2026-10-10 — coverage manifest and package facts
- **Coverage manifest** (`configs/coverage.yaml`, scanner `oa_rlaif.data.coverage`):
  - All 2,573 raw files plus 29 Bio-ML download files are assigned; 0 are unassigned.
  - 2,486 raw files are data-bearing.
  - 87 are support files: 61 provenance (code, licences, web components), 16 eval tooling (the Bio-ML scorer and submission fixtures), and 10 RF2 release metadata.
  - Data hidden in the Bio-ML site folder (results JSON 2022–2025, coherence files, explanatory pages) was reclassified as baseline metrics and knowledge passages.
  - The SNOMED map screenshot is transcribed in `configs/knowledge/`.
- **Parser fix:** the parent repo's parser silently dropped Anatomy synonyms and definitions, which are stored via named `genid` IRIs. The vendored parser recovers 344/345 mouse synonyms, 5,264/5,265 human synonyms and 840/841 definitions.
- **Lexical retrieval on Anatomy** (vendored q-gram retrieval, underscore-aware normalize): candidate recall @1/5/20/50 = 0.865 / 0.937 / 0.974 / 0.991 over all 1,516 references. This is a retrieval ceiling, not a matcher score.
- **Pinned training stack** (Linux lock compiled with uv on 2026-10-10): vllm 0.31.0 (which resolves torch 2.13.0), transformers 5.17.0, trl 1.15.0, peft 0.21.2, accelerate 1.15.0, datasets 5.1.0.
  - TRL 1.15 `SDFTTrainer` fields used: `prompt`, `privileged_context`, `teacher_model_kind=base|live|ema`, `distillation_mode=topk_logits`, `teacher_prompt_template`, `use_vllm` / `vllm_mode`.

## 2026-10-10 — DRAL-OA port: held-out check on Anatomy
- Fit the ported mapping layer (clf.py settings, train-split references only: 1,201 pairs; 6,042 graph nodes) in about 4.5 minutes on the Mac CPU.
- On 315 held-out Anatomy references, Hits@1:
  - lexical retrieval top-1: 0.867;
  - DRAL re-rank within the lexical top-20: 0.778;
  - DRAL over all 3,304 human classes: 0.740.
  The training loss fell from 0.25 to 0.0008, which suggests the easy random negatives lead to overfitting.
- **Implication:** on unseen references the DRAL mapping layer is a weaker ranker than lexical matching. That is consistent with DRAL-OA's official OAEI 2025 F 0.828 versus its internal 0.908 (clf.py scored its own training references). In the reward, the gold term carries correctness and r_DRAL gives graded partial credit, as approved. r_DRAL is not ground truth.
- All 364 ontology files were parsed with zero errors in 13.4 minutes. The profile cache is 384 MB.

## 2026-10-10 — local-only runtime (user instruction)
- The user wants everything to run locally except the A4B API.
- mini-swe-agent v2.4.6 is vendored in `third_party/mini-swe-agent` (commit a83fcae, MIT) and installed editable from source. It is not taken from PyPI at run time.
- Its "API" is `localhost:8000`, our own vLLM server. LiteLLM is just the client.
- `LITELLM_LOCAL_MODEL_COST_MAP=True` stops LiteLLM from fetching its price map.
- The sandbox runs with `--network none`. FastText `fast.bin` is shipped in the rsync bundle.
- **Remaining one-time downloads on Lambda:** Gemma 4 E4B weights (HF, gated licence), the pinned wheels and the Docker base image.

## 2026-10-10 — naming, verification, literature (user requests)
- **"textbook" renamed to "datasource"** everywhere: docs, configs, scripts (`15_datasource.sh`), module (`data/datasource.py`), the HF repo (`oa-rlaif-datasource`), coverage family and the figure. The raw file keeps its real name on disk.
- **Parse and reference verification added**, the standard OAEI/OBO practice:
  - every ontology is re-parsed with the OWL API (`tools/owl_verify`) and must match our declared-class count exactly;
  - ELK reports consistency;
  - references get the Beisswanger & Hahn (2012) ten quality checks.
  - Mac build: classes compiled against the parent repo's shaded OWL API + ELK + HermiT jar (offline). On Lambda: `mvn package`.
  - First check on Anatomy mouse.owl: OWL API 2,744 classes, equal to ours; 3,084 label assertions against our 3,081; ELK consistent, 0 unsatisfiable classes.
- **Literature section E (verification):** Beisswanger & Hahn 2012, OWL API, ROBOT, ELK, HermiT, Alignment API 4.0, OOPS!, LogMap UMLS assessment and MELT. 9 of 10 new entries are DOI-verified; Meilicke 2007 (AAAI) has no Crossref DOI and is dropped. Semantic Scholar shows 0 citations for HermiT (JAR 2014), likely a record-linking gap.
- **DH system output:** one systemAlignment.rdf (OAEI 2024, dha-unesco, tomato matcher) contains only the word "error". It is recorded as a failed-run baseline row, not parsed as an alignment.
- **Split diagnostics:** MultiFarm has one 1,877-node component (the same concepts linked across 10 languages), which landed in train, so the MultiFarm test share is 13%. It is kept whole to avoid cross-language leakage.
- **Properties added (parser v4).** The OWL API cross-check showed that MultiFarm/Conference references also align object and datatype properties. Our parser had kept only classes, so those correspondences were dropped as unresolved.
  - The parser now keeps declared object/datatype properties: kind, domain, range and super-properties.
  - Retrieval matches like with like (class↔class, property↔property).
  - The cache key includes the parser version, so all 364 files are re-parsed.
  - The OWL API verifier also cross-checks property counts.
- **Verifier sample** (9 files spanning RDF/XML, OWL, SKOS and EDOAL-paired ontologies): exact class-count match on all 9. ELK finds ENVO has 2 unsatisfiable classes (an ontology issue, reported).
- **The ICD silver-episode retrieval is now parallel** (process pool). It was the slowest build step at more than 15 minutes single-threaded.

## 2026-10-10 — the user's two questions
- **Q1: does the model understand each ontology's modelling assumptions?** Previously only implicitly.
  - Fixed with modelling profiles in every prompt and as SDFT knowledge prompts, plus the ELK Coherent column and assumption-sensitive evaluation slices.
  - The Anatomy reference alignment is coherent under ELK (0 unsatisfiable classes before and after).
- **Q2: what does the GGUF + agent output look like?** `oa_rlaif.eval.align` produces OAEI RDF, SSSOM, TSV and explanations, plus the OAEI results row when a reference is given. Backends: local vLLM, or a local OpenAI-compatible server (Ollama serving the GGUF).
  - Lexical top-1 baseline on Anatomy, full run: Runtime 8 s, Size 1,581, P 0.708, F 0.723, R 0.739, Recall+ 0.514, Coherent −.
  - For ICD-11↔SNOMED there is no reference, so no P/R/F; the proxy is agreement with the triangulated silver links.

## 2026-10-10 — A4B access, judge hosting, ICD retrieval, HF namespace
- **A4B API = Google AI Studio** (user's key; the key lives only in `.env`).
  - `google/gemma-4-26b-a4b-it:free` is OpenRouter naming and returns 404 there. AI Studio serves the model as `gemma-4-26b-a4b-it` at `https://generativelanguage.googleapis.com/v1beta/openai/`.
  - Gemma 4 on AI Studio always thinks: it returns `<thought>…</thought>` inline and rejects a thinking budget ("Thinking budget is not supported for this model"). `api_client.answer_text` strips the thought; `JUDGE_MAX_TOKENS=4096`.
  - Calls retry on 429/5xx with backoff and are paced client-side by `JUDGE_RPM=14`.
  - Live check: the judge scored the correct Anatomy pick 10 and a wrong pick 0 (2 calls, 16 s).
- **Paid AI Studio tier does not help.** Google's pricing page lists Gemma 4 as "Free of charge" (free tier) and "Not available" (paid tier).
  - Free limits are not published by Google. A third-party figure is about 15 RPM / 1,500 RPD, which is about 1.6% of the roughly 96K GRPO judge calls (12K prompts × 8).
  - OpenRouter paid (`google/gemma-4-26b-a4b-it`, $0.0675/M in, $0.225/M out from its public API) would cost about $20–35 in tokens. But synchronous reward scoring would idle 4×B200 for roughly $45–230.
  - **User decision: self-host the A4B judge on Lambda.** `google/gemma-4-26B-A4B-it` (51.6 GB, not gated) is served by vLLM on GPU 3 next to the policy (`VLLM_GPU_UTIL=0.40`, `JUDGE_GPU_UTIL=0.45`), with thinking off via `chat_template_kwargs`. `judge_up`/`judge_down` are in `scripts/_common.sh`; `60_grpo.sh` starts it.
  - AI Studio stays for datasource question generation on the Mac.
  - `judge_fraction` (configs/train.yaml) can judge a deterministic subset of prompts (whole groups) if needed; the default is 1.0.
  - Measured judge prompt size: mean 1,280 tokens, p90 2,001 (258 episodes).
- **HF namespace: `gonibeed`** (personal; `hf auth whoami` also shows org `fireline-science`). The user is logged in. Pushes still need an explicit OK in chat after the gates pass.
- **xlsx vs txt (WHO 10↔11 tables): mappings identical in all 5 pairs**, compared on the key columns (codes, Foundation/linearization URIs).
  - `10To11MapToMultipleCategories`: the `Depth` column differs in 7,607 of 15,630 rows. This is metadata only, reported and not used.
  - `foundation_10To11MapToOneCategory`: the xlsx has an extra empty trailing column.
- **ICD retrieval fixed.** SNOMED→ICD-11 gold is usually a *coarser* category (`classified_under`, often an "unspecified" residual), so lexical top-k missed it: test recall@20 was 0.196.
  - New retrieval adds icd11.owl synonyms, SNOMED parent labels (weight 0.8) and ancestor expansion (top 50 → ICD-11 ancestors up to depth 4, decay 0.7).
  - Tuned on a 1,500-concept **train** sample: recall@20 0.224 → 0.654. Test recall is reported after the rebuild, never used for tuning.
  - Source cards now show SNOMED parents and the ICD-10 3-character category.
- **Mac venv quirk:** macOS file sync sets the `hidden` flag on the venv's editable `.pth` files, and Python 3.11.16 skips hidden `.pth` files, so `import oa_rlaif` fails intermittently. `scripts/_common.sh` and pytest now set `PYTHONPATH` explicitly (src + vendored mini-swe-agent). Lambda is unaffected.

## 2026-10-10 — episode-file corruption found and fixed
- **Symptom:** the DRAL fit failed on invalid JSON. Five episode files (biodiv, bioml, dh, dh_multiling, multifarm) began with zero bytes and kept only their last few hundred lines. Anatomy (one pair) and every other data file were intact.
- **Cause (our bug):** in `data/build.py`, `handles.setdefault(family, open(path, "w"))` opened the file again, in truncate mode, for every pair. The first handle kept writing at its old offset, so multi-pair families got zero-filled holes.
  - The episode counts reported earlier come from the in-memory summary and remain correct; only the files were damaged.
  - Nothing downstream had used them yet: render, DRAL fit and the push had not run.
- **Fix:**
  - each family file is opened once (`ExitStack`);
  - all builder outputs are written atomically (`io.atomic_open`: `.tmp`, then rename);
  - `data/.build.lock` refuses a second concurrent build;
  - `verify.py` fails on any JSONL line that does not parse or contains zero bytes.
- The episodes stage was rebuilt, then the DRAL fit was rerun.

## 2026-10-10 — full parse and reference verification (first full run)
- **Ontologies (364):** 354 exact class-count matches with the OWL API.
  - **SNOMED CT:** 390,667 classes in our parser against 390,669 in the OWL API. The difference is OWL 2 punning: `762705008` and `762706009` are declared as both class and property. Property counts agree (143). ELK: consistent, 0 unsatisfiable.
  - **9 files reported "owlapi_failed"** (popconference-100-v1, the same 32 MB file in 8 pairs, and the taxon-dbpedia target) actually loaded. ELK found them **logically inconsistent**: the instance data contradicts the class axioms, a benchmark finding. Our verifier stopped at `getUnsatisfiableClasses()`. Fixed: inconsistent ontologies keep their counts and report `consistent_elk: false`, `unsatisfiable: -1`. Taxon target: 58 = 58 classes.
- **References (1,285):** 1,259 pass, 21 warn, 5 fail.
  - **taxon-dbpedia:** used the Alignment API long relation names (Equivalence / SubsumedBy / Subsumes). `read_alignment` now normalizes them to `=` / `<` / `>`. This was a reader gap, not a data defect.
  - **Hydrography (3 references):** use `//`, which is undefined in the benchmark and is not an Alignment API relation. The 15 cells (7 + 4 + 4) are excluded and counted, never relabelled. Complex tasks: 4,613 → 4,598.
  - **flopo-pto:** 1 target without a namespace (`PATO_0000103`), 14 source classes absent from the supplied FLOPO file (6.1%), 1 duplicate. The reference is reported, not edited.
  - All accepted deviations, with reasons, are in `configs/parse_exceptions.yaml`. The parse verification is re-run at the gate with the fixed tool.

## 2026-10-10 — search for the DRAL-OA RL code
- **Searched:** `~/Desktop/DRAL_OA` (29 .py files) and `~/Downloads/Advanced-Cybersecurity-OA` (64 .py, 60 .csv). Neither has the REINFORCE agent (Algorithm 1, Eq. 2–3) or Eq. 1 with a temperature. The temperature remains a stated assumption (0.1).
- **Advanced-Cybersecurity-OA** is the IoTSec↔UCO cybersecurity project. It has 18 variants of the same mapping-layer classifier `clf_*.py`: 5 security BERT encoders, plus LLM-enriched labels from GPT-4o/4.1/mini and Llama 3.2/3.3.
  - It confirms the code settings: Adam 3e-4, BCE, batch 32, no dropout.
  - A margin `ContrastiveLoss` (pairwise distance, margin 1.0) is defined but commented out. This is not Eq. 1 (InfoNCE).
  - Ranking there is 1.0·cos(mapping layer) + 0·cos(sentence embedding); Anatomy uses 0.5/0.5. `DralScorer(wv, wr)` is already configurable.
  - The scripts read `reference.csv`, which is not in the folder. With it, IoTSec↔UCO could become an unseen-domain zero-shot pair.
- **`~/Downloads/gnn-main-2` (checked 2026-10-10):** the 2023 GNN/OntoSim-style Anatomy matcher. It has no RL code. Pipeline:
  - FastText 300-d label vectors;
  - PyTorch-BigGraph graph embeddings (dot comparator, ranking loss, 1,000 uniform negatives, lr 0.01, 100 epochs);
  - word and meta similarity combined 0.5 / 0.5 (the origin of DRAL-OA's w_v = w_r = 0.5);
  - top-k = 5, then P/R/F.
  - It is the same code as `DRAL_OA/.../gnn-main`; that copy only adds an NLTK SSL workaround.
  - Its gold `reference.xml` has the same 1,516 cells as the raw Anatomy reference (a different serialization).
  - Not needed for OA-RLAIF; it is a possible earlier-method baseline.

## 2026-10-10 — cybersecurity zero-shot test pair (user: "lets use as test")
- **Pair `cyber/iotsec-uco`:** IoTSec → UCO from the user's Advanced-Cybersecurity-OA project and `~/Downloads/reference.csv`. It is an **unseen domain**, listed in `zero_shot_pairs`, never trained on, and has no DRAL layer.
- **Conversion** (`python -m oa_rlaif.data.cyber`, outputs in `data/external/cyber/`):
  - `uco_rdf.xml` is an OWL body without the rdf:RDF root or namespace/entity declarations. A wrapper was added; `&cbx;` is undefined in the source and maps to a marked placeholder namespace.
  - Reference: 91 rows, 90 unique, **69 pairs resolve to class IRIs**. Excluded: 1 duplicate, 1 malformed ID (`UCO#57Network`), and 20 IoT labels not present in `iotsec.owl` (probably a different version).
- **Evaluation:** a ranking test (pick in reference, any relation). The reference has no relation types and is mostly broader/narrower links: Vulnerability and Weakness each link to 17 IoT classes.
  - The prompt asks for the "most closely related (equivalent, broader or narrower)" target.
  - Never report equivalence P/R/F for this pair.
- **The user's own `clf_chatgpt_4_1_bert.py` run** (R@5 0.90, R@1 0.10) used the same links for training and test and a pool of ~24–28 UCO targets. A constant answer of the 5 most frequent targets gives R@5 0.70 / R@1 0.27 on all links.
- **Two fixes found through the cyber pair (2026-10-10):**
  1. **Label-less entities were invisible to retrieval.** UCO classes have no rdfs:label, and `CandidateGenerator` indexed labels only, so all 35 cyber test episodes had 0 candidates. `retrieval.candidates.label_texts` now falls back to the IRI local name, as cards already did. This applies to every family.
  2. **Zero-shot leakage of unlabeled episodes.** The zero-shot rule forced only reference entities into test. Unlabeled sources of flopo-pto (18), tadirah-unesco (1) and cyber (2) were hashed into train. The episode builder now sets `split = test` for every episode of a zero-shot pair; the verify gate would also have failed on it.
  - The episodes were rebuilt after both fixes, before render.
- **Anatomy ceiling (held-out):** 315 reference pairs, 302 reachable with one pick per source. Recall ceiling 0.959 (9 retrieval misses, 4 lost to multi-target sources); precision ceiling 1.0. The user expects P = R = 1.0; explained why reference-measured 1.0 is not reachable (reference gaps: 33 missing label-identical pairs). Ceiling raisers offered: k = 40 / semantic retrieval, multiple submissions per source.
- **MultiFarm held-out candidate recall is 0.274.** Lexical q-grams cannot cross scripts and languages. A multilingual embedding retriever is proposed (needs a one-time download, awaiting the user's OK).

## 2026-10-10 — user approved: Docker check, multilingual retrieval, Anatomy ceiling raisers
- **Docker sandbox check passed** (`tests/test_harness.py::test_agent_loop_in_docker_sandbox`, image `oa-rlaif-om:latest`, 213 MB, from python:3.11-slim). The agent ran `om profile` and `om candidates`, then submitted correctly. A socket probe inside the container was blocked (`--network none`).
- **Multilingual retrieval:** `retrieval/semantic.py`, intfloat/multilingual-e5-small on CPU, exact cosine with numpy (no faiss).
  - Train-sample candidate recall@20: MultiFarm lexical 0.298 → semantic 0.917; DH-multiling 0.556 → 0.789.
  - Anatomy, biodiv and DH stay lexical; fusion was not better on train.
  - Config: `episodes.retrieval_by_family`; sentence-transformers 6.1.0 is already in the Linux lock.
- **Anatomy k = 40:** train recall@k 0.978 → 0.990. Config: `episodes.k_by_family`.
- **Multi-target submissions:** `om submit C2,C5` (at most 3 distinct).
  - The gold term is F1(picks, gold present), so padding with a wrong target loses reward: one right plus one wrong = 0.667. r_DRAL is the mean over picks.
  - The SDFT demonstration lists every present gold.
  - Extraction (`alignment_io.select_equivalences`) lets a source keep several targets but never gives one target to two sources.
  - Agent-eval correctness is exact set match. Tests: `tests/test_multi_target.py`. 41 tests pass.
- **`scripts/run_lambda.sh`:** runs baseline → SDFT → eval → GRPO → eval --full → export (GGUF Q4_K_M + Modelfile) with budget gates. No push. The user must terminate the instance.
- **Datasource questions done (ASU qwen36-27b, thinking off):** 1,633 passages, 0 API errors. Kept 5,140 train and 1,229 test questions; 1,009 rejected as ungrounded; 7 test near-duplicates dropped.
  - All 16 textbook units have both train and test questions: textbook 1,488 / 375, surveys 3,209 / 743, other docs 443 / 111.
  - 156 passages produced nothing. Most are reference lists, plus the junk below.
- **Bug, fixed:** the SNOMED screenshot was read as binary text (56 junk passages). The transcription lookup failed on U+202F in the macOS screenshot filename.
  - **0 questions survived the grounding filter**, so there was no contamination.
  - The lookup now normalizes the name, and binary files (images, xlsx, zip) are never read as text. The passages were rebuilt (the screenshot is now 1 transcription passage); only new passages call the API (cache).

## 2026-10-10 — gates pass
- **Gates:** pytest 41 passed; `verify.py` 0 failures; the files git would commit are 323 (4.1 MB), with no data, keys or licensed text.
- **Verification fixes:**
  - **15 MultiFarm ar-fr references** have no resolvable cell: their target IRIs name entities absent from the target ontology, e.g. `confOf_ar#op-6477014-2447390` is a cmt_ar id. Documented, not edited.
  - **Bio-ML `local.*.cands.tsv`** is now role `candidate_pool`. These are ranking candidate lists, not references.
  - **Bio-ML landing pages, `.fld` assets and site `.astro` wrappers** are now provenance; they hold no tables or text.
  - **Empty files in the sources** are documented: 2 SNOMED Full files and 1 Complex-OM `class-eval.tsv`, each header-only.
  - **`trivial.rdf`** is now counted in the baselines stage and written to `data/eval/trivial_<family>.tsv`.
  - **UCO** has 3 punned properties; OWL API 103 against ours 100.
- **Held-out candidate recall after the rebuild:**

  | Family | Recall | Note |
  |---|---:|---|
  | Anatomy | 0.994 | k = 40 |
  | MultiFarm | 0.894 | semantic |
  | Cyber | 0.686 | |
  | DH | 0.976 | |
  | Biodiv | 0.961 | |
  | Bio-ML | 0.881 | |
  | DH-multiling | 0.500 | n = 8 |

  No zero-shot leakage.
- **Questions:** 5,144 train / 1,230 test; 1,578 passages.

## 2026-10-10 — pushed (user: "Lets push to github and hugging face")
- **GitHub `AGR19/OA-RLAIF` (private), 5 commits:** code, configs, docs, scripts, vendored mini-swe-agent. No data or secrets.
- **HF datasets (private, gonibeed):**

  | Repo | Size |
  |---|---:|
  | oa-rlaif-datasource | 0.02 GB |
  | oa-rlaif-sdft | 1.24 GB |
  | oa-rlaif-grpo | 1.50 GB |
  | oa-rlaif-dral-reward | 0.27 GB |
  | oa-rlaif-icd-triangulation | 0.18 GB |

  The grpo repo includes `raw/anatomy-dataset-2` for the full Anatomy alignment on Lambda.
- **Round-trip test:** pulled into an empty folder; every checked file is byte-identical to the local build. The Lambda data checks pass; training code loads 328,944 SDFT rows and 12,000 GRPO prompts; the reward functions run.
- **One command for Lambda:** `scripts/lambda_all.sh`, with resumable stages via `outputs/stages/*.done`.
  - Rehearsed on the Mac with stub stages: it stops for the HF login, runs the stages in order, stops on a failure without marking it done, resumes at the failed stage, and skips everything when done.
- **Fixes found while scripting:**
  1. SDFT data (328,944 rows × 2 epochs = 6,853 steps) would not fit the budget. Steps are now planned from the dry run (`plan_steps`: min(full epochs, time slot)), and the dry run times GRPO with the judge.
  2. The trained models were never evaluated as agents on Lambda; `50_eval.sh` now runs 200 Docker agent episodes per model.
  3. GGUF conversion used the system `python`.
  4. `00_check_env.sh` did not fail on missing items.
  5. The model push lacked the GGUF.
  6. Added an early base-model GGUF check.
  7. The SFT comparison arm is opt-in (`RUN_SFT_ARM=1`, ~3 GPU h).
- **The GGUF combines** base E4B + SDFT (merged) + GRPO trained on top: DRAL-OA reward + A4B judge, i.e. RLAIF.

## 2026-10-10 — SFT is compulsory; Mac end-to-end smoke test (user: "verify the full script ... do sft also make it compulsory not optional")
- **SFT comparison arm is compulsory** (supersedes "opt-in, `RUN_SFT_ARM=1`" above). `lambda_all.sh` stages 13–14 (`sft`, `eval_sft`) always run, after the GGUF push, so a late failure cannot lose the main model. Same rows as OPSD; steps = min(OPSD steps, 2 h slot) (`plan_sft_steps`). Time slots: OPSD 4 h, GRPO 4 h, SFT 2 h. Plan: 16 stages, ~15.1 h, ~$411.
- **`scripts/smoke_mac.sh`**: runs the Lambda stage order on the Mac with the real code paths. It uses a tiny random Gemma 4 (26.2M params, built from the cached E4B config), CPU training without vLLM, and a stub OpenAI server for the judge and the agent model. Data, reward, Docker sandbox, mini-SWE-agent loop and report are real. Outputs go to `outputs/smoke/`.
- **Bugs the smoke test found (each would have crashed on Lambda or on Ollama):**
  1. `SDFTConfig(warmup_ratio=...)` is rejected by TRL 1.15 / transformers 5. Fix: `warmup_steps=<ratio>` (a float < 1 is read as a ratio).
  2. `report_to=["tensorboard"]` without tensorboard in the lock. Fix: added tensorboard (lock additions only: absl-py, markdown, tensorboard, tensorboard-data-server, werkzeug).
  3. SFT with `assistant_only_loss=True` fails because the Gemma 4 chat template has no `{% generation %}` markers. Fix: conversational prompt/completion rows + `completion_only_loss=True`.
  4. Modelfile `PARAMETER stop "<end_of_turn>"` is Gemma 2/3. Gemma 4 ends a turn with `<turn|>` (id 106; E4B `eos_token_id` = [1, 106]). Fixed in `release.write_modelfile`.
- **Mac-only gap:** TRL 1.15 SFTTrainer always computes its loss with a Triton kernel, and PyTorch ships Triton on Linux only. The smoke run uses a plain-PyTorch stand-in (`dev/cpu_kernels.py`, only when `OA_SMOKE=1` and the kernel is missing). Lambda uses the real kernel (triton 3.7.1 in the lock).
- **Judge path check:** invalid replies score 0 without a judge call, so a model that writes only invalid replies never exercises the judge. `python -m oa_rlaif.dev.reward_check` scores gold replies through the GRPO rewards. It runs in the smoke test (stub judge) and in `35_dry_run.sh` right after `judge_up` (local A4B), before any GRPO step. Smoke result: judge 0.7 per valid reply (stub SCORE 7), 0 fallbacks; unlabeled episodes give R_task = 0.3·r_DRAL, as designed.

## 2026-10-10 — CoreWeave (user: "i am now running on coreweave now write scripts for it including installing docker")
- **One command:** `scripts/coreweave_all.sh`. The 16-stage list moved to `scripts/run_all.sh lambda|coreweave`; `lambda_all.sh` is a thin wrapper, so Lambda behaviour is unchanged. Runbook: `COREWEAVE_RUNBOOK.md`.
- **Docker (`scripts/coreweave_docker.sh`)** detects the machine type:
  - Docker already works (feature-tagged image, or a dind sidecar via `DOCKER_HOST`): nothing is installed beyond the CLI.
  - Virtual Server / node with systemd: Docker Engine from Docker's apt repo, `systemctl enable --now docker`, socket ACL for the user.
  - Kubernetes pod without systemd: Engine packages, then `dockerd` in the background. It falls back to vfs storage (data root on overlayfs) and to no bridge/iptables, because the sandbox runs with `--network none`. A non-privileged pod cannot run dockerd; the script prints the fixes: privileged pod, dind sidecar with the workspace at the same path, or a docker-tagged image.
  - The sandbox image builds with `--network none`. A self-test runs exactly as the harness does: no network, bind-mounted episode folder, `om`. `--start` restarts the daemon after a pod restart.
- **Price:** asked once (`coreweave_price.sh`), stored as `OA_GPU_USD_PER_HOUR`. Published CoreWeave figures (third-party trackers, 2026) are $68.80/h for 8×B200 on demand ($8.60/GPU-h), with spot from $34.11/h.
- **Budget guard now counts machine wall-clock time** since `outputs/stages/run.start`: `max(ledger, wall clock)`. Before, setup, downloads and gaps between stages were not counted. `OA_WALLCLOCK_BUDGET=0` turns this off.
- **`slot_scale`:** at the end of the dry run, slots × min(1, (hours left − 4.5 fixed h) / 10 slot h). It is stored in `step_plan.json["budget"]`, so later stages and the SFT row use the same plan. It stops if the scale is below 0.05. At Lambda's price the scale is 1.0; at $68.80/h it is about 0.05 (≈ 30 min of training in total).
- **GPU layout** (`_common.sh`) follows the GPU count and size: train on all GPUs but the last, vLLM on the last. On GPUs under 150 GB (H100/H200) the A4B judge gets its own GPU. `EVAL_TP` = min(4, GPUs).
- **Checks:** `/dev/shm` ≥ 8 GB (a pod default of 64 MB crashes NCCL and vLLM). At least 180 GB free disk, HF and uv caches in `.cache/` next to the repo. `00_check_env.sh` now fails, not passes, when the Docker daemon or sandbox image is missing (the Docker test used to be skipped silently).

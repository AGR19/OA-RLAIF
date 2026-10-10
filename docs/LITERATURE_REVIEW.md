# Literature review: RL / RLAIF fine-tuning, on-policy distillation, and AI-trained ontology alignment

Snapshot date: 2026-10-10

## How this review was built

- **Candidates:** 95 papers chosen for four questions:
  - Which RL and AI-feedback fine-tuning techniques are most cited?
  - Which on-policy distillation (OPD) methods put knowledge into weights?
  - Which ML/LLM-trained ontology-matching (OM) systems are the baselines?
  - How do researchers verify that ontologies and reference alignments are parsed and valid?
  - A fourth table lists the closest prior work to this project.
- **DOI checks:** every DOI was resolved at `doi.org` with CSL-JSON content negotiation (Crossref, or DataCite for arXiv `10.48550/arXiv.*` DOIs). The returned title had to match the expected title. When no DOI was known, Crossref was searched by title and the result had to pass the same match.
- **Venue DOIs:** where Semantic Scholar knows a venue DOI for an arXiv paper (for example ACL Anthology), that DOI was also resolved and title-checked, and is shown in preference to the arXiv DOI.
- **Citation counts:** from the Semantic Scholar Graph API (batch endpoint) on the snapshot date. Counts change daily, and Google Scholar counts are usually higher.
- **Failures:** anything that failed verification is listed under *Dropped*, never guessed. Items with no DOI (blogs, CEUR proceedings) are kept with a URL and marked "no DOI".
- **Venue tier:** the user's reference ranking is Google Scholar Metrics h5-index / h5-median: NeurIPS 371/637, ICLR 362/652, ICML 272/471, then AAAI. NeurIPS, ICLR and ICML papers generally have no publisher DOI, so their arXiv DataCite DOI is the citable DOI.
- **Reproduce:** `python tools/literature/verify_dois.py && python tools/literature/render_tables.py`. BibTeX for every verified DOI is in `tools/literature/references.bib`.

## A. RL and AI-feedback fine-tuning of language models (ranked by citations)

<!-- TABLE:rl -->
| # | Paper | Venue | Year | Citations | DOI |
|---|---|---|---|---:|---|
| 1 | Proximal Policy Optimization Algorithms | arXiv | 2017 | 31,981 | [10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347) |
| 2 | Training language models to follow instructions with human feedback | NeurIPS 2022 | 2022 | 24,834 | [10.52202/068431-2011](https://doi.org/10.52202/068431-2011) |
| 3 | LoRA: Low-Rank Adaptation of Large Language Models | ICLR 2022 | 2021 | 24,100 | [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685) |
| 4 | Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | NeurIPS 2023 (D&B) | 2023 | 11,985 | [10.52202/075280-2020](https://doi.org/10.52202/075280-2020) |
| 5 | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | NeurIPS 2023 | 2023 | 10,967 | [10.52202/075280-2338](https://doi.org/10.52202/075280-2338) |
| 6 | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | arXiv | 2024 | 9,744 | [10.48550/arXiv.2402.03300](https://doi.org/10.48550/arXiv.2402.03300) |
| 7 | Efficient Memory Management for Large Language Model Serving with PagedAttention | SOSP 2023 | 2023 | 8,986 | [10.1145/3600006.3613165](https://doi.org/10.1145/3600006.3613165) |
| 8 | Deep reinforcement learning from human preferences | NeurIPS 2017 | 2017 | 6,142 | [10.48550/arXiv.1706.03741](https://doi.org/10.48550/arXiv.1706.03741) |
| 9 | QLoRA: Efficient Finetuning of Quantized LLMs | NeurIPS 2023 | 2023 | 5,927 | [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314) |
| 10 | DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning | Nature 2025 | 2025 | 5,860 | [10.1038/s41586-025-09422-z](https://doi.org/10.1038/s41586-025-09422-z) |
| 11 | Let's Verify Step by Step | ICLR 2024 | 2023 | 4,877 | [10.48550/arXiv.2305.20050](https://doi.org/10.48550/arXiv.2305.20050) |
| 12 | Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback | arXiv | 2022 | 4,419 | [10.48550/arXiv.2204.05862](https://doi.org/10.48550/arXiv.2204.05862) |
| 13 | Constitutional AI: Harmlessness from AI Feedback | arXiv | 2022 | 3,818 | [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073) |
| 14 | Learning to summarize from human feedback | NeurIPS 2020 | 2020 | 3,543 | [10.48550/arXiv.2009.01325](https://doi.org/10.48550/arXiv.2009.01325) |
| 15 | DAPO: An Open-Source LLM Reinforcement Learning System at Scale | NeurIPS 2025 | 2025 | 2,862 | [10.48550/arXiv.2503.14476](https://doi.org/10.48550/arXiv.2503.14476) |
| 16 | Fine-Tuning Language Models from Human Preferences | arXiv | 2019 | 2,764 | [10.48550/arXiv.1909.08593](https://doi.org/10.48550/arXiv.1909.08593) |
| 17 | SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering | NeurIPS 2024 | 2024 | 2,109 | [10.48550/arXiv.2405.15793](https://doi.org/10.48550/arXiv.2405.15793) |
| 18 | Understanding R1-Zero-Like Training: A Critical Perspective | COLM 2025 | 2025 | 1,439 | [10.48550/arXiv.2503.20783](https://doi.org/10.48550/arXiv.2503.20783) |
| 19 | Scaling Laws for Reward Model Overoptimization | ICML 2023 | 2022 | 1,368 | [10.48550/arXiv.2210.10760](https://doi.org/10.48550/arXiv.2210.10760) |
| 20 | Simple statistical gradient-following algorithms for connectionist reinforcement learning | Machine Learning 1992 | 1992 | 1,117 | [10.1007/BF00992696](https://doi.org/10.1007/BF00992696) |
| 21 | Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback | TMLR 2023 | 2023 | 989 | [10.48550/arXiv.2307.15217](https://doi.org/10.48550/arXiv.2307.15217) |
| 22 | Tulu 3: Pushing Frontiers in Open Language Model Post-Training | arXiv | 2024 | 979 | [10.48550/arXiv.2411.15124](https://doi.org/10.48550/arXiv.2411.15124) |
| 23 | RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback | ICML 2024 | 2023 | 752 | [10.48550/arXiv.2309.00267](https://doi.org/10.48550/arXiv.2309.00267) |
| 24 | Self-Rewarding Language Models | ICML 2024 | 2024 | 751 | [10.48550/arXiv.2401.10020](https://doi.org/10.48550/arXiv.2401.10020) |
| 25 | Defining and Characterizing Reward Hacking | NeurIPS 2022 | 2022 | 258 | [10.48550/arXiv.2209.13085](https://doi.org/10.48550/arXiv.2209.13085) |
<!-- /TABLE:rl -->

**What this means for OA-RLAIF**
- **SFT → RL is the established pipeline** (InstructGPT; Stiennon et al.). RL from *AI* feedback with a written rubric comes from Constitutional AI. Lee et al. (ICML 2024) report that RLAIF is competitive with RLHF.
- **Our judge** follows the LLM-as-a-judge line (Zheng et al.). It carries only 20% of the reward, because of documented judge biases and reward over-optimization (Gao et al.; Skalse et al.).
- **GRPO** (DeepSeekMath; DeepSeek-R1, *Nature* 2025) removes the value network. The group-relative advantage replaces REINFORCE's return (Williams 1992), which the DRAL-OA agent used.
- **Dr. GRPO and DAPO** fix GRPO's length and normalisation biases and add dynamic sampling. We use `loss_type=dr_grpo` and drop zero-variance groups.
- **Verifiable rewards:** Tülu 3 (RLVR) supports rule-checked rewards. Our "gold match" term is that kind of reward, and the DRAL-OA term adds dense partial credit.

## B. On-policy distillation and self-distillation (knowledge into weights)

<!-- TABLE:opd -->
| # | Paper | Venue | Year | Citations | DOI |
|---|---|---|---|---:|---|
| 1 | Distilling the Knowledge in a Neural Network | NeurIPS 2014 DL workshop | 2015 | 26,690 | [10.48550/arXiv.1503.02531](https://doi.org/10.48550/arXiv.1503.02531) |
| 2 | Qwen3 Technical Report | arXiv (tech report) | 2025 | 8,949 | [10.48550/arXiv.2505.09388](https://doi.org/10.48550/arXiv.2505.09388) |
| 3 | A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning | AISTATS 2011 | 2010 | 4,769 | [10.48550/arXiv.1011.0686](https://doi.org/10.48550/arXiv.1011.0686) |
| 4 | Gemma 2: Improving Open Language Models at a Practical Size | arXiv (tech report) | 2024 | 2,540 | [10.48550/arXiv.2408.00118](https://doi.org/10.48550/arXiv.2408.00118) |
| 5 | Sequence-Level Knowledge Distillation | EMNLP 2016 | 2016 | 1,587 | [10.18653/v1/D16-1139](https://doi.org/10.18653/v1/D16-1139) |
| 6 | On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes | ICLR 2024 | 2023 | 1,034 | [10.48550/arXiv.2306.13649](https://doi.org/10.48550/arXiv.2306.13649) |
| 7 | Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models | arXiv 2026 | 2026 | 412 | [10.48550/arXiv.2601.18734](https://doi.org/10.48550/arXiv.2601.18734) |
| 8 | Reinforcement Learning via Self-Distillation | arXiv 2026 | 2026 | 302 | [10.48550/arXiv.2601.20802](https://doi.org/10.48550/arXiv.2601.20802) |
| 9 | Self-Distillation Enables Continual Learning | arXiv 2026 | 2026 | 237 | [10.48550/arXiv.2601.19897](https://doi.org/10.48550/arXiv.2601.19897) |
| 10 | DistiLLM: Towards Streamlined Distillation for Large Language Models | ICML 2024 | 2024 | 178 | [10.48550/arXiv.2402.03898](https://doi.org/10.48550/arXiv.2402.03898) |
| 11 | On-Policy Context Distillation for Language Models | arXiv 2026 | 2026 | 171 | [10.48550/arXiv.2602.12275](https://doi.org/10.48550/arXiv.2602.12275) |
| 12 | A Survey of On-Policy Distillation for Large Language Models | arXiv 2026 | 2026 | 151 | [10.48550/arXiv.2604.00626](https://doi.org/10.48550/arXiv.2604.00626) |
| 13 | Learning by Distilling Context | arXiv | 2022 | 137 | [10.48550/arXiv.2209.15189](https://doi.org/10.48550/arXiv.2209.15189) |
| 14 | MiniLLM: On-Policy Distillation of Large Language Models | ICLR 2024 | 2023 | 127 | [10.48550/arXiv.2306.08543](https://doi.org/10.48550/arXiv.2306.08543) |
| 15 | Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs? | arXiv 2026 | 2026 | 96 | [10.48550/arXiv.2603.24472](https://doi.org/10.48550/arXiv.2603.24472) |
| 16 | Privileged Solutions or Context-Induced Teacher Behavior? Dissecting On-Policy Self-Distillation | arXiv 2026 | 2026 | 11 | [10.48550/arXiv.2608.09228](https://doi.org/10.48550/arXiv.2608.09228) |
| 17 | Denser $\neq$ Better: Limits of On-Policy Self-Distillation for Continual Post-Training | arXiv 2026 | 2026 | 6 | [10.48550/arXiv.2607.01763](https://doi.org/10.48550/arXiv.2607.01763) |
| 18 | MixSD: Mixed Contextual Self-Distillation for Knowledge Injection | arXiv 2026 | 2026 | 3 | [10.48550/arXiv.2605.16865](https://doi.org/10.48550/arXiv.2605.16865) |
| 19 | OPD Before RL: Warm-Starting Rubric-Based RL with On-Policy Distillation | arXiv 2026 | 2026 | 0 | [10.48550/arXiv.2610.02781](https://doi.org/10.48550/arXiv.2610.02781) |
| 20 | On-Policy Distillation (Thinking Machines Lab: Connectionism) | Blog 2025 | 2025 | n/a | no DOI · [link](https://thinkingmachines.ai/blog/on-policy-distillation/) |
<!-- /TABLE:opd -->

**What this means for OA-RLAIF**
- **Why on-policy:** GKD (ICLR 2024) and MiniLLM (ICLR 2024) train the student on its *own* samples against a teacher's token distribution. This removes the train/inference mismatch of off-policy SFT (sequence-level KD, Kim & Rush 2016), and descends from DAgger (Ross et al. 2011).
- **The method we use: OPSD and SDFT (2026).** The teacher is the *same* model conditioned on privileged information: the datasource passage, the entity card, the gold mapping, or the ICD bridge rows. The student sees only the question.
  - This is how knowledge goes into weights **without RAG at inference**. It generalises context distillation (Snell et al. 2022) to on-policy training.
  - SDFT reports better knowledge acquisition than SFT and less forgetting.
  - TRL implements it as `SDFTTrainer`.
- **Risks we measure rather than assume:**
  - privileged-context distillation can hurt *thinking* models (we use a non-thinking model);
  - its gains may come from the context changing the teacher, not from the answer itself;
  - dense self-distillation may forget more than GRPO.
  We therefore track closed-book retention and run GRPO after OPD, as in *OPD Before RL* (2026).

## C. Ontology matching with ML/LLM training (ranked by citations)

<!-- TABLE:om -->
| # | Paper | Venue | Year | Citations | DOI |
|---|---|---|---|---:|---|
| 1 | node2vec: Scalable Feature Learning for Networks | KDD 2016 | 2016 | 12,677 | [10.1145/2939672.2939754](https://doi.org/10.1145/2939672.2939754) |
| 2 | Ontology Matching | Springer 2013 (book, 2nd ed.) | 2013 | 2,097 | [10.1007/978-3-642-38721-0](https://doi.org/10.1007/978-3-642-38721-0) |
| 3 | LogMap: Logic-Based and Scalable Ontology Matching | ISWC 2011 | 2011 | 653 | [10.1007/978-3-642-25073-6_18](https://doi.org/10.1007/978-3-642-25073-6_18) |
| 4 | Multilingual Knowledge Graph Embeddings for Cross-lingual Knowledge Alignment | IJCAI 2017 | 2017 | 641 | [10.24963/ijcai.2017/209](https://doi.org/10.24963/ijcai.2017/209) |
| 5 | Bootstrapping Entity Alignment with Knowledge Graph Embedding | IJCAI 2018 | 2018 | 614 | [10.24963/ijcai.2018/611](https://doi.org/10.24963/ijcai.2018/611) |
| 6 | Cross-lingual Knowledge Graph Alignment via Graph Convolutional Networks | EMNLP 2018 | 2018 | 600 | [10.18653/v1/D18-1032](https://doi.org/10.18653/v1/D18-1032) |
| 7 | Learning to match ontologies on the Semantic Web | VLDB Journal 2003 | 2003 | 584 | [10.1007/s00778-003-0104-2](https://doi.org/10.1007/s00778-003-0104-2) |
| 8 | Self-Alignment Pretraining for Biomedical Entity Representations | NAACL 2021 | 2021 | 543 | [10.18653/v1/2021.naacl-main.334](https://doi.org/10.18653/v1/2021.naacl-main.334) |
| 9 | The AgreementMakerLight Ontology Matching System | OTM/ODBASE 2013 | 2013 | 334 | [10.1007/978-3-642-41030-7_38](https://doi.org/10.1007/978-3-642-41030-7_38) |
| 10 | OWL2Vec*: embedding of OWL ontologies | Machine Learning 2021 | 2021 | 185 | [10.1007/s10994-021-05997-6](https://doi.org/10.1007/s10994-021-05997-6) |
| 11 | BERTMap: A BERT-Based Ontology Alignment System | AAAI 2022 | 2022 | 143 | [10.1609/aaai.v36i5.20510](https://doi.org/10.1609/aaai.v36i5.20510) |
| 12 | A Simple Standard for Sharing Ontological Mappings (SSSOM) | Database 2022 | 2022 | 121 | [10.1093/database/baac035](https://doi.org/10.1093/database/baac035) |
| 13 | OLaLa: Ontology Matching with Large Language Models | K-CAP 2023 | 2023 | 102 | [10.1145/3587259.3627571](https://doi.org/10.1145/3587259.3627571) |
| 14 | DeepAlignment: Unsupervised Ontology Matching with Refined Word Vectors | NAACL 2018 | 2018 | 95 | [10.18653/v1/N18-1072](https://doi.org/10.18653/v1/N18-1072) |
| 15 | Augmenting Ontology Alignment by Semantic Embedding and Distant Supervision | ESWC 2021 | 2021 | 70 | [10.1007/978-3-030-77385-4_23](https://doi.org/10.1007/978-3-030-77385-4_23) |
| 16 | Reinforcement Learning–based Collective Entity Alignment with Adaptive Features | ACM TOIS 2021 | 2021 | 67 | [10.1145/3446428](https://doi.org/10.1145/3446428) |
| 17 | Exploring Large Language Models for Ontology Alignment | ISWC 2023 (posters) | 2023 | 67 | [10.48550/arXiv.2309.07172](https://doi.org/10.48550/arXiv.2309.07172) |
| 18 | Agent-OM: Leveraging LLM Agents for Ontology Matching | PVLDB 2025 | 2023 | 52 | [10.14778/3712221.3712222](https://doi.org/10.14778/3712221.3712222) |
| 19 | LLMs4OM: Matching Ontologies with Large Language Models | ESWC 2024 (satellite) | 2024 | 51 | [10.48550/arXiv.2404.10317](https://doi.org/10.48550/arXiv.2404.10317) |
| 20 | DeepOnto: A Python Package for Ontology Engineering with Deep Learning | Semantic Web Journal 2024 | 2023 | 47 | [10.48550/arXiv.2307.03067](https://doi.org/10.48550/arXiv.2307.03067) |
| 21 | Applying Logical Constraints to Ontology Matching | KI 2007 | 2007 | 34 | [10.1007/978-3-540-74565-5_10](https://doi.org/10.1007/978-3-540-74565-5_10) |
| 22 | VeeAlign: Multifaceted Context Representation Using Dual Attention for Ontology Alignment | EMNLP 2021 | 2021 | 23 | [10.18653/v1/2021.emnlp-main.842](https://doi.org/10.18653/v1/2021.emnlp-main.842) |
| 23 | Deep reinforcement learning approach for ontology matching problem | Int. J. Data Science and Analytics 2023 | 2023 | 11 | [10.1007/s41060-023-00425-5](https://doi.org/10.1007/s41060-023-00425-5) |
| 24 | Truveta Mapper: A Zero-shot Ontology Alignment Framework | arXiv | 2023 | 11 | [10.48550/arXiv.2301.09767](https://doi.org/10.48550/arXiv.2301.09767) |
| 25 | Deep Reinforcement Learning Based Ontology Meta-Matching Technique | IEICE Trans. Inf. & Syst. 2023 | 2023 | 8 | [10.1587/transinf.2022dlp0050](https://doi.org/10.1587/transinf.2022dlp0050) |
| 26 | ONTOCONNECT: Domain-Agnostic Ontology Alignment using Graph Embedding with Negative Sampling | ICMLA 2021 | 2021 | 7 | [10.1109/icmla52953.2021.00155](https://doi.org/10.1109/icmla52953.2021.00155) |
| 27 | Deeper and Deeper: A Lightweight Semi-Supervised Deep Reinforcement Adaptive Learning-Based Ontology Alignment | AIxDKE 2024 | 2024 | 1 | [10.1109/AIxDKE63520.2024.00012](https://doi.org/10.1109/AIxDKE63520.2024.00012) |
| 28 | Ontology Matching | Springer 2007 (book) | 2007 | n/a | [10.1007/978-3-540-49612-0](https://doi.org/10.1007/978-3-540-49612-0) |
| 29 | Machine Learning-Friendly Biomedical Datasets for Equivalence and Subsumption Ontology Matching | ISWC 2022 | 2022 | n/a | [10.1007/978-3-031-19433-7_33](https://doi.org/10.1007/978-3-031-19433-7_33) |
| 30 | DRAL-OA: Results for OAEI 2025 | OM@ISWC 2025 (CEUR Vol-4144) | 2025 | n/a | no DOI · [link](https://ceur-ws.org/Vol-4144/om2025-oaei-paper6.pdf) |
<!-- /TABLE:om -->

**What this means for OA-RLAIF**
- **Foundations:** Euzenat & Shvaiko (the datasource used for SDFT) and their TKDE survey define the task, the evaluation measures and the alignment format that we output (OAEI RDF and SSSOM).
- **Strong non-LLM baselines** are LogMap (logic plus lexical, with repair) and AML. Learned matchers are BERTMap (AAAI 2022), DeepAlignment, VeeAlign and SapBERT-style encoders. Bio-ML (ISWC 2022) defines the 2026 benchmark and scorer that we use.
- **DRAL-OA** is the user's prior system, with an official OAEI 2025 Anatomy F of 0.828. It and other RL matchers (Xue 2023; Touati & Kemmar 2023; CEAFF for KG entity alignment) use RL over embeddings or similarity weights. None of them train an LLM policy.
- **LLM matchers:** OLaLa, LLMs4OM, Agent-OM and MILA use LLMs at inference time through prompting, retrieval or agents. The 2025 synthetic-data paper fine-tunes an LLM with SFT. None of them use on-policy distillation or GRPO.

## D. Closest prior work and novelty

<!-- TABLE:closest -->
| # | Paper | Venue | Year | Citations | DOI |
|---|---|---|---|---:|---|
| 1 | Ontology Matching with Large Language Models and Prioritized Depth-First Search | arXiv 2025 | 2025 | 31 | [10.48550/arXiv.2501.11441](https://doi.org/10.48550/arXiv.2501.11441) |
| 2 | OntoTune: Ontology-Driven Self-training for Aligning Large Language Models | WWW 2025 | 2025 | 20 | [10.1145/3696410.3714816](https://doi.org/10.1145/3696410.3714816) |
| 3 | OAEI-LLM-T: A TBox Benchmark Dataset for Understanding Large Language Model Hallucinations in Ontology Matching | arXiv 2025 | 2025 | 2 | [10.48550/arXiv.2503.21813](https://doi.org/10.48550/arXiv.2503.21813) |
| 4 | Improving LLM-based Ontology Matching with fine-tuning on synthetic data | arXiv 2025 | 2025 | 0 | [10.48550/arXiv.2511.22612](https://doi.org/10.48550/arXiv.2511.22612) |
| 5 | Surprising Effectiveness of Self-Demonstrations in Enhancing Schema-Ontology Mapping with LLMs | arXiv 2026 | 2026 | 0 | [10.48550/arXiv.2609.13776](https://doi.org/10.48550/arXiv.2609.13776) |
| 6 | Ontology Concept Overlap as a Training Signal: Knowledge-Grounded Reinforcement Learning for Clinical Question Answering | arXiv 2026 | 2026 | 0 | [10.48550/arXiv.2610.06360](https://doi.org/10.48550/arXiv.2610.06360) |
| 7 | From Matching to Retrieval: A New Role for LLMs in Ontology Alignment | OM@ISWC 2025 (CEUR Vol-4144) | 2025 | n/a | no DOI · [link](https://ceur-ws.org/Vol-4144/) |
<!-- /TABLE:closest -->

**Novelty statement (as of the snapshot date).** We searched arXiv, the OAEI/OM 2025 proceedings (CEUR Vol-4144), and the awesome-on-policy-distillation list (about 300 entries, October 2026). We found **no work that applies on-policy self-distillation or GRPO to train an LLM ontology matcher**. We also found none that combines a verifiable alignment reward with an embedding-based (DRAL-OA) reward and an AI judge.

The nearest works each cover one piece:
- LLM-based OM without RL (OLaLa, LLMs4OM, Agent-OM, MILA, *From Matching to Retrieval*);
- SFT of an LLM matcher on synthetic alignments (arXiv 2511.22612);
- ontology-driven self-training for knowledge, not matching (OntoTune, WWW 2025);
- an ontology used as a GRPO reward for *clinical QA*, not alignment (arXiv 2610.06360).

ICD-11↔SNOMED CT mappings composed through the WHO ICD-10 bridge and used to train a policy also have no direct precedent that we found. This is an absence-of-evidence claim, and it should be re-checked before submission.

## E. Ontology parsing and alignment verification (ranked by citations)

<!-- TABLE:verify -->
| # | Paper | Venue | Year | Citations | DOI |
|---|---|---|---|---:|---|
| 1 | The OWL API: A Java API for OWL ontologies | Semantic Web Journal 2011 | 2011 | 916 | [10.3233/sw-2011-0025](https://doi.org/10.3233/sw-2011-0025) |
| 2 | OOPS! (OntOlogy Pitfall Scanner!): An On-line Tool for Ontology Evaluation | IJSWIS 2014 | 2014 | 440 | [10.4018/ijswis.2014040102](https://doi.org/10.4018/ijswis.2014040102) |
| 3 | The Alignment API 4.0 | Semantic Web Journal 2011 | 2011 | 279 | [10.3233/sw-2011-0028](https://doi.org/10.3233/sw-2011-0028) |
| 4 | The Incredible ELK: From Polynomial Procedures to Efficient Reasoning with EL Ontologies | J. Automated Reasoning 2014 | 2013 | 217 | [10.1007/s10817-013-9296-3](https://doi.org/10.1007/s10817-013-9296-3) |
| 5 | ROBOT: A Tool for Automating Ontology Workflows | BMC Bioinformatics 2019 | 2019 | 211 | [10.1186/s12859-019-3002-3](https://doi.org/10.1186/s12859-019-3002-3) |
| 6 | Logic-based assessment of the compatibility of UMLS ontology sources | J. Biomedical Semantics 2011 | 2011 | 104 | [10.1186/2041-1480-2-s1-s2](https://doi.org/10.1186/2041-1480-2-s1-s2) |
| 7 | MELT - Matching EvaLuation Toolkit | SEMANTiCS 2019 | 2019 | 68 | [10.1007/978-3-030-33220-4_17](https://doi.org/10.1007/978-3-030-33220-4_17) |
| 8 | Towards valid and reusable reference alignments — ten basic quality checks for ontology alignments and their application to three different reference data sets | J. Biomedical Semantics 2012 | 2012 | 26 | [10.1186/2041-1480-3-S1-S4](https://doi.org/10.1186/2041-1480-3-S1-S4) |
| 9 | HermiT: An OWL 2 Reasoner | J. Automated Reasoning 2014 | 2014 | 0 | [10.1007/s10817-014-9305-1](https://doi.org/10.1007/s10817-014-9305-1) |
<!-- /TABLE:verify -->

**What this means for OA-RLAIF.** Every ontology file is parsed twice: once by our streaming Python parser and once by the **OWL API** (the library behind Protégé, ROBOT and HermiT). Declared class counts, or `skos:Concept` members for SKOS vocabularies, must match exactly. **ELK** checks consistency and unsatisfiable classes. Every reference alignment gets the **Beisswanger & Hahn (2012)** quality checks: standard format, URIs, explicit relation types, missing classes, multiple targets, trivial correspondences, and label-identical pairs without a correspondence. Measure range, duplicate and conflicting-relation checks are added, as in MELT's validation services and the Alignment API's parsers. Results are in `docs/PARSE_VERIFICATION.md` and are enforced by `data/verify.py`. Logical repair of predicted alignments follows LogMap/ALCOMO-style coherence checking.

## F. Schema and ontology matching surveys: the datasource survey corpus (ranked by citations)

These 18 surveys (1986–2023) are the user-supplied folder `Matching Surveys`. They serve two purposes:
- **Citations:** each row is DOI-verified like every other entry.
- **Training knowledge:** each PDF is split into passages and becomes SDFT knowledge prompts alongside the Ontology Matching textbook (`OA_PAPERS_DIR`; see `docs/PLAN.md`).

Together they trace the field:
- schema integration in databases (Batini 1986; Spaccapietra 1992; Kashyap & Sheth 1996);
- automatic schema matching (Rahm & Bernstein 2001; Shvaiko & Euzenat 2005);
- ontology mapping (Kalfoglou & Schorlemmer 2003; Noy 2004; Choi 2006);
- evaluation and open challenges (Shvaiko & Euzenat 2008, 2013; Euzenat et al. 2011);
- the large-scale, complex and background-knowledge directions that this project builds on (Ochieng & Kyanda 2018; Thiéblin et al. 2020; Portisch et al. 2022).

<!-- TABLE:survey -->
| # | Paper | Venue | Year | Citations | DOI |
|---|---|---|---|---:|---|
| 1 | A survey of approaches to automatic schema matching | VLDB Journal 2001 | 2001 | 4,016 | [10.1007/s007780100057](https://doi.org/10.1007/s007780100057) |
| 2 | Ontology mapping: the state of the art | Knowledge Engineering Review 2003 | 2003 | 1,559 | [10.1017/S0269888903000651](https://doi.org/10.1017/S0269888903000651) |
| 3 | A Survey of Schema-Based Matching Approaches | Journal on Data Semantics IV (LNCS 3730) 2005 | 2005 | 1,397 | [10.1007/11603412_5](https://doi.org/10.1007/11603412_5) |
| 4 | Ontology Matching: State of the Art and Future Challenges | IEEE TKDE 2013 | 2013 | 1,333 | [10.1109/TKDE.2011.253](https://doi.org/10.1109/TKDE.2011.253) |
| 5 | Semantic integration: a survey of ontology-based approaches | SIGMOD Record 2004 | 2004 | 1,246 | [10.1145/1041410.1041421](https://doi.org/10.1145/1041410.1041421) |
| 6 | A survey on ontology mapping | SIGMOD Record 2006 | 2006 | 688 | [10.1145/1168092.1168097](https://doi.org/10.1145/1168092.1168097) |
| 7 | Semantic and schematic similarities between database objects: a context-based approach | VLDB Journal 1996 | 1996 | 513 | [10.1007/s007780050029](https://doi.org/10.1007/s007780050029) |
| 8 | Ontology matching: A literature review | Expert Systems with Applications 2015 | 2015 | 392 | [10.1016/j.eswa.2014.08.032](https://doi.org/10.1016/j.eswa.2014.08.032) |
| 9 | Ten Challenges for Ontology Matching | ODBASE / OTM 2008 (LNCS 5332) | 2008 | 352 | [10.1007/978-3-540-88873-4_18](https://doi.org/10.1007/978-3-540-88873-4_18) |
| 10 | Ontology Alignment Evaluation Initiative: Six Years of Experience | Journal on Data Semantics XV (LNCS 6720) 2011 | 2011 | 320 | [10.1007/978-3-642-22630-4_6](https://doi.org/10.1007/978-3-642-22630-4_6) |
| 11 | Why Your Data Won’t Mix | ACM Queue 2005 | 2005 | 181 | [10.1145/1103822.1103836](https://doi.org/10.1145/1103822.1103836) |
| 12 | Model independent assertions for integration of heterogeneous schemas | VLDB Journal 1992 | 1992 | 175 | [10.1007/bf01228708](https://doi.org/10.1007/bf01228708) |
| 13 | Ontology Alignment—A Survey with Focus on Visually Supported Semi-Automatic Techniques | Future Internet 2010 | 2010 | 87 | [10.3390/fi2030238](https://doi.org/10.3390/fi2030238) |
| 14 | Survey on complex ontology matching | Semantic Web Journal 2020 | 2020 | 67 | [10.3233/SW-190366](https://doi.org/10.3233/SW-190366) |
| 15 | Background knowledge in ontology matching: A survey | Semantic Web Journal 2022 | 2022 | 40 | [10.3233/SW-223085](https://doi.org/10.3233/SW-223085) |
| 16 | Large-scale ontology matching: state-of-the-art analysis | ACM Computing Surveys 2018 | 2018 | 29 | [10.1145/3211871](https://doi.org/10.1145/3211871) |
| 17 | A comparative analysis of methodologies for database schema integration | ACM Computing Surveys 1986 | 1986 | n/a | [10.1145/27633.27634](https://doi.org/10.1145/27633.27634) |
| 18 | Semantic Integration Research in the Database Community: A Brief Survey | AI Magazine 2005 | 2005 | n/a | no DOI · [link](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/1801) |
<!-- /TABLE:survey -->

## Dropped or unverified entries

<!-- DROPPED -->
- `bodenreider2005mice` — Of mice and men: aligning mouse and human anatomies: no DOI found by Crossref title search
- `meilicke2007repair` — Repairing Ontology Mappings: no DOI found by Crossref title search
<!-- /DROPPED -->

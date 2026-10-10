# Review of the original 16-step plan, adapted to ontology alignment

The original plan, pasted on 2026-10-10, was generic. It used Gemma 4 E4B, mini-SWE-agent with Docker, an A4B judge, Unsloth SFT and then GRPO. This document goes through it step by step:
- **Keep** means unchanged.
- **Change** means adapted.
- **Why** gives the evidence.

The current plan is `docs/PLAN.md`. Decisions are logged in `MEMORY.md`.

## Setup

**1. Define success.** *Change: make it measurable.* "Task finished" is defined at three levels:
- **Episode:** the submitted target is in the gold set, or NONE when the gold target is not among the candidates.
- **System:** OAEI P/R/F1 (plus Recall+ on Anatomy) on held-out splits and zero-shot pairs. Beat DRAL-OA's official OAEI 2025 Anatomy F of **0.828**.
- **New mappings:** recovery rate on held-out ICD bridge links.

*Why:* "100% correct" can be neither reached nor measured. Reference alignments are incomplete and noisy, and Bio-ML had to repair its references. The aim is to maximize precision with calibrated abstention.

**2. 80/20 split.** *Change: split by group, not at random.*
- The split is 80/20 by source-entity group within each pair.
- A concept is grouped across all languages (MultiFarm, DH-multiling). SNOMED IRIs are grouped across Bio-ML pairs.
- Whole pairs are held out for zero-shot testing.
- The DRAL mapping layer is fit on train references only.

*Why:* a random row split leaks the same concept into both train and test, because it appears in several language pairs or ontology pairs. The original DRAL-OA `clf.py` evaluated on its own training references. Bio-ML's test gold is hidden, so its held-out set is carved from the public train/valid references.

**3. mini-SWE-agent with Docker.** *Keep it, for baseline, evaluation and deployment.*
- An `om` CLI (`show`, `candidates`, `submit`) runs inside a Docker image that holds the ontologies.
- Actions use mini-SWE-agent v2's bash-block format, and the task ends with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.

*Change:* GRPO trains on the final decision turn, in the same message format, rather than on full multi-turn Docker rollouts. Multi-turn agentic RL is out of scope for v1. *Why:* it costs and throughput fit the $500 budget, and the train and deploy formats stay identical. Lambda on-demand instances ship with Docker preinstalled.

**4. Serve A4B as the judge.** *Change: call it through an API* (the user's choice), with any OpenAI-compatible endpoint.
- Temperature 0, cached, with a neutral fallback on timeout.
- The judge never sees gold.
- Self-hosting A4B on vLLM is needed only if A4B becomes a logit-level distillation teacher; that is off by default.

## Reward

**5. Reward function with partial credit.** *Change:* the task reward is the gold match plus the user's **DRAL-OA reward**:
- `R_task = 0.5·1[pick ∈ gold] + 0.5·r_DRAL`;
- `r_DRAL` is Eq. 1 of the DRAL-OA papers: `softmax_T(sim(z_s, z_pick))` over the prompt's candidates, with `sim = 0.5·cos(mapping-layer) + 0.5·cos(label)`;
- bad format or a hallucinated IRI scores 0;
- silver ICD episodes get weight 0.5, and unlabeled episodes use only the DRAL term, at weight 0.3.

*Why:* the gold term stops the policy from copying the mapping layer's mistakes, and the DRAL term gives dense partial credit for near misses.

**6. A4B scores the trajectory 0–10 against a rubric.** *Keep.*
- The rubric covers: evidence cited correctly (lexical, structural or semantic, following the OM datasource's classification of matching techniques), thought consistent with the answer, no invented IRIs, and calibrated confidence.

**7. Combine as `0.8 × function + 0.2 × judge`.** *Keep:* `R = 0.8·R_task + 0.2·J/10`.
- *Why:* judge-only rewards are prone to over-optimization (Gao et al. 2023) and to hacking (Skalse et al. 2022). A minority weight for the judge is a standard safeguard.

## Baseline

**8. Base E4B in mini-SWE-agent on the held-out set.** *Keep,* and run the same set as a vLLM batch for speed. The baseline also fixes X, the ICD bridge-recovery target in the CLAUDE.md goal.

## SFT

**9. Convert data to mini-SWE-agent-style conversations in Gemma 4's chat format.** *Keep the format.*
- Use the non-thinking `gemma-4` template, as Unsloth recommends for small models.
- Each SDFT prompt carries the student-visible prompt plus a `privileged_context` field for the teacher.

**10. Fine-tune E4B with LoRA in Unsloth for 1–3 epochs.** *Change:*
- **SDFT/OPSD instead of plain SFT**, at the user's request for 2026 knowledge injection with no RAG.
- **TRL instead of Unsloth.**
- About 2 epochs.

*Why:*
- Unsloth's Gemma 4 guide says GRPO must disable vLLM fast inference, and Unsloth is single-GPU oriented. TRL uses all 4 B200s and a vLLM server.
- OPSD (2026) and SDFT (2026) beat SFT on knowledge acquisition with less forgetting, building on GKD (ICLR 2024).
- Plain SFT stays only as a comparison row.

**11. Evaluate on the held-out set.** *Keep.* Also measure closed-book datasource QA and ontology-fact retention, to check that knowledge actually landed in the weights.

## RLAIF (GRPO)

**12. Start from the SFT checkpoint.** *Change:* start from the **SDFT** checkpoint.

**13. GRPO with the combined reward, temperature ≈ 1.0, 4–8 attempts.** *Keep*, with 8 generations and temperature 1.0. *Add:*
- `loss_type=dr_grpo` and `mask_truncated_completions`;
- drop zero-variance groups (DAPO dynamic sampling);
- vLLM server on GPU 3, training on GPUs 0–2.

**14. Watch the average reward and read samples.** *Keep.* *Add:*
- log each reward term separately;
- dump samples every N steps;
- alarms on NONE rate, out-of-list IRIs, length drift, and disagreement between the judge and correctness.

## Finish

**15. Evaluate and compare with steps 8 and 11.** *Keep.* *Add:*
- full-mapping runs (Anatomy as whole ontologies; ICD-11 MMS → SNOMED as SSSOM);
- comparison with DRAL-OA's 0.828 and the lexical baseline.

**16. Merge, export to GGUF, run in Ollama, use in mini-SWE-agent.** *Keep.*
- Unsloth's `save_pretrained_gguf` (or llama.cpp convert) with the same Gemma 4 chat template in the Ollama Modelfile.
- Push the adapters to private HF.

## Added by this review
- **Data coverage gate:** all 2,573 raw files have a role, and `verify.py` fails otherwise.
- **ICD-11↔SNOMED triangulation:** silver, typed links through the WHO ICD-10 bridge.
- **Budget guard:** $500 total, with a stop at $450.
- **Progress bars and ETAs** for every stage.
- **Private GitHub and HF publishing**, behind verification gates.

# Reward specification: from DRAL-OA to GRPO

## Sources
- **DRAL-OA (AIxDKE 2024).** Zaeifi, Mosallanezhad, Bansal. DOI [10.1109/AIxDKE63520.2024.00012](https://doi.org/10.1109/AIxDKE63520.2024.00012). It gives Eq. 1 (contrastive reward), Eq. 2–3 (REINFORCE) and Algorithm 1.
- **DRAL-OA OAEI 2025 results.** Gonibeed Ravishankar, Zaefi, Bansal, OM@ISWC 2025 (CEUR Vol-4144). Official Anatomy result: P 0.83, R 0.827, F 0.828.
- **The authors' code**, `old bert 2/clf.py`, which is ported in `src/oa_rlaif/reward/dral_mapping_layer.py`. It contains the mapping layer only; the REINFORCE agent and Eq. 1 are not in the released code.

## What the paper defines

| Element | DRAL-OA |
|---|---|
| State | Source-class embedding E′ (FastText or BERT label vector ⊕ node2vec relation vector) |
| Action | Add or subtract σ = 0.01 on one dimension of E′ (\|E′\| × 2 actions) |
| Reward | Eq. 1: L_ij = −log( exp(sim(z_i,z_j)/T) / Σ_{k≠i} exp(sim(z_i,z_k)/T) ), with sim = cosine |
| Optimiser | REINFORCE (Eq. 2–3), discount λ = 0.99, 2,000 episodes, T = 20 steps, lr 1e-4 |
| Ranking | combined similarity w_v·cos(label) + w_r·cos(relation), with w_v = w_r = 0.5 as the best setting (Table I) |

## What GRPO uses
In OA-RLAIF the policy is the LLM and the action is a choice among the episode's candidates: C1..Ck, or NONE. Eq. 1 turns that choice into a reward.

- **Labeled episodes:** r_DRAL(pick) = exp(−L_{s,pick}) = softmax over k ∈ candidates of sim(s,k)/T, evaluated at the pick.
- **NONE:** r_DRAL(NONE) = 1 − max_k softmax_T(sim(s,k)). This is high when no candidate looks like a match.
- **Similarity:** sim(i,j) = 0.5·cos(z_i, z_j) + 0.5·cos(t_i, t_j), where z is the mapping-layer embedding (128-d) and t is the FastText label vector (300-d), exactly as clf.py and Table I combine them.
- **Temperature:** T is not reported in the papers. It is the config value `grpo.dral_temperature` (default 0.1) and is varied in sensitivity checks.

The task reward and the total:

```
labeled   : R_task = 0.5·1[pick ∈ gold  or  NONE when gold ∉ candidates] + 0.5·r_DRAL
silver    : R_task = 0.5 · (0.5·1[pick ∈ composed links] + 0.5·r_DRAL)          (ICD triangulation)
unlabeled : R_task = 0.3·r_DRAL                                                  (semi-supervised, as DRAL-OA)
invalid   : R_task = 0 and judge = 0       (no THOUGHT, ≠1 bash block, index out of range, conf ∉ [0,1])
total     : R = 0.8·R_task + 0.2·J/10       (J = A4B judge score, rubric in configs/judge_rubric.md)
```

GRPO's group-relative advantage Â_i = R_i − mean(R_group) replaces REINFORCE's discounted return G_t. Each prompt is a single decision, so λ has no effect. The loss is Dr. GRPO (`loss_type=dr_grpo`, `scale_rewards=none`) with truncated completions masked.

## Faithful port of the mapping layer (clf.py), with the differences stated

| | clf.py | OA-RLAIF |
|---|---|---|
| Graph | parents (weight 100), children/restrictions/disjoint (weight 1) | parents (100), disjoint (1). The parser keeps named parents and equivalences, not restriction fillers |
| node2vec | dim 100, walk 30, 20 walks, window 20, min_count 1, batch_words 256, p = q = 1 | same. Vectorised first-order weighted walks. 5 walks for graphs >50K nodes, restricted to episode entities + parents (SNOMED / NCIT scale) |
| Features | [node2vec ; label vector], each standardised within itself | same (FastText `fast.bin`, 300-d) |
| Model | 400→256 tanh→128 embedder; concat→64 ReLU→1 sigmoid; BCE; Adam 3e-4; 20 epochs; batch 32; 50,000 pairs at 50% positives; no dropout | same as clf.py (default). The paper's settings (Adam 1e-5, dropout 0.2) are config options for a sensitivity run |
| Positives | every reference pair | **train-split references only** |
| Evaluation | top-k over targets that occur in the reference, scored on the same references | held-out references, ranked over all targets |

## Paper (AIxDKE 2024) against the released code, re-checked 2026-10-10
The paper defines the parts that the released code lacks: Eq. 1, Eq. 2–3 and Algorithm 1 (the REINFORCE agent). We use Eq. 1 as the reward. We do not port the REINFORCE agent, because GRPO replaces it. In DRAL-OA the agent perturbs one embedding value by ±σ per step. In OA-RLAIF the policy is the LLM, which picks a candidate.

| Item | Paper | `clf.py` | OA-RLAIF |
|---|---|---|---|
| Mapping-layer optimiser | Adam, lr 1e-5 | Adam, lr 3e-4 | 3e-4 (code), 1e-5 as an option |
| Embedding-layer dropout | p = 0.2 | none | none (code), 0.2 as an option |
| node2vec | 100 / 30 / 20 / window 20 / min count 1 / batch words 256 | same | same |
| Label vectors | BERT (768-d) beats FastText (300-d) in Table I | FastText `fast.bin` | FastText (local file, no download) |
| Agent | REINFORCE, 2,000 episodes, T = 20 steps, λ = 0.99, lr 1e-4, σ = 0.01, α = β = 0.5, policy MLP with 320 hidden units | absent | replaced by GRPO (single-step episodes, so λ has no effect) |
| Eq. 1 temperature | symbol T, value **not reported** (T also names the 20-step horizon) | absent | `grpo.dral_temperature` = 0.1, varied in sensitivity checks |
| α, β | listed in Algorithm 1 and set to 0.5, but their role is not defined | absent | not used |

A search of the authors' local code folders (`DRAL_OA`, `Advanced-Cybersecurity-OA`) on 2026-10-10 found no REINFORCE agent and no Eq. 1 implementation. The cybersecurity variants define a margin contrastive loss (margin 1.0) but train with BCE, and they rank with weights 1.0 / 0.0 instead of 0.5 / 0.5 (`DralScorer(wv, wr)` accepts both).

What the paper does not give (and is therefore a stated assumption here): the temperature value, the role of α and β, the "external resource" used for 50% of the semi-supervised training data, and how the negatives k in the Eq. 1 denominator are chosen. OA-RLAIF uses the episode's candidate list as the denominator set.

## Measured on Anatomy (held-out, 2026-10-10)
315 held-out references; mapping layer fit on 1,201 train references.

| Ranker | Hits@1 |
|---|---:|
| Lexical retrieval alone | 0.867 |
| DRAL re-ranking of the lexical top-20 | 0.778 |
| DRAL over all 3,304 targets | 0.740 |

On unseen references the mapping layer is weaker than lexical matching. That is why the gold term carries correctness and r_DRAL supplies graded partial credit. r_DRAL is not used as ground truth.

## Reward-hacking safeguards
- **Strict format gate:** see the invalid case above.
- **Separate logging** of every term (gold, DRAL, judge) and of the invalid rate.
- **Completion dumps** every 50 steps.
- **Alarms** on NONE rate, invalid rate, length drift, and judge–correctness divergence (`train/grpo.py: hacking_monitor`).
- **Judge isolation:** the judge never sees gold, scores are cached, and it falls back to a neutral 5 on API failure (counted).

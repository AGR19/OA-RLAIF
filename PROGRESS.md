# OA-RLAIF build progress

_Updated: 2026-10-10 (afternoon). Estimates are agent working time on the Mac. The Lambda run starts only after the build and your OK._

```
Build   [█████████████░░░░░░░]  11 / 17 steps done, 4 running    ~2 h remaining
Lambda  [░░░░░░░░░░░░░░░░░░░░]  not started      ~16 h, ~$440 (judge is local: no API cost)
```

| # | Step | Status | Est. |
|---|------|--------|------|
| 0 | Save plan, CLAUDE.md, MEMORY.md | done | — |
| 1 | Method figure (TikZ → PDF/PNG/SVG) | done — judge box now reads "local" | — |
| 2 | Repo review | done — `docs/REPO_REVIEW.md` | — |
| 3 | Literature review + DOI checks | done — 105 verified, 4 no-DOI (URL recorded), 2 dropped. Section F adds the 18 "Matching Surveys". | — |
| 4 | Plan review | done — `docs/PLAN_REVIEW.md` | — |
| 5 | Package, progress + budget utils | done — 35 tests pass | — |
| 6 | Data rebuild, all 2,573 raw files | done. Passages rebuilt: textbook 16/16 units (461), 18 surveys (978), other docs (194) | — |
| 7 | Datasource questions | **running** — ASU `qwen36-27b`, thinking off, 1,633 passages × 5 questions; question-level test split | ~70 m |
| 8 | ICD-11↔SNOMED triangulation | done — held-out recall@20 SNOMED→ICD-11 0.196 → 0.614; ICD-10→ICD-11 0.668 → 0.782 | — |
| 9 | DRAL-OA reward port + fit | port, Eq. 1, tests and Anatomy check done; fit-all queued | 45 m |
| 10 | A4B judge | done — served locally on Lambda GPU 3 (vLLM, thinking off) | — |
| 11 | Harness: om CLI, Docker, mini-SWE | done | — |
| 12 | Train / eval / export scripts | done | — |
| 12b | Parse + reference verifier | code done; 9-file sample matches; full run queued | 30 m |
| 12c | Modelling profiles + `align` output modes | code done; profile build queued | 10 m |
| 12d | Thesis chapter "RLAIF" (ASD-STE100) | **running** — background agent writes into `~/Downloads/adishesh-Thesis/` | — |
| 13 | Verification gates (coverage, leakage, tests, scan) | queued | 60 m |
| 14 | Push: private GitHub + private HF (`gonibeed`) | queued — needs your OK | 15 m |

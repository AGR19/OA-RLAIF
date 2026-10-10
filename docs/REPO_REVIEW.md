# Review of the existing `oa-sft-sm-1` repository (state on 2026-10-10)

## What was built
- **A complete audited pipeline for Inkling-Small SFT on Tinker:**
  - **Parsers:** a streaming OWL/RDF/Turtle/OWL-Functional/OWL-XML parser (`src/data/ontology_parser.py`) and an alignment reader keeping EDOAL expressions (`src/data/extractors/common.py`).
  - **Evaluation:** OAEI P/R/F1/Recall+ plus ranking metrics, with a parity test against the Bio-ML scoring kit (`src/training/eval_metrics.py`).
  - **Matching support:** q-gram + SapBERT retrieval (`src/inference/candidate_generator.py`), a hard-negative sampler, and a Java OWL API + HermiT/ELK coherence checker (`tools/reasoner`).
  - **Tests:** 19 test files with 231 test functions.
- **A byte-preserving corpus of all supplied files**, under `OA Finetune Small Model/datasets/Finetuning Dataset/` (41 GB):
  - `complete_corpus.sqlite`: 38.96M unique records;
  - task views: `mapping_prediction` (12,711 examples), `atomic_equivalence`, and `clinical_composition` (433K ICD-11→SNOMED paths).
- **Bio-ML 2026** downloaded and checksum-verified into `data/external/bio-ml`.
- **LaTeX reporting**: a 62-page snapshot paper of dataset statistics and experiment status.

## What was measured
- **A lexical q-gram baseline on the Anatomy test split:** P 1.0, R 0.159, F 0.274, Hits@1 0.774.
- **Tinker pilots were stopped after 9 and 2 updates**, with $2.11 reserved of a $20 cap. There are no neural evaluation results.

## Issues found
| Issue | Effect | How OA-RLAIF handles it |
|---|---|---|
| The parser dropped synonyms/definitions stored through `genid` IRIs (Anatomy) | Lost 5,264 human and 344 mouse synonyms and 840 definitions | Fixed in the vendored parser; regression test `tests/test_ontology.py` |
| `normalize()` kept underscores inside tokens | `Gray_Matter_of_Spinal_Cord` could not match `spinal cord grey matter` | Underscores and camelCase are now word breaks |
| Absolute home-directory provenance paths in SQLite views and docs | Views do not port to another machine | All provenance is relative to the raw folder; test `test_no_absolute_user_paths` |
| Test-heavy splits (1,426 / 149 / 11,136) | Unsuitable for training | Group-level 80/20 by connected components of gold links, plus zero-shot pairs |
| Anatomy `trivial.rdf` overlaps the reference | Leakage if used as an input | Used only as the lexical baseline and Recall+ set |
| `complete_tasks.sqlite` has an un-checkpointed 4.4 GB WAL | Unknown completeness | Not used; data is rebuilt from raw |
| `configs/data_splits.yaml` points `bioml_root` at the organiser repo (no TSVs) | Bio-ML silently missing | Bio-ML read from `data/external/bio-ml` and covered by the manifest |
| Tinker coupling (7 scripts, 6 tests, the base model hard-coded) | Cannot run elsewhere | TRL + PEFT + vLLM; no Tinker code vendored |
| One DH system output contains only "error" | Crashes naive parsing | Recorded as a failed-run baseline row |

## Reused (vendored into `RLAIF/src/oa_rlaif/`)
`models.py`, `ontology_parser.py` (fixed), `read_alignment` / `orient`, `AlignmentEvaluator`, `CandidateGenerator` (fixed normalisation), plus the design of `parse_mapping_tsv` and `fetch_bioml.py`. The Java reasoner's shaded jar is used to compile `tools/owl_verify` offline.

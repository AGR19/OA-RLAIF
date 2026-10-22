"""Hugging Face cards for the OA-RLAIF repos: what each repo is, how it was made, and real samples.

    python -m oa_rlaif.data.hf_cards                  # write the cards to outputs/hf_cards/<repo>/README.md
    python -m oa_rlaif.data.hf_cards --push [REPO..]  # upload them as README.md to the private repos

Naming (2026-10-22): **weights** = a LoRA adapter that needs the base model; **model** = a complete model that runs as
it is (safetensors or GGUF); **data** / **results** = datasets. The dataset cards show real rows, read from data/
when the card is built: no licensed text is stored in git, only in the private repos. The raw-data repo
(the raw ontology folders) is never changed by this tool.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..paths import data_dir, out_dir

BASE = "google/gemma-4-E4B-it"
CODE = "github.com/AGR19/OA-RLAIF (private)"
NEVER_CHANGED = {"oa-rlaif-raw-data"}  # the raw ontology folders: the user asked that nothing there is changed

# repo -> (kind, one line) for the table on every card
FAMILY = [
    ("oa-rlaif-e4b-sft-weights", "weights (LoRA)", f"Stage 1, SFT: adapter for `{BASE}`"),
    ("oa-rlaif-e4b-opsd-weights", "weights (LoRA)", f"Stage 1, OPSD: adapter for `{BASE}`"),
    ("oa-rlaif-e4b-grpo-weights", "weights (LoRA)", "Stage 2, GRPO: adapter for base + OPSD (merged)"),
    ("oa-rlaif-e4b-sft-model-safetensors", "model (safetensors, bf16)", "base + SFT"),
    ("oa-rlaif-e4b-opsd-grpo-model-gguf", "model (GGUF Q4_K_M)", "base + OPSD + GRPO"),
    ("oa-rlaif-e4b-all-combined-model-safetensors", "model (safetensors, bf16)", "base + OPSD → GRPO + SFT"),
    ("oa-rlaif-e4b-all-combined-model-gguf", "model (GGUF Q4_K_M)", "base + OPSD → GRPO + SFT, quantized"),
    ("oa-rlaif-datasource-qa-data", "data", "OM datasource passages → questions and answers"),
    ("oa-rlaif-opsd-sft-data", "data", "Stage 1 rows: the same rows train OPSD and SFT"),
    ("oa-rlaif-grpo-data", "data", "Stage 2 prompts with their episodes; evaluation sets; splits"),
    ("oa-rlaif-dral-reward-data", "data", "DRAL-OA vectors for the GRPO reward term r_DRAL"),
    ("oa-rlaif-icd-bridge-data", "data", "SNOMED CT → ICD-10 → ICD-11 composed links"),
    ("oa-rlaif-raw-data", "data", "the raw ontology folders (no PDFs), Bio-ML, parsed-ontology cache"),
    ("oa-rlaif-heldout-eval-results", "results", "the earlier evaluation (held-out decisions)"),
    ("oa-rlaif-full-eval-results", "results", "the full-alignment evaluation (full_eval/)"),
]
DATASETS = {name for name, kind, _ in FAMILY if kind in ("data", "results")}
LICENCE = ("**Licence and use.** Private research use only; do not make this repo public. The weights follow the "
           "Gemma terms of use. The training data contains or derives from licensed sources: the OM datasource "
           "(Springer © 2007), SNOMED CT (affiliate licence) and UMLS-derived Bio-ML references.")
LORA = ("LoRA r = 32, α = 64, dropout 0, on the language model's attention and MLP projections (q, k, v, o, gate, "
        "up, down); the vision and audio towers stay frozen and unused. bf16. Trained with thinking off (the chat "
        "template's `enable_thinking=False`).")


def family_table(this: str) -> str:
    rows = ["| Repo | Kind | What |", "|---|---|---|"]
    rows += [f"| {'**' + n + '**' if n == this else n} | {k} | {w} |" for n, k, w in FAMILY]
    return "\n".join(rows)


def front(meta: dict) -> str:
    lines = ["---"]
    for key, value in meta.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines += [f"- {v}" for v in value]
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines + ["---", ""])


def cut(text: str, n: int = 600) -> str:
    text = str(text)
    return text if len(text) <= n else text[:n].rstrip() + " …"


def block(text: str, n: int = 600) -> str:
    return "````text\n" + cut(text, n) + "\n````"


# ---- real rows from data/ ----------------------------------------------------------------------------------------------
def first_rows(path: Path, key, wanted: list[str] | None = None, limit: int = 2_000_000) -> dict:
    """The first row for each key value (all keys, or only `wanted`), streaming the JSONL file."""
    found: dict = {}
    if not path.exists():
        return found
    with path.open() as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            k = key(row)
            if k not in found and (wanted is None or k in wanted):
                found[k] = row
                if wanted is not None and len(found) == len(wanted):
                    break
            if i > limit:
                break
    return found


def count_rows(path: Path, key) -> dict:
    counts: dict = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                k = key(json.loads(line))
                counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def prompt_text(messages: list[dict], role: str) -> str:
    return "\n".join(m["content"] for m in messages if m["role"] == role)


def sft_target(row: dict) -> str:
    """What SFT trains on for a row (oa_rlaif.train.sdft.run_sft): the reference answer, else the example reply."""
    ctx = row["privileged_context"]
    if "Reference answer:" in ctx:
        return ctx.rsplit("Reference answer:", 1)[-1].strip()
    return ctx.split("Example of a correct reply:\n", 1)[-1]


# ---- shared sections -----------------------------------------------------------------------------------------------------
REWARD = """## The GRPO reward

For each prompt the policy samples G = 8 replies. Each reply gets

    R = 0.8 · R_task + 0.2 · J / 10

**Format gate.** A reply must be one `THOUGHT:` line and exactly one `om submit` command whose choice is a valid
candidate index (C1..Ck, at most 3) or NONE. Any other reply gets R = 0, and the judge is not called.

**R_task, the task reward**, depends on the kind of episode:

| Episode kind | R_task |
|---|---|
| gold (the reference has the answer) | 0.5 · F1(picks, gold) + 0.5 · r_DRAL |
| silver (ICD-11 links composed through WHO ICD-10, not gold) | the same, × 0.5 |
| unlabeled (no reference) | 0.3 · r_DRAL |

F1 compares the submitted targets with the gold targets: one correct pick = 1; extra or missing picks lower it
(recall counts at most 3 gold targets, the most one submission can name). NONE scores 1 only when no candidate is a
gold target. If r_DRAL cannot be computed (a class without a stored vector), R_task = F1.

**r_DRAL, the DRAL-OA reward (Eq. 1).** For the source class s and each candidate k of the episode:

    sim(s, k) = 0.5 · cos(z_s, z_k) + 0.5 · cos(t_s, t_k)
    p_k       = softmax over the episode's candidates of sim(s, k) / T,   T = 0.1
    r_DRAL(pick) = p_pick (mean over the picks)        r_DRAL(NONE) = 1 − max_k p_k

z is the DRAL-OA mapping-layer embedding of a class and t its label vector; both are stored in
`oa-rlaif-dral-reward-data`. The mapping layer was fit on the training references only. r_DRAL gives graded credit:
a wrong pick that DRAL-OA finds close to the source earns more than a far one.

**J, the judge.** `google/gemma-4-26B-A4B-it`, served locally next to the policy (thinking off), grades each valid
reply from 0 to 10 without seeing the answer: evidence 0–4 (lexical, structural, semantic evidence that supports the
choice), consistency 0–3 (the choice follows from the THOUGHT), faithfulness 0–2 (nothing invented), calibration 0–1
(the confidence fits the evidence). Rubric: `configs/judge_rubric.md` in the code repo.

**Update.** Dr. GRPO: advantage Â_i = R_i − mean(R) over the 8 replies of the prompt (no division by the standard
deviation), β = 0 (no KL term), truncated replies masked."""

STAGES = f"""## How the OA-RLAIF models are trained

| Stage | Starts from | Data | Loss |
|---|---|---|---|
| SFT (Stage 1) | `{BASE}` | `oa-rlaif-opsd-sft-data` | cross-entropy on the reference answer y* (off-policy) |
| OPSD (Stage 1) | `{BASE}` | `oa-rlaif-opsd-sft-data` (the same rows) | the student samples a reply y; a frozen copy of the base model, also shown the privileged context c (the gold answer and its evidence), scores every token; loss = Σ_t KL(student ‖ teacher) |
| GRPO (Stage 2) | base + OPSD, merged | `oa-rlaif-grpo-data`, `oa-rlaif-dral-reward-data` | group-relative policy gradient with the reward below |

All three train LoRA adapters only. {LORA}
**All combined** = base + OPSD → GRPO (merged), then the SFT adapter's weight change added on top (task arithmetic:
the SFT adapter was trained on the plain base)."""

TRAINING = """| Setting | SFT | OPSD | GRPO |
|---|---|---|---|
| Learning rate, schedule | 1e-4, cosine, 3% warm-up (as OPSD) | 1e-4, cosine, 3% warm-up | 5e-6, linear decay (trainer default), no warm-up |
| Batch per GPU × accumulation | 8 × 4 | 8 × 4 | 8 × 2, G = 8 replies per prompt |
| Completion length | 384 tokens | 384 tokens (one sample per prompt, temperature 1.0) | 256 tokens (temperature 1.0) |
| Other | completion-only loss | teacher = frozen base with the privileged context; top-100 logit distillation | Dr. GRPO, β = 0, rewards not scaled |
| Steps (from each adapter's `training_args.bin`) | 437 | 539 | 1,068 |

Hardware: 4 × B200 on Vast.ai (3 GPUs train; 1 GPU serves vLLM rollouts and, in GRPO, the A4B judge)."""

PROMPT = """## Prompt and reply format

The model decides one source class at a time from k numbered candidates (40 for Anatomy, 20 elsewhere), with both
ontologies' modelling profiles in the prompt:

```text
[system] You are OA-RLAIF, an ontology alignment agent. Decide which candidate target entity corresponds to the
source entity. … Reply with one line starting 'THOUGHT:' that names your evidence, then exactly one bash code block.
[user]   Task: find the target entity that the source entity is equivalent to.
         Alignment: anatomy/mouse-human
         ONTOLOGY PROFILES (modelling assumptions) …
         SOURCE
         IRI: http://mouse.owl#MA_0002211
         name: left renal vein
         parents: renal vein
         CANDIDATES
         [C1] … [C40] (each with name, synonyms, parents, children, definition)
```

An example reply (the format the mini-SWE-agent harness executes):

````text
THOUGHT: C7 has the same name and the parent renal vein matches.
```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && om submit C7 --relation = --confidence 0.95
```
````

Use thinking off (`enable_thinking=False`), as in training, and temperature 0."""

ANATOMY = """## Measured result: whole Anatomy alignment (mouse → human)

All combined as a mini-SWE-agent (the code repo's `om_agent` console; this model imported into Ollama at int8;
thinking on; 40 candidates per class): every one of the 2,737 mouse classes aligned, then scored against the OAEI
Anatomy reference with DRAL-OA's clf.py top-1 (only the classes that have a partner in the reference are scored; the
model's own choice is a hit when it is a partner).

*Scores against the reference*

| Scoring | Size | Precision | F-Measure | Recall | Recall+ |
|---|---:|---:|---:|---:|---:|
| clf.py top-1 (1,497 of 1,497 reference classes) | 1,496 | 0.941 | 0.937 | 0.934 | 0.863 |

The model was trained on part of this reference (the 80/20 split). The full-alignment evaluation of every model on
every dataset is in `oa-rlaif-full-eval-results`."""


def model_card(name: str) -> str:
    kinds = {
        "oa-rlaif-e4b-sft-weights": ("adapter", "peft", "Stage 1, SFT: a LoRA adapter for the base model."),
        "oa-rlaif-e4b-opsd-weights": ("adapter", "peft", "Stage 1, OPSD (on-policy self-distillation): a LoRA "
                                                         "adapter for the base model."),
        "oa-rlaif-e4b-grpo-weights": ("adapter", "peft", "Stage 2, GRPO with the DRAL-OA reward and the A4B judge "
                                                         "(RLAIF): a LoRA adapter for base + OPSD, merged."),
        "oa-rlaif-e4b-sft-model-safetensors": ("merge", "transformers", "The SFT model: the base model with the SFT "
                                                                       "adapter merged in. Safetensors, bf16."),
        "oa-rlaif-e4b-opsd-grpo-model-gguf": ("quantized", "gguf", "The OA-RLAIF model: base + OPSD + GRPO, merged and "
                                                                    "quantized to GGUF Q4_K_M (5.3 GB)."),
        "oa-rlaif-e4b-all-combined-model-safetensors": ("merge", "transformers", "All combined: base + OPSD → GRPO + "
                                                                                 "SFT, merged. Safetensors, bf16."),
        "oa-rlaif-e4b-all-combined-model-gguf": ("quantized", "gguf", "All combined (base + OPSD → GRPO + SFT), "
                                                                       "quantized to GGUF Q4_K_M."),
    }
    relation, library, line = kinds[name]
    meta = {"license": "other", "license_name": "gemma-and-private-data", "base_model": BASE,
            "base_model_relation": relation, "pipeline_tag": "text-generation", "language": ["en"],
            "tags": ["ontology-matching", "ontology-alignment", "oa-rlaif", "gemma4", "private"]}
    if library != "gguf":
        meta["library_name"] = library
    else:
        meta["tags"].append("gguf")
    kind = {"adapter": "**Weights** (a LoRA adapter): it needs the base model.",
            "merge": "**Model**: complete; load it like the base model.",
            "quantized": "**Model**: complete; runs as it is (llama.cpp, Ollama)."}[relation]
    use = {
        "oa-rlaif-e4b-sft-weights": f"""```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

base = AutoModelForImageTextToText.from_pretrained("{BASE}", dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "gonibeed/oa-rlaif-e4b-sft-weights")   # or .merge_and_unload()
tokenizer = AutoProcessor.from_pretrained("{BASE}").tokenizer
```
The merged result is `oa-rlaif-e4b-sft-model-safetensors`.""",
        "oa-rlaif-e4b-opsd-weights": f"""```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

base = AutoModelForImageTextToText.from_pretrained("{BASE}", dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "gonibeed/oa-rlaif-e4b-opsd-weights")
tokenizer = AutoProcessor.from_pretrained("{BASE}").tokenizer
```
GRPO continues from this model: merge it (`model.merge_and_unload()`) before applying `oa-rlaif-e4b-grpo-weights`.""",
        "oa-rlaif-e4b-grpo-weights": f"""This adapter was trained on base + OPSD, not on the plain base. Merge OPSD first:
```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

base = AutoModelForImageTextToText.from_pretrained("{BASE}", dtype=torch.bfloat16)
opsd = PeftModel.from_pretrained(base, "gonibeed/oa-rlaif-e4b-opsd-weights").merge_and_unload()
model = PeftModel.from_pretrained(opsd, "gonibeed/oa-rlaif-e4b-grpo-weights")
tokenizer = AutoProcessor.from_pretrained("{BASE}").tokenizer
```
(`adapter_config.json` names the local folder of the merged OPSD model it was trained on.) The merged and quantized
result is `oa-rlaif-e4b-opsd-grpo-model-gguf`.""",
        "oa-rlaif-e4b-sft-model-safetensors": """```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained("gonibeed/oa-rlaif-e4b-sft-model-safetensors", dtype=torch.bfloat16)
tokenizer = AutoProcessor.from_pretrained("gonibeed/oa-rlaif-e4b-sft-model-safetensors").tokenizer
```
vLLM: `vllm serve gonibeed/oa-rlaif-e4b-sft-model-safetensors`.""",
        "oa-rlaif-e4b-all-combined-model-safetensors": """```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

repo = "gonibeed/oa-rlaif-e4b-all-combined-model-safetensors"
model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16)
tokenizer = AutoProcessor.from_pretrained(repo).tokenizer
```
vLLM: `vllm serve gonibeed/oa-rlaif-e4b-all-combined-model-safetensors`. The local mini-SWE-agent console of the
code repo (`om_agent/`) runs this model with Ollama. `eval/` and `report/` hold its earlier evaluation.""",
        "oa-rlaif-e4b-opsd-grpo-model-gguf": """```bash
hf download gonibeed/oa-rlaif-e4b-opsd-grpo-model-gguf --local-dir oa-rlaif-e4b
cd oa-rlaif-e4b && ollama create oa-rlaif-e4b -f Modelfile && ollama run oa-rlaif-e4b
```
`Modelfile`: temperature 0, context 8,192, stop `<turn|>`.""",
        "oa-rlaif-e4b-all-combined-model-gguf": """```bash
hf download gonibeed/oa-rlaif-e4b-all-combined-model-gguf --local-dir oa-rlaif-all-combined
cd oa-rlaif-all-combined && ollama create oa-rlaif-all-combined -f Modelfile && ollama run oa-rlaif-all-combined
```
Converted from `oa-rlaif-e4b-all-combined-model-safetensors` on Hugging Face Jobs with llama.cpp b11206
(`convert_hf_to_gguf.py` to f16, then `llama-quantize` Q4_K_M). `Modelfile`: temperature 0, context 16,384, stop
`<turn|>`.""",
    }[name]
    grpo_inside = name in ("oa-rlaif-e4b-grpo-weights", "oa-rlaif-e4b-opsd-grpo-model-gguf",
                           "oa-rlaif-e4b-all-combined-model-safetensors", "oa-rlaif-e4b-all-combined-model-gguf")
    measured = name in ("oa-rlaif-e4b-all-combined-model-safetensors", "oa-rlaif-e4b-all-combined-model-gguf")
    parts = [front(meta), f"# {name}", "", f"{line} {kind}", "",
             "OA-RLAIF aligns ontologies: for each class of a source ontology it picks the corresponding class of a "
             "target ontology among retrieved candidates, or NONE. Base model: Gemma 4 E4B. Code: " + CODE + ".", "",
             "## The OA-RLAIF repos", "", family_table(name), "", "## How to use", "", use, "", STAGES, "",
             "### Training settings", "", TRAINING, "", PROMPT, ""]
    if grpo_inside:
        parts += [REWARD, ""]
    if measured:
        parts += [ANATOMY, ""]
    parts += ["## Limitations", "",
              "- Trained and evaluated with thinking off. With thinking on, the model often writes its answer inside "
              "the thinking text; the code repo's `oa_rlaif.eval.thinking.split` recovers it.",
              "- Decides among retrieved candidates: a correct target that retrieval misses cannot be chosen.",
              "- The image and audio parts of Gemma 4 are present in the weights but were neither trained nor used. The "
              "Hub may list the base architecture as image-text-to-text; this model is used for text generation only.",
              "", LICENCE, ""]
    return "\n".join(parts)


# ---- dataset cards ----------------------------------------------------------------------------------------------------
def opsd_sft_example(task: str, row: dict) -> str:
    system = prompt_text(row["prompt"], "system")
    user = prompt_text(row["prompt"], "user")
    return "\n".join([
        f"#### `{task}` (row `{row['id']}`, family `{row['family']}`)", "",
        "**1. Prompt x** (what the student sees; system prompt, then the user turn):", "",
        block(f"[system] {cut(system, 160)}\n[user] {user}", 900), "",
        "**2. Privileged context c** (only the OPSD teacher sees it):", "",
        block(row["privileged_context"], 700), "",
        "**3. How each stage uses the row:**", "",
        "- **OPSD**: the student generates its own reply to x; the frozen teacher reads x + c and gives its "
        "probability for every token of that reply; the student is trained to match the teacher token by token "
        "(KL divergence).",
        "- **SFT**: the student is trained directly on this target y* (cross-entropy on the completion only):", "",
        block(sft_target(row), 500), ""])


def card_opsd_sft() -> str:
    train = data_dir() / "sdft" / "train.jsonl"
    counts = count_rows(train, lambda r: r["task"])
    test = sum(1 for _ in (data_dir() / "sdft" / "test.jsonl").open()) if (data_dir() / "sdft" / "test.jsonl").exists() else None
    order = ["decision", "decision_agent", "fact_parents", "fact_synonyms", "fact_definition", "knowledge_qa",
             "logical_definition", "complex_edoal", "profile_qa"]
    def kind(r):  # the decision example: an Anatomy row whose answer is a real match, not NONE
        if r["task"] != "decision":
            return r["task"]
        match = r["family"] == "anatomy" and "the correct answer is NONE" not in r["privileged_context"]
        return "decision" if match else "decision (other)"

    examples = first_rows(train, kind, [t for t in order if t in counts])
    groups = {"decision": "alignment decision (C# or NONE among candidates)",
              "decision_agent": "the same decision as a mini-SWE-agent episode",
              "fact_": "ontology facts (parents, synonyms, definitions, codes, SNOMED CT attributes, maps, history)",
              "knowledge_qa": "OM datasource questions (oa-rlaif-datasource-qa-data)",
              "logical_definition": "OWL logical definitions", "complex_edoal": "complex (EDOAL) correspondences",
              "profile_qa": "ontology modelling profiles"}
    summary = {}
    for task, n in counts.items():
        g = next((k for k in groups if task == k or (k.endswith("_") and task.startswith(k))), "other")
        summary[g] = summary.get(g, 0) + n
    table = ["| Row type | Train rows |", "|---|---:|"] + [f"| {groups.get(g, g)} | {n:,} |" for g, n in summary.items()]
    meta = {"license": "other", "license_name": "private-licensed-sources", "pretty_name": "OA-RLAIF Stage 1 rows (OPSD and SFT)",
            "language": ["en", "multilingual"], "tags": ["ontology-matching", "oa-rlaif", "opsd", "sft", "private"],
            "configs": ["config_name: default\n  data_files:\n  - split: train\n    path: sdft/train.jsonl\n  - split: test\n    path: sdft/test.jsonl"]}
    return "\n".join([
        front(meta),
        "# oa-rlaif-opsd-sft-data", "",
        "The Stage 1 training rows. **The same rows train both Stage 1 arms**: SFT (`oa-rlaif-e4b-sft-weights`) and "
        "OPSD (`oa-rlaif-e4b-opsd-weights`). Code: " + CODE + " (`python -m oa_rlaif.data.build`).", "",
        "## The OA-RLAIF repos", "", family_table("oa-rlaif-opsd-sft-data"), "",
        "## Files", "",
        f"- `sdft/train.jsonl`: {sum(counts.values()):,} rows (the 80% training split); `sdft/test.jsonl`: "
        f"{test:,} rows (held out)." if test else "- `sdft/train.jsonl`, `sdft/test.jsonl`",
        "- `facts/`, `complex/`: the fact and complex-correspondence pools the rows were drawn from.",
        "- Row fields: `id`, `task`, `family` (the dataset or ontology), `prompt` (chat messages = x), "
        "`privileged_context` (c, for the OPSD teacher; it ends with the reference answer or an example reply).", "",
        "## What the rows contain", "", *table, "",
        "## One example per row type", "",
        "Real rows from `sdft/train.jsonl` (long fields shortened with …).", "",
        *[opsd_sft_example(t, examples[t]) for t in order if t in examples],
        STAGES, "", LICENCE, ""])


def reward_example() -> str:
    """A worked GRPO reward on a real Anatomy episode, computed with the training code's reward functions."""
    from ..reward.dral_reward import DralScorer
    from ..reward.task_reward import task_reward

    path = data_dir() / "grpo" / "train.jsonl"
    row = first_rows(path, lambda r: (r["family"], r["kind"]), [("anatomy", "gold")]).get(("anatomy", "gold"))
    if row is None:
        return ""
    episode = json.loads(row["episode"]) if isinstance(row["episode"], str) else row["episode"]
    scorer = DralScorer(sorted((data_dir() / "dral").glob("*.npz")), temperature=0.1)
    cands = [c["iri"] for c in episode["candidates"]]
    probs = scorer.probabilities(episode["source_iri"], cands)
    gold = list(episode.get("gold") or [])
    gold_i = cands.index(gold[0]) + 1 if gold and gold[0] in cands else None
    name = lambda iri: next((ln[6:] for c in episode["candidates"] if c["iri"] == iri  # noqa: E731
                             for ln in c["card"].splitlines() if ln.startswith("name: ")), iri)
    order = sorted(range(len(cands)), key=lambda i: -probs[i]) if probs is not None else list(range(len(cands)))
    wrong_i = next(i + 1 for i in order if cands[i] not in gold)

    def reply(choice):
        return (f"THOUGHT: example.\n```mswea_bash_command\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && "
                f"om submit {choice} --relation = --confidence 0.9\n```")

    rows = ["| Reply | F1 | r_DRAL | R_task | R if J = 8 | R if J = 2 |", "|---|---:|---:|---:|---:|---:|"]
    cases = [(f"C{gold_i} = gold “{name(gold[0])}”", f"C{gold_i}")] if gold_i else []
    cases += [(f"C{wrong_i} = closest wrong “{name(cands[wrong_i - 1])}”", f"C{wrong_i}"), ("NONE", "NONE"),
              ("C99 (no such candidate)", "C99")]
    for label, choice in cases:
        r = task_reward(reply(choice), episode, scorer)
        if not r["valid"]:
            rows.append(f"| {label} | – | – | 0 | 0 | 0 |")
            continue
        task = r["task"]
        rows.append(f"| {label} | {r['gold']:.2f} | {r['dral']:.3f} | {task:.3f} | {0.8 * task + 0.2 * 0.8:.3f} | "
                    f"{0.8 * task + 0.2 * 0.2:.3f} |")
    top = "\n".join(f"| C{i + 1} | {name(cands[i])} | {probs[i]:.3f} | {'gold' if cands[i] in gold else ''} |"
                    for i in order[:5]) if probs is not None else ""
    return "\n".join([
        "## Worked example: the reward of one real episode", "",
        f"Episode `{episode['id']}` ({episode['pair_id']}, kind gold): source `{episode['source_iri']}` "
        f"“{next((ln[6:] for ln in episode['source_card'].splitlines() if ln.startswith('name: ')), '')}”, "
        f"{len(cands)} candidates. The five candidates DRAL-OA scores highest (p_k = softmax of sim / 0.1 over all "
        f"{len(cands)}):", "", "| Candidate | Name | p_k | |", "|---|---|---:|---|", top, "",
        "Rewards computed with the training code (`oa_rlaif.reward.task_reward`, the stored DRAL-OA vectors). J is "
        "the judge's grade; the two J columns are examples, not measured grades:", "", *rows, ""])


def card_grpo() -> str:
    train = data_dir() / "grpo" / "train.jsonl"
    counts = count_rows(train, lambda r: (r["family"], r["kind"]))
    test = data_dir() / "grpo" / "test.jsonl"
    n_test = sum(1 for _ in test.open()) if test.exists() else None
    fam: dict = {}
    for (f, k), n in counts.items():
        fam.setdefault(f, {}).update({k: n})
    table = ["| Family | gold | silver | unlabeled |", "|---|---:|---:|---:|"]
    table += [f"| {f} | {v.get('gold', 0):,} | {v.get('silver', 0):,} | {v.get('unlabeled', 0):,} |" for f, v in fam.items()]
    row = first_rows(train, lambda r: (r["family"], r["kind"]), [("anatomy", "gold")]).get(("anatomy", "gold"))
    sample = ""
    if row:
        ep = json.loads(row["episode"]) if isinstance(row["episode"], str) else row["episode"]
        sample = "\n".join([
            "## One row", "", f"Row `{row['id']}` (family `{row['family']}`, kind `{row['kind']}`).", "",
            "**prompt** (what the policy sees; shortened):", "",
            block(f"[system] {cut(prompt_text(row['prompt'], 'system'), 200)}\n[user] {prompt_text(row['prompt'], 'user')}", 1100), "",
            "**episode** (kept by the reward function, never shown to the policy): the source, the candidates in the "
            "order shown (C1..Ck) and the gold answer:", "",
            block(json.dumps({"source_iri": ep["source_iri"], "gold": ep.get("gold"), "kind": ep.get("kind"),
                              "split": ep.get("split"),
                              "candidates": [c["iri"] for c in ep["candidates"][:6]] + ["…"]}, indent=1), 900), "",
            "The policy samples 8 replies to the prompt; each reply is scored with the reward below.", ""])
    meta = {"license": "other", "license_name": "private-licensed-sources", "pretty_name": "OA-RLAIF Stage 2 (GRPO) prompts",
            "language": ["en", "multilingual"], "tags": ["ontology-matching", "oa-rlaif", "grpo", "rlaif", "private"],
            "configs": ["config_name: default\n  data_files:\n  - split: train\n    path: grpo/train.jsonl\n  - split: test\n    path: grpo/test.jsonl"]}
    return "\n".join([
        front(meta), "# oa-rlaif-grpo-data", "",
        "Stage 2 (GRPO) prompts with the episode each reward is computed from, plus the evaluation sets, the episode "
        "pools and the 80/20 split. Trains `oa-rlaif-e4b-grpo-weights`. Code: " + CODE + ".", "",
        "## The OA-RLAIF repos", "", family_table("oa-rlaif-grpo-data"), "",
        "## Files", "",
        f"- `grpo/train.jsonl`: {sum(counts.values()):,} prompts" + (f"; `grpo/test.jsonl`: {n_test:,}." if n_test else "."),
        "  Fields: `id`, `family`, `kind` (gold / silver / unlabeled), `prompt` (chat messages), `episode` (JSON: "
        "source, candidates, gold; for the reward only).",
        "- `episodes/<family>.jsonl`: every decision episode per dataset family, with its split.",
        "- `eval/`: the earlier evaluation sets (held-out decisions per family, zero-shot pairs, closed-book "
        "questions), `trivial_anatomy.tsv`, `skip_list.tsv`.",
        "- `splits/components.tsv`: the group-level 80/20 split (a source entity is never in two splits).",
        "- `baselines/`: outputs of other matching systems, used as hard negatives, never as gold.",
        "- `raw/anatomy-dataset-2/`, `external/cyber/`: the ontologies the earlier full Anatomy alignment and the "
        "IoTSec → UCO pair read.", "",
        "## Prompts per family and kind (train)", "", *table, "",
        "gold = the reference has the answer; silver = an ICD-11 link composed through WHO ICD-10 "
        "(`oa-rlaif-icd-bridge-data`); unlabeled = no reference (only r_DRAL rewards it).", "",
        sample, REWARD, "", reward_example(), LICENCE, ""])


def card_dral() -> str:
    import numpy as np

    files = sorted((data_dir() / "dral").glob("*.npz"))
    rows = ["| File | Classes | z | t |", "|---|---:|---|---|"]
    for p in files:
        z = np.load(p)
        rows.append(f"| `dral/{p.name}` | {len(z['iris']):,} | {z['z'].shape[1]}-d | {z['t'].shape[1]}-d |")
    meta = {"license": "other", "license_name": "private-licensed-sources", "pretty_name": "OA-RLAIF DRAL-OA reward vectors",
            "tags": ["ontology-matching", "oa-rlaif", "grpo", "reward", "private"]}
    return "\n".join([
        front(meta), "# oa-rlaif-dral-reward-data", "",
        "The vectors behind r_DRAL, the DRAL-OA term of the GRPO reward. One file per ontology group, each with "
        "`iris` (class IRIs), `z` (the DRAL-OA mapping-layer embedding of each class) and `t` (its label vector). "
        "The mapping layer was fit on the training references only. Read by `oa_rlaif.reward.dral_reward.DralScorer` "
        "during GRPO. Code: " + CODE + ".", "",
        "## The OA-RLAIF repos", "", family_table("oa-rlaif-dral-reward-data"), "",
        "## Files", "", *rows, "",
        "```python\nimport numpy as np\nd = np.load(\"dral/anatomy_mouse-human.npz\")\nd[\"iris\"][:3], d[\"z\"].shape, d[\"t\"].shape\n```", "",
        REWARD, "", reward_example(), LICENCE, ""])


def card_datasource() -> str:
    k = data_dir() / "knowledge"
    q = first_rows(k / "questions.jsonl", lambda r: 0, [0]).get(0)
    p = first_rows(k / "passages.jsonl", lambda r: 0, [0]).get(0)
    n_q = sum(1 for _ in (k / "questions.jsonl").open()) if (k / "questions.jsonl").exists() else 0
    n_p = sum(1 for _ in (k / "passages.jsonl").open()) if (k / "passages.jsonl").exists() else 0
    row = first_rows(data_dir() / "sdft" / "train.jsonl", lambda r: r["task"], ["knowledge_qa"]).get("knowledge_qa")
    meta = {"license": "other", "license_name": "private-licensed-sources", "pretty_name": "OA-RLAIF datasource questions",
            "language": ["en"], "tags": ["ontology-matching", "oa-rlaif", "question-answering", "private"]}
    parts = [front(meta), "# oa-rlaif-datasource-qa-data", "",
             "The OM datasource as training material: passages from the ontology-matching textbook, 18 survey papers "
             "and the benchmark documentation, and the questions and answers written from them by "
             "`gemma-4-26b-a4b-it` (Google AI Studio). The questions become the `knowledge_qa` rows of "
             "`oa-rlaif-opsd-sft-data` (Stage 1, so the knowledge goes into the weights) and the closed-book test in "
             "`oa-rlaif-grpo-data/eval/`. Code: " + CODE + ".", "",
             "## The OA-RLAIF repos", "", family_table("oa-rlaif-datasource-qa-data"), "",
             "## Files", "",
             f"- `knowledge/passages.jsonl`: {n_p:,} passages; `knowledge/questions.jsonl`: {n_q:,} questions with "
             "answers; `knowledge/question_coverage.json`: questions per source; `knowledge/generated/`: the "
             "generator's raw outputs.", "",
             "## From passage to training row", ""]
    if p:
        parts += ["**1. A passage** (`knowledge/passages.jsonl`, shortened):", "", block(json.dumps(p, ensure_ascii=False, indent=1), 700), ""]
    if q:
        parts += ["**2. A question written from a passage** (`knowledge/questions.jsonl`):", "", block(json.dumps(q, ensure_ascii=False, indent=1), 700), ""]
    if row:
        parts += ["**3. The Stage 1 row it becomes** (`oa-rlaif-opsd-sft-data`, task `knowledge_qa`): the prompt has "
                  "the question only; the passage and the answer are the privileged context, which only the OPSD "
                  "teacher sees; SFT trains on the reference answer.", "", opsd_sft_example("knowledge_qa", row)]
    return "\n".join(parts + [LICENCE, ""])


def card_icd() -> str:
    path = data_dir() / "icd" / "triangulated.jsonl"
    row = first_rows(path, lambda r: 0, [0]).get(0)
    stats_path = data_dir() / "icd" / "triangulation_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    meta = {"license": "other", "license_name": "snomed-affiliate-and-who", "pretty_name": "OA-RLAIF ICD bridge links",
            "tags": ["ontology-matching", "oa-rlaif", "icd-11", "snomed-ct", "private"]}
    s = stats
    return "\n".join([
        front(meta), "# oa-rlaif-icd-bridge-data", "",
        "SNOMED CT → ICD-11 links composed through WHO ICD-10: SNOMED CT → ICD-10 (the SNOMED CT map), then ICD-10 → "
        "ICD-11 (the WHO 10To11 tables). No SNOMED CT → ICD-11 reference exists; these links are a silver standard. "
        "They give the 2,400 silver GRPO prompts (`oa-rlaif-grpo-data`, family icd) and the agreement score of the "
        "complete SNOMED CT → ICD-11 map in the full evaluation. Code: " + CODE + ".", "",
        "## The OA-RLAIF repos", "", family_table("oa-rlaif-icd-bridge-data"), "",
        "## Files", "",
        f"- `icd/triangulated.jsonl`: {s.get('links', 0):,} links ({s.get('snomed_concepts', 0):,} SNOMED CT concepts → "
        f"{s.get('icd11_targets', 0):,} ICD-11 categories; {s.get('strong_both', 0):,} with both legs strong; "
        f"{s.get('test_links', 0):,} in the test split).",
        "- `icd/icd11_snomed.sssom.tsv`: the same links as SSSOM; `icd/triangulation_stats.json`, `icd/summary.json`; "
        "`icd/xlsx_crosscheck.json`: the WHO tables' TXT and XLSX versions agree.", "",
        "## One link", "", block(json.dumps(row, ensure_ascii=False, indent=1), 1200) if row else "", "",
        "A link is **strong** when the SNOMED CT → ICD-10 leg and the ICD-10 → ICD-11 leg are both 1:1 or "
        "classification-exact; relation `classified_under` means the ICD-11 category classifies the concept (not "
        "clinical equivalence).", "", LICENCE, ""])


def card_heldout_results() -> str:
    meta = {"license": "other", "license_name": "private-licensed-sources", "pretty_name": "OA-RLAIF earlier evaluation",
            "tags": ["ontology-matching", "oa-rlaif", "evaluation", "private"]}
    return "\n".join([
        front(meta), "# oa-rlaif-heldout-eval-results", "",
        "The earlier evaluation of base, SFT, OPSD and OPSD → GRPO (2026-10-12/13): held-out decisions (the 20% "
        "test split of every family), zero-shot pairs, ICD-10 → ICD-11, SNOMED CT → ICD-11 bridge recovery, "
        "closed-book knowledge, mini-SWE-agent runs and the whole Anatomy alignment. Start with `report.md`. The "
        "full-alignment evaluation (every class, whole references, no held-out view) is `oa-rlaif-full-eval-results`. "
        "Code: " + CODE + ".", "",
        "## The OA-RLAIF repos", "", family_table("oa-rlaif-heldout-eval-results"), "",
        "## Files", "",
        "- `report.md` / `.tex` / `.json`: the tables.",
        "- `<model>/` for base, sft, sdft (= OPSD), grpo: `metrics.json` and every prediction "
        "(`heldout_<family>.predictions.jsonl`, `zero_shot`, `closed_book_*`, `icd11_snomed_predicted.sssom.tsv`).",
        "- `agent_<model>.json`: the mini-SWE-agent runs; `full_align_anatomy/`: the whole Anatomy alignment "
        "(alignment.rdf, SSSOM, summary).", "", LICENCE, ""])


def card(name: str) -> str:
    if name.startswith("oa-rlaif-e4b-"):
        return model_card(name)
    return {"oa-rlaif-opsd-sft-data": card_opsd_sft, "oa-rlaif-grpo-data": card_grpo,
            "oa-rlaif-dral-reward-data": card_dral, "oa-rlaif-datasource-qa-data": card_datasource,
            "oa-rlaif-icd-bridge-data": card_icd, "oa-rlaif-heldout-eval-results": card_heldout_results}[name]()


CARD_REPOS = [n for n, _, _ in FAMILY if n not in NEVER_CHANGED and n != "oa-rlaif-full-eval-results"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oa_rlaif.data.hf_cards")
    parser.add_argument("repos", nargs="*", help="default: every repo this tool writes")
    parser.add_argument("--push", action="store_true", help="upload README.md to the private repos")
    args = parser.parse_args(argv)
    repos = args.repos or CARD_REPOS
    blocked = [r for r in repos if r in NEVER_CHANGED or r not in CARD_REPOS]
    if blocked:
        raise SystemExit(f"not written by this tool: {blocked}")
    out = out_dir() / "hf_cards"
    texts = {}
    for name in repos:
        texts[name] = card(name)
        path = out / name / "README.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(texts[name])
        print(f"[cards] {path} ({len(texts[name]):,} characters)")
    if not args.push:
        return 0
    from huggingface_hub import HfApi

    from .hf_push import namespace

    api, ns = HfApi(), namespace()
    for name, text in texts.items():
        kind = "dataset" if name in DATASETS else "model"
        repo = f"{ns}/{name}"
        if not api.repo_info(repo, repo_type=kind).private:
            raise SystemExit(f"{repo} is not private; refusing to upload")
        api.upload_file(path_or_fileobj=text.encode(), path_in_repo="README.md", repo_id=repo, repo_type=kind,
                        commit_message="Model/dataset card: naming, how it was made, examples")
        print(f"[cards] pushed {repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

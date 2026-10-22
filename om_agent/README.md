# om_agent: the OA-RLAIF mini-SWE-agent harness for ontology matching

The trained model as a mini-SWE-agent (2.4.6) that aligns **every** class of a source ontology to a target ontology,
one agent episode per class. The same code runs in two places:

- **Locally** (Mac, Ollama): the console `oa_agent.sh` and the batch run `run_matching.sh`, with the all-merged model
  (base Gemma 4 E4B + OPSD + GRPO + SFT) served by this folder's own Ollama on port 11435.
- **In the full evaluation** (GPU box, vLLM): `full_eval/agent_run.py` imports `Matcher` from `test_matching_ai.py` and
  runs it as the method "All merged + mini-SWE-agent", with thinking off and on (see `full_eval/README.md`).

```
./oa_agent.sh                  # the console (below)
./run_matching.sh              # every class in batch, no questions (writes output/mouse-human)
./om-score output/s2           # the scoring skill
```

## One episode

1. Scripted exploration, run for real in the task folder (the steps that come before the decision in the trained
   agent rows): `om profile`, `om show <class>`, `om candidates <class> --k N`.
2. The model takes over with the trained system prompt (`harness.prompts.AGENT_SYSTEM`). It may run more `om`
   commands and ends with `om submit C#[,C#] | NONE --relation .. --confidence ..`. Step limit 6
   (`configs/mini_swe_agent.yaml`).
3. The task folder holds only the public episode (`harness.sandbox.public_episode`): never the reference.

Where the commands run (`--sandbox`): `local` (this machine, in the class's task folder; the console's default),
`docker` (the Docker sandbox, no network) or `user` (the unprivileged sandbox user of `scripts/sandbox_setup.sh`, in a
task folder outside the repo; used on Vast.ai container instances, which cannot run Docker).

**Thinking** is on by default here (`/thinking off` or `--no-think` turns it off); the full evaluation runs both. The
model was trained and evaluated with thinking off. With thinking on it often writes its usual answer inside the
thinking channel; the answer is then taken from there (`oa_rlaif.eval.thinking.split`, shared with the full
evaluation) and the console says so.

## The console

`./oa_agent.sh` shows the model (name, where it is loaded from, format, thinking) and the agent's skills. It then asks
for the source and target ontology (type a path or drag the file into the window; a file from elsewhere is copied into
`ontologies/`), a **reference alignment (optional)** (Enter = skip) and the output folder (default
`output/<source>-<target>`). Nothing runs until you say so: it asks before it starts, and the agent asks before every
command (answer 2 once to allow the rest of the session; 3 stops). Ctrl-C stops after the current class; running
again in the same output folder continues where it stopped. The console shows an estimate of the time left from the
speed measured on this machine.

The reference is used only for scoring; the agent never sees it.

## Scoring: the Top-k evaluation (k = 1, 3, 5)

Precision@k, Recall@k and F-measure@k, as in DRAL-OA's `clf.py` (`precision_at_k` / `recall_at_k`). Only the source
classes that have a partner in the reference are scored. Each class's ranked answers are the model's own choice first,
then its other candidates in retrieval order; NONE = nothing found. Per class c, with F_c its top k and R_c its
partners:

    Precision@k = |F_c ∩ R_c| / min(k, |R_c|)        Recall@k = |F_c ∩ R_c| / |R_c|

averaged over the scored classes; F-measure@k = 2PR / (P + R). With a reference, `results.md` holds the Top-1, Top-3
and Top-5 rows and `topk.svg` the chart (Precision, Recall and F-measure vs. Top-k).

**The `om-score` skill** scores any run folders, side by side, and never writes into them (output in `scores/`):

```
./om-score output/s2
./om-score output/s1 output/s2 --reference ontologies/reference.rdf
```

In the console, `/score` runs it for the current run, `/score output/s1 output/s2` for others.

## Outputs

`output/<source>-<target>/`: `alignment.rdf` (matches with confidence >= 0.5, 1:1, MELT/OAEI format),
`decisions.tsv` (every decided class with labels, for checking by hand), `decisions.jsonl` (every decision with the
agent's commands, model calls, tokens and thinking), `summary.json`, `run.json`, `results.md` and `topk.svg` (with a
reference) and `tasks/` (each class's full trajectory).

"Every class" = the source ontology's own classes: for mouse.owl, the 2,737 anatomy classes (the file also declares
owl:Thing, 8 oboInOwl bookkeeping classes and the UNDEFINED_is_a/part_of placeholders; those are not matched).

## Files

| File | What |
|---|---|
| `test_matching_ai.py` | the matcher (DRAL-OA's MELT matcher with the FastText/GNN/RL part replaced by our model); `Matcher` is also the full evaluation's agent |
| `om_score.py`, `om-score` | the scoring skill: Top-k evaluation (k = 1, 3, 5), table and chart |
| `oa_agent.py`, `oa_agent.sh` | the console: banner (model, server, weights, format, agent), skills, live transcript per class, slash commands |
| `run_matching.sh` | batch run of the matcher (no console) |
| `setup_ollama.sh` | used by both: starts this folder's Ollama (port 11435) and imports the model once |
| `Modelfile` | Ollama import (int8, temperature 0, 16,384-token window, stop `<turn|>`) from `cache/import` |
| `fix_kv_shared.py` | builds `cache/import` = `model/all_merged` (unchanged, linked) + the 54 tensors of the KV-shared layers 24-41 and `global_head_dim`, which Ollama requires but transformers 5.x does not save; the tensors are never used (zeros give identical output) |

Runtime folders (not in git): `model/all_merged/` (the weights, from the private HF repo `oa-rlaif-e4b-all-combined-model-safetensors`),
`ollama-models/`, `ontologies/`, `output/`, `scores/`, `logs/`, `cache/`.

The scripts find the OA-RLAIF repo as this folder's parent; a copy outside the repo uses `OA_RLAIF_DIR` (default
`~/Desktop/oa-sft-sm-1/RLAIF`) and that repo's `.venv`.

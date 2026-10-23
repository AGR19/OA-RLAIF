#!/usr/bin/env python3
"""test_matching_ai.py: align two ontologies with OUR model as a mini-SWE-agent, one agent episode per class.

Based on DRAL-OA's MELT matcher (DRAL_OA/.../oaei-resources/pythonMatcher.py), with DRAL-OA's FastText + GNN +
reinforcement-learning classifier replaced by the OA-RLAIF model ("all merged": base Gemma 4 E4B + OPSD + GRPO + SFT)
served locally by Ollama. EVERY declared class of the source ontology is handled; none is skipped or sampled:

  1. both ontologies are parsed (labels, synonyms, parents, children, definitions) and profiled
  2. candidates: the top-k target classes of the same kind for each source class (q-gram index; k=40 as in training)
  3. one mini-SWE-agent episode per source class, run locally on this machine by default (mini-SWE-agent's local
     environment: `om` and the shell run directly here, in that class's task folder; `--sandbox docker` runs the
     same episode in the Docker sandbox instead, with no network). The agent first runs `om profile`, `om show <class>` and `om candidates
     <class>` (the steps that come before the decision in the trained agent rows); then the model takes over: it
     may run more `om` commands and ends with `om submit C#[,C#] | NONE --relation .. --confidence ..`
  4. every decision is saved the moment it is made (decisions.jsonl); running again continues where it stopped
  5. matches at or above --threshold, greedy 1:1 by confidence -> alignment.rdf (MELT/OAEI format)

Easiest: ./run_matching.sh in this folder (starts this folder's Ollama, imports the model once, runs every class).
Directly, with the RLAIF venv (it has mini-SWE-agent, rdflib and the OA-RLAIF code) and Ollama serving the model:

  ~/Desktop/oa-sft-sm-1/RLAIF/.venv/bin/python "test_matching_ai.py" \\
      file:///path/mouse.owl file:///path/human.owl [file:///path/reference.rdf] [--workers 2] [--limit 20]

Same interface as the MELT external Python matcher: source URL, target URL, optional input alignment; prints the file
URL of the alignment. With an input alignment (optional) it also scores the result with the Top-k evaluation
(k = 1, 3, 5): Precision@k, Recall@k and F-measure@k as in DRAL-OA's clf.py (precision_at_k / recall_at_k: only the
source classes that have a partner in the reference; the model's own choice first, then its other candidates in
retrieval order). Scoring only: unlike DRAL-OA, nothing is trained on it, and the agent never sees it. Output folder (--out): alignment.rdf, decisions.jsonl (every class and its
decision), decisions.tsv (the same with labels, for checking by hand), summary.json, results.md (with an input
alignment) and tasks/ (each episode's folder and trajectory, including the model's thinking).

Thinking is ON by default (--no-think turns it off). The model was trained and evaluated with thinking off.

The same agent runs the full evaluation on the GPU box (full_eval/agent_run.py): --backend vllm talks to a local
`vllm serve` (thinking switched with chat_template_kwargs, the thinking text split off by oa_rlaif.eval.thinking), and
--sandbox user runs every command as the unprivileged sandbox user (harness.sandbox; machines without Docker).
This folder is om_agent/ in the OA-RLAIF repo; a copy outside the repo finds the repo through OA_RLAIF_DIR.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.sax.saxutils import escape, quoteattr

HERE = Path(__file__).resolve().parent
# The OA-RLAIF repo: this folder's parent when it is om_agent/ in the repo, else OA_RLAIF_DIR, else the Mac copy.
RLAIF = Path(os.environ.get("OA_RLAIF_DIR") or (HERE.parent if (HERE.parent / "src" / "oa_rlaif").is_dir()
                                                 else Path.home() / "Desktop" / "oa-sft-sm-1" / "RLAIF"))
sys.path[:0] = [str(RLAIF / "src"), str(RLAIF / "third_party" / "mini-swe-agent" / "src")]
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def own_folders() -> None:
    """The console, the matcher and om-score keep everything they write in this folder: outputs, task folders,
    caches, mini-SWE-agent's config. Called by their entry points before any OA-RLAIF or mini-SWE-agent code is
    imported (not on import: full_eval imports the Matcher in its own process). The OA-RLAIF code is only read."""
    os.environ.setdefault("OA_OUT_DIR", str(HERE / "output"))          # progress bookkeeping
    os.environ.setdefault("OA_DATA_DIR", str(HERE / "cache" / "data"))  # ontology-profile cache
    os.environ.setdefault("MSWEA_GLOBAL_CONFIG_DIR", str(HERE / "cache" / "mini-swe-agent"))

DEFAULT_MODEL = "oa-rlaif-all-merged"
# this folder's own Ollama server (run_matching.sh starts it on port 11435 with its model store in ollama-models/)
OLLAMA = os.environ.get("OLLAMA_API_BASE", "http://127.0.0.1:11435/v1")


# ---- ontologies and episodes ----------------------------------------------------------------------------------------
def path_of(location: str) -> Path:
    return Path(unquote(urlparse(location).path)) if location.startswith("file:") else Path(location)


def own_classes(onto) -> set[str]:
    """The ontology's own classes: declared, in its main namespace, not an OBO relation placeholder (UNDEFINED_part_of).
    Leaves out owl:Thing and the oboInOwl bookkeeping classes (Synonym, DbXref, ...) that the OBO-to-OWL files declare."""
    from collections import Counter

    declared = [i for i, e in onto.entities.items() if e.declared]
    space = Counter(i.rsplit("#", 1)[0] if "#" in i else i.rsplit("/", 1)[0] for i in declared).most_common(1)[0][0]
    return {i for i in declared if i.startswith(space) and not i.rsplit("#", 1)[-1].startswith("UNDEFINED_")}


def build_episodes(source_path: Path, target_path: Path, k: int, pair_id: str) -> list[dict]:
    """One episode per source class (own_classes): its card, both ontology profiles and its k candidates."""
    from oa_rlaif.data.cards import card, children_index, kind_of
    from oa_rlaif.data.ontology_profiles import summary_text
    from oa_rlaif.eval.align import build_profile_from
    from oa_rlaif.ontology.parser import OntologyParser
    from oa_rlaif.retrieval.candidates import CandidateGenerator

    source, target = OntologyParser().parse(source_path), OntologyParser().parse(target_path)
    profile_s, profile_t = summary_text(build_profile_from(source_path)), summary_text(build_profile_from(target_path))
    generator = CandidateGenerator(target)
    kids_s, kids_t = children_index(source), children_index(target)
    episodes = []
    targets = own_classes(target)
    for iri in sorted(own_classes(source)):
        kind = kind_of(source, iri)
        ranked = [c for c, _ in generator.candidates(source.entities[iri], k=k * 4)
                  if c in targets and kind_of(target, c) == kind][:k]
        episodes.append({"id": iri, "pair_id": pair_id, "relation": "=", "source_iri": iri,
                         "source_card": card(source, iri, children=kids_s),
                         "source_profile": profile_s, "target_profile": profile_t,
                         "candidates": [{"iri": c, "card": card(target, c, children=kids_t)} for c in ranked]})
    return episodes, source, target


# ---- the agent --------------------------------------------------------------------------------------------------------
def make_agent_class():
    from minisweagent.agents.default import DefaultAgent

    class SeededAgent(DefaultAgent):
        """mini-SWE-agent whose first steps are the exploration of the trained agent rows (om profile, om show,
        om candidates), run for real in the sandbox; every later step is the model's own."""

        def __init__(self, *args, scripted: list[tuple[str, str]] = (), on_message=None, approve=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._scripted = list(scripted)
            self._on_message = on_message  # live view (oa_agent.py): every message as it is added
            self._approve = approve  # oa_agent.py: asks the user before each command runs; False stops the episode

        def execute_actions(self, message: dict) -> list[dict]:
            from minisweagent.exceptions import UserInterruption

            for action in message.get("extra", {}).get("actions", []):
                if self._approve and not self._approve(action["command"]):
                    raise UserInterruption({"role": "exit", "content": "The user declined the command.",
                                            "extra": {"exit_status": "Declined", "submission": ""}})
            return super().execute_actions(message)

        def add_messages(self, *messages: dict) -> list[dict]:
            added = super().add_messages(*messages)
            if self._on_message:
                for message in messages:
                    self._on_message(message)
            return added

        def step(self) -> list[dict]:
            if self._scripted:
                thought, command = self._scripted.pop(0)
                message = self.model.format_message(
                    role="assistant", content=f"THOUGHT: {thought}\n```mswea_bash_command\n{command}\n```")
                message.setdefault("extra", {})["actions"] = [{"command": command}]
                message["extra"]["scripted"] = True  # the exploration of the trained agent rows, not a model call
                self.add_messages(message)
                return self.execute_actions(message)
            return super().step()

    return SeededAgent


def make_model_class():
    from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

    from oa_rlaif.eval.thinking import split

    class Model(LitellmTextbasedModel):
        """Earlier turns go back to the model without their thinking, as Gemma 4's chat template does. Each call's
        thinking, tokens and answer are kept in self.calls (one Model per episode)."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.calls: list[dict] = []

        def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
            return [{k: v for k, v in m.items() if k not in ("reasoning_content", "reasoning", "thinking_blocks")}
                    for m in super()._prepare_messages_for_api(messages)]

        def _parse_actions(self, response) -> list[dict]:
            # Thinking text arrives apart from the answer (Ollama: the reasoning field) or inside it between Gemma 4's
            # channel markers (vLLM with special tokens kept). Trained with thinking off, the model, when thinking is
            # on, often writes its whole usual answer (THOUGHT + command) inside the thinking channel and never closes
            # it; the answer is then taken from there (oa_rlaif.eval.thinking.split; the console says so).
            message = response.choices[0].message
            parts = split(message.content or "", getattr(message, "reasoning_content", None) or "")
            message.content = parts["answer"]
            if parts["thinking"]:
                message.reasoning_content = parts["thinking"]
            usage = getattr(response, "usage", None)
            self.calls.append({"answer": parts["answer"], "thinking": parts["thinking"],
                               "answer_in_thinking": parts["answer_in_thinking"],
                               "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                               "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                               "finish_reason": response.choices[0].finish_reason})
            return super()._parse_actions(response)

    return Model


class Matcher:
    """backend "ollama": this folder's Ollama (the console); "vllm": a local `vllm serve` (full_eval/agent_run.py).
    sandbox "local": the agent's commands run here, in the class's task folder; "docker": in the Docker sandbox, no
    network; "user": as the unprivileged sandbox user in a task folder outside the repo (harness.sandbox)."""

    def __init__(self, model_name: str, out: Path, workers: int, sandbox: str = "local", think: bool = True,
                 backend: str = "ollama", api_base: str | None = None, save_trajectory: bool = True,
                 keep_task_dirs: bool = True):
        from oa_rlaif.eval.thinking import settings
        from oa_rlaif.harness.prompts import AGENT_SYSTEM
        from oa_rlaif.io import read_yaml

        self.cfg = read_yaml(RLAIF / "configs" / "mini_swe_agent.yaml")
        model_cfg = dict(self.cfg["model"])
        model_cfg.pop("model_class", None)
        # Thinking ON (default here): Gemma 4 reasons first; the reasoning counts against max_tokens (512 in the
        # config, enough only without thinking). Thinking OFF reproduces the training prompt exactly (checked by token
        # counts with the real tokenizer). Ollama switches it with reasoning_effort, vLLM with the chat template.
        extra_body = {"reasoning_effort": "high" if think else "none"} if backend == "ollama" else settings(think)
        kwargs = {**model_cfg.get("model_kwargs", {}), "api_base": api_base or OLLAMA,
                  "api_key": "ollama" if backend == "ollama" else "EMPTY", "extra_body": extra_body}
        if think:
            kwargs["max_tokens"] = max(kwargs.get("max_tokens", 0), 4096)
        model_cfg.update(model_name=f"openai/{model_name}", model_kwargs=kwargs)
        self.model_cfg, self.Model, self.think = model_cfg, make_model_class(), think
        self.agent_cfg = {**self.cfg["agent"], "system_template": AGENT_SYSTEM}
        self.Agent = make_agent_class()
        self.out, self.workers, self.sandbox = out, workers, sandbox
        self.save_trajectory, self.keep_task_dirs = save_trajectory, keep_task_dirs
        self.lock = threading.Lock()

    def task_dir(self, episode: dict) -> Path:
        """The class's task folder with the public episode only (the agent can read every file in it)."""
        from oa_rlaif.harness.sandbox import SANDBOX_USER, prepare_task_dir, public_episode

        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", episode["id"])[-120:]
        if self.sandbox == "user":
            import pwd

            task_dir = prepare_task_dir(self.out.name, name)
            (task_dir / "episode.json").write_text(json.dumps(public_episode(episode)))
            entry = pwd.getpwnam(SANDBOX_USER)
            os.chown(task_dir / "episode.json", entry.pw_uid, entry.pw_gid)
            return task_dir
        task_dir = self.out / "tasks" / name
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "episode.json").write_text(json.dumps(public_episode(episode)))
        return task_dir

    def run_episode(self, episode: dict, on_message=None, approve=None, model_name: str | None = None,
                    model_kwargs: dict | None = None, note: str = "") -> dict:
        """One class. model_name / model_kwargs override this Matcher's model for this call (dm_harness: the
        JevAgentHarness model names and a per-class session id); note is added to the task text (dm_harness: the
        decision model's hint when a step is redone)."""
        import shutil

        from oa_rlaif.eval.agent_eval import make_environment, parse_submission
        from oa_rlaif.harness.prompts import agent_task

        started = time.time()
        task_dir = self.task_dir(episode)
        src, k = episode["source_iri"], len(episode["candidates"])
        scripted = [("check the modelling assumptions of both ontologies first.", "om profile"),
                    ("inspect the source entity.", f"om show {src}"),
                    ("retrieve candidate targets.", f"om candidates {src} --k {k}")]
        env = make_environment(self.sandbox, task_dir, self.cfg)
        model_cfg = self.model_cfg
        if model_name or model_kwargs:
            model_cfg = {**model_cfg, "model_kwargs": {**model_cfg["model_kwargs"], **(model_kwargs or {})}}
            if model_name:
                model_cfg["model_name"] = f"openai/{model_name}"
        model = self.Model(**model_cfg)
        agent = self.Agent(model, env, scripted=scripted, on_message=on_message, approve=approve, **self.agent_cfg)
        try:
            outcome = agent.run(task=agent_task(episode) + (f"\n\n{note}" if note else ""))
        except Exception as error:  # noqa: BLE001 (one class failing must not stop the others)
            outcome = {"exit_status": f"error: {type(error).__name__}: {error}"[:300], "submission": ""}
        finally:
            getattr(env, "cleanup", lambda: None)()
        if self.save_trajectory:
            folder = self.out / "tasks" / task_dir.name
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "trajectory.json").write_text(json.dumps(agent.messages, default=str)[:2_000_000])
        if not self.keep_task_dirs:
            shutil.rmtree(task_dir, ignore_errors=True)
        decision = parse_submission(outcome.get("submission", ""))
        thought = next((m["content"].split("\n", 1)[0] for m in reversed(agent.messages)
                        if m.get("role") == "assistant"), "")
        targets = decision["targets"] if decision else []
        status = outcome.get("exit_status")
        commands = [a["command"] for m in agent.messages if m.get("role") == "assistant"
                    and not m.get("extra", {}).get("scripted") for a in m.get("extra", {}).get("actions", [])]
        return {"source": src, "decision": ("NONE" if targets == ["NONE"] else "match") if decision
                else "declined" if status == "Declined" else "no submission",
                "targets": [] if targets == ["NONE"] else targets,
                "relation": decision["relation"] if decision else None,
                "confidence": decision["confidence"] if decision else None,
                "exit": status, "model_calls": agent.n_calls, "thinking": self.think, "thought": thought[:300],
                # what the agent did beyond the scripted exploration, and what it cost
                "commands": commands, "extra_commands": sum("om submit" not in c for c in commands),
                "format_errors": sum(m.get("extra", {}).get("interrupt_type") == "FormatError" for m in agent.messages),
                "prompt_tokens": sum(c["prompt_tokens"] for c in model.calls),
                "completion_tokens": sum(c["completion_tokens"] for c in model.calls),
                "thinking_chars": sum(len(c["thinking"]) for c in model.calls),
                "answer_in_thinking": sum(c["answer_in_thinking"] for c in model.calls),
                "turns": [{"answer": c["answer"], "thinking": c["thinking"], "finish_reason": c["finish_reason"]}
                          for c in model.calls],
                "seconds": round(time.time() - started, 2)}

    def run(self, episodes: list[dict], retries: int) -> list[dict]:
        log_path = self.out / "decisions.jsonl"
        done = {}
        if log_path.exists():  # resume: classes already decided are not asked again
            for line in log_path.read_text().splitlines():
                record = json.loads(line)
                if record["decision"] in ("match", "NONE"):
                    done[record["source"]] = record
        todo = [e for e in episodes if e["source_iri"] not in done]
        logging.info("%d classes: %d already decided, %d to do (workers %d)", len(episodes), len(done), len(todo), self.workers)
        started, finished = time.time(), 0
        with ThreadPoolExecutor(self.workers) as pool, log_path.open("a") as log:
            futures = {pool.submit(self.run_episode, e): e for e in todo}
            for future in as_completed(futures):
                record = future.result()
                for _ in range(retries):
                    if record["decision"] in ("match", "NONE"):
                        break
                    record = self.run_episode(futures[future])
                with self.lock:
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                done[record["source"]] = record
                finished += 1
                rate = (time.time() - started) / finished
                print(f"\r[{finished}/{len(todo)}] {record['decision']:<14} eta {rate * (len(todo) - finished) / 60:6.1f} min",
                      end="", flush=True)
        print()
        return [done[e["source_iri"]] for e in episodes if e["source_iri"] in done]


# ---- outputs ------------------------------------------------------------------------------------------------------------
def write_alignment(path: Path, alignments, onto1: str, onto2: str) -> None:
    """MELT / Alignment API RDF, the same format as DRAL-OA's AlignmentFormat.serialize_mapping_to_file."""
    cells = "".join(f"""
  <map>
    <Cell>
      <entity1 rdf:resource={quoteattr(a.source_iri)}/>
      <entity2 rdf:resource={quoteattr(a.target_iri)}/>
      <relation>{escape(a.relation)}</relation>
      <measure rdf:datatype="xsd:float">{a.confidence:.4f}</measure>
    </Cell>
  </map>""" for a in alignments)
    path.write_text(f"""<?xml version="1.0" encoding="utf-8"?>
<rdf:RDF xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment"
  xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
  xmlns:xsd="http://www.w3.org/2001/XMLSchema#">
<Alignment>
  <xml>yes</xml>
  <level>0</level>
  <type>??</type>
  <onto1><Ontology rdf:about={quoteattr(onto1)}/></onto1>
  <onto2><Ontology rdf:about={quoteattr(onto2)}/></onto2>{cells}
</Alignment>
</rdf:RDF>
""", encoding="utf-8")


IGNORED_RELATIONS = {"?", "%"}  # unknown / incompatible cells are not links


def reference_partners(reference_path: Path, source, target) -> dict[str, set]:
    """Every reference link, read from this run's source side (as DRAL-OA's clf.py builds link_labels from
    reference.csv): source class -> its partner(s). All relations count, as in clf.py."""
    from collections import defaultdict

    from oa_rlaif.ontology.alignment_io import orient, read_alignment

    partners: dict[str, set] = defaultdict(set)
    for a in read_alignment(reference_path):
        if a.atomic and a.relation not in IGNORED_RELATIONS:
            a = orient(a, source, target)
            if a.status != "unresolved_endpoint":
                partners[a.source_iri].add(a.target_iri)
    return dict(partners)


def rankings_of(records: list[dict], candidates: dict[str, list[str]]) -> dict[str, list[str]]:
    """Each class's ranked answers for Top-k: the model's own choice first, then its other candidates in retrieval
    order (the order they were shown, C1 first). NONE or no submission = nothing found."""
    out = {}
    for r in records:
        if r["decision"] != "match":
            out[r["source"]] = []
            continue
        first = r["targets"][:1]
        out[r["source"]] = first + [c for c in candidates.get(r["source"], []) if c not in first]
    return out


def score(records: list[dict], candidates: dict[str, list[str]], reference_path: Path, source, target) -> dict:
    """The Top-k evaluation (k = 1, 3, 5) of DRAL-OA's clf.py (precision_at_k / recall_at_k), the only score:
    only the source classes that have a partner in the reference are scored; per class c, with F_c its top k and
    R_c its partners, precision@k = |F_c ∩ R_c| / min(k, |R_c|), recall@k = |F_c ∩ R_c| / |R_c|, averaged over the
    scored classes; F-measure@k = 2PR / (P + R). Reference classes not decided yet are left out and counted."""
    from oa_rlaif.eval.metrics import AlignmentEvaluator

    partners = reference_partners(reference_path, source, target)
    decided = {r["source"] for r in records}
    scored = {s: t for s, t in partners.items() if s in decided}
    result = AlignmentEvaluator().topk(rankings_of(records, candidates), scored)
    result.update(reference_classes=len(partners), reference_classes_decided=len(scored))
    return result


def results_table(topk: dict, model: str) -> str:
    """The Top-k evaluation (k = 1, 3, 5) in a results table: one row per k."""
    lines = [f"**{topk['method']}**: {topk['classes_scored']:,} reference classes scored "
             f"(of {topk['reference_classes']:,}), model {model}", "",
             "| Top-k | Precision | Recall | F-measure | Classes with a hit |", "|---|---:|---:|---:|---:|"]
    for k, v in topk["k"].items():
        lines.append(f"| Top-{k} | {v['Precision']:.3f} | {v['Recall']:.3f} | {v['F-measure']:.3f} | {v['classes_with_a_hit']:,} |")
    return "\n".join(lines) + "\n"


def label(onto, iri: str) -> str:
    from oa_rlaif.data.cards import names

    return (names(onto.entities[iri]) or [""])[0] if iri in onto.entities else ""


def write_decisions_tsv(path: Path, records: list[dict], kept: set, source, target, threshold: float) -> None:
    """Every decided class with labels, for checking by hand: one row per proposed target (one row for NONE)."""
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source_iri", "source_label", "decision", "target_iri", "target_label", "relation",
                         "confidence", "in_alignment_rdf"])
        for r in records:
            for t in r["targets"] or [""]:
                writer.writerow([r["source"], label(source, r["source"]), r["decision"], t, label(target, t) if t else "",
                                 r["relation"] or "", "" if r["confidence"] is None else r["confidence"],
                                 "yes" if (r["source"], t) in kept else "no"])


def write_outputs(out: Path, records: list[dict], n_classes: int, source_url: str, target_url: str, source, target,
                  reference_path: Path | None, model: str, k: int, threshold: float, runtime: float,
                  candidates: dict[str, list[str]] | None = None) -> dict:
    """alignment.rdf (matches >= threshold, greedy 1:1 by confidence), decisions.tsv, summary.json and, with a
    reference, the Top-k evaluation (k = 1, 3, 5): results.md and topk.svg."""
    from oa_rlaif.ontology.alignment_io import select_equivalences
    from oa_rlaif.ontology.models import Alignment

    picks = [Alignment(r["source"], t, {"classified_under": "<"}.get(r["relation"], r["relation"] or "="),
                       r["confidence"] if r["confidence"] is not None else 1.0, "oa-rlaif-agent")
             for r in records if r["decision"] == "match" and (r["confidence"] or 0) >= threshold for t in r["targets"]]
    alignments = select_equivalences(picks)
    write_alignment(out / "alignment.rdf", alignments, source_url, target_url)
    write_decisions_tsv(out / "decisions.tsv", records, {a.pair for a in alignments}, source, target, threshold)
    counts = {d: sum(r["decision"] == d for r in records) for d in ("match", "NONE", "no submission")}
    summary = {"source": str(path_of(source_url)), "target": str(path_of(target_url)), "model": model,
               "classes": n_classes, "decided": len(records), **counts, "mappings": len(alignments),
               "runtime_s": round(runtime), "k": k, "threshold": threshold}
    if reference_path:
        from oa_rlaif.eval.topk_chart import write_topk_chart

        topk = score(records, candidates or {}, reference_path, source, target)
        summary["reference"] = str(reference_path)
        summary["scores"] = topk
        (out / "results.md").write_text(results_table(topk, model))
        write_topk_chart(out / "topk.svg", {model: topk},
                         f"{path_of(source_url).stem} -> {path_of(target_url).stem}: {topk['method']}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def match(source_url: str, target_url: str, input_alignment_url: str | None, args) -> str:

    started = time.time()
    source_path, target_path = path_of(source_url), path_of(target_url)
    pair_id = args.pair_id or ("anatomy/mouse-human" if (source_path.stem, target_path.stem) == ("mouse", "human")
                               else f"{source_path.stem}-{target_path.stem}")
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    logging.info("Match %s to %s with %s (%s %s); pair %s; agent runs %s; output %s", source_path.name,
                 target_path.name, args.model, args.backend, args.api_base or OLLAMA, pair_id, args.sandbox, out)
    episodes, source, target = build_episodes(source_path, target_path, args.k, pair_id)
    if args.limit:
        episodes = episodes[: args.limit]
    records = Matcher(args.model, out, args.workers, args.sandbox, think=not args.no_think, backend=args.backend,
                      api_base=args.api_base).run(episodes, args.retries)
    summary = write_outputs(out, records, len(episodes), source_url, target_url, source, target,
                            path_of(input_alignment_url) if input_alignment_url else None, args.model, args.k,
                            args.threshold, time.time() - started,
                            candidates={e["source_iri"]: [c["iri"] for c in e["candidates"]] for e in episodes})
    if "scores" in summary:
        t = summary["scores"]
        for k, v in t["k"].items():
            logging.info("%s, Top-%d (%d reference classes): Precision %.3f  Recall %.3f  F-measure %.3f", t["method"], k,
                         t["classes_scored"], v["Precision"], v["Recall"], v["F-measure"])
    logging.info("classes %d: %d matched, %d NONE, %d without a submission; %d mappings written", len(episodes),
                 summary["match"], summary["NONE"], summary["no submission"], summary["mappings"])
    return (out / "alignment.rdf").resolve().as_uri()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source_url")
    parser.add_argument("target_url")
    parser.add_argument("input_alignment_url", nargs="?", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="served model name (Ollama, or vLLM's --served-model-name)")
    parser.add_argument("--backend", choices=["ollama", "vllm"], default="ollama")
    parser.add_argument("--api-base", default=None, help=f"default: this folder's Ollama ({OLLAMA})")
    parser.add_argument("--out", default=str(HERE / "output"))
    parser.add_argument("--k", type=int, default=40, help="candidates per class (training used 40)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=2, help="episodes in parallel (Ollama serves them)")
    parser.add_argument("--retries", type=int, default=1, help="extra tries for a class that ends without a submission")
    parser.add_argument("--limit", type=int, default=0, help="first N classes only (a quick test)")
    parser.add_argument("--sandbox", choices=["local", "docker", "user"], default="local",
                        help="local: the agent's commands run on this machine (default); docker: in the Docker "
                             "sandbox; user: as the sandbox user (scripts/sandbox_setup.sh)")
    parser.add_argument("--no-think", action="store_true", help="thinking off (as in training); default: on")
    parser.add_argument("--pair-id", default=None, help="the 'Alignment:' line of the prompt (default: from file names)")
    args = parser.parse_args(argv)
    print(match(args.source_url, args.target_url, args.input_alignment_url, args))
    return 0


if __name__ == "__main__":
    own_folders()
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)
    sys.exit(main(sys.argv[1:]))

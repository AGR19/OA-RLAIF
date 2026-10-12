"""Render built data into trainer-ready JSONL.

    python -m oa_rlaif.data.render

data/sdft/{train,test}.jsonl  TRL SDFTTrainer rows: prompt (chat messages, student view) + privileged_context
                              (teacher-only). Tasks: decisions (gold + silver), mini-SWE-agent trajectories,
                              ontology facts, SNOMED RF2 facts, complex alignments, datasource/doc questions.
data/grpo/{train,test}.jsonl  TRL GRPOTrainer rows: prompt + episode (JSON string read by the reward functions).
                              Kinds: gold, silver (ICD triangulation), unlabeled (DRAL-only reward).
data/eval/*.jsonl             held-out decision episodes, zero-shot pairs, closed-book facts and questions.
outputs/coverage/render_summary.json  rows and estimated tokens per task, after caps.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from ..io import read_jsonl, read_yaml
from ..harness.prompts import (AGENT_SYSTEM, KNOWLEDGE_SYSTEM, agent_task, decision_messages, decision_user,
                               privileged_context)
from ..paths import config_dir, data_dir, out_dir
from ..progress import stage

WHO_ICD10_PROFILE = ("WHO ICD-10 (from the WHO 10<->11 mapping tables): statistical classification of chapters, blocks and "
                     "3-4 character categories; categories are mutually exclusive and residual codes (.8 other, .9 "
                     "unspecified) collect leftovers.")


def pair_profiles() -> dict[str, tuple[str, str]]:
    """pair_id -> (source profile text, target profile text) from data/ontology_profiles.json."""
    from ..paths import relative_to_raw
    from .families import discover

    path = data_dir() / "ontology_profiles.json"
    if not path.exists():
        return {}
    texts = {k: v["summary"] for k, v in json.loads(path.read_text()).items()}
    out = {}
    for spec in discover():
        source = texts.get(relative_to_raw(spec.source), "") if spec.source else ""
        target = texts.get(relative_to_raw(spec.target), "") if spec.target else ""
        out[spec.pair_id] = (source, target)
    icd11 = next((v for k, v in texts.items() if k.endswith("icd11.owl")), "")
    snomed = next((v for k, v in texts.items() if "snomed-ct-us" in k), "")
    out["icd/who-icd10-to-icd11"] = (WHO_ICD10_PROFILE, icd11)
    out["icd/snomed-to-icd11"] = (snomed, icd11)
    return out


def tokens(text: str) -> int:
    return max(1, len(text) // 4)


def agent_trajectory(episode: dict) -> list[dict]:
    """Synthetic multi-turn mini-SWE-agent context ending just before the final decision turn."""
    src = episode["source_iri"]
    candidates = decision_user(episode).split("CANDIDATES", 1)[1].strip()
    return [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": agent_task(episode)},
        {"role": "assistant", "content": "THOUGHT: check the modelling assumptions of both ontologies first.\n```mswea_bash_command\nom profile\n```"},
        {"role": "user", "content": "<returncode>0</returncode>\n<output>\nSource ontology: " + (episode.get("source_profile") or "unknown")
                                    + "\nTarget ontology: " + (episode.get("target_profile") or "unknown") + "\n</output>"},
        {"role": "assistant", "content": f"THOUGHT: inspect the source entity.\n```mswea_bash_command\nom show {src}\n```"},
        {"role": "user", "content": "<returncode>0</returncode>\n<output>\n" + episode["source_card"] + "\n</output>"},
        {"role": "assistant", "content": f"THOUGHT: retrieve candidate targets.\n```mswea_bash_command\nom candidates {src} --k {len(episode['candidates'])}\n```"},
        {"role": "user", "content": "<returncode>0</returncode>\n<output>\n" + candidates + "\n</output>"},
    ]


def knowledge_row(item: dict, task: str) -> dict:
    return {"id": item["id"], "task": task, "family": item.get("ontology") or item.get("source", task),
            "prompt": [{"role": "system", "content": KNOWLEDGE_SYSTEM}, {"role": "user", "content": item["prompt"]}],
            "privileged_context": f"{item['privileged_context']}\n\nReference answer: {item['answer']}"}


def render() -> dict:
    cfg = read_yaml(config_dir() / "render.yaml")
    rng = random.Random(cfg["seed"])
    base = data_dir()
    sdft = {"train": [], "test": []}
    grpo = {"train": [], "test": []}
    evals = defaultdict(list)
    zero_shot = set(read_yaml(config_dir() / "data.yaml").get("zero_shot_pairs", []))

    profiles = pair_profiles()
    episode_files = sorted((base / "episodes").glob("*.jsonl"))
    with stage("render_episodes", total=len(episode_files)) as bar:
        for path in episode_files:
            for ep in read_jsonl(path):
                ep["source_profile"], ep["target_profile"] = profiles.get(ep["pair_id"], ("", ""))
                split = ep["split"]
                if split == "test":
                    evals["zero_shot" if ep["pair_id"] in zero_shot else f"heldout_{ep['family']}"].append(ep)
                row = {"id": ep["id"], "family": ep["family"], "kind": ep["kind"], "prompt": decision_messages(ep),
                       "episode": json.dumps(ep, ensure_ascii=False)}
                grpo[split].append(row)
                if ep["kind"] in ("gold", "silver"):
                    agent = rng.random() < cfg["agent_trajectory_rate"]
                    sdft[split].append({"id": ep["id"], "task": "decision_agent" if agent else "decision",
                                        "family": ep["family"],
                                        "prompt": agent_trajectory(ep) if agent else decision_messages(ep),
                                        "privileged_context": privileged_context(ep)})
            bar.advance()

    knowledge_inputs = [("facts/facts.jsonl", "fact"), ("facts/snomed_rf2_facts.jsonl", "fact_rf2"),
                        ("facts/profile_facts.jsonl", "profile_qa"),
                        ("complex/complex.jsonl", "complex"), ("knowledge/questions.jsonl", "knowledge_qa")]
    seen_complex: set[str] = set()
    with stage("render_knowledge", total=len(knowledge_inputs)) as bar:
        for relative, task in knowledge_inputs:
            path = base / relative
            if path.exists():
                for item in read_jsonl(path):
                    if task == "complex" and item.get("dedupe_key"):
                        if item["dedupe_key"] in seen_complex:
                            continue
                        seen_complex.add(item["dedupe_key"])
                    row = knowledge_row(item, item.get("task", task))
                    sdft[item["split"]].append(row)
                    if item["split"] == "test":
                        evals[f"closed_book_{task}"].append(item)
            bar.advance()

    # caps per task for the SDFT budget (training split only; test is kept whole for evaluation)
    caps = cfg["sdft_caps"]
    by_task = defaultdict(list)
    for row in sdft["train"]:
        by_task[row["task"] + ":" + row["family"].split("/")[0]].append(row)
    capped = []
    for key, rows in sorted(by_task.items()):
        task = key.split(":")[0]
        cap = caps.get(key, caps.get(task, caps["default"]))
        rng.shuffle(rows)
        capped += rows[:cap]
    rng.shuffle(capped)
    sdft["train"] = capped
    grpo_cap = cfg["grpo_train_cap"]
    rng.shuffle(grpo["train"])
    kinds = Counter()
    kept = []
    for row in grpo["train"]:
        limit = cfg["grpo_kind_share"][row["kind"]] * grpo_cap
        if kinds[row["kind"]] < limit:
            kinds[row["kind"]] += 1
            kept.append(row)
    grpo["train"] = kept

    def write(path: Path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for split in ("train", "test"):
        write(base / "sdft" / f"{split}.jsonl", sdft[split])
        write(base / "grpo" / f"{split}.jsonl", grpo[split])
    for name, rows in evals.items():
        write(base / "eval" / f"{name}.jsonl", rows)

    def prompt_tokens(row):
        return sum(tokens(m["content"]) for m in row["prompt"]) + tokens(row.get("privileged_context", ""))

    summary = {
        "sdft_train_rows": len(sdft["train"]), "sdft_test_rows": len(sdft["test"]),
        "sdft_train_prompt_tokens_est": sum(prompt_tokens(r) for r in sdft["train"]),
        "sdft_train_by_task": Counter(r["task"] for r in sdft["train"]),
        "grpo_train_rows": len(grpo["train"]), "grpo_train_by_kind": Counter(r["kind"] for r in grpo["train"]),
        "grpo_test_rows": len(grpo["test"]), "eval_sets": {k: len(v) for k, v in evals.items()},
    }
    (out_dir() / "coverage").mkdir(parents=True, exist_ok=True)
    (out_dir() / "coverage" / "render_summary.json").write_text(json.dumps(summary, indent=1, default=dict))
    return summary


if __name__ == "__main__":
    print(json.dumps(render(), indent=1, default=dict))

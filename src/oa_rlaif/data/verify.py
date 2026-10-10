"""Verification gate run before any push (python -m oa_rlaif.data.verify). Exit code 1 on any failure.

Checks
  coverage   every raw/Bio-ML file has a role; every data-bearing file produced >= 1 parsed record, and every
             training-role file produced >= 1 training record
  leakage    no train decision shares a component (or ICD bridge category) with a test decision;
             zero-shot pairs never appear in training files
  schema     every SDFT row has prompt + privileged_context; every GRPO row's episode parses; gold answers
             round-trip through the strict submit parser
Writes docs/DATA_CARD.md (counts per family/task/file) and outputs/coverage/verification.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

from ..harness.prompts import gold_answer, parse_decision
from ..io import read_jsonl, read_tsv, read_yaml
from ..paths import ROOT, config_dir, data_dir, out_dir
from .coverage import build_manifest, load_rules
from .icd_triangulation import bridge_split

TRAINING_ROLES = {"ontology", "gold_reference", "complex_reference", "icd_bridge", "snomed_facts", "snomed_mapping",
                  "knowledge_passage"}


def verify() -> dict:
    failures: list[str] = []
    _, _, data_bearing = load_rules()
    manifest = build_manifest()
    unassigned = [r["path"] for r in manifest if r["role"] == "UNASSIGNED"]
    if unassigned:
        failures.append(f"{len(unassigned)} unassigned files, e.g. {unassigned[:3]}")
    derived = {r["path"]: (int(r["parsed_records"]), int(r["training_records"]))
               for r in read_tsv(out_dir() / "coverage" / "derived_counts.tsv")}

    exceptions_cfg = read_yaml(config_dir() / "parse_exceptions.yaml")
    documented = set(exceptions_cfg.get("references") or {}) | set(exceptions_cfg.get("empty_files") or {})

    def key(row):
        return {"raw": "", "bioml": "bio-ml/", "papers": "matching_surveys/"}[row["source"]] + row["path"]

    per_file = []
    for row in manifest:
        parsed, training = derived.get(key(row), (0, 0))
        per_file.append({**row, "parsed": parsed, "training": training})
        if row["role"] in data_bearing and parsed < 1 and key(row) not in documented:
            failures.append(f"no parsed records: {key(row)} ({row['role']})")
        elif (row["role"] in TRAINING_ROLES and training < 1 and row["role"] != "ontology"
              and key(row) not in documented):  # documented unusable references (configs/parse_exceptions.yaml)
            failures.append(f"no training records: {key(row)} ({row['role']})")
    ontology_rows = [r for r in per_file if r["role"] == "ontology"]
    if not any(r["training"] for r in ontology_rows):
        failures.append("no ontology produced training facts")

    # integrity: every JSONL line parses and no file has zero-filled regions (truncated or overlapping writes)
    for path in sorted(data_dir().rglob("*.jsonl")):
        if "generated" in path.parts:
            continue
        with path.open("rb") as f:
            for number, line in enumerate(f, 1):
                try:
                    if b"\x00" in line:
                        raise ValueError("zero bytes")
                    json.loads(line)
                except ValueError as error:
                    failures.append(f"damaged JSONL: {path.relative_to(data_dir())} line {number} ({str(error)[:40]})")
                    break

    # leakage: components of train vs test decisions
    components = {r["iri"]: r["component"] for r in read_tsv(data_dir() / "splits" / "components.tsv")}
    zero_shot = set(read_yaml(config_dir() / "data.yaml").get("zero_shot_pairs", []))
    train_components, test_components = set(), set()
    episodes = Counter()
    for path in sorted((data_dir() / "episodes").glob("*.jsonl")):
        for ep in read_jsonl(path):
            episodes[(ep["family"], ep["split"], ep["kind"])] += 1
            if ep["family"] == "icd":
                code = ep["source_iri"].rsplit("/", 1)[-1]
                if ep["pair_id"] == "icd/who-icd10-to-icd11" and bridge_split(code) != ep["split"]:
                    failures.append(f"ICD bridge split mismatch for {code}")
                continue
            if ep["kind"] == "unlabeled":
                continue
            comp = components.get(ep["source_iri"], ep["source_iri"])
            (train_components if ep["split"] == "train" else test_components).add(comp)
            if ep["pair_id"] in zero_shot and ep["split"] == "train":
                failures.append(f"zero-shot pair in train: {ep['pair_id']}")
    overlap = train_components & test_components
    if overlap:
        failures.append(f"{len(overlap)} components appear in both train and test")

    # schema + gold round-trip
    sdft_rows = Counter()
    for split in ("train", "test"):
        path = data_dir() / "sdft" / f"{split}.jsonl"
        if path.exists():
            for row in read_jsonl(path):
                sdft_rows[(split, row["task"])] += 1
                if not row.get("prompt") or not row.get("privileged_context"):
                    failures.append(f"sdft row without prompt/privileged_context: {row['id']}")
                    break
    bad_roundtrip = 0
    grpo_rows = Counter()
    for split in ("train", "test"):
        path = data_dir() / "grpo" / f"{split}.jsonl"
        if path.exists():
            for row in read_jsonl(path):
                ep = json.loads(row["episode"])
                grpo_rows[(split, ep["kind"])] += 1
                if ep["kind"] != "unlabeled":
                    d = parse_decision(gold_answer(ep), ep)
                    expected = ep["gold"][0] if ep["gold"] else "NONE"
                    if not d.valid or (d.choice != expected and d.choice not in ep["gold"]):
                        bad_roundtrip += 1
                if split == "train" and ep["pair_id"] in zero_shot:
                    failures.append(f"zero-shot pair in GRPO train: {ep['pair_id']}")
    if bad_roundtrip:
        failures.append(f"{bad_roundtrip} gold answers fail the submit-parser round trip")

    # parse + reference verification (OWL API cross-parse, ELK, Beisswanger & Hahn checks)
    pv_path = out_dir() / "coverage" / "parse_verification.json"
    exceptions = read_yaml(config_dir() / "parse_exceptions.yaml")
    if not pv_path.exists():
        failures.append("parse verification has not been run (python -m oa_rlaif.data.parse_verify)")
    else:
        pv = json.loads(pv_path.read_text())
        for row in pv["ontologies"]["files"]:
            if row["status"] in ("mismatch", "owlapi_failed") and row["path"] not in (exceptions.get("ontologies") or {}):
                if row["status"] == "owlapi_failed" and row["bytes"] >= 300e6:
                    continue  # large files are re-checked on Lambda with a bigger heap; reported in PARSE_VERIFICATION.md
                failures.append(f"ontology {row['status']}: {row['path']} (ours {row['ours_classes']}, diff {row.get('class_diff')})")
        for row in pv["references"]["files"]:
            if row["status"] == "fail" and row["reference"] not in (exceptions.get("references") or {}):
                failures.append(f"reference check failed: {row['reference']} {row.get('parse_error', '')}")

    report = {"ok": not failures, "failure_count": len(failures), "failures": failures[:50], "files": len(manifest),
              "episodes": {"/".join(k): v for k, v in sorted(episodes.items())},
              "sdft_rows": {"/".join(k): v for k, v in sorted(sdft_rows.items())},
              "grpo_rows": {"/".join(k): v for k, v in sorted(grpo_rows.items())},
              "components_train": len(train_components), "components_test": len(test_components)}
    (out_dir() / "coverage" / "verification.json").write_text(json.dumps(report, indent=1))
    write_data_card(per_file, report)
    return report


def write_data_card(per_file: list[dict], report: dict) -> None:
    by_family = defaultdict(lambda: Counter())
    for r in per_file:
        by_family[(r["source"], r["family"], r["role"])]["files"] += 1
        by_family[(r["source"], r["family"], r["role"])]["parsed"] += r["parsed"]
        by_family[(r["source"], r["family"], r["role"])]["training"] += r["training"]
    lines = ["# Data card (generated by `python -m oa_rlaif.data.verify`)", "",
             f"Verification: **{'PASS' if report['ok'] else 'FAIL'}**. Files covered: {report['files']}.", "",
             "Licensing: the datasource (Springer © 2007) and SNOMED CT / UMLS-derived rows are private-use only and are "
             "never committed to git; datasets go to private Hugging Face repos only.", "",
             "## Per family and role", "", "| Source | Family | Role | Files | Parsed records | Training records |",
             "|---|---|---|---:|---:|---:|"]
    for (source, family, role), c in sorted(by_family.items()):
        lines.append(f"| {source} | {family} | {role} | {c['files']} | {c['parsed']:,} | {c['training']:,} |")
    lines += ["", "## Decision episodes", "", "| family/split/kind | episodes |", "|---|---:|"]
    lines += [f"| {k} | {v:,} |" for k, v in report["episodes"].items()]
    lines += ["", "## Trainer rows", "", "| SDFT split/task | rows |", "|---|---:|"]
    lines += [f"| {k} | {v:,} |" for k, v in report["sdft_rows"].items()]
    lines += ["", "| GRPO split/kind | rows |", "|---|---:|"] + [f"| {k} | {v:,} |" for k, v in report["grpo_rows"].items()]
    if report["failures"]:
        lines += ["", "## Failures", ""] + [f"- {f}" for f in report["failures"]]
    (ROOT / "docs" / "DATA_CARD.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    result = verify()
    print(json.dumps({k: v for k, v in result.items() if k in ("ok", "failures", "components_train", "components_test")}, indent=1))
    sys.exit(0 if result["ok"] else 1)

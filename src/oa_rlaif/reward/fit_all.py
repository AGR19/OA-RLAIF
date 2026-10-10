"""Fit the DRAL-OA mapping layer for every decision family (python -m oa_rlaif.reward.fit_all). Runs on the Mac.

Groups: anatomy, dh, dh_multiling, multifarm, biodiv (pooled per family, full graphs), each Bio-ML pair and the
ICD SNOMED->ICD-11 silver task (graphs restricted to episode entities + their parents). Positives are the gold
targets of TRAIN-split episodes only. Output: data/dral/<group>.npz (iris, z, t) used by reward/dral_reward.py.
The ICD-10 -> ICD-11 crosswalk has no ICD-10 ontology graph, so its episodes get the gold term only.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from ..data.families import discover
from ..data.profiles import load_profile
from ..io import read_jsonl
from ..paths import data_dir, out_dir, raw_dir
from ..progress import stage
from .dral_mapping_layer import DralConfig, fit_group

POOLED = {"anatomy", "dh", "dh_multiling", "multifarm", "biodiv"}


def fasttext_path() -> Path:
    value = os.environ.get("OA_FASTTEXT_BIN")
    if not value:
        raise SystemExit("Set OA_FASTTEXT_BIN in .env to the DRAL-OA fast.bin (FastText 300d) file")
    return Path(value).expanduser()


def groups() -> dict[str, dict]:
    pairs = {p.pair_id: p for p in discover()}
    found: dict[str, dict] = defaultdict(lambda: {"pairs": set(), "train": [], "targets": set(), "entities": set()})
    for path in sorted((data_dir() / "episodes").glob("*.jsonl")):
        for ep in read_jsonl(path):
            if ep["pair_id"] == "icd/who-icd10-to-icd11":
                continue
            key = ep["family"] if ep["family"] in POOLED else ep["pair_id"].replace("/", "_")
            g = found[key]
            g["pairs"].add(ep["pair_id"])
            g["entities"].add(ep["source_iri"])
            for cand in ep["candidates"]:
                g["targets"].add(cand["iri"])
                g["entities"].add(cand["iri"])
            if ep["split"] == "train" and ep["kind"] in ("gold", "silver"):
                g["train"] += [(ep["source_iri"], t) for t in ep.get("gold_all", [])]
    for key, g in found.items():
        files = set()
        for pid in g["pairs"]:
            spec = pairs.get(pid)
            if spec is not None:
                files.update(p for p in (spec.source, spec.target) if p is not None)
        if key.startswith("icd_"):
            icd = raw_dir() / "ICD-10 to SNOWMEDCT"
            files.update([icd / "icd11.owl", *icd.glob("snomed-ct-us-*.owl")])
        g["files"] = sorted(files)
    return found


def main() -> int:
    import fasttext

    model = fasttext.load_model(str(fasttext_path()))
    report = {}
    todo = groups()
    with stage("fit_dral", total=len(todo)) as bar:
        for key, g in sorted(todo.items(), key=lambda kv: len(kv[1]["entities"])):
            out = data_dir() / "dral" / f"{key}.npz"
            if out.exists():
                report[key] = {"cached": True}
                bar.advance()
                continue
            profiles = [load_profile(p) for p in g["files"]]
            big = sum(len(p.entities) for p in profiles) > 50_000
            cfg = DralConfig(num_walks=5 if big else 20)  # documented deviation for >50K-node graphs
            report[key] = fit_group(profiles, g["train"], g["targets"], model, out, cfg,
                                    restrict=g["entities"] if big else None)
            report[key].update(pairs=len(g["pairs"]), num_walks=cfg.num_walks, restricted=big)
            print(key, json.dumps(report[key]), flush=True)
            bar.advance()
    (out_dir() / "dral_fit_report.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

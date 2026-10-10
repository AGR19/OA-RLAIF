"""Discover every ontology pair in the raw folder (and the Bio-ML download) with its reference files,
system outputs and language metadata. One PairSpec per alignment task; nothing here parses ontologies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import bioml_dir, data_dir, raw_dir

LANG = re.compile(r"^[a-z]{2}-[a-z]{2}$")


@dataclass
class PairSpec:
    pair_id: str
    family: str
    source: Path | None  # None when source cards come from a table (WHO ICD-10)
    target: Path | None
    references: list[Path]
    ref_kind: str = "simple"  # simple | complex | crosswalk | bioml
    source_lang: str = ""
    target_lang: str = ""
    system_outputs: list[Path] = field(default_factory=list)
    trivial: Path | None = None
    bioml_cands: dict[str, Path] = field(default_factory=dict)
    note: str = ""


def _dh_pairs(base: Path, family: str, results_index: dict[str, list[Path]]) -> list[PairSpec]:
    pairs = []
    for d in sorted(p for p in base.iterdir() if p.is_dir() and (p / "reference.rdf").exists()):
        key = d.name.split("_", 1)[1] if family == "dh" else d.name
        langs = d.name.rsplit("_", 1)[1].split("-") if family == "dh_multiling" else ["", ""]
        pairs.append(PairSpec(
            pair_id=f"{family}/{d.name}", family=family, source=d / "source.rdf", target=d / "target.rdf",
            references=[d / "reference.rdf"], source_lang=langs[0], target_lang=langs[1],
            system_outputs=results_index.get(key, [])))
    return pairs


def _dh_results_index(raw: Path) -> dict[str, list[Path]]:
    """Map OAEI DH test-case names (2024 and 2025 layouts) to their systemAlignment.rdf files."""
    index: dict[str, list[Path]] = {}
    for repo in ("DH-benchmark-main", "DH-benchmark-multiling-main"):
        for path in (raw / repo / "results").rglob("systemAlignment.rdf"):
            case = path.parent.parent.name  # .../<test case>/<system>/systemAlignment.rdf
            if LANG.match(case):
                case = f"idai-pactols_{case}"
            index.setdefault(case, []).append(path)
    return index


def _multifarm(raw: Path) -> list[PairSpec]:
    base = raw / "dataset-2015-testing"
    pairs = []
    for ref in sorted((base / "ref").rglob("*.rdf")):
        parts = ref.stem.split("-")  # <o1>-<o2>-<l1>-<l2>
        if len(parts) != 4:
            continue
        o1, o2, l1, l2 = parts
        source, target = base / "ont" / l1 / f"{o1}-{l1}.owl", base / "ont" / l2 / f"{o2}-{l2}.owl"
        pairs.append(PairSpec(f"multifarm/{ref.parent.name}/{ref.stem}", "multifarm", source, target, [ref],
                              source_lang=l1, target_lang=l2,
                              note="same-ontology" if o1 == o2 else "cross-ontology"))
    return pairs


def _complex(raw: Path) -> list[PairSpec]:
    base = raw / "complex-OM-benchmark-main"
    pairs = []
    for ref in sorted(list(base.rglob("reference.rdf")) + list(base.rglob("reference.edoal"))):
        if "2025_OAEI" in ref.parts:
            continue
        d = ref.parent
        source = next(iter(sorted(d.glob("source.*"))), None)
        target = next(iter(sorted(d.glob("target.*"))), None)
        pairs.append(PairSpec(f"complex_om/{d.relative_to(base).as_posix()}", "complex_om", source, target, [ref],
                              ref_kind="complex"))
    for ref in sorted((base / "biomedical").glob("*_reference.owl")):
        pairs.append(PairSpec(f"complex_om/biomedical/{ref.stem}", "complex_om", None, None, [ref],
                              ref_kind="complex_owl", note="logical definitions across HP/MP/WBP"))
    return pairs


def _bioml() -> list[PairSpec]:
    base = bioml_dir()
    if not base.exists():
        return []
    onto = {p.name.split("-", 1)[0]: p for p in (base / "ontologies").glob("*.owl")}
    snomed = next(iter(sorted((raw_dir() / "ICD-10 to SNOWMEDCT").glob("snomed-ct-us-*.owl"))), None)
    onto.setdefault("SNOMED", snomed)
    pairs = []
    for d in sorted(p for p in base.iterdir() if p.is_dir() and (p / "refs_equiv").exists()):
        src, tgt = d.name.split("-")
        pairs.append(PairSpec(
            f"bioml/{d.name}", "bioml", onto.get(src), onto.get(tgt),
            sorted((d / "refs_equiv").glob("*.tsv")), ref_kind="bioml",
            bioml_cands={s: d / f"local.{s}.cands.tsv" for s in ("train", "valid", "test") if (d / f"local.{s}.cands.tsv").exists()},
            note="SNOMED from the raw US 2026-09 OWL; unresolved Bio-ML IRIs are counted and dropped" if src == "SNOMED" else ""))
    return pairs


def _cyber() -> list[PairSpec]:
    """IoTSec -> UCO zero-shot test pair, built by `python -m oa_rlaif.data.cyber` (absent until converted)."""
    base = data_dir() / "external" / "cyber"
    if not (base / "reference.rdf").exists():
        return []
    return [PairSpec("cyber/iotsec-uco", "cyber", base / "iotsec.owl", base / "uco.owl", [base / "reference.rdf"],
                     note="unseen-domain ranking test; reference links carry no relation type")]


def discover() -> list[PairSpec]:
    raw = raw_dir()
    results = _dh_results_index(raw)
    anatomy = raw / "anatomy-dataset-2"
    icd = raw / "ICD-10 to SNOWMEDCT"
    pairs = [
        PairSpec("anatomy/mouse-human", "anatomy", anatomy / "mouse.owl", anatomy / "human.owl",
                 [anatomy / "reference.rdf"], trivial=anatomy / "trivial.rdf"),
        *_dh_pairs(raw / "DH-benchmark-main", "dh", results),
        *_dh_pairs(raw / "DH-benchmark-multiling-main", "dh_multiling", results),
        *_multifarm(raw),
        PairSpec("biodiv/envo-sweet", "biodiv", raw / "biodiv/envo.owl", raw / "biodiv/sweet.owl",
                 [raw / "biodiv/envo-sweet.rdf"]),
        PairSpec("biodiv/flopo-pto", "biodiv", raw / "biodiv/flopo.owl", raw / "biodiv/pto.owl",
                 [raw / "biodiv/flopo-pto.rdf"]),
        *_complex(raw),
        *_bioml(),
        *_cyber(),
        PairSpec("icd/who-icd10-to-icd11", "icd", None, icd / "icd11.owl",
                 sorted((icd / "ICD 10 to 11 Mapings" / "mapping").glob("*.txt")), ref_kind="crosswalk",
                 note="WHO ICD-10 -> ICD-11 classification crosswalk; not clinical equivalence"),
    ]
    return pairs


def ontology_files(pairs: list[PairSpec]) -> list[Path]:
    """Every ontology a pair references, plus raw ontologies that belong to no pair (facts only)."""
    files = {p for pair in pairs for p in (pair.source, pair.target) if p is not None}
    icd = raw_dir() / "ICD-10 to SNOWMEDCT"
    files.update(p for p in [icd / "ICD10CM.ttl", *icd.glob("snomed-ct-us-*.owl")] if p.exists())
    return sorted(files)

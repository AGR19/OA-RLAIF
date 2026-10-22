"""Cybersecurity zero-shot test pair: IoTSec -> UCO (an unseen domain; never used for training).

Inputs: the raw folder's cybersecurity/ (moved there from the Advanced-Cybersecurity-OA project on 2026-10-22):
  ontologies/iotsec.owl, ontologies/uco_rdf.xml
  reference.csv       label_1 = UCO#n, label_2 = IOT#n
  id-tables/          UCO_vectorized_BERT.csv, IOT_vectorized_BERT.csv: `data_num` / `self_word` map the reference IDs
                      (UCO#n, IOT#n) to class labels
Without that folder, the older layout is read: OA_CYBER_DIR (the Advanced-Cybersecurity-OA folder) and
OA_CYBER_REFERENCE (reference.csv), set in .env.

Outputs in data/external/cyber/ (gitignored):
  iotsec.owl      copy of the source ontology
  uco.owl         uco_rdf.xml wrapped as valid RDF/XML: the original file is an OWL body with no rdf:RDF root and
                  no namespace or entity declarations. The wrapper adds them; &cbx; is undefined in the file and
                  gets the marked placeholder urn:oa-rlaif:unresolved-cbx# (it affects the class Observable only)
  reference.rdf   Alignment API RDF, IoTSec class -> UCO class, relation "related": the reference gives no relation
                  types and is mostly broader/narrower links, so evaluation credits the pick and ignores the relation
  id_map.tsv, conversion_report.json   every ID, its label, its IRI, and every excluded row with the reason

    python -m oa_rlaif.data.cyber
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from ..io import atomic_open, write_tsv
from ..ontology.alignment_io import write_oaei_rdf
from ..ontology.models import Alignment
from ..ontology.parser import OntologyParser
from ..paths import data_dir
from ..retrieval.candidates import normalize

PAIR_ID = "cyber/iotsec-uco"
UCO_BASE = "http://ffrdc.ebiquity.umbc.edu/ns/ontology/"
WRAPPER_HEAD = (
    '<?xml version="1.0"?>\n'
    '<!DOCTYPE rdf:RDF [ <!ENTITY owl "http://www.w3.org/2002/07/owl#"> '
    '<!ENTITY rdfs "http://www.w3.org/2000/01/rdf-schema#"> <!ENTITY xsd "http://www.w3.org/2001/XMLSchema#"> '
    '<!ENTITY cbx "urn:oa-rlaif:unresolved-cbx#"> ]>\n'
    '<!-- OA-RLAIF wrapper: the source file uco_rdf.xml has no rdf:RDF root and no namespace/entity declarations. -->\n'
    '<rdf:RDF xmlns="http://www.w3.org/2002/07/owl#" xmlns:owl="http://www.w3.org/2002/07/owl#" '
    'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" '
    f'xmlns:xsd="http://www.w3.org/2001/XMLSchema#" xml:base="{UCO_BASE}">\n')


def cyber_dir() -> Path:
    return data_dir() / "external" / "cyber"


def _key(text: str) -> str:
    """Label key that ignores spacing and case ("I Pv4Address" == "IPv4Address" == "ipv4 address")."""
    return normalize(text).replace(" ", "")


def _labels(table: Path) -> dict[str, str]:
    csv.field_size_limit(sys.maxsize)
    with table.open(newline="", encoding="utf-8") as f:
        return {r["data_num"].strip(): r["self_word"].strip() for r in csv.DictReader(f)}


def _class_index(profile) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for iri, e in profile.entities.items():
        if e.declared and not any(k == "kind" for k, _ in e.annotations):  # classes only
            keys = {_key(label.text) for label in e.labels} | {_key(iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1])}
            for k in keys:
                index.setdefault(k, set()).add(iri)
    return index


def label_tables(source_dir: Path) -> tuple[Path, Path]:
    """(UCO table, IoTSec table): the raw folder's id-tables/, or the Advanced-Cybersecurity-OA project layout."""
    if (source_dir / "id-tables").is_dir():
        return source_dir / "id-tables" / "UCO_vectorized_BERT.csv", source_dir / "id-tables" / "IOT_vectorized_BERT.csv"
    tables = source_dir / "cybersecurity ontology alignment-final" / "parsing ontology"
    return tables / "UCO/vectorizing/UCO_vectorized_BERT.csv", tables / "iotsec/vectorizing/IOT_vectorized_BERT.csv"


def convert(source_dir: Path, reference_csv: Path) -> dict:
    out = cyber_dir()
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_dir / "ontologies" / "iotsec.owl", out / "iotsec.owl")
    body = (source_dir / "ontologies" / "uco_rdf.xml").read_text(encoding="utf-8")
    body = body.split("?>", 1)[1] if body.lstrip().startswith("<?xml") else body
    with atomic_open(out / "uco.owl") as f:
        f.write(WRAPPER_HEAD + body + "\n</rdf:RDF>\n")

    uco_table, iot_table = label_tables(source_dir)
    labels = {"UCO": _labels(uco_table), "IOT": _labels(iot_table)}
    index = {"UCO": _class_index(OntologyParser().parse(out / "uco.owl")),
             "IOT": _class_index(OntologyParser().parse(out / "iotsec.owl"))}

    def resolve(identifier: str) -> tuple[str, str, str]:
        side = identifier.split("#", 1)[0]
        label = labels.get(side, {}).get(identifier)
        if label is None:
            return "", "", "id_not_in_table"
        iris = index[side].get(_key(label), set())
        if len(iris) != 1:
            return label, "", "label_not_in_ontology" if not iris else "label_ambiguous"
        return label, next(iter(iris)), "resolved"

    with reference_csv.open(newline="", encoding="utf-8-sig") as f:
        rows = [(r["label_1"].strip(), r["label_2"].strip()) for r in csv.DictReader(f)]
    seen, kept, excluded, id_rows = set(), [], [], {}
    for uco_id, iot_id in rows:
        if (uco_id, iot_id) in seen:
            excluded.append({"uco": uco_id, "iot": iot_id, "reason": "duplicate_row"})
            continue
        seen.add((uco_id, iot_id))
        u, i = resolve(uco_id), resolve(iot_id)
        id_rows[uco_id], id_rows[iot_id] = u, i
        if u[2] != "resolved" or i[2] != "resolved":
            reason = u[2] if u[2] != "resolved" else i[2]
            excluded.append({"uco": uco_id, "iot": iot_id, "reason": f"{'uco' if u[2] != 'resolved' else 'iot'}:{reason}"})
            continue
        kept.append(Alignment(i[1], u[1], "related", 1.0, "reference.csv", f"{uco_id}|{iot_id}"))

    write_oaei_rdf(out / "reference.rdf", kept)
    write_tsv(out / "id_map.tsv", ["id", "label", "iri", "status"],
              ([k, *v] for k, v in sorted(id_rows.items())))
    report = {
        "pair_id": PAIR_ID, "direction": "source IoTSec -> target UCO (as in the original clf scripts)",
        "reference_rows": len(rows), "unique_rows": len(seen), "kept_pairs": len(kept),
        "excluded": Counter(e["reason"] for e in excluded), "excluded_rows": excluded,
        "uco_targets_with_most_links": Counter(a.target_iri for a in kept).most_common(5),
        "relation_note": "the reference gives no relation types; most links are broader/narrower, so the pair is "
                         "evaluated as a ranking test (pick in reference, any relation), never as equivalence P/R/F",
        "repair_note": "uco_rdf.xml lacks the rdf:RDF root and namespace/entity declarations; a wrapper was added. "
                       "&cbx; is undefined in the source and maps to urn:oa-rlaif:unresolved-cbx#",
    }
    (out / "conversion_report.json").write_text(json.dumps(report, indent=1, default=list))
    return {k: v for k, v in report.items() if k != "excluded_rows"}


def main() -> int:
    from ..paths import raw_dir

    source = raw_dir() / "cybersecurity"
    reference = source / "reference.csv"
    if not (source / "ontologies" / "uco_rdf.xml").exists():
        source = Path(os.environ.get("OA_CYBER_DIR", "")).expanduser()
        reference = Path(os.environ.get("OA_CYBER_REFERENCE") or source / "reference.csv").expanduser()
    if not (source / "ontologies" / "uco_rdf.xml").exists() or not reference.exists():
        raise SystemExit("No cybersecurity inputs: expected <raw folder>/cybersecurity/ (ontologies/, reference.csv, "
                         "id-tables/), or OA_CYBER_DIR and OA_CYBER_REFERENCE in .env")
    print(json.dumps(convert(source, reference), indent=1, default=list))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

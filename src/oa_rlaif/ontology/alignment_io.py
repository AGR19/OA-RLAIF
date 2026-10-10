"""Alignment readers/writers. read_alignment and orient are vendored from the parent repo
src/data/extractors/common.py; OAEI RDF and SSSOM writers are new."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

from lxml import etree

from ..io import read_tsv
from .models import Alignment

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"


def read_alignment(path):
    """Retain every reference cell, including expressions and uncertain relations."""
    path = Path(path)
    for a in _read_raw(path):
        relation = normalize_relation(a.relation)
        yield a if relation == a.relation else replace(a, relation=relation)


def select_equivalences(picks: list[Alignment]) -> list[Alignment]:
    """Greedy extraction by confidence: an equivalence target is kept for one source only. A source may keep
    several targets when the model submitted them together (multi-target sources, e.g. 19 in the Anatomy reference).
    Non-equivalence picks (<, >, related) are kept as they are."""
    owner: dict[str, str] = {}
    kept = []
    for a in sorted(picks, key=lambda a: -a.confidence):
        if a.relation == "=":
            if owner.get(a.target_iri, a.source_iri) != a.source_iri:
                continue
            owner[a.target_iri] = a.source_iri
        kept.append(a)
    return kept


# Alignment API long relation names (EDOAL files, e.g. the Complex-OM taxon track) mapped to the standard symbols.
RELATION_ALIASES = {"Equivalence": "=", "EquivRelation": "=", "SubsumedBy": "<", "SubsumedRelation": "<",
                    "Subsumes": ">", "SubsumeRelation": ">", "Incompatible": "%", "IncompatRelation": "%"}


def normalize_relation(relation: str) -> str:
    text = (relation or "").strip()
    return RELATION_ALIASES.get(text.rsplit(".", 1)[-1], text)


def _read_raw(path):
    path = Path(path)
    if path.suffix.lower() in (".tsv", ".csv"):
        for i, row in enumerate(read_tsv(path), 2):

            def get(*keys, default="", row=row):
                return next((row[k] for k in keys if k in row), default)

            source = get("SrcEntity", "source_iri", "source", "Entity1")
            target = get("TgtEntity", "target_iri", "target", "Entity2")
            score = float(get("Score", "confidence", default="1"))
            if not math.isfinite(score):
                raise ValueError(f"Nonfinite score at {path}:{i}")
            yield Alignment(
                source,
                target,
                get("Relation", "relation", default="="),
                score,
                str(path),
                str(i),
            )
        return
    i = 0
    for _, cell in etree.iterparse(
        str(path),
        events=("end",),
        tag="{*}Cell",
        resolve_entities=False,
        no_network=True,
    ):
        i += 1
        children = {etree.QName(e).localname: e for e in cell if isinstance(e.tag, str)}

        def endpoint(name, children=children):
            e = children.get(name)
            if e is None:
                return "", "", "missing_endpoint"
            if e.get(RDF + "resource"):
                return e.get(RDF + "resource"), "", "accepted"
            nested = list(e)
            if (
                len(nested) == 1
                and etree.QName(nested[0]).localname == "Class"
                and nested[0].get(RDF + "about")
                and len(nested[0]) == 0
            ):
                return nested[0].get(RDF + "about"), "", "accepted"
            return "", etree.tostring(e, encoding="unicode"), "expression"

        s, se, ss = endpoint("entity1")
        t, te, ts = endpoint("entity2")
        rel = children.get("relation")
        measure = children.get("measure")
        score = float(measure.text) if measure is not None and measure.text else 1.0
        if not math.isfinite(score):
            raise ValueError(f"Nonfinite measure: {path}:{i}")
        yield Alignment(
            s,
            t,
            (rel.text or "=").strip() if rel is not None else "=",
            score,
            str(path),
            str(i),
            se,
            te,
            "accepted"
            if ss == ts == "accepted"
            else "expression"
            if "expression" in (ss, ts)
            else "invalid",
        )
        cell.clear()
        while cell.getprevious() is not None:
            del cell.getparent()[0]


def orient(mapping, source, target):
    from dataclasses import replace

    if not mapping.atomic:
        return mapping
    a, b = mapping.pair
    if a in source.entities and b in target.entities:
        return mapping
    if b in source.entities and a in target.entities:
        return replace(
            mapping,
            source_iri=b,
            target_iri=a,
            relation={"<": ">", ">": "<"}.get(mapping.relation, mapping.relation),
        )
    return replace(mapping, status="unresolved_endpoint")


RDF_HEADER = """<?xml version="1.0" encoding="utf-8"?>
<rdf:RDF xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment"
         xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:xsd="http://www.w3.org/2001/XMLSchema#">
<Alignment>
  <xml>yes</xml>
  <level>0</level>
  <type>??</type>
"""


def _xml_attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def write_oaei_rdf(path, alignments) -> int:
    """Write atomic correspondences in the OAEI Alignment API RDF/XML format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        f.write(RDF_HEADER)
        for a in alignments:
            if not a.atomic:
                continue
            relation = {"<": "&lt;", ">": "&gt;"}.get(a.relation, _xml_attr(a.relation))
            f.write(
                "  <map>\n    <Cell>\n"
                f'      <entity1 rdf:resource="{_xml_attr(a.source_iri)}"/>\n'
                f'      <entity2 rdf:resource="{_xml_attr(a.target_iri)}"/>\n'
                f"      <relation>{relation}</relation>\n"
                f'      <measure rdf:datatype="xsd:float">{a.confidence:.6f}</measure>\n'
                "    </Cell>\n  </map>\n"
            )
            count += 1
        f.write("</Alignment>\n</rdf:RDF>\n")
    return count


SSSOM_PREDICATE = {"=": "skos:exactMatch", "<": "skos:broadMatch", ">": "skos:narrowMatch",
                   "related": "skos:relatedMatch", "classified_under": "skos:broadMatch"}
SSSOM_COLUMNS = ["subject_id", "subject_label", "predicate_id", "object_id", "object_label",
                 "mapping_justification", "confidence", "mapping_tool", "comment"]


def write_sssom(path, rows, metadata: dict) -> int:
    """Write SSSOM TSV (Matentzoglu et al. 2022) with a YAML metadata header.

    `rows` are dicts keyed by SSSOM_COLUMNS; relation symbols are mapped with SSSOM_PREDICATE upstream.
    Note the SSSOM direction: `A skos:broadMatch B` means B is broader than A.
    """
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "".join("#" + line + "\n" for line in yaml.safe_dump(metadata, sort_keys=False).splitlines())
    count = 0
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        f.write("\t".join(SSSOM_COLUMNS) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(c, "")).replace("\t", " ").replace("\n", " ") for c in SSSOM_COLUMNS) + "\n")
            count += 1
    return count

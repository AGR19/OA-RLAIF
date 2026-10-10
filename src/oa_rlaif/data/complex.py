"""Complex-OM tasks: EDOAL correspondences (kept as expressions, never flattened to class pairs) and the
biomedical HP/MP/WBP logical definitions. Both become SDFT prompts whose privileged context is the gold
expression; evaluation compares normalised expressions.
"""

from __future__ import annotations

from collections import Counter

import hashlib
import re
from pathlib import Path

from lxml import etree

from ..ontology.alignment_io import read_alignment
from ..paths import relative_to_raw
from .cards import card, local_name

EDOAL = "http://ns.inria.org/edoal/1.0/"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


def _iri_text(iri: str) -> str:
    return local_name(iri) or iri


def render_edoal(xml: str) -> str:
    """Compact functional rendering of an EDOAL entity, e.g. AttributeDomainRestriction(onAttribute=…, class=…)."""
    if not xml:
        return ""
    if not xml.lstrip().startswith("<"):
        return xml
    # Entity references (&edoal; &irstea; ...) are declared in the source file's DOCTYPE, which an extracted cell
    # no longer carries; drop the prefix so IRIs render by local name.
    xml = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#)[\w.-]+;", "", xml)
    root = etree.fromstring(xml.encode(), etree.XMLParser(recover=True))
    if root is None:
        return ""

    def walk(e) -> str:
        name = etree.QName(e).localname
        about = e.get(f"{{{RDF}}}about") or e.get(f"{{{RDF}}}resource")
        children = [c for c in e if isinstance(c.tag, str)]
        literal = (e.text or "").strip()
        if name in ("entity1", "entity2"):
            return " ".join(walk(c) for c in children)
        if about and not children:
            return f"{name}({_iri_text(about)})" if name not in ("Class", "Relation", "Property") or True else _iri_text(about)
        if not children:
            value = e.get("string") or e.get("value") or literal
            return f"{name}({value})" if value else name
        parts = []
        for child in children:
            child_name = etree.QName(child).localname
            grand = [g for g in child if isinstance(g.tag, str)]
            if len(grand) == 1 and child_name[0].islower():
                parts.append(f"{child_name}={walk(grand[0])}")
            else:
                parts.append(walk(child))
        head = f"{name}[{_iri_text(about)}]" if about else name
        return f"{head}({', '.join(parts)})"

    return walk(root)


KNOWN_RELATIONS = {"=", "<", ">", "%", "HasInstance", "InstanceOf"}
EXCLUDED_RELATIONS: Counter = Counter()  # reference cells skipped for an undefined relation, reported in the build summary


def complex_tasks(spec, source_profile, target_profile, split_of) -> tuple[list[dict], dict[str, int]]:
    tasks, counts = [], {}
    for ref in spec.references:
        key = relative_to_raw(ref)
        counts[key] = 0
        for a in read_alignment(ref):
            counts[key] += 1
            if a.relation not in KNOWN_RELATIONS:  # e.g. "//" in Hydrography: undefined by the benchmark, never guessed
                EXCLUDED_RELATIONS[f"{key}|{a.relation}"] += 1
                continue
            left = a.source_iri and card(source_profile, a.source_iri) if source_profile and a.source_iri else render_edoal(a.source_expression)
            right = _iri_text(a.target_iri) if a.target_iri else render_edoal(a.target_expression)
            if not left or not right:
                continue
            anchor = a.source_iri or hashlib.sha1(a.source_expression.encode()).hexdigest()[:12]
            prompt = (f"Complex ontology alignment from '{spec.pair_id.split('/')[-1]}'. Source entity:\n{left}\n\n"
                      "Give the corresponding target entity or EDOAL expression and the relation (=, <, >).")
            tasks.append({
                "id": hashlib.sha1(f"{key}|{a.record}".encode()).hexdigest()[:16],
                "task": "complex_edoal", "pair_id": spec.pair_id, "split": split_of(a.source_iri or anchor),
                "prompt": prompt, "answer": f"relation: {a.relation}\ntarget: {right}",
                "privileged_context": f"Gold correspondence ({a.relation}): {right}",
                "dedupe_key": hashlib.sha1(f"{spec.pair_id.split('/')[-1]}|{left}|{right}".encode()).hexdigest()[:16],
                "provenance": key,
            })
    return tasks, counts


def logical_definition_tasks(path: Path, split_of) -> tuple[list[dict], int]:
    """HP/MP/WBP reference OWLs: one task per class with an owl:equivalentClass expression."""
    from rdflib import OWL, RDFS, BNode, Graph, URIRef

    graph = Graph()
    graph.parse(str(path), format="xml")
    labels = {s: str(o) for s, o in graph.subject_objects(RDFS.label)}

    def show(node) -> str:
        if isinstance(node, URIRef):
            return labels.get(node) or _iri_text(str(node))
        prop = graph.value(node, OWL.onProperty)
        if prop is not None:
            some = graph.value(node, OWL.someValuesFrom)
            only = graph.value(node, OWL.allValuesFrom)
            filler = some if some is not None else only
            return f"{show(prop)} {'some' if some is not None else 'only'} ({show(filler)})"
        members = graph.value(node, OWL.intersectionOf)
        if members is not None:
            from rdflib.collection import Collection
            return " and ".join(show(m) for m in Collection(graph, members))
        members = graph.value(node, OWL.unionOf)
        if members is not None:
            from rdflib.collection import Collection
            return " or ".join(show(m) for m in Collection(graph, members))
        return "?"

    tasks, key = [], relative_to_raw(path)
    for subject, expression in graph.subject_objects(OWL.equivalentClass):
        if not isinstance(subject, URIRef) or not isinstance(expression, BNode):
            continue
        name = labels.get(subject) or _iri_text(str(subject))
        rendered = show(expression)
        tasks.append({
            "id": hashlib.sha1(f"{key}|{subject}".encode()).hexdigest()[:16],
            "task": "logical_definition", "pair_id": f"complex_om/biomedical/{path.stem}",
            "split": split_of(str(subject)),
            "prompt": f"In {path.stem.split('_')[0].upper()}, give the logical (equivalent-class) definition of '{name}'.",
            "answer": rendered, "privileged_context": f"{name} ≡ {rendered}", "provenance": key,
        })
    return tasks, len(tasks)

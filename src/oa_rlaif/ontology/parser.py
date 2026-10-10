"""Streaming ontology readers (vendored from the parent repo src/data/ontology_parser.py).

Changes from the original: no project-root path guards; synonyms and definitions stored on blank nodes
(e.g. Anatomy `oboInOwl:hasRelatedSynonym` -> genid node -> rdfs:label) are now kept; SKOS altLabel and
language tags are preserved as before.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from lxml import etree
from rdflib import OWL, RDF, RDFS, SKOS, BNode, Graph, Literal, URIRef
from rdflib.store import Store

from ..io import read_tsv, read_yaml, sha256, write_tsv, write_yaml

from .models import Entity, Label, OntologyProfile

LABELS = {
    "label",
    "prefLabel",
    "altLabel",
    "hasExactSynonym",
    "hasRelatedSynonym",
    "hasBroadSynonym",
    "hasNarrowSynonym",
}
DEFINITIONS = {"definition", "hasDefinition", "IAO_0000115", "description"}
PARSER_VERSION = 4  # 4: object/datatype properties kept as entities (kind, domain, range, super-properties)
PROPERTY_TYPES = {"ObjectProperty", "DatatypeProperty", "DataProperty"}


def local(iri):
    return str(iri).rsplit("#", 1)[-1].rsplit("/", 1)[-1]


class ProfileStore(Store):
    """RDFLib parser sink; does not retain an in-memory RDF graph."""

    def __init__(self, profile):
        super().__init__()
        self.profile = profile
        self.count = 0
        self.namespaces_map = {}
        self.bnode_links = {}  # indirection node -> list of (subject IRI, property local name)
        self.bnode_text = {}  # node -> rdfs:label literals (resolved and dropped after parsing)

    def bind(self, prefix, namespace, override=True):
        self.namespaces_map[prefix] = namespace

    def namespace(self, prefix):
        return self.namespaces_map.get(prefix)

    def prefix(self, namespace):
        return next((k for k, v in self.namespaces_map.items() if v == namespace), None)

    def namespaces(self):
        return iter(self.namespaces_map.items())

    def add(self, triple, context, quoted=False):
        self.count += 1
        s, p, o = triple
        key = local(p)
        if key == "label" and isinstance(o, Literal) and isinstance(s, (BNode, URIRef)):
            # Any node may be an indirection target (Anatomy uses named genid IRIs, others blank nodes).
            self.bnode_text.setdefault(s, []).append(o)
        if isinstance(s, BNode):
            return
        if not isinstance(s, URIRef):
            return
        if isinstance(o, (BNode, URIRef)) and (key in LABELS or key in DEFINITIONS):
            self.bnode_links.setdefault(o, []).append((str(s), key))
            return
        if p == RDF.type and o in (OWL.Class, RDFS.Class, SKOS.Concept):
            self.entity(s).declared = True
        elif p == RDF.type and o in (OWL.ObjectProperty, OWL.DatatypeProperty):
            entity = self.entity(s)
            entity.declared = True
            entity.annotations.add(("kind", "object property" if o == OWL.ObjectProperty else "datatype property"))
        elif p == RDFS.subPropertyOf and isinstance(o, URIRef):
            self.entity(s).parents.add(str(o))
        elif p in (RDFS.domain, RDFS.range) and isinstance(o, URIRef):
            self.entity(s).annotations.add(("domain" if p == RDFS.domain else "range", str(o)))
        elif key in LABELS and isinstance(o, Literal):
            self.entity(s).labels.add(Label(str(o), o.language or "", key))
        elif key in DEFINITIONS and isinstance(o, Literal):
            self.entity(s).definitions.add(Label(str(o), o.language or "", key))
        elif p in (RDFS.subClassOf, SKOS.broader) and isinstance(o, URIRef):
            self.entity(s).parents.add(str(o))
        elif p == SKOS.narrower and isinstance(o, URIRef):
            self.entity(o).parents.add(str(s))
        elif key.lower() == "cui":
            self.entity(s).cuis.add(str(o))
        elif key in ("notation", "icd_mms_code", "code") or (
            p == SKOS.exactMatch and "icd11.code" in str(o)
        ):
            self.entity(s).codes.add(str(o))
        elif p == OWL.disjointWith and isinstance(o, URIRef):
            self.profile.disjoint.add(tuple(sorted((str(s), str(o)))))
        elif p == OWL.equivalentClass and isinstance(o, URIRef):
            self.entity(s).parents.add(str(o))
            self.entity(o).parents.add(str(s))
        elif isinstance(o, Literal) and ("umls" in str(p) or "ICD10CM/" in str(p)):
            self.entity(s).annotations.add((str(p), str(o)))

    def resolve_bnodes(self):
        """Attach blank-node labels to the subjects that referenced them (order-independent)."""
        for node, links in self.bnode_links.items():
            for literal in self.bnode_text.get(node, []):
                for subject, key in links:
                    target = self.entity(subject).definitions if key in DEFINITIONS else self.entity(subject).labels
                    target.add(Label(str(literal), literal.language or "", key))
        self.bnode_links.clear()
        self.bnode_text.clear()

    def entity(self, iri):
        iri = str(iri)
        if iri not in self.profile.entities:
            self.profile.entities[iri] = Entity(iri)
        return self.profile.entities[iri]


TOKEN = re.compile(
    r'\s+|#[^\n]*|"(?:[^"\\]|\\.)*"(?:@[\w-]+|\^\^(?:<[^>]*>|[^\s()]+))?|<[^>]*>|[()]|[^\s()]+'
)


def functional_forms(path):
    """Yield one top-level OWL axiom at a time, including multiline nested expressions."""
    stack = []
    pending = None
    for line_number, line in enumerate(Path(path).open(), 1):
        for match in TOKEN.finditer(line):
            token = match.group()
            if token.isspace() or token.startswith("#"):
                continue
            if token == "(":
                if pending is None:
                    raise ValueError(f"Missing constructor at {path}:{line_number}")
                stack.append([pending])
                pending = None
            elif token == ")":
                if pending is not None:
                    stack[-1].append(pending)
                    pending = None
                if not stack:
                    raise ValueError(f"Unbalanced OWL at {path}:{line_number}")
                form = stack.pop()
                if not stack or (len(stack) == 1 and stack[0][0] == "Ontology"):
                    yield form
                else:
                    stack[-1].append(form)
            else:
                if pending is not None and stack:
                    stack[-1].append(pending)
                pending = token
    if stack:
        raise ValueError(f"Unclosed OWL expression: {path}")


def form_text(form):
    return (
        form
        if isinstance(form, str)
        else form[0] + "(" + " ".join(form_text(x) for x in form[1:]) + ")"
    )


class OntologyParser:
    def parse(self, path, ontology_id=None):
        path = Path(path).resolve()
        p = OntologyProfile(ontology_id or path.stem, path.name)
        with path.open("rb") as f:
            head = f.read(4096).lstrip()
        if head.startswith(b"Prefix(") or head.startswith(b"Ontology("):
            self._functional(path, p)
        elif (
            b"<Ontology" in head and b"www.w3.org/2002/07/owl#" in head and b"<rdf:RDF" not in head
        ):
            self._owlxml(path, p)
        else:
            store = ProfileStore(p)
            logging.getLogger("rdflib.term").setLevel(logging.ERROR)
            Graph(store=store).parse(
                str(path), format="turtle" if path.suffix.lower() == ".ttl" else "xml"
            )
            store.resolve_bnodes()
            p.metadata["triples_seen"] = store.count
            # Referenced parents are retained; individual labels must not become classes.
        declared = {k for k, e in p.entities.items() if e.declared}
        properties = {k for k in declared if any(a[0] == "kind" for a in p.entities[k].annotations)}
        parents = {x for k in declared for x in p.entities[k].parents}
        keep = declared | parents
        p.entities = {k: e for k, e in p.entities.items() if k in keep}
        for iri in parents:
            p.entities.setdefault(iri, Entity(iri))
        p.metadata.update(
            sha256=sha256(path),
            classes=len(declared - properties),
            properties=len(properties),
            parser_version=PARSER_VERSION,
            entities=len(p.entities),
            bytes=path.stat().st_size,
            source_file=path.name,
        )
        return p

    def _functional(self, path, p):
        prefixes = {}
        sink = ProfileStore(p)

        def iri(value):
            if not isinstance(value, str):
                return None
            if value.startswith("<"):
                return value[1:-1]
            if ":" in value:
                prefix, tail = value.split(":", 1)
                if prefix in prefixes:
                    return prefixes[prefix] + tail
            return None

        def supers(expression):
            if isinstance(expression, str):
                target = iri(expression)
                return [target] if target else []
            if expression[0] == "ObjectIntersectionOf":
                return [v for x in expression[1:] for v in supers(x)]
            return []

        counts = {}
        for form in functional_forms(path):
            op = form[0]
            counts[op] = counts.get(op, 0) + 1
            if op == "Prefix":
                raw = "".join(str(x) for x in form[1:])
                prefix, value = raw.split("=", 1)
                prefixes[prefix.rstrip(":")] = value.strip("<>")
            elif op == "Declaration" and isinstance(form[1], list) and form[1][0] == "Class":
                sink.entity(iri(form[1][1])).declared = True
            elif op == "Declaration" and isinstance(form[1], list) and form[1][0] in ("ObjectProperty", "DataProperty"):
                target = iri(form[1][1])
                if target:
                    entity = sink.entity(target)
                    entity.declared = True
                    entity.annotations.add(("kind", "object property" if form[1][0] == "ObjectProperty" else "datatype property"))
            elif op == "AnnotationAssertion" and len(form) >= 4:
                prop, subject, value = form[-3:]
                subject = iri(subject)
                prop = iri(prop)
                if subject and prop and isinstance(value, str):
                    m = re.fullmatch(r'"((?:[^"\\]|\\.)*)"(?:@([\w-]+)|\^\^.*)?', value)
                    if m:
                        import json

                        text = json.loads('"' + m[1] + '"')
                        sink.add(
                            (URIRef(subject), URIRef(prop), Literal(text, lang=m[2])),
                            None,
                        )
            elif op in ("SubClassOf", "EquivalentClasses"):
                terms = [x for x in form[1:] if not (isinstance(x, list) and x[0] == "Annotation")]
                if len(terms) >= 2:
                    for j, left in enumerate(terms if op == "EquivalentClasses" else terms[:1]):
                        s = iri(left)
                        if s:
                            for k, right in enumerate(terms):
                                if k != j:
                                    sink.entity(s).parents.update(supers(right))
            elif op == "DisjointClasses":
                named = [iri(x) for x in form[1:]]
                for i, a in enumerate(named):
                    for b in named[i + 1 :]:
                        if a and b:
                            p.disjoint.add(tuple(sorted((a, b))))
        p.metadata["construct_counts"] = counts

    def _owlxml(self, path, p):
        prefixes = {}
        sink = ProfileStore(p)

        def expand(e):
            value = e.get("IRI") or e.get("abbreviatedIRI")
            if value and ":" in value and not value.startswith(("http:", "https:", "urn:")):
                a, b = value.split(":", 1)
                return prefixes.get(a + ":", a + ":") + b
            return value

        for _, e in etree.iterparse(
            str(path), events=("end",), resolve_entities=False, no_network=True
        ):
            name = etree.QName(e).localname
            if name == "Prefix":
                prefixes[e.get("name")] = e.get("IRI")
            parent = e.getparent()
            if parent is not None and etree.QName(parent).localname == "Ontology":
                children = list(e)
                if (
                    name == "Declaration"
                    and children
                    and etree.QName(children[0]).localname == "Class"
                ):
                    sink.entity(expand(children[0])).declared = True
                elif name == "Declaration" and children and etree.QName(children[0]).localname in ("ObjectProperty", "DataProperty"):
                    entity = sink.entity(expand(children[0]))
                    entity.declared = True
                    kind = "object property" if etree.QName(children[0]).localname == "ObjectProperty" else "datatype property"
                    entity.annotations.add(("kind", kind))
                elif (
                    name == "SubClassOf"
                    and len(children) == 2
                    and all(etree.QName(x).localname == "Class" for x in children)
                ):
                    sink.entity(expand(children[0])).parents.add(expand(children[1]))
                elif name == "AnnotationAssertion" and len(children) == 3:
                    prop = expand(children[0])
                    sub = children[1].text
                    val = children[2]
                    if etree.QName(val).localname == "Literal":
                        sink.add(
                            (
                                URIRef(sub),
                                URIRef(prop),
                                Literal(
                                    val.text or "",
                                    lang=val.get("{http://www.w3.org/XML/1998/namespace}lang"),
                                ),
                            ),
                            None,
                        )
                e.clear()
                while e.getprevious() is not None:
                    del parent[0]

    def save(self, profile, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        write_tsv(
            directory / "entities.tsv",
            ["iri", "declared", "name"],
            ((e.iri, int(e.declared), e.name) for e in profile.entities.values()),
        )
        write_tsv(
            directory / "labels.tsv",
            ["iri", "text", "language", "kind"],
            (
                (e.iri, label.text, label.language, label.kind)
                for e in profile.entities.values()
                for label in sorted(e.labels | e.definitions)
            ),
        )
        write_tsv(
            directory / "parents.tsv",
            ["child", "parent"],
            ((e.iri, x) for e in profile.entities.values() for x in sorted(e.parents)),
        )
        write_tsv(
            directory / "identifiers.tsv",
            ["iri", "kind", "value"],
            (
                (e.iri, k, v)
                for e in profile.entities.values()
                for k, vs in [("cui", e.cuis), ("code", e.codes)]
                for v in sorted(vs)
            ),
        )
        write_tsv(
            directory / "annotations.tsv",
            ["iri", "property", "value"],
            ((e.iri, k, v) for e in profile.entities.values() for k, v in sorted(e.annotations)),
        )
        write_tsv(directory / "disjoint.tsv", ["left", "right"], sorted(profile.disjoint))
        write_yaml(
            directory / "profile.yaml",
            {
                **profile.metadata,
                "ontology_id": profile.ontology_id,
                "path": profile.path,
                "cache_format": 3,
            },
        )

    def load(self, directory):
        directory = Path(directory)
        metadata = read_yaml(directory / "profile.yaml")
        p = OntologyProfile(metadata.pop("ontology_id"), metadata.pop("path"), metadata=metadata)
        for row in read_tsv(directory / "entities.tsv"):
            p.entities[row["iri"]] = Entity(row["iri"], declared=row["declared"] == "1")
        for row in read_tsv(directory / "labels.tsv"):
            label = Label(row["text"], row["language"], row["kind"])
            entity = p.entities[row["iri"]]
            (entity.definitions if row["kind"] in DEFINITIONS else entity.labels).add(label)
        for row in read_tsv(directory / "parents.tsv"):
            p.entities[row["child"]].parents.add(row["parent"])
        for row in read_tsv(directory / "identifiers.tsv"):
            (
                p.entities[row["iri"]].cuis
                if row["kind"] == "cui"
                else p.entities[row["iri"]].codes
            ).add(row["value"])
        for row in read_tsv(directory / "annotations.tsv"):
            p.entities[row["iri"]].annotations.add((row["property"], row["value"]))
        if (directory / "disjoint.tsv").exists():
            p.disjoint = {
                tuple(sorted((r["left"], r["right"]))) for r in read_tsv(directory / "disjoint.tsv")
            }
        return p

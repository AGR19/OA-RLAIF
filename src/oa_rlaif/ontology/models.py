"""Ontology data classes (vendored unchanged from the parent repo src/data/models.py)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, order=True)
class Label:
    text: str
    language: str = ""
    kind: str = "label"


@dataclass
class Entity:
    iri: str
    labels: set[Label] = field(default_factory=set)
    definitions: set[Label] = field(default_factory=set)
    parents: set[str] = field(default_factory=set)
    cuis: set[str] = field(default_factory=set)
    codes: set[str] = field(default_factory=set)
    annotations: set[tuple[str, str]] = field(default_factory=set)
    declared: bool = False

    @property
    def name(self):
        labels = sorted(
            self.labels,
            key=lambda x: (x.language not in ("en", ""), x.kind != "prefLabel", x.text),
        )
        return labels[0].text if labels else self.iri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


@dataclass
class OntologyProfile:
    ontology_id: str
    path: str
    entities: dict[str, Entity] = field(default_factory=dict)
    axioms: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    disjoint: set[tuple[str, str]] = field(default_factory=set)


@dataclass(frozen=True)
class Alignment:
    source_iri: str
    target_iri: str
    relation: str = "="
    confidence: float = 1.0
    provenance: str = ""
    record: str = ""
    source_expression: str = ""
    target_expression: str = ""
    status: str = "accepted"

    @property
    def pair(self):
        return self.source_iri, self.target_iri

    @property
    def atomic(self):
        return bool(
            self.source_iri
            and self.target_iri
            and not self.source_expression
            and not self.target_expression
        )

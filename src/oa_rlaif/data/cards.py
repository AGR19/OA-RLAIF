"""Compact, deterministic text cards for ontology entities (what the model sees about a class)."""

from __future__ import annotations

from ..ontology.models import Entity, OntologyProfile

SYNONYM_KINDS = {"altLabel", "hasExactSynonym", "hasRelatedSynonym", "hasBroadSynonym", "hasNarrowSynonym"}


def local_name(iri: str) -> str:
    return iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def display(text: str) -> str:
    return " ".join(text.replace("_", " ").split())


def names(entity: Entity, language: str = "") -> tuple[str, list[str]]:
    """Preferred name (language first, then English, then any) and up to 5 distinct synonyms. Blank labels are
    ignored (IoTSec writes empty rdfs:label elements); with no other label the name is the IRI's local name."""
    labels = sorted((x for x in entity.labels if x.text.strip()), key=lambda x: (
        x.language != language if language else False, x.language not in ("en", ""),
        x.kind in SYNONYM_KINDS, x.kind != "prefLabel", x.text))
    if not labels:
        return local_name(entity.iri), []
    preferred = display(labels[0].text)
    seen, synonyms = {preferred.casefold()}, []
    for label in labels[1:]:
        text = display(label.text)
        if text.casefold() not in seen:
            seen.add(text.casefold())
            synonyms.append(text + (f" @{label.language}" if label.language and label.language != language else ""))
        if len(synonyms) == 5:
            break
    return preferred, synonyms


def name_of(profile: OntologyProfile, iri: str, language: str = "") -> str:
    entity = profile.entities.get(iri)
    return names(entity, language)[0] if entity else local_name(iri)


def card(profile: OntologyProfile, iri: str, language: str = "", children: dict | None = None,
         max_definition: int = 240) -> str:
    """One-line-per-field card: IRI, name, synonyms, definition, parents (and a few children)."""
    entity = profile.entities.get(iri)
    if entity is None:
        return f"IRI: {iri}\nname: {local_name(iri)}"
    preferred, synonyms = names(entity, language)
    lines = [f"IRI: {iri}", f"name: {preferred}"]
    kind = next((v for k, v in entity.annotations if k == "kind"), "")
    if kind:
        lines.append(f"kind: {kind}")
        for field in ("domain", "range"):
            values = [name_of(profile, v, language) for k, v in sorted(entity.annotations) if k == field][:2]
            if values:
                lines.append(f"{field}: " + "; ".join(values))
    if synonyms:
        lines.append("synonyms: " + "; ".join(synonyms))
    definitions = sorted(entity.definitions, key=lambda d: (d.language not in (language, "en", ""), d.text))
    if definitions:
        text = display(definitions[0].text)
        lines.append("definition: " + (text[:max_definition] + "…" if len(text) > max_definition else text))
    parents = [name_of(profile, p, language) for p in sorted(entity.parents) if p in profile.entities][:3]
    if parents:
        lines.append("parents: " + "; ".join(parents))
    if children is not None:
        kids = [name_of(profile, c, language) for c in sorted(children.get(iri, ()))][:3]
        if kids:
            lines.append("children: " + "; ".join(kids))
    if entity.codes:
        lines.append("codes: " + "; ".join(sorted(entity.codes)[:3]))
    return "\n".join(lines)


def children_index(profile: OntologyProfile) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for iri, entity in profile.entities.items():
        for parent in entity.parents:
            index.setdefault(parent, set()).add(iri)
    return index


def kind_of(profile: OntologyProfile, iri: str) -> str:
    entity = profile.entities.get(iri)
    return next((v for k, v in entity.annotations if k == "kind"), "class") if entity else "class"

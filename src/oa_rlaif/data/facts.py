"""Ontology-fact prompts for SDFT knowledge injection.

For a sample of classes in every ontology the student is asked a closed-book question (no card in its prompt);
the frozen teacher sees the entity card as privileged context, so the answer must come from the weights at
inference. Question types: parents, synonyms, definition, code lookup, reverse code lookup.
Populated variants of the same TBox (Complex-OM popconference 0-100%) are deduplicated by class set.
"""

from __future__ import annotations

import hashlib
import random

from ..ontology.models import OntologyProfile
from .cards import card, children_index, display, name_of, names

TEMPLATES = {
    "parents": "In the ontology '{onto}', which class or classes are the direct parents of '{name}'?",
    "synonyms": "In the ontology '{onto}', what other names or synonyms are used for '{name}'?",
    "definition": "In the ontology '{onto}', how is '{name}' defined?",
    "code": "In '{onto}', what is the code of '{name}'?",
    "reverse_code": "In '{onto}', which concept has the code {code}?",
}


def tbox_signature(profile: OntologyProfile) -> str:
    declared = sorted(i for i, e in profile.entities.items() if e.declared)
    return hashlib.sha1("\n".join(declared).encode()).hexdigest()


TITLES = {
    "anatomy-dataset-2/mouse.owl": "Adult Mouse Anatomy (MA)",
    "anatomy-dataset-2/human.owl": "NCI Thesaurus anatomy (human)",
    "icd11.owl": "ICD-11 Foundation", "ICD10CM.ttl": "ICD-10-CM", "snomed-ct-us": "SNOMED CT US Edition",
    "envo.owl": "Environment Ontology (ENVO)", "sweet.owl": "SWEET", "flopo.owl": "Flora Phenotype Ontology (FLOPO)",
    "pto.owl": "Plant Trait Ontology (PTO)", "Thesaurus.owl": "NCI Thesaurus (NCIT)", "doid.owl": "Disease Ontology (DOID)",
    "fma.owl": "Foundational Model of Anatomy (FMA)",
}


def ontology_title(profile: OntologyProfile) -> str:
    oid = profile.ontology_id
    for key, title in TITLES.items():
        if key in oid:
            return title
    parts = oid.split("/")
    if len(parts) >= 2 and parts[-1] in ("source.rdf", "target.rdf", "source.owl", "target.owl"):
        side = parts[-2].split("_", 1)[-1].split("-")
        index = 0 if parts[-1].startswith("source") else -1
        return f"{side[index] if side else parts[-2]} ({parts[0].split('-')[0]})"
    return parts[-1].rsplit(".", 1)[0]


def build_facts(profile: OntologyProfile, cap: int, split_of, seed: int = 7, language: str = "") -> list[dict]:
    rng = random.Random(f"{seed}|{profile.ontology_id}")
    declared = sorted(i for i, e in profile.entities.items() if e.declared)
    chosen = declared if len(declared) <= cap else rng.sample(declared, cap)
    kids = children_index(profile)
    title = ontology_title(profile)
    facts = []
    for iri in sorted(chosen):
        entity = profile.entities[iri]
        preferred, synonyms = names(entity, language)
        context = card(profile, iri, language, kids)
        answers = {}
        parents = [name_of(profile, p, language) for p in sorted(entity.parents) if p in profile.entities]
        if parents:
            answers["parents"] = "; ".join(parents[:5])
        if synonyms:
            answers["synonyms"] = "; ".join(synonyms)
        if entity.definitions:
            answers["definition"] = display(sorted(entity.definitions, key=lambda d: d.text)[0].text)[:400]
        codes = sorted(entity.codes)
        if codes:
            answers["code"] = "; ".join(codes[:3])
            answers["reverse_code"] = preferred
        for kind, answer in answers.items():
            prompt = TEMPLATES[kind].format(onto=title, name=preferred, code=codes[0] if codes else "")
            facts.append({
                "id": hashlib.sha1(f"{profile.ontology_id}|{iri}|{kind}".encode()).hexdigest()[:16],
                "task": f"fact_{kind}", "ontology": profile.ontology_id, "iri": iri,
                "split": split_of(iri), "prompt": prompt, "answer": answer,
                "privileged_context": context,
            })
    return facts

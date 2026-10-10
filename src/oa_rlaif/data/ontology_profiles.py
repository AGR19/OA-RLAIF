"""Modelling profiles: the assumptions behind each ontology, made explicit for the model.

An alignment decision depends on what each ontology's constructs *mean*: an ICD category is a mutually
exclusive classification bucket (with residual "other / unspecified" codes); SKOS `broader` in a thesaurus is
not subsumption; OBO anatomy mixes is-a with part-of restrictions; SNOMED CT distinguishes primitive from
fully defined concepts and tags each with a semantic category. This module derives a compact profile for every
ontology from its parsed profile plus a scan of the raw file, writes data/ontology_profiles.json, and renders a
short text block that is (1) shown in every decision prompt for the source and target ontologies and
(2) turned into SDFT knowledge prompts so the conventions are also learned into the weights.

    python -m oa_rlaif.data.ontology_profiles
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from ..paths import data_dir, raw_dir, relative_to_raw
from ..progress import stage
from .facts import ontology_title
from .families import discover, ontology_files
from .profiles import load_profile

RESIDUAL = re.compile(r"\b(other|unspecified|not elsewhere classified|nec|nos|not otherwise specified|unknown)\b", re.I)
SCAN_BYTES = 40_000_000  # raw-file scan budget per ontology (head of file) for construct counts
CONSTRUCTS = {
    "skos_concept": re.compile(rb"skos:Concept|skos/core#Concept"),
    "skos_broader": re.compile(rb"skos:broader|skos/core#broader"),
    "some_values_from": re.compile(rb"someValuesFrom|ObjectSomeValuesFrom"),
    "part_of": re.compile(rb"part_of|partOf|BFO_0000050|UNDEFINED_part_of|PartOf"),
    "disjoint": re.compile(rb"disjointWith|DisjointClasses|AllDisjointClasses"),
    "equivalent_class": re.compile(rb"equivalentClass|EquivalentClasses"),
    "individuals": re.compile(rb"owl:NamedIndividual|NamedIndividual\("),
}


def scan_constructs(path: Path) -> dict[str, int]:
    with path.open("rb") as f:
        head = f.read(SCAN_BYTES)
    counts = {name: len(rx.findall(head)) for name, rx in CONSTRUCTS.items()}
    counts["scanned_fraction"] = round(min(1.0, SCAN_BYTES / max(1, path.stat().st_size)), 3)
    return counts


def depth_stats(profile) -> tuple[int, float]:
    parents = {i: [p for p in e.parents if p in profile.entities] for i, e in profile.entities.items() if e.declared}
    memo: dict[str, int] = {}

    def depth(iri: str, seen: frozenset = frozenset()) -> int:
        if iri in memo:
            return memo[iri]
        ps = [p for p in parents.get(iri, []) if p not in seen]
        value = 0 if not ps else 1 + max(depth(p, seen | {iri}) for p in ps[:3])
        memo[iri] = value
        return value

    import sys
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
    depths = [depth(i) for i in list(parents)[:200000]]
    return (max(depths) if depths else 0), (round(sum(depths) / len(depths), 2) if depths else 0.0)


def build_profile(path: Path) -> dict:
    profile = load_profile(path)
    declared = [e for e in profile.entities.values() if e.declared]
    classes = [e for e in declared if not any(k == "kind" for k, _ in e.annotations)]
    properties = len(declared) - len(classes)
    n = max(1, len(classes))
    labels = [label for e in classes for label in e.labels]
    languages = Counter(label.language or "untagged" for label in labels)
    synonyms = sum(1 for e in classes if any(lbl.kind not in ("label", "prefLabel") for lbl in e.labels))
    with_labels = sum(1 for e in classes if e.labels)
    residual = sum(1 for e in classes if any(RESIDUAL.search(lbl.text) for lbl in e.labels))
    poly = sum(1 for e in classes if len([p for p in e.parents if p in profile.entities]) > 1)
    max_depth, mean_depth = depth_stats(profile)
    constructs = scan_constructs(path)
    skos = constructs["skos_concept"] > 0 or constructs["skos_broader"] > 0
    return {
        "ontology": relative_to_raw(path), "title": ontology_title(profile),
        "formalism": "SKOS thesaurus" if skos and constructs["some_values_from"] == 0 else "OWL ontology",
        "hierarchy": "skos:broader (thesaurus broader term; NOT logical subsumption)" if skos
                     else "rdfs:subClassOf (is-a subsumption)",
        "classes": len(classes), "properties": properties,
        "polyhierarchy_share": round(poly / n, 3), "max_depth": max_depth, "mean_depth": mean_depth,
        "part_of_restrictions": constructs["part_of"], "existential_restrictions": constructs["some_values_from"],
        "disjointness_axioms": constructs["disjoint"], "equivalence_axioms": constructs["equivalent_class"],
        "residual_category_share": round(residual / n, 3),
        "definition_coverage": round(sum(1 for e in classes if e.definitions) / n, 3),
        "synonym_coverage": round(synonyms / n, 3), "label_coverage": round(with_labels / n, 3),
        "languages": dict(languages.most_common(5)), "scan": constructs,
    }


def summary_text(p: dict) -> str:
    """Compact, model-facing description of the modelling assumptions (3-5 lines)."""
    lines = [f"{p['title']}: {p['formalism']}, {p['classes']:,} classes"
             + (f", {p['properties']:,} properties" if p["properties"] else "") + ".",
             f"Hierarchy = {p['hierarchy']}; polyhierarchy {p['polyhierarchy_share']:.0%}, depth up to {p['max_depth']}."]
    extras = []
    if p["part_of_restrictions"]:
        extras.append("part-of relations are modelled separately from is-a (a part is not a subclass)")
    if p["disjointness_axioms"]:
        extras.append("declares disjoint classes")
    if p["residual_category_share"] >= 0.02:
        extras.append(f"{p['residual_category_share']:.0%} residual categories ('other', 'unspecified', 'NOS'), "
                      "which only match residual categories with the same scope")
    if p["label_coverage"] < 0.5:
        extras.append("most classes have no labels; names come from IRI local names")
    if p["definition_coverage"] >= 0.2:
        extras.append(f"definitions for {p['definition_coverage']:.0%} of classes")
    languages = [k for k in p["languages"] if k != "untagged"]
    if languages:
        extras.append("labels in " + ", ".join(languages))
    if extras:
        lines.append("Conventions: " + "; ".join(extras) + ".")
    for key in ("snomed_note", "icd_note"):
        if p.get(key):
            lines.append(p[key])
    return "\n".join(lines)


SPECIAL_NOTES = {
    "snomed-ct-us": ("snomed_note", "SNOMED CT is a description-logic (EL++) terminology: concepts carry a semantic tag "
                     "(disorder, finding, procedure, body structure, ...) and are primitive or fully defined; "
                     "a disorder never matches a body structure or a procedure."),
    "icd11.owl": ("icd_note", "ICD-11 is a statistical classification: categories are mutually exclusive buckets, "
                  "residual categories ('other specified', 'unspecified') collect leftovers, and the Foundation layer is "
                  "richer than the MMS linearization used for coding."),
    "ICD10CM.ttl": ("icd_note", "ICD-10-CM is a US clinical modification of WHO ICD-10: codes extend WHO categories with "
                    "more specific subcategories; inclusion notes are synonyms of the category, not separate concepts."),
}


def build_all() -> dict[str, dict]:
    files = ontology_files(discover())
    out: dict[str, dict] = {}
    with stage("ontology_profiles", total=len(files)) as bar:
        for path in files:
            p = build_profile(path)
            for key, (field, note) in SPECIAL_NOTES.items():
                if key in path.name:
                    p[field] = note
            p["summary"] = summary_text(p)
            out[p["ontology"]] = p
            bar.advance()
    (data_dir() / "ontology_profiles.json").write_text(json.dumps(out, indent=1))
    return out


def profile_questions(profiles: dict[str, dict]) -> list[dict]:
    """SDFT knowledge prompts about each ontology's modelling conventions (teacher sees the profile)."""
    rows = []
    for key, p in profiles.items():
        context = p["summary"]
        for kind, question in (
            ("formalism", f"What kind of artefact is '{p['title']}' (OWL ontology or SKOS thesaurus), and what does its hierarchy relation mean for alignment?"),
            ("conventions", f"Which modelling conventions of '{p['title']}' should an ontology matcher respect?"),
        ):
            rows.append({"id": hashlib.sha1(f"profile|{key}|{kind}".encode()).hexdigest()[:16], "task": "profile_qa",
                         "ontology": key, "split": "train", "prompt": question, "answer": context,
                         "privileged_context": f"Modelling profile:\n{context}"})
    return rows


def main() -> int:
    profiles = build_all()
    rows = profile_questions(profiles)
    with (data_dir() / "facts" / "profile_facts.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"ontologies": len(profiles), "profile_prompts": len(rows)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

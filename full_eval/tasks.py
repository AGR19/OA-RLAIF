"""The full-alignment tasks and what "every class" means for each.

Tasks are the same dataset families as the earlier held-out evaluation (outputs/eval/report.md, Table 1), now as whole
alignments: Anatomy, Bio-ML (three whole-ontology pairs), MultiFarm, DH, DH multilingual, Biodiv, the zero-shot pairs,
the WHO ICD-10 -> ICD-11 crosswalk, and the complete SNOMED CT -> ICD-11 map.

Every class of a task's source ontology is decided by the model (C# or NONE), except entities that are not classes of
the ontology itself (owl:Thing, oboInOwl / OWL / RDF / SKOS vocabulary, OBO relation placeholders such as
UNDEFINED_part_of) and deprecated classes (owl:deprecated true, or an 'obsolete' label). OAEI removes the same
vocabulary correspondences before scoring, and Bio-ML drops correspondences with a deprecated endpoint.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from oa_rlaif.data.families import PairSpec, discover
from oa_rlaif.io import read_yaml
from oa_rlaif.paths import config_dir, data_dir, out_dir, relative_to_raw

META_PREFIXES = (
    "http://www.w3.org/2002/07/owl#", "http://www.geneontology.org/formats/oboInOwl#",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/2004/02/skos/core#", "http://purl.org/dc/elements/1.1/", "http://purl.org/dc/terms/",
    "http://www.w3.org/2001/XMLSchema#", "http://protege.stanford.edu/plugins/owl/protege#",
)
OBSOLETE = re.compile(r"^\s*obsolete[\s_:]", re.I)
EQUIVALENCE = {"=", "<=", ">=", ""}  # Bio-ML's <= / >= are equivalences on the repaired basis (as in training)

# Report columns, in the order of the earlier Table 1.
COLUMNS = ["Anatomy", "Bio-ML", "MultiFarm", "DH", "DH-ML", "Biodiv", "ICD-10→11", "SNOMED→ICD-11", "Zero-shot"]
# The two very large tasks (whole-ontology pairs); everything else is "small" (minutes per model).
GIANT = {"bioml__NCIT-DOID", "bioml__SNOMED-FMA", "bioml__SNOMED-NCIT", "icd__snomed-to-icd11"}
# ICD's own scope for SNOMED CT: the SNOMED CT -> ICD-10 map covers these three hierarchies (clinical finding, event,
# situation with explicit context). FULL_EVAL_SNOMED_SCOPE=all maps every active concept instead.
SNOMED_ICD_ROOTS = {"404684003": "Clinical finding", "272379006": "Event", "243796009": "Situation with explicit context"}


@dataclass
class Task:
    slug: str                 # folder name, e.g. "anatomy__mouse-human", "multifarm", "icd__snomed-to-icd11"
    family: str
    column: str
    specs: list               # PairSpec list (empty for the two ICD tasks)
    giant: bool = False

    @property
    def pair_ids(self) -> list[str]:
        return [s.pair_id for s in self.specs] or [f"icd/{self.slug.split('__', 1)[1]}"]


def zero_shot_pairs() -> set[str]:
    return set(read_yaml(config_dir() / "data.yaml").get("zero_shot_pairs", []))


def tasks(families: set[str] | None = None) -> list[Task]:
    """Every task, in the order of the earlier report. Complex-OM pairs other than the zero-shot cree-swo are not
    part of it: their references are complex (EDOAL) correspondences that a C#/NONE decision cannot express."""
    specs = discover()
    zero = zero_shot_pairs()
    by_family: dict[str, list[PairSpec]] = {}
    for s in specs:
        if s.family == "complex_om" and s.pair_id not in zero:
            continue
        if s.family == "icd":
            continue
        if s.source is None or s.target is None:
            continue
        by_family.setdefault(s.family, []).append(s)
    out: list[Task] = []
    for s in by_family.get("anatomy", []):
        out.append(Task("anatomy__mouse-human", "anatomy", "Anatomy", [s]))
    for s in by_family.get("bioml", []):
        slug = "bioml__" + s.pair_id.split("/", 1)[1]
        out.append(Task(slug, "bioml", "Bio-ML", [s], giant=True))
    for family, column in (("multifarm", "MultiFarm"), ("dh", "DH"), ("dh_multiling", "DH-ML"), ("biodiv", "Biodiv")):
        if by_family.get(family):
            out.append(Task(family, family, column, by_family[family]))
    for family in ("complex_om", "cyber"):
        for s in by_family.get(family, []):
            out.append(Task(family + "__" + s.pair_id.split("/")[-1], family, "Zero-shot", [s]))
    out.append(Task("icd__who-icd10-to-icd11", "icd10", "ICD-10→11", []))
    out.append(Task("icd__snomed-to-icd11", "snomed_icd11", "SNOMED→ICD-11", [], giant=True))
    return [t for t in out if families is None or t.family in families or t.slug in families]


def ontology_paths(task_list: list[Task]) -> list[Path]:
    """Every ontology file the tasks read (sources, targets, and ICD-11 for the two ICD tasks)."""
    from oa_rlaif.data.icd_triangulation import icd_dir

    files = {p for t in task_list for s in t.specs for p in (s.source, s.target) if p}
    if any(t.family in ("icd10", "snomed_icd11") for t in task_list):
        files.add(icd_dir() / "icd11.owl")
    return sorted(files)


# ---- parsed ontologies -------------------------------------------------------------------------------------------------
def inputs_dir() -> Path:
    """Where full_eval/pull_inputs.py puts the private inputs repo (raw folders at its root)."""
    return data_dir() / "full_eval_inputs"


def load_profile_any(path: Path):
    """The parsed ontology: the local profile cache (data/profiles), else the packed cache from the inputs repo
    (profiles/index.json maps the raw-relative path to its cache folder; cache keys contain file mtimes, which a
    download does not keep), else parse the file now and cache it."""
    from oa_rlaif.data.profiles import cache_dir, is_cached, load_profile
    from oa_rlaif.ontology.parser import OntologyParser

    if is_cached(path):
        return load_profile(path)
    index_path = inputs_dir() / "profiles" / "index.json"
    if index_path.exists():
        entry = json.loads(index_path.read_text()).get(relative_to_raw(path))
        if entry and entry["size"] == path.stat().st_size and (inputs_dir() / "profiles" / entry["key"] / "profile.yaml").exists():
            return OntologyParser().load(inputs_dir() / "profiles" / entry["key"])
    profile = OntologyParser().parse(path, relative_to_raw(path))
    OntologyParser().save(profile, cache_dir(path))
    return profile


# ---- which entities are "every class" ----------------------------------------------------------------------------------
def deprecated_iris(path: Path) -> set[str]:
    """IRIs marked owl:deprecated true, read by streaming the raw file (the parser does not keep this flag).
    RDF/XML: the rdf:about of the enclosing element (entity references like &obo; expanded); functional syntax:
    AnnotationAssertion(owl:deprecated <iri> "true"). Cached under outputs/full_eval/cache/deprecated/."""
    cache = out_dir() / "full_eval" / "cache" / "deprecated" / (relative_to_raw(path).replace("/", "__") + ".json")
    if cache.exists():
        return set(json.loads(cache.read_text()))
    found: set[str] = set()
    entities: dict[str, str] = {}
    prefixes: dict[str, str] = {}
    about = None
    entity_decl = re.compile(r'<!ENTITY\s+(\w[\w.-]*)\s+"([^"]*)"')
    about_attr = re.compile(r'rdf:about="([^"]*)"')
    rdf_deprecated = re.compile(r"<owl:deprecated[^>]*>\s*true\s*</owl:deprecated>", re.I)
    functional = re.compile(r'AnnotationAssertion\(\s*owl:deprecated\s+(\S+)\s+"true"', re.I)
    prefix_decl = re.compile(r"Prefix\(\s*([\w-]*):?=<([^>]*)>\)")

    def expand(value: str) -> str:
        for name, iri in entities.items():
            value = value.replace(f"&{name};", iri)
        return value

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "<!ENTITY" in line:
                for name, iri in entity_decl.findall(line):
                    entities[name] = iri
            if line.startswith("Prefix("):
                for name, iri in prefix_decl.findall(line):
                    prefixes[name] = iri
            match = about_attr.search(line)
            if match:
                about = expand(match.group(1))
            if "deprecated" not in line:
                continue
            if rdf_deprecated.search(line) and about:
                found.add(about)
            for value in functional.findall(line):
                if value.startswith("<"):
                    found.add(value[1:-1])
                elif ":" in value:
                    prefix, tail = value.split(":", 1)
                    if prefix in prefixes:
                        found.add(prefixes[prefix] + tail)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(sorted(found)))
    return found


def is_meta(iri: str) -> bool:
    return iri.startswith(META_PREFIXES) or iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1].startswith("UNDEFINED_")


def scope(profile, deprecated: set[str]) -> list[str]:
    """The entities to align: declared classes and properties of the ontology, without vocabulary/placeholder
    entities and without deprecated ones. Sorted, so every model sees the same order."""
    out = []
    for iri, e in profile.entities.items():
        if not e.declared or is_meta(iri) or iri in deprecated:
            continue
        if any(OBSOLETE.match(label.text) for label in e.labels):
            continue
        out.append(iri)
    return sorted(out)


# ---- references ---------------------------------------------------------------------------------------------------------
def reference_pairs(spec: PairSpec, source, target) -> set[tuple[str, str]]:
    """Atomic equivalence correspondences of the pair's reference(s), oriented source -> target. Non-equivalence
    relations are left out, as OAEI does for Anatomy and DH ("only equivalence relationships")."""
    from oa_rlaif.ontology.alignment_io import orient, read_alignment

    pairs = set()
    for ref in spec.references:
        for a in read_alignment(ref):
            if not a.atomic or a.relation not in EQUIVALENCE:
                continue
            a = orient(a, source, target)
            if a.status != "unresolved_endpoint":
                pairs.add(a.pair)
    return pairs


def split_of() -> dict[str, str]:
    """Training split of every entity that has a gold link (data/splits/components.tsv): 'train' or 'test'."""
    from oa_rlaif.io import read_tsv

    path = data_dir() / "splits" / "components.tsv"
    return {row["iri"]: row["split"] for row in read_tsv(path)} if path.exists() else {}


def unit_name(model: str, task: Task) -> str:
    return f"{model}/{task.slug}"

"""ICD-11 <- WHO ICD-10 <- SNOMED CT triangulation, ICD crosswalk episodes and SNOMED RF2 facts.

Legs
  A  SNOMED s -> WHO ICD-10 c   RF2 extended map refset 447562003 (US 2026-03 release)
       strong: active, mapGroup 1, mapPriority 1, mapRule TRUE, category 447637006 (properly classified)
       weak:   any other rule (IFA / OTHERWISE / context dependent); "cannot be classified" rows are skipped
  B  WHO ICD-10 c -> ICD-11 m     WHO 2026-Jan tables
       strong: 10To11MapToOneCategory (one category), relation from foundation_10To11 (Equivalent/Subclass)
       weak:   10To11MapToMultipleCategories (c split across several ICD-11 categories)

A composed link (m, s) is typed `classified_under` (SSSOM: m skos:narrowMatch s). It is promoted to silver
`exactMatch` only when both legs are strong and unique (c has one strong SNOMED source and one ICD-11 target)
and the SNOMED preferred term and the ICD-11 title agree lexically. Nothing here is presented as clinical
equivalence; every link carries its legs and rule texts as evidence.

Bridge split: 20% of ICD-10 three-character categories are held out. Their links are test-only, so the
bridge-recovery evaluation asks the model for ICD-11 classifications it never saw for those categories.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from ..io import atomic_open
from ..ontology.alignment_io import SSSOM_COLUMNS, write_sssom
from ..ontology.models import Entity, Label
from ..paths import raw_dir, relative_to_raw
from ..progress import stage
from ..retrieval.candidates import CandidateGenerator, normalize
from .cards import card
from .profiles import load_profile
from .splits import stable_fraction

SCT = "http://snomed.info/id/"
ICD10 = "http://id.who.int/icd/release/10/"
WHO_MAP = "447562003"
CM_MAP = "6011000124106"
PROPERLY_CLASSIFIED = "447637006"
CANNOT_CLASSIFY = "447638001"
FSN, SYNONYM, DEFINITION_TYPE = "900000000000003001", "900000000000013009", "900000000000550004"
IS_A = "116680003"
US_LANG, PREFERRED = "900000000000509007", "900000000000548007"
ASSOCIATION_NAMES = {"900000000000527005": "SAME AS", "900000000000526001": "REPLACED BY",
                     "900000000000523009": "POSSIBLY EQUIVALENT TO", "900000000000528000": "WAS A"}
SIMPLE_MAPS = {"900000000000497000": "CTV3", "446608001": "ICD-O"}


def icd_dir() -> Path:
    return raw_dir() / "ICD-10 to SNOWMEDCT"


def rf2_dir() -> Path:
    return next(icd_dir().glob("SnomedCT_*"))


def _rf2(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)


def _table(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = csv.reader(f, delimiter="\t")
        header = next(rows)
        for row in rows:
            yield header, row


def clean_code(code: str) -> str:
    return code.strip().rstrip("?").replace("†", "").replace("*", "").strip()


def bridge_split(icd10_code: str, test_fraction: float = 0.2) -> str:
    category = clean_code(icd10_code)[:3]
    return "test" if stable_fraction(category, "icd-bridge") < test_fraction else "train"


# ------------------------------------------------------------------ WHO tables (leg B)

def read_who(counts: dict) -> tuple[dict, dict, dict]:
    """Returns (c -> {m: strength}), ICD-10 titles, ICD-11 titles/codes."""
    mapping = Path(icd_dir() / "ICD 10 to 11 Mapings" / "mapping")
    leg_b: dict[str, dict[str, str]] = defaultdict(dict)
    icd10_title: dict[str, str] = {}
    icd11_info: dict[str, dict] = {}
    foundation_relation: dict[tuple[str, str], str] = {}
    for header, row in _table(mapping / "foundation_10To11MapToOneCategory.txt"):
        counts[relative_to_raw(mapping / "foundation_10To11MapToOneCategory.txt")] += 1
        if len(row) >= 8 and row[2]:
            foundation_relation[(row[2], row[5])] = row[7]
    for name, strength in (("10To11MapToOneCategory.txt", "strong"), ("10To11MapToMultipleCategories.txt", "weak")):
        for header, row in _table(mapping / name):
            counts[relative_to_raw(mapping / name)] += 1
            record = dict(zip([h.strip() for h in header], row))
            code = record.get("icd10Code", "").strip()
            uri = (record.get("ICD-11 FoundationURI") or record.get("ICD-11 Foundation URI") or "").strip()
            if not code or not uri:
                continue
            icd10_title[code] = record.get("icd10Title", "")
            icd11_info.setdefault(uri, {"code": record.get("icd11Code", ""), "title": record.get("icd11Title", ""),
                                        "chapter": record.get("icd11Chapter", "")})
            relation = foundation_relation.get((code, uri), "")
            current = leg_b[code].get(uri)
            value = "strong" if strength == "strong" else "weak"
            if relation:
                value += f":{relation}"
            if current is None or current.startswith("weak"):
                leg_b[code][uri] = value
    for name in ("11To10MapToOneCategory.txt", "foundation_11To10MapToOneCategory.txt"):
        for header, row in _table(mapping / name):
            counts[relative_to_raw(mapping / name)] += 1
            record = dict(zip([h.strip() for h in header], row))
            uri = (record.get("Foundation URI") or "").strip()
            if uri and uri not in icd11_info:
                icd11_info[uri] = {"code": record.get("icd11Code", ""), "title": record.get("icd11Title", ""),
                                   "chapter": record.get("icd11Chapter", "")}
    readme = mapping / "readme.txt"
    if readme.exists():
        counts[relative_to_raw(readme)] += 1
    return leg_b, icd10_title, icd11_info


KEY_COLUMNS = ("icd10code", "icd11code", "foundationuri", "icd11foundationuri", "linearizationreleaseuri",
               "linearizationuri", "linearization(releaseuri)")


def crosscheck_xlsx(counts: dict) -> dict:
    """Every .xlsx table must carry the same MAPPINGS as its .txt twin; per-column differences are reported.

    Mapping-defining columns (codes, Foundation/linearization URIs) must be identical row by row; metadata columns
    (e.g. depth) may differ between the two WHO exports and are listed with counts, never silently ignored.
    """
    from openpyxl import load_workbook

    mapping = Path(icd_dir() / "ICD 10 to 11 Mapings" / "mapping")
    report = {}
    for xlsx in sorted(mapping.glob("*.xlsx")):
        sheet = load_workbook(xlsx, read_only=True).active
        x_rows = [tuple("" if v is None else str(v).strip() for v in r) for r in sheet.iter_rows(values_only=True)]
        counts[relative_to_raw(xlsx)] += len(x_rows)
        with xlsx.with_suffix(".txt").open(encoding="utf-8-sig") as f:
            t_rows = [tuple(c.strip() for c in line.rstrip("\n").split("\t")) for line in f]
        header = [h.lower().replace(" ", "").replace("-", "") for h in t_rows[0]]
        key_idx = [i for i, h in enumerate(header) if any(k in h for k in KEY_COLUMNS)]
        diffs: dict[str, int] = {}
        key_mismatch = 0
        for a, b in zip(x_rows[1:], t_rows[1:]):
            if any((a[i] if i < len(a) else "") != (b[i] if i < len(b) else "") for i in key_idx):
                key_mismatch += 1
            for i in range(max(len(a), len(b))):
                u, v = (a[i] if i < len(a) else ""), (b[i] if i < len(b) else "")
                if u != v:
                    name = t_rows[0][i] if i < len(t_rows[0]) else f"unnamed_col_{i}"
                    diffs[name] = diffs.get(name, 0) + 1
        report[xlsx.name] = {"xlsx_rows": len(x_rows) - 1, "txt_rows": len(t_rows) - 1,
                             "key_columns": [t_rows[0][i] for i in key_idx], "key_mismatched_rows": key_mismatch,
                             "column_differences": diffs, "mappings_identical": key_mismatch == 0 and len(x_rows) == len(t_rows)}
    return report


# ------------------------------------------------------------------ SNOMED RF2

class Snomed:
    def __init__(self, counts: dict, sample_concepts: int, seed: int):
        self.counts = counts
        self.root = rf2_dir() / "Snapshot"
        self.rng = random.Random(seed)
        self.sample_concepts = sample_concepts

    def count(self, path: Path) -> None:
        self.counts[relative_to_raw(path)] += 0

    def stream(self, path: Path):
        key = relative_to_raw(path)
        for row in _rf2(path):
            self.counts[key] += 1
            yield row

    def load(self, map_concepts: set[str]):
        term = self.root / "Terminology"
        active = []
        for row in self.stream(next(term.glob("sct2_Concept_*"))):
            if row["active"] == "1":
                active.append(row["id"])
        self.active = set(active)
        self.sample = set(self.rng.sample(active, min(self.sample_concepts, len(active))))
        need = set(map_concepts) | self.sample
        self.parents: dict[str, set[str]] = defaultdict(set)
        self.attributes: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for row in self.stream(next(term.glob("sct2_Relationship_Snapshot*"))):
            if row["active"] == "1" and row["sourceId"] in need:
                if row["typeId"] == IS_A:
                    self.parents[row["sourceId"]].add(row["destinationId"])
                elif row["sourceId"] in self.sample:
                    self.attributes[row["sourceId"]].append((row["typeId"], row["destinationId"]))
        self.stated = sum(1 for _ in self.stream(next(term.glob("sct2_StatedRelationship_*"))))
        self.concrete: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for row in self.stream(next(term.glob("sct2_RelationshipConcreteValues_*"))):
            if row["active"] == "1" and row["sourceId"] in self.sample:
                self.concrete[row["sourceId"]].append((row["typeId"], row["value"]))
        self.owl: dict[str, str] = {}
        for row in self.stream(next(term.glob("sct2_sRefset_OWLExpression*"))):
            if row["active"] == "1" and row["referencedComponentId"] in self.sample:
                self.owl[row["referencedComponentId"]] = row["owlExpression"]
        self.ident = sum(1 for _ in self.stream(next(term.glob("sct2_Identifier_*"))))
        # history
        content = self.root / "Refset" / "Content"
        self.replaced: dict[str, tuple[str, str]] = {}
        for row in self.stream(next(content.glob("der2_cRefset_Association*"))):
            if row["active"] == "1" and row["refsetId"] in ASSOCIATION_NAMES:
                self.replaced.setdefault(row["referencedComponentId"], (ASSOCIATION_NAMES[row["refsetId"]], row["targetComponentId"]))
        self.inactivation = sum(1 for _ in self.stream(next(content.glob("der2_cRefset_AttributeValue*"))))
        self.simple_refsets: dict[str, set[str]] = defaultdict(set)
        for row in self.stream(next(content.glob("der2_Refset_Simple*"))):
            if row["active"] == "1" and row["referencedComponentId"] in self.sample:
                self.simple_refsets[row["referencedComponentId"]].add(row["refsetId"])
        self.simple_maps: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for row in self.stream(next((self.root / "Refset" / "Map").glob("der2_sRefset_SimpleMap*"))):
            if row["active"] == "1" and row["refsetId"] in SIMPLE_MAPS and row["referencedComponentId"] in self.sample:
                self.simple_maps[row["referencedComponentId"]].append((SIMPLE_MAPS[row["refsetId"]], row["mapTarget"]))
        # preferred terms (US English language refset) and all descriptions for every concept we name
        preferred: set[str] = set()
        for row in self.stream(next((self.root / "Refset" / "Language").glob("der2_cRefset_Language*"))):
            if row["active"] == "1" and row["refsetId"] == US_LANG and row["acceptabilityId"] == PREFERRED:
                preferred.add(row["referencedComponentId"])
        names = need | {t for v in self.attributes.values() for t, d in v} | {d for v in self.attributes.values() for t, d in v}
        names |= {p for v in self.parents.values() for p in v} | {t for v in self.concrete.values() for t, _ in v}
        names |= {r for v in self.simple_refsets.values() for r in v} | {t for _, t in self.replaced.values()}
        names |= set(self.replaced)
        self.fsn: dict[str, str] = {}
        self.pt: dict[str, str] = {}
        self.synonyms: dict[str, list[str]] = defaultdict(list)
        for row in self.stream(next(term.glob("sct2_Description_*"))):
            if row["active"] != "1" or row["conceptId"] not in names:
                continue
            if row["typeId"] == FSN:
                self.fsn[row["conceptId"]] = row["term"]
            elif row["id"] in preferred:
                self.pt[row["conceptId"]] = row["term"]
            else:
                self.synonyms[row["conceptId"]].append(row["term"])
        self.definitions: dict[str, str] = {}
        for row in self.stream(next(term.glob("sct2_TextDefinition_*"))):
            if row["active"] == "1" and row["conceptId"] in need:
                self.definitions[row["conceptId"]] = row["term"]
        for path in (self.root / "Refset" / "Metadata").glob("*.txt"):
            sum(1 for _ in self.stream(path))
        full = rf2_dir() / "Full"
        self.full_rows = {relative_to_raw(p): sum(1 for _ in self.stream(p)) for p in full.rglob("*.txt")}
        for extra in rf2_dir().glob("*"):
            if extra.is_file() and extra.name != ".DS_Store":
                self.counts[relative_to_raw(extra)] += 1

    def name(self, cid: str) -> str:
        return self.pt.get(cid) or self.fsn.get(cid, cid).rsplit(" (", 1)[0]

    def entity(self, cid: str) -> Entity:
        e = Entity(SCT + cid, declared=True)
        e.labels.add(Label(self.name(cid), "en", "prefLabel"))
        for s in self.synonyms.get(cid, [])[:8]:
            e.labels.add(Label(s, "en", "altLabel"))
        if cid in self.definitions:
            e.definitions.add(Label(self.definitions[cid], "en", "definition"))
        e.parents = {SCT + p for p in self.parents.get(cid, ())}
        return e


def read_leg_a(counts: dict) -> tuple[dict, dict, dict]:
    """SNOMED -> WHO ICD-10 (strong/weak) and the auxiliary ICD-10-CM map."""
    path = next((rf2_dir() / "Snapshot" / "Refset" / "Map").glob("der2_iisssccRefset_ExtendedMap*"))
    key = relative_to_raw(path)
    leg_a: dict[str, dict[str, dict]] = defaultdict(dict)
    cm: dict[str, set[str]] = defaultdict(set)
    for row in _rf2(path):
        counts[key] += 1
        if row["active"] != "1":
            continue
        if row["refsetId"] == CM_MAP and row["mapTarget"]:
            cm[row["referencedComponentId"]].add(clean_code(row["mapTarget"]))
            continue
        if row["refsetId"] != WHO_MAP or row["mapCategoryId"] == CANNOT_CLASSIFY or not row["mapTarget"]:
            continue
        code = clean_code(row["mapTarget"])
        strong = (row["mapGroup"] == "1" and row["mapPriority"] == "1" and row["mapRule"] == "TRUE"
                  and row["mapCategoryId"] == PROPERLY_CLASSIFIED)
        entry = leg_a[row["referencedComponentId"]].get(code)
        if entry is None or (strong and not entry["strong"]):
            leg_a[row["referencedComponentId"]][code] = {"strong": strong, "rule": row["mapRule"],
                                                         "advice": row["mapAdvice"], "group": row["mapGroup"],
                                                         "priority": row["mapPriority"]}
    return leg_a, cm, {}


def triangulate(leg_a, leg_b, sct: Snomed, icd11_info) -> list[dict]:
    sources_per_code: dict[str, set[str]] = defaultdict(set)
    for s, codes in leg_a.items():
        for c, info in codes.items():
            if info["strong"]:
                sources_per_code[c].add(s)
    links = []
    for s, codes in leg_a.items():
        for c, a in codes.items():
            targets = leg_b.get(c) or leg_b.get(c.split(".")[0]) or {}
            for m, b in targets.items():
                strong_b = b.startswith("strong")
                unique = len(sources_per_code.get(c, ())) == 1 and len(targets) == 1
                title = icd11_info.get(m, {}).get("title", "")
                similarity = difflib.SequenceMatcher(None, normalize(sct.name(s)), normalize(title)).ratio()
                exact = a["strong"] and strong_b and unique and similarity >= 0.85
                confidence = (0.9 if a["strong"] else 0.5) * (0.9 if strong_b else 0.6)
                links.append({
                    "icd11": m, "icd11_code": icd11_info.get(m, {}).get("code", ""), "icd11_title": title,
                    "snomed": SCT + s, "snomed_label": sct.name(s), "icd10": c,
                    "relation": "exactMatch" if exact else "classified_under",
                    "confidence": round(confidence * (0.5 + 0.5 * similarity) if not exact else 0.9 * similarity, 4),
                    "leg_a": {"strong": a["strong"], "rule": a["rule"], "advice": a["advice"]},
                    "leg_b": b, "label_similarity": round(similarity, 3),
                    "split": bridge_split(c),
                })
    return links


class _Profile:  # minimal OntologyProfile duck type for retrieval/cards (picklable)
    def __init__(self, entities):
        self.ontology_id, self.entities, self.metadata = "icd11-who-tables", entities, {}


# Retrieval for classification targets. ICD categories are coarser than SNOMED concepts (classified_under), so the
# gold is often an ancestor of the lexically closest category. Settings were tuned on a 1,500-concept TRAIN sample
# (never on test): recall@20 0.224 titles only -> 0.227 + icd11.owl synonyms -> 0.535 + ancestor expansion
# -> 0.654 with SNOMED parent labels, decay 0.7, expansion from the top 50, depth 4.
EXPANSION = {"lexical_k": 60, "parent_k": 30, "parent_weight": 0.8, "expand_top": 50, "depth": 4, "decay": 0.7}


def _icd11_profile(icd11_info: dict, owl=None) -> _Profile:
    """ICD-11 targets from the WHO tables, enriched with icd11.owl labels, synonyms and (in-table) parents."""
    entities = {}
    for uri, info in icd11_info.items():
        e = Entity(uri, declared=True)
        e.labels.add(Label(info["title"], "en", "label"))
        if info["code"]:
            e.codes.add(info["code"])
        source = owl.entities.get(uri) if owl is not None else None
        if source is not None:
            for label in source.labels:
                if label.kind != "label":
                    e.labels.add(Label(label.text, "en", label.kind))
            for definition in list(source.definitions)[:1]:
                e.definitions.add(definition)
        entities[uri] = e
    if owl is not None:
        for uri, ancestors in _ancestors(icd11_info, owl, depth=1).items():
            entities[uri].parents = {a for a, _ in ancestors}
    return _Profile(entities)


def _ancestors(icd11_info: dict, owl, depth: int) -> dict[str, list[tuple[str, int]]]:
    """In-table ICD-11 ancestors up to `depth` Foundation levels (walking through Foundation-only nodes)."""
    parents = {i: list(e.parents) for i, e in owl.entities.items()}
    out = {}
    for uri in icd11_info:
        found, frontier, seen = [], [uri], {uri}
        for d in range(1, depth + 1):
            nxt = []
            for node in frontier:
                for q in parents.get(node, ()):
                    if q not in seen:
                        seen.add(q)
                        nxt.append(q)
                        if q in icd11_info:
                            found.append((q, d))
            frontier = nxt
        out[uri] = found
    return out


def icd_candidates(generator: CandidateGenerator, ancestors: dict, entity: Entity, parent: Entity | None, k: int) -> list[str]:
    x = EXPANSION
    scores: dict[str, float] = defaultdict(float)
    for c, s in generator.candidates(entity, k=x["lexical_k"]):
        scores[c] = max(scores[c], s)
    if parent is not None and parent.labels:
        for c, s in generator.candidates(parent, k=x["parent_k"]):
            scores[c] = max(scores[c], x["parent_weight"] * s)
    inherited: dict[str, float] = defaultdict(float)
    for c, s in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[: x["expand_top"]]:
        for a, d in ancestors.get(c, ()):
            if d <= x["depth"]:
                inherited[a] += s * x["decay"] ** d
    for a, v in inherited.items():
        scores[a] = max(scores.get(a, 0.0), v)
    return [c for c, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def _episode_chunk(args) -> list[dict]:
    """Worker: build ICD-11 retrieval once, then one episode per item."""
    icd11, ancestors, items, k, seed = args
    generator = CandidateGenerator(icd11)
    out = []
    for pair_id, kind, source_entity, parent_entity, source_card, gold, split, extra in items:
        rng = random.Random(f"{seed}|{pair_id}|{source_entity.iri}")
        cands = icd_candidates(generator, ancestors, source_entity, parent_entity, k)
        retrieved = any(g in cands for g in gold)
        if split == "train":
            missing = [g for g in gold if g not in cands]
            if missing and rng.random() < 0.7:
                cands = cands[: k - len(missing)] + missing
        present = [g for g in gold if g in cands]
        order = list(cands)
        random.Random(hashlib.sha1(f"{pair_id}|{source_entity.iri}".encode()).hexdigest()).shuffle(order)
        out.append({"id": hashlib.sha1(f"{pair_id}|{source_entity.iri}".encode()).hexdigest()[:16],
                    "family": "icd", "pair_id": pair_id, "split": split, "kind": kind,
                    "source_iri": source_entity.iri, "source_card": source_card,
                    "candidates": [{"iri": c, "card": card(icd11, c)} for c in order],
                    "gold": present, "gold_all": sorted(gold), "gold_retrieved": retrieved,
                    "relation": extra.pop("relation", "classified_under"),
                    "none_reason": "" if present else "retrieval_miss", "hard_negatives": [],
                    "source_lang": "en", "target_lang": "en", "provenance": extra})
    return out


def _parallel_episodes(icd11: _Profile, ancestors: dict, items: list, k: int, seed: int,
                       workers: int | None = None) -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor

    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    size = max(1, len(items) // (workers * 4))
    chunks = [items[i:i + size] for i in range(0, len(items), size)]
    episodes: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool, stage("icd_episodes", total=len(chunks)) as bar:
        for result in pool.map(_episode_chunk, [(icd11, ancestors, chunk, k, seed) for chunk in chunks]):
            episodes.extend(result)
            bar.advance()
    return episodes


def build(base: Path, assigner=None, seed: int = 13, sample_concepts: int = 20000,
          max_silver_episodes: int = 40000, k: int = 20) -> dict[str, tuple[int, int]]:
    counts: dict[str, int] = defaultdict(int)
    training: dict[str, int] = defaultdict(int)
    out = base / "icd"
    out.mkdir(parents=True, exist_ok=True)
    with stage("icd_triangulation", total=8) as bar:
        leg_b, icd10_title, icd11_info = read_who(counts)
        bar.advance()
        xlsx = crosscheck_xlsx(counts)
        (out / "xlsx_crosscheck.json").write_text(json.dumps(xlsx, indent=1))
        bar.advance()
        leg_a, cm, _ = read_leg_a(counts)
        bar.advance()
        sct = Snomed(counts, sample_concepts, seed)
        sct.load(set(leg_a))
        bar.advance()
        links = triangulate(leg_a, leg_b, sct, icd11_info)
        with atomic_open(out / "triangulated.jsonl") as f:
            for link in links:
                f.write(json.dumps(link, ensure_ascii=False) + "\n")
        predicate = {"exactMatch": "skos:exactMatch", "classified_under": "skos:narrowMatch"}
        write_sssom(out / "icd11_snomed.sssom.tsv", ({
            "subject_id": link["icd11"], "subject_label": link["icd11_title"],
            "predicate_id": predicate[link["relation"]], "object_id": link["snomed"],
            "object_label": link["snomed_label"], "mapping_justification": "semapv:CompositeMatching",
            "confidence": link["confidence"], "mapping_tool": "oa-rlaif icd_triangulation",
            "comment": f"via ICD-10 {link['icd10']}; legA {'strong' if link['leg_a']['strong'] else 'weak'} "
                       f"({link['leg_a']['rule'][:40]}); legB {link['leg_b']}"} for link in links),
            {"mapping_set_id": "https://github.com/AGR19/OA-RLAIF/icd11_snomed_triangulated",
             "license": "SNOMED CT affiliate licence applies; private use only",
             "subject_source": "ICD-11 Foundation (WHO 2026-01 tables)", "object_source": "SNOMED CT US 2026-03",
             "comment": "Composed through WHO ICD-10. classified_under is NOT clinical equivalence."})
        stats = {"links": len(links), "exact": sum(l["relation"] == "exactMatch" for l in links),
                 "classified_under": sum(l["relation"] == "classified_under" for l in links),
                 "strong_both": sum(l["leg_a"]["strong"] and l["leg_b"].startswith("strong") for l in links),
                 "snomed_concepts": len({l["snomed"] for l in links}), "icd11_targets": len({l["icd11"] for l in links}),
                 "test_links": sum(l["split"] == "test" for l in links), "xlsx_crosscheck": xlsx,
                 "icd10cm_auxiliary_concepts": len(cm), "full_release_rows": sum(sct.full_rows.values())}
        (out / "triangulation_stats.json").write_text(json.dumps(stats, indent=1))
        bar.advance()

        # (a) WHO crosswalk ICD-10 -> ICD-11 and (b) silver SNOMED -> ICD-11, retrieved in parallel workers
        owl = load_profile(icd_dir() / "icd11.owl")
        icd11 = _icd11_profile(icd11_info, owl)
        ancestors = _ancestors(icd11_info, owl, EXPANSION["depth"])
        items = []
        for code, targets in sorted(leg_b.items()):
            e = Entity(ICD10 + code, declared=True)
            e.labels.add(Label(icd10_title.get(code, code), "en", "label"))
            e.codes.add(code)
            category = code.split(".")[0]
            parent = None
            source_card = f"IRI: {ICD10 + code}\nname: {icd10_title.get(code, code)}"
            if category != code and category in icd10_title:
                parent = Entity(ICD10 + category, declared=True)
                parent.labels.add(Label(icd10_title[category], "en", "label"))
                source_card += f"\nparents: {icd10_title[category]}"
            source_card += f"\ncodes: {code}"
            items.append(("icd/who-icd10-to-icd11", "gold", e, parent, source_card, sorted(targets), bridge_split(code),
                          {"references": ["ICD-10 to SNOWMEDCT/ICD 10 to 11 Mapings/mapping/10To11MapToOneCategory.txt"],
                           "relation": "classified_under"}))
        by_snomed: dict[str, set[str]] = defaultdict(set)
        split_of: dict[str, str] = {}
        for link in links:
            if link["leg_a"]["strong"] and link["leg_b"].startswith("strong"):
                by_snomed[link["snomed"]].add(link["icd11"])
                split_of[link["snomed"]] = "test" if link["split"] == "test" or split_of.get(link["snomed"]) == "test" else "train"
        rng = random.Random(seed)
        chosen = sorted(by_snomed)
        if len(chosen) > max_silver_episodes:
            chosen = sorted(rng.sample(chosen, max_silver_episodes))
        map_file = relative_to_raw(next((rf2_dir() / "Snapshot/Refset/Map").glob("der2_iisssccRefset_ExtendedMap*")))
        for iri in chosen:
            entity = sct.entity(iri[len(SCT):])
            context = {iri: entity}
            parent = Entity(iri + "#parents", declared=True)
            for p in entity.parents:
                context[p] = Entity(p, declared=True)
                context[p].labels.add(Label(sct.name(p[len(SCT):]), "en", "prefLabel"))
                parent.labels.add(Label(sct.name(p[len(SCT):]), "en", "label"))
            source_card = card(_Profile(context), iri)
            items.append(("icd/snomed-to-icd11", "silver", entity, parent, source_card, sorted(by_snomed[iri]), split_of[iri],
                          {"references": [map_file], "relation": "classified_under"}))
        bar.advance()
        episodes = _parallel_episodes(icd11, ancestors, items, k, seed)
        with atomic_open(base / "episodes" / "icd.jsonl") as f:
            for e in episodes:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        bar.advance()

        # (c) SNOMED RF2 facts (closed-book knowledge; privileged context = the concept card)
        facts = []

        cards_cache: dict[str, str] = {}

        def fact(kind, cid, prompt, answer, source):
            if cid not in cards_cache:
                cards_cache[cid] = card(_Profile({SCT + cid: sct.entity(cid)}), SCT + cid)
            facts.append({"id": hashlib.sha1(f"rf2|{kind}|{cid}".encode()).hexdigest()[:16], "task": f"fact_{kind}",
                          "ontology": "SNOMED CT US 2026-03 (RF2)", "iri": SCT + cid,
                          "split": "test" if stable_fraction(cid, "rf2-facts") < 0.2 else "train",
                          "prompt": prompt, "answer": answer, "privileged_context": cards_cache[cid]})
            training[source] += 1

        term = rf2_dir() / "Snapshot" / "Terminology"
        src = {name: relative_to_raw(next(term.glob(f"{name}*"))) for name in
               ("sct2_Description_", "sct2_TextDefinition_", "sct2_Relationship_Snapshot", "sct2_RelationshipConcreteValues_",
                "sct2_sRefset_OWLExpression", "sct2_Concept_", "sct2_StatedRelationship_")}
        lang = relative_to_raw(next((rf2_dir() / "Snapshot/Refset/Language").glob("*")))
        simple = relative_to_raw(next((rf2_dir() / "Snapshot/Refset/Map").glob("der2_sRefset_SimpleMap*")))
        assoc = relative_to_raw(next((rf2_dir() / "Snapshot/Refset/Content").glob("der2_cRefset_Association*")))
        simple_ref = relative_to_raw(next((rf2_dir() / "Snapshot/Refset/Content").glob("der2_Refset_Simple*")))
        for cid in sorted(sct.sample):
            name = sct.name(cid)
            if sct.synonyms.get(cid):
                fact("synonyms", cid, f"In SNOMED CT, what other terms are used for '{name}'?", "; ".join(sct.synonyms[cid][:6]), src["sct2_Description_"])
                training[lang] += 1
            if cid in sct.fsn:
                fact("fsn", cid, f"What is the fully specified name of the SNOMED CT concept '{name}'?", sct.fsn[cid], src["sct2_Concept_"])
            if cid in sct.definitions:
                fact("definition", cid, f"How does SNOMED CT define '{name}'?", sct.definitions[cid], src["sct2_TextDefinition_"])
            if sct.parents.get(cid):
                fact("parents", cid, f"In SNOMED CT, what is '{name}' a kind of?", "; ".join(sct.name(p) for p in sorted(sct.parents[cid])[:4]), src["sct2_StatedRelationship_"])
            for type_id, dest in sct.attributes.get(cid, [])[:3]:
                fact(f"attr_{type_id}", cid, f"In SNOMED CT, what is the {sct.name(type_id).lower()} of '{name}'?", sct.name(dest), src["sct2_Relationship_Snapshot"])
            for type_id, value in sct.concrete.get(cid, [])[:2]:
                fact(f"value_{type_id}", cid, f"In SNOMED CT, what is the {sct.name(type_id).lower()} of '{name}'?", value.lstrip("#"), src["sct2_RelationshipConcreteValues_"])
            if cid in sct.owl:
                expression = sct.owl[cid]
                for token in set(t for t in expression.replace("(", " ").replace(")", " ").split() if t.startswith(":")):
                    expression = expression.replace(token, f"[{sct.name(token[1:])}]")
                fact("owl", cid, f"Give the SNOMED CT OWL axiom for '{name}'.", expression[:800], src["sct2_sRefset_OWLExpression"])
            for system, target in sct.simple_maps.get(cid, [])[:2]:
                fact(f"map_{system}", cid, f"What is the {system} code for the SNOMED CT concept '{name}'?", target, simple)
            refsets = sorted(sct.simple_refsets.get(cid, ()))
            if refsets:
                fact("refsets", cid, f"Which SNOMED CT simple reference sets include '{name}'?", "; ".join(sct.name(r) for r in refsets[:4]), simple_ref)
        for cid, (kind, target) in sorted(sct.replaced.items())[:5000]:
            fact("history", cid, f"The SNOMED CT concept {cid} ({sct.name(cid)}) is inactive. What is it associated with?",
                 f"{kind} {sct.name(target)} ({target})", assoc)
        with (base / "facts" / "snomed_rf2_facts.jsonl").open("w") as f:
            for item in facts:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        bar.advance()

    # training-record attribution for coverage
    who = "ICD-10 to SNOWMEDCT/ICD 10 to 11 Mapings/mapping/"
    for name in ("10To11MapToOneCategory.txt", "10To11MapToMultipleCategories.txt", "11To10MapToOneCategory.txt",
                 "foundation_10To11MapToOneCategory.txt", "foundation_11To10MapToOneCategory.txt"):
        training[who + name] += len(episodes)
    training[relative_to_raw(next((rf2_dir() / "Snapshot/Refset/Map").glob("der2_iisssccRefset_ExtendedMap*")))] += len(links)
    result = {key: (counts[key], training.get(key, 0)) for key in set(counts) | set(training)}
    recall: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for e in episodes:
        recall[f"{e['pair_id']}|{e['split']}"][0] += e["gold_retrieved"]
        recall[f"{e['pair_id']}|{e['split']}"][1] += 1
    stats["retrieval_recall_at_k"] = {key: round(hit / n, 4) for key, (hit, n) in sorted(recall.items())}
    stats["retrieval_settings"] = {"k": k, **EXPANSION}
    (out / "triangulation_stats.json").write_text(json.dumps(stats, indent=1))
    (out / "summary.json").write_text(json.dumps({"stats": stats, "episodes": len(episodes), "facts": len(facts)}, indent=1))
    return result

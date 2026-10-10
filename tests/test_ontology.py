"""Parser, alignment I/O and retrieval checks. Raw-data tests skip when the raw folder is absent."""

from __future__ import annotations

import pytest

from oa_rlaif.ontology.alignment_io import read_alignment, write_oaei_rdf, write_sssom
from oa_rlaif.ontology.models import Alignment
from oa_rlaif.ontology.parser import OntologyParser
from oa_rlaif.paths import raw_dir
from oa_rlaif.retrieval.candidates import normalize

ANATOMY = raw_dir() / "anatomy-dataset-2"
needs_raw = pytest.mark.skipif(not ANATOMY.exists(), reason="raw ontology folder not available")


def test_normalize_breaks_underscores_and_camel_case():
    assert normalize("Gray_Matter_of_Spinal_Cord") == "gray matter of spinal cord"
    assert normalize("hasRelatedSynonym") == "has related synonym"


def test_genid_synonyms_are_attached(tmp_path):
    owl = tmp_path / "t.owl"
    owl.write_text("""<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
 xmlns:owl="http://www.w3.org/2002/07/owl#" xmlns:oboInOwl="http://www.geneontology.org/formats/oboInOwl#">
 <owl:Class rdf:about="http://x#A"><rdfs:label>trunk</rdfs:label>
  <oboInOwl:hasRelatedSynonym rdf:resource="http://x#genid1"/>
  <oboInOwl:hasDefinition><rdf:Description><rdfs:label>the torso</rdfs:label></rdf:Description></oboInOwl:hasDefinition>
 </owl:Class>
 <rdf:Description rdf:about="http://x#genid1"><rdfs:label>body</rdfs:label></rdf:Description>
</rdf:RDF>""")
    profile = OntologyParser().parse(owl, "t")
    entity = profile.entities["http://x#A"]
    assert {label.text for label in entity.labels} == {"trunk", "body"}
    assert {label.text for label in entity.definitions} == {"the torso"}
    assert "http://x#genid1" not in profile.entities


def test_rdf_and_sssom_round_trip(tmp_path):
    rows = [Alignment("http://a#1", "http://b#2", "=", 0.9), Alignment("http://a#3", "http://b#4", "<", 0.5)]
    path = tmp_path / "out.rdf"
    assert write_oaei_rdf(path, rows) == 2
    back = list(read_alignment(path))
    assert [(a.source_iri, a.target_iri, a.relation) for a in back] == [
        ("http://a#1", "http://b#2", "="), ("http://a#3", "http://b#4", "<")]
    n = write_sssom(tmp_path / "m.sssom.tsv", [{"subject_id": "a:1", "predicate_id": "skos:exactMatch",
                                               "object_id": "b:2", "mapping_justification": "semapv:ManualMappingCuration"}],
                    {"mapping_set_id": "https://example.org/test"})
    assert n == 1
    assert (tmp_path / "m.sssom.tsv").read_text().startswith("#mapping_set_id")


@needs_raw
@pytest.mark.raw
def test_anatomy_counts_match_known_files():
    mouse = OntologyParser().parse(ANATOMY / "mouse.owl", "mouse")
    human = OntologyParser().parse(ANATOMY / "human.owl", "human")
    assert mouse.metadata["classes"] == 2744 and human.metadata["classes"] == 3304
    synonyms = sum(label.kind == "hasRelatedSynonym" for e in human.entities.values() for label in e.labels)
    assert synonyms > 5000
    refs = list(read_alignment(ANATOMY / "reference.rdf"))
    assert len(refs) == 1516


def test_properties_are_kept_with_kind_domain_range(tmp_path):
    owl = tmp_path / "p.owl"
    owl.write_text("""<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
 xmlns:owl="http://www.w3.org/2002/07/owl#">
 <owl:Class rdf:about="http://x#Paper"><rdfs:label>Paper</rdfs:label></owl:Class>
 <owl:Class rdf:about="http://x#Person"/>
 <owl:ObjectProperty rdf:about="http://x#writtenBy"><rdfs:label>written by</rdfs:label>
  <rdfs:domain rdf:resource="http://x#Paper"/><rdfs:range rdf:resource="http://x#Person"/>
  <rdfs:subPropertyOf rdf:resource="http://x#relatedTo"/></owl:ObjectProperty>
 <owl:ObjectProperty rdf:about="http://x#relatedTo"/>
 <owl:DatatypeProperty rdf:about="http://x#title"/>
</rdf:RDF>""")
    from oa_rlaif.data.cards import card, kind_of

    profile = OntologyParser().parse(owl, "p")
    assert profile.metadata["classes"] == 2 and profile.metadata["properties"] == 3
    assert kind_of(profile, "http://x#writtenBy") == "object property"
    assert kind_of(profile, "http://x#title") == "datatype property"
    assert kind_of(profile, "http://x#Paper") == "class"
    text = card(profile, "http://x#writtenBy")
    assert "domain: Paper" in text and "range: Person" in text and "parents: relatedTo" in text

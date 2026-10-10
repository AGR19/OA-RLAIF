"""Alignment output helpers and modelling-profile summaries."""

from __future__ import annotations

from oa_rlaif.data.ontology_profiles import summary_text
from oa_rlaif.eval.align import string_equiv
from oa_rlaif.ontology.parser import OntologyParser

OWL = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
 xmlns:owl="http://www.w3.org/2002/07/owl#">
 <owl:Class rdf:about="http://{p}#A"><rdfs:label>{a}</rdfs:label></owl:Class>
 <owl:Class rdf:about="http://{p}#B"><rdfs:label>{b}</rdfs:label></owl:Class>
</rdf:RDF>"""


def test_string_equiv_matches_normalised_labels(tmp_path):
    (tmp_path / "s.owl").write_text(OWL.format(p="s", a="Spinal_Cord", b="liver"))
    (tmp_path / "t.owl").write_text(OWL.format(p="t", a="spinal cord", b="Kidney"))
    s = OntologyParser().parse(tmp_path / "s.owl")
    t = OntologyParser().parse(tmp_path / "t.owl")
    assert string_equiv(s, t) == {("http://s#A", "http://t#A")}


def test_profile_summary_flags_thesaurus_and_residual_categories():
    profile = {"title": "Thes", "formalism": "SKOS thesaurus", "classes": 100, "properties": 0,
               "hierarchy": "skos:broader (thesaurus broader term; NOT logical subsumption)",
               "polyhierarchy_share": 0.1, "max_depth": 4, "part_of_restrictions": 0, "disjointness_axioms": 0,
               "residual_category_share": 0.05, "label_coverage": 1.0, "definition_coverage": 0.0,
               "languages": {"de": 90, "en": 10}}
    text = summary_text(profile)
    assert "NOT logical subsumption" in text and "residual categories" in text and "labels in de, en" in text

"""Diagnose split components: sizes, families spanned, and what zero-shot forcing pulls into test."""
import collections
import json

from oa_rlaif.data.build import SIMPLE_KINDS
from oa_rlaif.data.families import discover
from oa_rlaif.data.splits import SplitAssigner
from oa_rlaif.io import read_yaml
from oa_rlaif.ontology.alignment_io import read_alignment
from oa_rlaif.paths import config_dir

cfg = read_yaml(config_dir() / "data.yaml")
zero = set(cfg["zero_shot_pairs"])
pairs = discover()
for include_complex in (True, False):
    a = SplitAssigner(cfg["test_fraction"], cfg["split_salt"])
    family_of = collections.defaultdict(set)
    sources_by_family = collections.defaultdict(set)
    for p in pairs:
        kinds = SIMPLE_KINDS | ({"complex"} if include_complex else set())
        if p.ref_kind not in kinds:
            continue
        for ref in p.references:
            for x in read_alignment(ref):
                if x.source_iri and x.target_iri and x.relation in ("=", "<=", ">="):
                    a.add_link(x.source_iri, x.target_iri)
                    family_of[x.source_iri].add(p.family); family_of[x.target_iri].add(p.family)
                    if p.ref_kind in SIMPLE_KINDS:
                        sources_by_family[p.family].add(x.source_iri)
                if p.pair_id in zero:
                    for i in (x.source_iri, x.target_iri):
                        if i:
                            a.force_test(i)
    a.freeze()
    sizes = a.component_sizes()
    biggest = sorted(sizes.items(), key=lambda kv: -kv[1])[:5]
    span = {root: set() for root, _ in biggest}
    for iri in list(a.uf.parent):
        r = a.uf.find(iri)
        if r in span:
            span[r] |= family_of.get(iri, set())
    print(f"\n=== include_complex={include_complex}: components={len(sizes)}")
    for root, n in biggest:
        print(f"  size {n:6d} families {sorted(span[root])} forced_test={root in a._forced_roots}")
    for fam, srcs in sorted(sources_by_family.items()):
        test = sum(a.split(s) == "test" for s in srcs)
        print(f"  {fam:14s} sources {len(srcs):6d}  test share {test/len(srcs):.2f}")

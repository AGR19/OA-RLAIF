"""Sanity check of the DRAL-OA port on Anatomy: fit on train-split references, rank held-out sources."""
import json
import time
from pathlib import Path

import fasttext
import numpy as np

from oa_rlaif.data.build import assign_splits
from oa_rlaif.data.families import discover
from oa_rlaif.io import read_yaml
from oa_rlaif.ontology.alignment_io import orient, read_alignment
from oa_rlaif.ontology.parser import OntologyParser
from oa_rlaif.paths import ROOT, config_dir, data_dir
from oa_rlaif.retrieval.candidates import CandidateGenerator
from oa_rlaif.reward.dral_mapping_layer import DralConfig, fit_pair
from oa_rlaif.reward.dral_reward import DralScorer

FASTTEXT = Path.home() / "Desktop/DRAL_OA/melt-master/examples/externalPythonMatcherWeb/oaei-resources/fast.bin"
cfg = read_yaml(config_dir() / "data.yaml")
pairs = discover()
anatomy = next(p for p in pairs if p.pair_id == "anatomy/mouse-human")
assigner = assign_splits([p for p in pairs if p.family in ("anatomy",)], cfg)
mouse = OntologyParser().parse(anatomy.source, "mouse")
human = OntologyParser().parse(anatomy.target, "human")
refs = [orient(a, mouse, human) for a in read_alignment(anatomy.references[0])]
train = [(a.source_iri, a.target_iri) for a in refs if assigner.split(a.source_iri) == "train"]
test = [(a.source_iri, a.target_iri) for a in refs if assigner.split(a.source_iri) == "test"]
print("train", len(train), "test", len(test))
t0 = time.time()
ft = fasttext.load_model(str(FASTTEXT))
print("fasttext loaded", round(time.time() - t0), "s")
out = data_dir() / "dral" / "anatomy_mouse-human.npz"
info = fit_pair(mouse, human, train, ft, out, DralConfig())
print("fit", info, round(time.time() - t0), "s")
scorer = DralScorer([out])
gen = CandidateGenerator(human)
hits_lex = hits_dral = hits_all = 0
targets = [t for t in human.entities if human.entities[t].declared and scorer.known(t)]
for s, t in test:
    cands = [c for c, _ in gen.candidates(mouse.entities[s], 20)]
    hits_lex += cands[:1] == [t]
    probs = scorer.probabilities(s, cands)
    if probs is not None:
        hits_dral += cands[int(np.nanargmax(probs))] == t
    sims = scorer.similarities(s, targets)
    hits_all += targets[int(np.nanargmax(sims))] == t
n = len(test)
result = {"test_refs": n, "hits1_lexical_top20": round(hits_lex / n, 3),
          "hits1_dral_rerank_top20": round(hits_dral / n, 3), "hits1_dral_all_targets": round(hits_all / n, 3)}
print(json.dumps(result))
(ROOT / "outputs" / "dral_anatomy_check.json").write_text(json.dumps({**result, **info}, indent=1))

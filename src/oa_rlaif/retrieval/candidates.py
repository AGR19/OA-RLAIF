"""Candidate retrieval (vendored from the parent repo src/inference/candidate_generator.py).

Changes: underscores and camelCase are word breaks in normalize(); the embedding device and cache
directory are arguments instead of a project-root path.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np



def normalize(value):
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value).replace("_", " ")
    return " ".join(re.findall(r"[^\W_]+", value.casefold()))


def label_texts(entity) -> list[str]:
    """Labels, or the IRI local name when an entity has none (e.g. UCO) or only blank ones (IoTSec writes empty
    rdfs:label elements): cards already show it, retrieval must too."""
    texts = [label.text for label in entity.labels or [] if label.text.strip()]
    if not texts:
        texts = [entity.iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]]
    return texts


def grams(value, q=3):
    value = " " + normalize(value) + " "
    return {value[i : i + q] for i in range(max(1, len(value) - q + 1))}


class CandidateGenerator:
    def __init__(
        self,
        target,
        q=3,
        semantic=False,
        embedding_model="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        cache_dir="data/embeddings",
        max_postings=5000,
        device="cpu",
    ):
        self.target = target
        self.q = q
        self.index = defaultdict(set)
        self.exact = defaultdict(set)
        self.sizes = {}
        self.ids = sorted(target.entities)
        self.children = defaultdict(set)
        for i, e in target.entities.items():
            for p in e.parents:
                self.children[p].add(i)
        self.max_postings = max_postings
        self.semantic = semantic
        self.model = None
        self.ann = None
        for iri in self.ids:
            e = target.entities[iri]
            features = set()
            for text in label_texts(e):
                features.update(grams(text, q))
                self.exact[normalize(text)].add(iri)
            self.sizes[iri] = len(features)
            for feature in features:
                self.index[feature].add(iri)
        self.device = device
        if semantic:
            self._semantic(embedding_model, cache_dir)

    def _semantic(self, name, directory):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name, device=self.device)
        faiss.omp_set_num_threads(1)
        path = Path(directory) / (self.target.metadata.get("sha256", self.target.ontology_id) + ".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        # The cache key also includes encoder identity; never reuse across model changes.
        import hashlib

        path = path.with_stem(path.stem + "-" + hashlib.sha256(name.encode()).hexdigest()[:12])
        if path.exists() and path.with_suffix(".complete").exists():
            vectors = np.load(path, mmap_mode="r")
        else:
            vectors = None
            for start in range(0, len(self.ids), 128):
                batch = self.model.encode(
                    [self.target.entities[i].name for i in self.ids[start : start + 128]],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).astype("float32")
                if vectors is None:
                    vectors = np.lib.format.open_memmap(
                        path.with_suffix(".pending.npy"),
                        mode="w+",
                        dtype="float32",
                        shape=(len(self.ids), batch.shape[1]),
                    )
                vectors[start : start + len(batch)] = batch
            if vectors is None:
                raise ValueError("Empty semantic target ontology")
            vectors.flush()
            path.with_suffix(".pending.npy").replace(path)
            path.with_suffix(".complete").write_text(str(vectors.shape))
        self.ann = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        self.ann.hnsw.efSearch = 64
        self.ann.add(np.asarray(vectors))

    def candidates(self, entity, k=50):
        features = set()
        exact = set()
        for text in label_texts(entity):
            features.update(grams(text, self.q))
            exact.update(self.exact.get(normalize(text), ()))
        overlap = Counter()
        # Rare postings first; ubiquitous q-grams cannot create a full Cartesian scan.
        for g in sorted(features, key=lambda g: len(self.index.get(g, ()))):
            posting = self.index.get(g, ())
            if len(posting) <= self.max_postings:
                overlap.update(posting)
        scores = {i: c / max(1, len(features) + self.sizes[i] - c) for i, c in overlap.items()}
        for i in exact:
            scores[i] = 1.0
        if self.ann is not None:
            embedding = self.model.encode([entity.name], normalize_embeddings=True).astype(
                "float32"
            )
            values, indices = self.ann.search(embedding, min(k, len(self.ids)))
            for j, value in zip(indices[0], values[0], strict=True):
                if j >= 0:
                    scores[self.ids[j]] = max(scores.get(self.ids[j], 0), float(value))
        return sorted(scores.items(), key=lambda x: (-round(x[1], 12), x[0]))[:k]

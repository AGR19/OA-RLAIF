"""Multilingual semantic retrieval for cross-lingual pairs (MultiFarm, DH multilingual).

Lexical q-grams cannot match labels across scripts (e.g. Chinese <-> Arabic): held-out MultiFarm candidate recall was
0.274 at k = 20. This index embeds each entity's labels with a multilingual sentence encoder (default
intfloat/multilingual-e5-small, 384-d, run locally on CPU; the "query: " prefix is the e5 convention for symmetric
similarity) and searches by exact cosine with numpy, which is enough for ontologies of a few thousand classes.
`fuse` merges a lexical and a semantic ranking by reciprocal rank fusion (no score calibration needed).

Vectors are cached per ontology profile and encoder in data/embeddings/.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np

from ..paths import data_dir
from .candidates import label_texts

DEFAULT_MODEL = "intfloat/multilingual-e5-small"


@lru_cache(maxsize=2)
def encoder(name: str = DEFAULT_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cpu")


def entity_text(entity) -> str:
    texts = list(dict.fromkeys(" ".join(t.replace("_", " ").split()) for t in label_texts(entity)))
    return "query: " + "; ".join(texts[:3])


class SemanticIndex:
    def __init__(self, profile, model: str = DEFAULT_MODEL, cache_key: str | None = None):
        self.model_name = model
        self.ids = sorted(i for i, e in profile.entities.items() if e.declared)
        texts = [entity_text(profile.entities[i]) for i in self.ids]
        digest = hashlib.sha256(("\n".join(self.ids) + "|" + model).encode()).hexdigest()[:16]
        path = data_dir() / "embeddings" / f"{cache_key or profile.ontology_id}-{digest}.npy".replace("/", "_")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self.vectors = np.load(path)
        else:
            self.vectors = encoder(model).encode(texts, batch_size=128, normalize_embeddings=True,
                                                 show_progress_bar=False).astype("float32")
            np.save(path, self.vectors)

    def search(self, entity, k: int = 80) -> list[tuple[str, float]]:
        query = encoder(self.model_name).encode([entity_text(entity)], normalize_embeddings=True,
                                                show_progress_bar=False).astype("float32")[0]
        scores = self.vectors @ query
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top]


def fuse(*rankings: list[tuple[str, float]], constant: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: score(c) = sum over rankings of 1 / (constant + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (iri, _) in enumerate(ranking, 1):
            scores[iri] = scores.get(iri, 0.0) + 1.0 / (constant + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))

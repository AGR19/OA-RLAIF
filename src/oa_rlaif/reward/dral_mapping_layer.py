"""Port of the DRAL-OA mapping layer (`old bert 2/clf.py` in the DRAL_OA repository).

Faithful to clf.py:
  * graph over source+target classes: parent edges weight 100, child / disjoint edges weight 1
    (clf.py also drew restriction edges; our parser keeps named parents, equivalences and disjointness only)
  * node2vec with clf.py's settings (dimensions 100, walk_length 30, num_walks 20, window 20, min_count 1,
    batch_words 256). clf.py used node2vec's defaults p = q = 1, i.e. first-order weighted random walks,
    which is what `weighted_walks` implements before gensim skip-gram training.
  * features = [node2vec(100) ; text vector(300, FastText fast.bin)], each standardised within the vector
    (clf.py's normalize_vector)
  * MLP embedder 400 -> 256 (tanh) -> 128 and a classifier on the concatenated pair -> 64 (ReLU) -> 1 (sigmoid),
    BCE, Adam lr 3e-4, 20 epochs, batch 32, 50,000 pairs of which ~50% are reference positives
  * similarity used for ranking: 0.5 * cos(embedder) + 0.5 * cos(text)   (paper Table I: wv = wr = 0.5)

Changed on purpose:
  * positives come from TRAIN-split references only (clf.py used every reference, then scored on them)
  * embeddings for all classes are exported so Eq. 1 can score any candidate set
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..retrieval.candidates import normalize


@dataclass
class DralConfig:
    dimensions: int = 100
    walk_length: int = 30
    num_walks: int = 20
    window: int = 20
    min_count: int = 1
    batch_words: int = 256
    parent_weight: float = 100.0
    other_weight: float = 1.0
    embed_dim: int = 128
    hidden: int = 256
    lr: float = 3e-4          # clf.py; the AIxDKE 2024 paper reports 1e-5 (sensitivity setting)
    dropout: float = 0.0      # clf.py has none; the paper reports p = 0.2 in the embedding layer
    epochs: int = 20
    batch: int = 32
    dataset_size: int = 50_000
    seed: int = 42
    workers: int = max(1, (os.cpu_count() or 2) - 1)


def build_graph(profiles, cfg: DralConfig, restrict: set[str] | None = None):
    """CSR adjacency over all classes of the given profiles (optionally a restricted node set + 1 hop)."""
    index: dict[str, int] = {}
    edges: dict[tuple[int, int], float] = {}

    def node(iri):
        if iri not in index:
            index[iri] = len(index)
        return index[iri]

    def add(a, b, w):
        ia, ib = node(a), node(b)
        edges[(ia, ib)] = max(edges.get((ia, ib), 0.0), w)
        edges[(ib, ia)] = max(edges.get((ib, ia), 0.0), w)

    keep = None
    if restrict is not None:  # the entities in episodes plus their direct parents (1-hop neighbourhood)
        keep = set(restrict)
        for profile in profiles:
            for iri in restrict:
                entity = profile.entities.get(iri)
                if entity is not None:
                    keep.update(entity.parents)
    for profile in profiles:
        for iri, entity in profile.entities.items():
            if keep is not None and iri not in keep:
                continue
            node(iri)
            for parent in entity.parents:
                if keep is None or parent in keep:
                    add(parent, iri, cfg.parent_weight)  # clf.py: graph.add_edge(parent, self, weight=100)
        for a, b in profile.disjoint:
            if keep is None or (a in keep and b in keep):
                add(a, b, cfg.other_weight)
    nodes = [None] * len(index)
    for iri, i in index.items():
        nodes[i] = iri
    n = len(nodes)
    order = sorted(edges)
    indptr = np.zeros(n + 1, dtype=np.int64)
    for a, _ in order:
        indptr[a + 1] += 1
    indptr = np.cumsum(indptr)
    indices = np.array([b for _, b in order], dtype=np.int64)
    weights = np.array([edges[e] for e in order], dtype=np.float64)
    return nodes, indptr, indices, weights


def weighted_walks(indptr, indices, weights, cfg: DralConfig) -> list[list[str]]:
    """First-order weighted random walks (node2vec with p = q = 1), all walkers stepped together in numpy."""
    rng = np.random.default_rng(cfg.seed)
    n = len(indptr) - 1
    degree = np.diff(indptr)
    # per-node cumulative weights, normalised, laid out in CSR order
    cumulative = weights.copy()
    for start, end in zip(indptr[:-1], indptr[1:]):
        if end > start:
            block = np.cumsum(cumulative[start:end])
            cumulative[start:end] = block / block[-1]
    walks = []
    for _ in range(cfg.num_walks):
        current = rng.permutation(n)
        path = [current]
        alive = degree[current] > 0
        for _ in range(cfg.walk_length - 1):
            nxt = current.copy()
            idx = np.nonzero(alive)[0]
            if len(idx) == 0:
                break
            nodes_now = current[idx]
            u = rng.random(len(idx))
            starts, ends = indptr[nodes_now], indptr[nodes_now + 1]
            # vectorised searchsorted within each node's CSR slice
            offsets = np.array([np.searchsorted(cumulative[s:e], x) for s, e, x in zip(starts, ends, u)], dtype=np.int64) \
                if len(idx) < 2048 else _batched_search(cumulative, starts, ends, u)
            nxt[idx] = indices[starts + np.minimum(offsets, ends - starts - 1)]
            path.append(nxt)
            current = nxt
            alive = degree[current] > 0
        stacked = np.stack(path, axis=1)
        lengths = np.full(n, stacked.shape[1])
        walks.extend([[str(x) for x in row[:length]] for row, length in zip(stacked, lengths)])
    return walks


def _batched_search(cumulative, starts, ends, u):
    """searchsorted per CSR slice without a Python loop: offset = count of entries < u in the slice."""
    lengths = ends - starts
    max_len = int(lengths.max()) if len(lengths) else 0
    out = np.zeros(len(starts), dtype=np.int64)
    for j in range(max_len):  # loop over slice positions (small: max degree), not over walkers
        valid = j < lengths
        positions = starts[valid] + j
        out[valid] += cumulative[positions] < u[valid]
    return out


def node2vec_embeddings(profiles, cfg: DralConfig, restrict: set[str] | None = None) -> tuple[list[str], np.ndarray]:
    from gensim.models import Word2Vec

    nodes, indptr, indices, weights = build_graph(profiles, cfg, restrict)
    walks = weighted_walks(indptr, indices, weights, cfg)
    model = Word2Vec(walks, vector_size=cfg.dimensions, window=cfg.window, min_count=cfg.min_count,
                     batch_words=cfg.batch_words, workers=cfg.workers, sg=1, seed=cfg.seed)
    vectors = np.stack([model.wv[str(i)] if str(i) in model.wv else np.zeros(cfg.dimensions, dtype=np.float32)
                        for i in range(len(nodes))])
    return nodes, vectors.astype(np.float32)


def text_vectors(profiles, iris: list[str], model) -> np.ndarray:
    """FastText sentence vectors of each class's preferred label (DRAL-OA pre-processing: normalised tokens)."""
    from ..data.cards import names

    lookup = {}
    for profile in profiles:
        lookup.update(profile.entities)
    out = np.zeros((len(iris), model.get_dimension()), dtype=np.float32)
    for i, iri in enumerate(iris):
        entity = lookup.get(iri)
        text = normalize(names(entity)[0]) if entity is not None else normalize(iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1])
        out[i] = model.get_sentence_vector(text or "unknown")
    return out


def standardise(matrix: np.ndarray) -> np.ndarray:
    """clf.py normalize_vector: StandardScaler fit on each vector's own values."""
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    return (matrix - mean) / np.where(std == 0, 1, std)


def train_mapping_layer(features: np.ndarray, index: dict[str, int], positives: list[tuple[str, str]],
                        target_iris: list[str], cfg: DralConfig):
    """BCE training of the clf.py MLP on 50% reference positives and 50% random negatives."""
    import torch

    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    positives = [(s, t) for s, t in positives if s in index and t in index]
    if not positives:
        raise ValueError("no train-split reference pairs with embeddings")
    gold = set(positives)
    sources = [s for s, _ in positives]
    pairs, labels = [], []
    for _ in range(cfg.dataset_size):
        if rng.random() < 0.5:
            s, t = rng.choice(positives)
            pairs.append((index[s], index[t]))
            labels.append(1.0)
        else:
            while True:
                s, t = rng.choice(sources), rng.choice(target_iris)
                if (s, t) not in gold and t in index:
                    break
            pairs.append((index[s], index[t]))
            labels.append(0.0)

    x = torch.tensor(features, dtype=torch.float32)

    class MLP(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            layers = [torch.nn.Linear(input_dim, cfg.hidden), torch.nn.Tanh()]
            if cfg.dropout > 0:
                layers.append(torch.nn.Dropout(cfg.dropout))
            self.embedder = torch.nn.Sequential(*layers, torch.nn.Linear(cfg.hidden, cfg.embed_dim))
            self.clf = torch.nn.Sequential(torch.nn.Linear(cfg.embed_dim * 2, 64), torch.nn.ReLU(),
                                           torch.nn.Linear(64, 1), torch.nn.Sigmoid())

        def forward(self, a, b):
            return self.clf(torch.cat([self.embedder(a), self.embedder(b)], dim=1)).squeeze(-1)

    model = MLP(features.shape[1])
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = torch.nn.BCELoss()
    pair_tensor = torch.tensor(pairs, dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    history = []
    for epoch in range(cfg.epochs):
        permutation = torch.randperm(len(pairs))
        total = 0.0
        for start in range(0, len(pairs), cfg.batch):
            batch = permutation[start:start + cfg.batch]
            a, b = x[pair_tensor[batch, 0]], x[pair_tensor[batch, 1]]
            loss = loss_fn(model(a, b), label_tensor[batch])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch)
        history.append(total / len(pairs))
    with torch.no_grad():
        embedded = model.embedder(x).numpy()
    return model, embedded, history


def fit_group(profiles, train_pairs: list[tuple[str, str]], target_iris: set[str], fasttext_model, out: Path,
              cfg: DralConfig | None = None, restrict: set[str] | None = None) -> dict:
    """Fit one mapping layer over a group of ontologies (a pair, or a pooled family) and save iris / z / t."""
    cfg = cfg or DralConfig()
    nodes, graph_vectors = node2vec_embeddings(profiles, cfg, restrict)
    texts = text_vectors(profiles, nodes, fasttext_model)
    features = np.concatenate([standardise(graph_vectors), standardise(texts)], axis=1)
    index = {iri: i for i, iri in enumerate(nodes)}
    targets = [i for i in target_iris if i in index]
    _, embedded, history = train_mapping_layer(features, index, train_pairs, targets, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, iris=np.array(nodes), z=embedded.astype(np.float32), t=texts.astype(np.float32))
    return {"nodes": len(nodes), "train_pairs": len(train_pairs), "loss_first": history[0], "loss_last": history[-1]}


def fit_pair(source_profile, target_profile, train_pairs, fasttext_model, out: Path, cfg: DralConfig | None = None) -> dict:
    return fit_group([source_profile, target_profile], train_pairs, set(target_profile.entities), fasttext_model, out, cfg)

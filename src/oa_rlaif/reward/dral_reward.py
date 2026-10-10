"""DRAL-OA Eq. 1 as a reward for a chosen candidate.

    L_ij = -log( exp(sim(z_i, z_j)/T) / sum_k exp(sim(z_i, z_k)/T) )         (Eq. 1, DRAL-OA papers)
    r_DRAL(pick) = exp(-L_{s,pick}) = softmax_T over the episode's candidates of sim(s, k)
    r_DRAL(NONE) = 1 - max_k softmax_T(sim(s, k))       (high when no candidate looks like a match)

sim(i, j) = 0.5 * cos(z_i, z_j) + 0.5 * cos(t_i, t_j): z is the mapping-layer embedding and t the FastText
label vector, combined exactly as in clf.py / Table I (wv = wr = 0.5). The temperature T is not reported in
the papers; it is a configuration value (default 0.1), flagged in docs/REWARD_SPEC.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class DralScorer:
    def __init__(self, npz_paths: list[Path], temperature: float = 0.1, wv: float = 0.5, wr: float = 0.5):
        self.temperature, self.wv, self.wr = temperature, wv, wr
        self.index: dict[str, tuple[int, int]] = {}
        self.z: list[np.ndarray] = []
        self.t: list[np.ndarray] = []
        for n, path in enumerate(npz_paths):
            data = np.load(path, allow_pickle=False)
            z = data["z"] / np.maximum(np.linalg.norm(data["z"], axis=1, keepdims=True), 1e-9)
            t = data["t"] / np.maximum(np.linalg.norm(data["t"], axis=1, keepdims=True), 1e-9)
            self.z.append(z.astype(np.float32))
            self.t.append(t.astype(np.float32))
            for i, iri in enumerate(data["iris"].tolist()):
                self.index.setdefault(iri, (n, i))

    def known(self, iri: str) -> bool:
        return iri in self.index

    def _vectors(self, iri: str) -> tuple[np.ndarray, np.ndarray]:
        n, i = self.index[iri]
        return self.z[n][i], self.t[n][i]

    def similarities(self, source: str, candidates: list[str]) -> np.ndarray | None:
        if source not in self.index:
            return None
        zs, ts = self._vectors(source)
        sims = np.full(len(candidates), np.nan, dtype=np.float64)
        for k, cand in enumerate(candidates):
            if cand in self.index:
                zc, tc = self._vectors(cand)
                sims[k] = self.wv * float(zs @ zc) + self.wr * float(ts @ tc)
        return sims

    def probabilities(self, source: str, candidates: list[str]) -> np.ndarray | None:
        sims = self.similarities(source, candidates)
        if sims is None or np.all(np.isnan(sims)):
            return None
        logits = np.where(np.isnan(sims), -np.inf, sims / self.temperature)
        logits -= np.max(logits)
        weights = np.exp(logits)
        return weights / weights.sum()

    def reward(self, source: str, candidates: list[str], choice: str) -> float | None:
        """Eq. 1 reward in [0, 1] for `choice` (an IRI among candidates, or "NONE"); None if unscorable."""
        probs = self.probabilities(source, candidates)
        if probs is None:
            return None
        if choice == "NONE":
            return float(1.0 - probs.max())
        if choice not in candidates:
            return 0.0
        return float(probs[candidates.index(choice)])

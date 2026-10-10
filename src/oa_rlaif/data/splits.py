"""Leakage-free 80/20 splits by connected component of gold equivalence links.

Every gold `=` correspondence from every family (except the ICD crosswalk, which has its own bridge split)
is an edge. A component is one real-world concept across languages, ontologies and pairs (MultiFarm IRIs
differ per language, so IRI grouping alone would leak). A component is test if its stable hash falls in the
test fraction, or if it touches a zero-shot pair. Every episode inherits the split of its source entity.
"""

from __future__ import annotations

import hashlib


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        parent = self.parent.setdefault(x, x)
        if parent != x:
            root = self.find(parent)
            self.parent[x] = root
            return root
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def stable_fraction(key: str, salt: str = "oa-rlaif-v1") -> float:
    digest = hashlib.sha256(f"{salt}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class SplitAssigner:
    def __init__(self, test_fraction: float = 0.2, salt: str = "oa-rlaif-v1"):
        self.uf = UnionFind()
        self.test_fraction = test_fraction
        self.salt = salt
        self.forced_test: set[str] = set()
        self._members: dict[str, int] | None = None

    def add_link(self, a: str, b: str) -> None:
        if a and b:
            self.uf.union(a, b)

    def force_test(self, iri: str) -> None:
        self.forced_test.add(iri)

    def component(self, iri: str) -> str:
        return self.uf.find(iri)

    def split(self, iri: str) -> str:
        root = self.uf.find(iri)
        if self._forced_roots is None:
            self._forced_roots = {self.uf.find(x) for x in self.forced_test}
        if root in self._forced_roots:
            return "test"
        return "test" if stable_fraction(root, self.salt) < self.test_fraction else "train"

    _forced_roots: set[str] | None = None

    def freeze(self) -> None:
        self._forced_roots = {self.uf.find(x) for x in self.forced_test}

    def component_sizes(self) -> dict[str, int]:
        sizes: dict[str, int] = {}
        for node in list(self.uf.parent):
            root = self.uf.find(node)
            sizes[root] = sizes.get(root, 0) + 1
        return sizes

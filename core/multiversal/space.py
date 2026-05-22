"""
SemanticSpace — the ℝⁿ manifold all meanings live in.

All modalities project into the same space.
Operations happen here: interpolation, composition, projection,
decomposition, axis measurement, nearest-neighbor search.

The embedding model defines n. Nothing else does.
"""
from __future__ import annotations
import numpy as np
from typing import Callable
from core.multiversal.nmeaning import NMeaning


class SemanticSpace:
    def __init__(self, embed_fn: Callable[[str], list[float]]):
        """
        embed_fn: takes a string, returns a float list of length n.
        Typically: OllamaEmbeddings(...).embed_query
        """
        self._embed = embed_fn
        self._dim: int | None = None
        self._corpus: list[NMeaning] = []
        self._axis_cache: dict[str, np.ndarray] = {}

    # ── Embedding ──────────────────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        v = np.array(self._embed(text), dtype=np.float64)
        if self._dim is None:
            self._dim = len(v)
        return v / (np.linalg.norm(v) + 1e-10)

    @property
    def dim(self) -> int | None:
        return self._dim

    # ── Axis measurement ───────────────────────────────────────────────

    def axis_vector(self, axis_name: str) -> np.ndarray:
        """
        Embed an axis as the difference between its positive and negative poles.
        E.g. axis "certainty" = embed("certain") − embed("uncertain").
        Cached per session.
        """
        if axis_name not in self._axis_cache:
            pos = self.embed(axis_name)
            neg = self.embed(f"absence of {axis_name}")
            v = pos - neg
            n = np.linalg.norm(v)
            self._axis_cache[axis_name] = v / (n + 1e-10)
        return self._axis_cache[axis_name]

    def axis_value(self, meaning: NMeaning, axis_name: str) -> float:
        """
        Measure where meaning falls on a named semantic axis.
        Returns a value in [-1, 1].
        """
        ax = self.axis_vector(axis_name)
        return float(np.dot(meaning.vector / (np.linalg.norm(meaning.vector) + 1e-10), ax))

    def measure_dims(self, meaning: NMeaning, axes: list[str]) -> dict[str, float]:
        return {ax: self.axis_value(meaning, ax) for ax in axes}

    # ── Manifold operations ────────────────────────────────────────────

    def interpolate(self, a: NMeaning, b: NMeaning, t: float = 0.5) -> NMeaning:
        """
        Spherical linear interpolation (slerp) between two meanings.
        t=0 → a, t=1 → b, t=0.5 → midpoint on the great circle.
        """
        u = a.vector / (np.linalg.norm(a.vector) + 1e-10)
        v = b.vector / (np.linalg.norm(b.vector) + 1e-10)
        dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
        theta = np.arccos(dot)
        if abs(theta) < 1e-6:
            return a
        result = (np.sin((1 - t) * theta) * u + np.sin(t * theta) * v) / np.sin(theta)
        return NMeaning(
            vector=result,
            source_text=f"interpolate({a.source_text!r}, {b.source_text!r}, t={t:.2f})",
            source_modality="operation",
            label=f"t={t:.2f} between [{a.label or a.source_text[:20]}] and [{b.label or b.source_text[:20]}]",
        )

    def compose(self, meanings: list[NMeaning], weights: list[float] | None = None) -> NMeaning:
        """
        Weighted centroid in embedding space — the 'sum' of meanings.
        Renormalized to stay on the unit hypersphere.
        """
        if weights is None:
            weights = [1.0] * len(meanings)
        w = np.array(weights, dtype=np.float64)
        w /= w.sum()
        result = sum(w[i] * m.vector for i, m in enumerate(meanings))
        result /= (np.linalg.norm(result) + 1e-10)
        return NMeaning(
            vector=result,
            source_text=" ⊕ ".join(m.source_text[:20] for m in meanings),
            source_modality="operation",
            label="composed",
        )

    def subtract(self, base: NMeaning, remove: NMeaning) -> NMeaning:
        """
        Analogy operation: base − remove + zero ≈ 'base without the remove concept'.
        Classic word2vec-style: king − man + woman = queen.
        """
        zero = np.zeros_like(base.vector)
        result = base.vector - remove.vector
        result /= (np.linalg.norm(result) + 1e-10)
        return NMeaning(
            vector=result,
            source_text=f"{base.source_text} − {remove.source_text}",
            source_modality="operation",
            label="subtracted",
        )

    def analogy(self, a: NMeaning, b: NMeaning, c: NMeaning) -> NMeaning:
        """
        a is to b as c is to ? → result = b − a + c.
        """
        result = b.vector - a.vector + c.vector
        result /= (np.linalg.norm(result) + 1e-10)
        return NMeaning(
            vector=result,
            source_text=f"{a.source_text!r}:{b.source_text!r}::{c.source_text!r}:?",
            source_modality="operation",
            label="analogy",
        )

    def project(self, meaning: NMeaning, axes: list[str]) -> NMeaning:
        """
        Project meaning onto the subspace defined by named axes.
        Strips all semantic content except the given dimensions.
        """
        axis_vecs = np.array([self.axis_vector(ax) for ax in axes])
        basis = self._gram_schmidt(axis_vecs)
        result = np.zeros_like(meaning.vector)
        for b in basis:
            result += np.dot(meaning.vector, b) * b
        n = np.linalg.norm(result)
        if n < 1e-10:
            result = meaning.vector.copy()
            n = np.linalg.norm(result)
        return NMeaning(
            vector=result / n,
            dims={ax: self.axis_value(meaning, ax) for ax in axes},
            source_text=meaning.source_text,
            source_modality=meaning.source_modality,
            label=f"projected onto [{', '.join(axes)}]",
        )

    def decompose(self, meaning: NMeaning, axes: list[str]) -> dict[str, float]:
        """
        Decompose meaning into components along named axes.
        Returns the signed magnitude along each axis.
        """
        return {ax: float(np.dot(meaning.vector, self.axis_vector(ax))) for ax in axes}

    def pca_axes(self, meanings: list[NMeaning], n_components: int = 3) -> np.ndarray:
        """
        Find the principal semantic axes across a set of meanings.
        Returns (n_components, dim) array of principal directions.
        """
        matrix = np.array([m.vector for m in meanings])
        matrix -= matrix.mean(axis=0)
        _, _, vt = np.linalg.svd(matrix, full_matrices=False)
        return vt[:n_components]

    # ── Corpus / nearest neighbor ──────────────────────────────────────

    def add(self, meaning: NMeaning) -> None:
        self._corpus.append(meaning)

    def nearest(self, meaning: NMeaning, k: int = 5) -> list[tuple[NMeaning, float]]:
        """Find k nearest meanings in the corpus by cosine similarity."""
        if not self._corpus:
            return []
        scored = [
            (m, meaning.cosine_similarity(m))
            for m in self._corpus
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ── Internal ───────────────────────────────────────────────────────

    @staticmethod
    def _gram_schmidt(vectors: np.ndarray) -> list[np.ndarray]:
        basis = []
        for v in vectors:
            w = v.copy().astype(np.float64)
            for b in basis:
                w -= np.dot(w, b) * b
            n = np.linalg.norm(w)
            if n > 1e-10:
                basis.append(w / n)
        return basis

"""
NMeaning — n-dimensional meaning.

The vector IS the meaning. Named dims are discovered projections onto it,
not the primary structure. Nothing is hardcoded. The space defines itself.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class NMeaning:
    # ── Primary representation: a point in ℝⁿ ─────────────────────────
    vector: np.ndarray                      # shape (n,) — unit-normalized

    # ── Named semantic dimensions (open set, discovered dynamically) ───
    # Maps axis name → position on that axis (-1.0 … +1.0)
    dims: dict[str, float] = field(default_factory=dict)

    # ── Per-dimension uncertainty (optional) ───────────────────────────
    uncertainty: np.ndarray | None = None   # shape (n,)

    # ── Provenance ─────────────────────────────────────────────────────
    source_text:     str           = ""
    source_modality: str           = "text"
    source_language: str | None    = None

    # ── Manifold metadata ─────────────────────────────────────────────
    # Human-readable label for the nearest interpretable concept
    label: str = ""

    @property
    def dim(self) -> int:
        return len(self.vector)

    def norm(self) -> float:
        return float(np.linalg.norm(self.vector))

    def unit(self) -> "NMeaning":
        n = self.norm()
        if n < 1e-10:
            return self
        return NMeaning(
            vector=self.vector / n,
            dims=self.dims,
            uncertainty=self.uncertainty / n if self.uncertainty is not None else None,
            source_text=self.source_text,
            source_modality=self.source_modality,
            source_language=self.source_language,
            label=self.label,
        )

    def cosine_similarity(self, other: "NMeaning") -> float:
        a = self.vector / (np.linalg.norm(self.vector) + 1e-10)
        b = other.vector / (np.linalg.norm(other.vector) + 1e-10)
        return float(np.clip(np.dot(a, b), -1.0, 1.0))

    def distance(self, other: "NMeaning") -> float:
        """Angular distance in [0, 1]."""
        sim = self.cosine_similarity(other)
        return float(np.arccos(np.clip(sim, -1.0, 1.0)) / np.pi)

    def summary(self, top_dims: int = 8) -> str:
        lines = [
            f"dim={self.dim}  label={self.label!r}",
            f"source=({self.source_modality}) {self.source_text[:60]!r}",
        ]
        if self.dims:
            sorted_dims = sorted(self.dims.items(), key=lambda x: abs(x[1]), reverse=True)
            dim_str = "  ".join(f"{k}:{v:+.2f}" for k, v in sorted_dims[:top_dims])
            lines.append(f"dims: {dim_str}")
        return "\n  ".join(lines)

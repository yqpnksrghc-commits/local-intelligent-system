"""
Channel — the path meaning travels between participants.

A channel has:
  attenuation — overall signal loss (0=lossless, 1=total loss)
  warp        — rotation in semantic space (perspective shift in transit)
  axes_passed — dimensional filter (only these axes transit; None = all)

Channels model the reality that meaning transforms in transit:
  a concept sent in Mandarin to a French speaker through
  a poetic channel arrives differently than through a technical channel.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from core.multiversal.nmeaning import NMeaning
from core.multiversal.space import SemanticSpace


@dataclass
class Channel:
    id:           str
    sender:       str
    receiver:     str

    # Signal loss: 0.0 = perfect, 1.0 = nothing arrives
    attenuation:  float           = 0.0

    # Dimensional filter — axes that transit this channel
    # None = all dimensions pass
    axes_passed:  list[str] | None = None

    # Semantic warp: a direction the channel biases toward
    # e.g. a "poetic" channel warps toward aesthetic dimensions
    warp_axis:    str | None      = None
    warp_strength: float          = 0.0

    stats: dict = field(default_factory=lambda: {"sent": 0, "received": 0})

    def transmit(self, meaning: NMeaning, space: SemanticSpace) -> NMeaning:
        """Apply channel characteristics to a meaning in transit."""
        self.stats["sent"] += 1
        v = meaning.vector.copy()

        # attenuation
        if self.attenuation > 0:
            noise = np.random.normal(0, self.attenuation * 0.1, v.shape)
            v = v * (1 - self.attenuation) + noise

        # dimensional filter
        if self.axes_passed:
            filtered = space.project(
                NMeaning(vector=v, source_text=meaning.source_text,
                         source_modality=meaning.source_modality),
                self.axes_passed,
            )
            v = filtered.vector

        # semantic warp: nudge toward warp_axis
        if self.warp_axis and self.warp_strength > 0:
            warp_vec = space.axis_vector(self.warp_axis)
            v = v + self.warp_strength * warp_vec
            n = np.linalg.norm(v)
            if n > 1e-10:
                v = v / n

        n = np.linalg.norm(v)
        if n < 1e-10:
            return meaning

        self.stats["received"] += 1
        return NMeaning(
            vector=v / n,
            dims=meaning.dims,
            source_text=meaning.source_text,
            source_modality=meaning.source_modality,
            source_language=meaning.source_language,
            label=f"{meaning.label} via {self.id}",
        )

    def summary(self) -> str:
        axes = ", ".join(self.axes_passed) if self.axes_passed else "all"
        warp = f"{self.warp_axis}×{self.warp_strength:.2f}" if self.warp_axis else "none"
        return (f"Channel({self.id}: {self.sender}→{self.receiver} "
                f"attenuation={self.attenuation:.2f} axes=[{axes}] warp={warp})")

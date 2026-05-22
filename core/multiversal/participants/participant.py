"""
Participant — an entity with a position in ℝⁿ, a preferred modality,
and a receptivity profile they control.

Receptivity is the semantic volume control:
  1.0 — full resolution, all dimensions received
  0.5 — only dimensions with magnitude > threshold pass through
  0.0 — no meaning received (deaf)

Bandwidth filters which semantic axes the participant is tuned to.
A participant focused on temporal meaning will receive the temporal
components of any incoming signal, regardless of its source modality.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from core.multiversal.nmeaning import NMeaning
from core.multiversal.participants.message import Message


@dataclass
class Participant:
    name:               str
    preferred_modality: str          = "text"
    preferred_language: str          = "english"

    # Semantic state — their current position in ℝⁿ
    # Starts at None; forms as they send/receive meaning
    state:              NMeaning | None = None

    # Receptivity: 0.0 (silent) → 1.0 (full)
    # The receiver sets this to a level that is comfortable
    receptivity:        float        = 0.8

    # Bandwidth: named axes the participant is tuned to
    # None = all axes (omnidirectional reception)
    bandwidth:          list[str] | None = None

    # Resonance threshold: messages below this similarity to
    # current state are attenuated further (comfort filter)
    resonance_threshold: float       = 0.0

    # History
    inbox:   list[Message] = field(default_factory=list)
    outbox:  list[Message] = field(default_factory=list)

    def set_volume(self, level: float) -> None:
        """Set receptivity to level in [0, 1]."""
        self.receptivity = max(0.0, min(1.0, level))
        print(f"[{self.name}] receptivity → {int(self.receptivity * 100)}%")

    def set_bandwidth(self, axes: list[str] | None) -> None:
        """Tune to specific semantic axes. None = omnidirectional."""
        self.bandwidth = axes
        desc = ", ".join(axes) if axes else "omnidirectional"
        print(f"[{self.name}] bandwidth → {desc}")

    def receive(self, incoming: NMeaning, space) -> NMeaning | None:
        """
        Apply receptivity and bandwidth filter to incoming meaning.
        Returns the meaning as actually received, or None if below threshold.
        """
        import numpy as np

        # resonance check: how aligned is this with current state?
        if self.state is not None:
            resonance = self.state.cosine_similarity(incoming)
            if resonance < self.resonance_threshold:
                effective_receptivity = self.receptivity * max(0.0, resonance)
            else:
                effective_receptivity = self.receptivity
        else:
            effective_receptivity = self.receptivity

        if effective_receptivity < 0.01:
            return None

        # bandwidth projection: filter to tuned axes only
        if self.bandwidth:
            received = space.project(incoming, self.bandwidth)
        else:
            received = incoming

        # receptivity scaling: attenuate the vector magnitude
        # (lower receptivity = noisier, lower-fidelity reception)
        noise_level = (1.0 - effective_receptivity) * 0.15
        noise = np.random.normal(0, noise_level, received.vector.shape)
        v = received.vector * effective_receptivity + noise
        n = np.linalg.norm(v)
        if n < 1e-10:
            return None

        received_meaning = NMeaning(
            vector=v / n,
            dims={k: val * effective_receptivity for k, val in received.dims.items()},
            source_text=received.source_text,
            source_modality=received.source_modality,
            source_language=received.source_language,
            label=received.label,
        )

        # update participant state: blend toward received meaning
        self._update_state(received_meaning, space)
        return received_meaning

    def _update_state(self, received: NMeaning, space) -> None:
        """Shift participant's position in semantic space toward received meaning."""
        if self.state is None:
            self.state = received
        else:
            # slow drift: 90% current state, 10% new meaning
            self.state = space.interpolate(self.state, received, t=0.1)
            self.state.label = f"{self.name}'s state"

    def resonance_with(self, meaning: NMeaning) -> float:
        if self.state is None:
            return 0.5
        return self.state.cosine_similarity(meaning)

    def summary(self) -> str:
        state_str = self.state.label if self.state else "uninitiated"
        bw = ", ".join(self.bandwidth) if self.bandwidth else "all"
        return (f"Participant({self.name!r}  modality={self.preferred_modality}"
                f"  receptivity={int(self.receptivity*100)}%"
                f"  bandwidth=[{bw}]"
                f"  state={state_str!r}"
                f"  inbox={len(self.inbox)}  outbox={len(self.outbox)})")

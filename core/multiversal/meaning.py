"""
The Semantic Intermediate Representation (SIR).

Every modality — language, music, mathematics, symbol, gesture, code —
encodes into this structure. Every decoder reads from it.
Meaning is the invariant. Modality is the surface.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Meaning:
    # ── Predicate-argument structure (universal deep grammar) ──────────
    predicate:  str           = ""       # the core action / state / relation
    agent:      Optional[str] = None     # who/what initiates
    patient:    Optional[str] = None     # who/what receives the action
    theme:      Optional[str] = None     # what is transferred or described
    location:   Optional[str] = None     # where
    time:       Optional[str] = None     # when
    manner:     Optional[str] = None     # how
    cause:      Optional[str] = None     # why

    # ── Pragmatic layer ────────────────────────────────────────────────
    speech_act:  str   = "assertion"    # assertion | question | command | exclamation
    modality:    str   = "factual"      # factual | possible | necessary | conditional
    force:       float = 1.0            # certainty  0.0 (doubt) → 1.0 (certainty)
    perspective: str   = "third"        # first | second | third | universal

    # ── Affective / aesthetic layer (VAD model) ────────────────────────
    valence:    float = 0.0    # negative −1.0 ←→ positive +1.0
    arousal:    float = 0.5    # calm 0.0 ←→ excited 1.0
    dominance:  float = 0.5    # submissive 0.0 ←→ dominant 1.0

    # ── Temporal structure ─────────────────────────────────────────────
    tense:  str = "present"    # past | present | future | timeless
    aspect: str = "simple"     # simple | progressive | perfect | habitual

    # ── Semantic tags (open set) ───────────────────────────────────────
    tags: list[str] = field(default_factory=list)

    # ── Semantic embedding (language-model vector) ─────────────────────
    embedding: Optional[list[float]] = None

    # ── Provenance ─────────────────────────────────────────────────────
    source_text:     str           = ""
    source_modality: str           = "text"
    source_language: Optional[str] = None

    def summary(self) -> str:
        parts = [f"[{self.speech_act.upper()}]", f"predicate={self.predicate!r}"]
        for role in ("agent", "patient", "theme", "cause", "manner"):
            val = getattr(self, role)
            if val:
                parts.append(f"{role}={val!r}")
        parts.append(
            f"VAD=({self.valence:+.2f}, {self.arousal:.2f}, {self.dominance:.2f})"
        )
        parts.append(f"tense={self.tense}  aspect={self.aspect}  force={self.force:.2f}")
        if self.tags:
            parts.append(f"tags={self.tags}")
        return "\n  ".join(parts)

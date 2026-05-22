"""
Dynamics — meaning as motion through ℝⁿ.

A conversation is not a sequence of static points.
It is a trajectory. The path carries information:
  velocity     — how fast meaning is shifting
  acceleration — is the shift speeding up or slowing down?
  curvature    — is the conversation turning or going straight?
  convergence  — are participants moving toward each other?
  oscillation  — is meaning cycling back on itself?

Diffusion models how meaning propagates through a participant network
when no one is explicitly sending — the passive spread of state.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from core.multiversal.nmeaning import NMeaning


# ── Trajectory ────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryPoint:
    meaning:   NMeaning
    timestamp: datetime = field(default_factory=datetime.utcnow)
    label:     str = ""


class Trajectory:
    """
    A participant's or session's path through ℝⁿ over time.
    The arc of a conversation made geometric.
    """

    def __init__(self, owner: str = ""):
        self.owner  = owner
        self.points: list[TrajectoryPoint] = []

    def append(self, meaning: NMeaning, label: str = "") -> None:
        self.points.append(TrajectoryPoint(meaning, label=label))

    def __len__(self) -> int:
        return len(self.points)

    # ── Kinematic metrics ─────────────────────────────────────────────

    def velocity(self) -> float:
        """
        Mean angular speed over the last 5 steps.
        High = meaning is shifting rapidly.
        """
        pts = self.points[-6:]
        if len(pts) < 2:
            return 0.0
        dists = [pts[i].meaning.distance(pts[i+1].meaning) for i in range(len(pts)-1)]
        return float(np.mean(dists))

    def acceleration(self) -> float:
        """
        Change in velocity — is the shift speeding up or slowing down?
        Positive = accelerating away. Negative = decelerating (converging).
        """
        pts = self.points[-7:]
        if len(pts) < 3:
            return 0.0
        dists = [pts[i].meaning.distance(pts[i+1].meaning) for i in range(len(pts)-1)]
        if len(dists) < 2:
            return 0.0
        return float(dists[-1] - dists[0])

    def curvature(self) -> float:
        """
        How much the direction of travel is changing.
        0 = straight line through semantic space.
        1 = tight spiral, meaning cycling back on itself.
        """
        pts = self.points[-4:]
        if len(pts) < 3:
            return 0.0
        directions = []
        for i in range(len(pts) - 1):
            d = pts[i+1].meaning.vector - pts[i].meaning.vector
            n = np.linalg.norm(d)
            if n > 1e-10:
                directions.append(d / n)
        if len(directions) < 2:
            return 0.0
        dots = [float(np.dot(directions[i], directions[i+1]))
                for i in range(len(directions)-1)]
        return float(1.0 - np.mean(dots))

    def is_converging(self, window: int = 5) -> bool:
        """Is the trajectory settling — velocity decreasing?"""
        if len(self.points) < window + 1:
            return False
        recent   = [self.points[-i].meaning.distance(self.points[-i-1].meaning)
                    for i in range(1, window)]
        older    = [self.points[-window-i].meaning.distance(self.points[-window-i-1].meaning)
                    for i in range(1, min(window, len(self.points) - window))]
        if not older:
            return False
        return float(np.mean(recent)) < float(np.mean(older))

    def is_oscillating(self, window: int = 6) -> bool:
        """Is meaning cycling — returning to near-previous positions?"""
        pts = self.points[-window:]
        if len(pts) < 4:
            return False
        sims = []
        for i in range(len(pts)):
            for j in range(i + 2, len(pts)):
                sims.append(pts[i].meaning.cosine_similarity(pts[j].meaning))
        return float(np.max(sims)) > 0.92

    def centroid(self) -> NMeaning | None:
        """Mean position of the trajectory — the semantic 'home base'."""
        if not self.points:
            return None
        vecs = np.array([p.meaning.vector for p in self.points])
        mean = vecs.mean(axis=0)
        mean /= (np.linalg.norm(mean) + 1e-10)
        return NMeaning(vector=mean, label=f"{self.owner} centroid",
                        source_modality="trajectory")

    def summary(self) -> str:
        if len(self.points) < 2:
            return f"Trajectory({self.owner!r}  steps={len(self.points)})"
        return (
            f"Trajectory({self.owner!r}  steps={len(self.points)}"
            f"  vel={self.velocity():.3f}"
            f"  accel={self.acceleration():+.3f}"
            f"  curve={self.curvature():.3f}"
            f"  {'converging' if self.is_converging() else 'diverging'}"
            f"{'  oscillating' if self.is_oscillating() else ''})"
        )


# ── Diffusion ─────────────────────────────────────────────────────────────────

class DiffusionModel:
    """
    Passive meaning propagation through the participant network.
    Without explicit sends, participants' states drift toward neighbors.
    Models the background field of mutual influence in any communicating group.
    """

    def __init__(self, diffusion_rate: float = 0.05):
        self.rate = diffusion_rate   # how fast states bleed into each other

    def step(self, participants: dict, space) -> dict[str, float]:
        """
        One diffusion step. Each participant absorbs a fraction of the
        weighted mean of all other participants' states.
        Returns dict of {name: shift_magnitude}.
        """
        active = {n: p for n, p in participants.items() if p.state is not None}
        if len(active) < 2:
            return {}

        shifts = {}
        for name, participant in active.items():
            others = [p.state for n, p in active.items() if n != name]
            weights = [participant.receptivity] * len(others)
            if not others:
                continue
            field = space.compose(others, weights)
            new_state = space.interpolate(participant.state, field, t=self.rate)
            shift = participant.state.distance(new_state)
            participant.state = new_state
            participant.state.label = f"{name} (diffused)"
            shifts[name] = shift

        return shifts

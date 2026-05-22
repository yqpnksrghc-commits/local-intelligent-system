"""
Topology — the shape of the semantic space inhabited by a session.

Finds attractors (stable meanings the conversation orbits),
basins of attraction (which attractor each meaning falls toward),
semantic clusters, and the overall landscape of the session.

No external clustering library required — uses pure numpy k-means.
"""
from __future__ import annotations
import numpy as np
from core.multiversal.nmeaning import NMeaning
from core.multiversal.space import SemanticSpace


# ── K-means in ℝⁿ ─────────────────────────────────────────────────────────────

def _kmeans(vectors: np.ndarray, k: int, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Spherical k-means on unit vectors. Returns (centroids, labels)."""
    n = len(vectors)
    k = min(k, n)
    idx = np.random.choice(n, k, replace=False)
    centroids = vectors[idx].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # assign
        sims = vectors @ centroids.T          # (n, k)
        new_labels = sims.argmax(axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # update centroids
        for c in range(k):
            members = vectors[labels == c]
            if len(members):
                mean = members.mean(axis=0)
                norm = np.linalg.norm(mean)
                centroids[c] = mean / (norm + 1e-10)

    return centroids, labels


# ── Attractor detection ───────────────────────────────────────────────────────

def find_attractors(meanings: list[NMeaning], n: int = 3) -> list[NMeaning]:
    """
    Find n semantic attractors in a set of meanings via spherical k-means.
    Attractors are the stable points the conversation orbits.
    """
    if len(meanings) < n:
        return meanings
    vecs = np.array([m.vector / (np.linalg.norm(m.vector) + 1e-10) for m in meanings])
    centroids, labels = _kmeans(vecs, n)

    attractors = []
    for i, c in enumerate(centroids):
        members = [meanings[j] for j in range(len(meanings)) if labels[j] == i]
        # label the attractor with the most central member's label
        sims = [float(np.dot(c, m.vector / (np.linalg.norm(m.vector) + 1e-10)))
                for m in members]
        best = members[int(np.argmax(sims))] if members else None
        a = NMeaning(
            vector=c,
            label=f"attractor[{i}]: {best.label if best else '?'}",
            source_modality="topology",
            dims={"cluster_size": float(len(members))},
        )
        attractors.append(a)
    return attractors


def basin_of(meaning: NMeaning, attractors: list[NMeaning]) -> NMeaning:
    """Which attractor does this meaning fall toward?"""
    if not attractors:
        return meaning
    sims = [meaning.cosine_similarity(a) for a in attractors]
    return attractors[int(np.argmax(sims))]


# ── Landscape analysis ────────────────────────────────────────────────────────

def landscape(meanings: list[NMeaning], space: SemanticSpace,
              n_attractors: int = 3) -> dict:
    """
    Full topological analysis of a set of meanings.
    Returns attractors, participant assignments, density, and spread.
    """
    if not meanings:
        return {}

    vecs = np.array([m.vector / (np.linalg.norm(m.vector) + 1e-10) for m in meanings])

    # spread: mean pairwise distance
    n = len(vecs)
    if n > 1:
        sims = vecs @ vecs.T
        mask = ~np.eye(n, dtype=bool)
        spread = float(1.0 - np.mean(sims[mask]))
    else:
        spread = 0.0

    # centroid of the whole landscape
    mean_vec = vecs.mean(axis=0)
    mean_norm = np.linalg.norm(mean_vec)
    centroid = NMeaning(
        vector=mean_vec / (mean_norm + 1e-10),
        label="session centroid",
        source_modality="topology",
    )

    # attractors
    attractors = find_attractors(meanings, n_attractors) if n >= n_attractors else meanings

    # basin assignment
    assignments = {m.label or m.source_text[:30]: basin_of(m, attractors).label
                   for m in meanings}

    # density: inverse of mean nearest-neighbor distance
    density = 1.0 - spread

    return {
        "spread":      spread,
        "density":     density,
        "centroid":    centroid,
        "attractors":  attractors,
        "assignments": assignments,
        "n_meanings":  n,
    }


# ── Convergence field ─────────────────────────────────────────────────────────

def convergence_field(participant_states: list[NMeaning]) -> dict:
    """
    Measure whether participant states are converging or diverging over time.
    Requires at least two states.
    """
    if len(participant_states) < 2:
        return {"status": "insufficient_data"}

    vecs = np.array([s.vector / (np.linalg.norm(s.vector) + 1e-10)
                     for s in participant_states])
    sim_matrix = vecs @ vecs.T
    n = len(vecs)
    mask = ~np.eye(n, dtype=bool)
    mean_sim = float(np.mean(sim_matrix[mask]))
    min_sim  = float(np.min(sim_matrix[mask]))
    max_sim  = float(np.max(sim_matrix[mask]))

    return {
        "mean_similarity": mean_sim,
        "min_similarity":  min_sim,
        "max_similarity":  max_sim,
        "cohesion":        mean_sim,        # 1=perfectly aligned, 0=orthogonal
        "status": "converging" if mean_sim > 0.7 else
                  ("diverging" if mean_sim < 0.3 else "mixed"),
    }

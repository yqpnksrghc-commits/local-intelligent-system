"""
Signal encoder — maps any 1D signal into ℝⁿ.

Extracts a feature vector from the signal, then maps it into the semantic
space by constructing a semantic description from the features and embedding it.

Works for: audio waveforms, sensor streams, time series, any numeric sequence.
"""
from __future__ import annotations
import numpy as np
from core.multiversal.nmeaning import NMeaning
from core.multiversal.space import SemanticSpace


def _extract_features(signal: np.ndarray, sr: int = 16000) -> dict[str, float]:
    """Extract modality-agnostic signal features."""
    s = signal.astype(np.float64)
    s_norm = s / (np.max(np.abs(s)) + 1e-10)

    # time domain
    rms        = float(np.sqrt(np.mean(s_norm ** 2)))
    zcr        = float(np.mean(np.diff(np.sign(s_norm)) != 0))
    crest      = float(np.max(np.abs(s_norm)) / (rms + 1e-10))
    skewness   = float(np.mean(s_norm ** 3) / (np.std(s_norm) ** 3 + 1e-10))
    kurtosis   = float(np.mean(s_norm ** 4) / (np.std(s_norm) ** 4 + 1e-10)) - 3

    # frequency domain
    fft        = np.fft.rfft(s_norm)
    power      = np.abs(fft) ** 2
    freqs      = np.fft.rfftfreq(len(s_norm), 1.0 / sr)
    total_pow  = power.sum() + 1e-10
    centroid   = float(np.sum(freqs * power) / total_pow)
    bandwidth  = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_pow))
    rolloff_idx= np.searchsorted(np.cumsum(power), 0.85 * total_pow)
    rolloff    = float(freqs[min(rolloff_idx, len(freqs) - 1)])
    flatness   = float(np.exp(np.mean(np.log(power + 1e-10))) / (np.mean(power) + 1e-10))

    # entropy
    p = power / total_pow
    entropy    = float(-np.sum(p * np.log(p + 1e-10)))

    # temporal envelope
    frame_size = max(1, len(s_norm) // 20)
    frames     = [s_norm[i:i+frame_size] for i in range(0, len(s_norm), frame_size)]
    energies   = [float(np.sqrt(np.mean(f**2))) for f in frames if len(f) > 0]
    env_slope  = float(np.polyfit(range(len(energies)), energies, 1)[0]) if len(energies) > 1 else 0.0

    return {
        "rms": rms, "zcr": zcr, "crest_factor": crest,
        "skewness": skewness, "kurtosis": kurtosis,
        "spectral_centroid": centroid / (sr / 2),
        "spectral_bandwidth": bandwidth / (sr / 2),
        "spectral_rolloff": rolloff / (sr / 2),
        "spectral_flatness": flatness,
        "entropy": entropy / np.log(len(power) + 2),
        "envelope_slope": env_slope,
    }


def _features_to_description(features: dict[str, float]) -> str:
    """Convert signal features to a semantic description for embedding."""
    parts = []

    energy = features.get("rms", 0)
    if energy > 0.7:   parts.append("intense high-energy signal")
    elif energy > 0.4: parts.append("moderate energy signal")
    else:              parts.append("quiet low-energy signal")

    zcr = features.get("zcr", 0)
    if zcr > 0.3:      parts.append("rapid oscillation high frequency")
    elif zcr < 0.05:   parts.append("slow smooth low frequency")

    entropy = features.get("entropy", 0.5)
    if entropy > 0.8:  parts.append("complex disordered noise")
    elif entropy < 0.3:parts.append("ordered structured pattern")

    slope = features.get("envelope_slope", 0)
    if slope > 0.01:   parts.append("growing increasing intensity")
    elif slope < -0.01:parts.append("decaying fading diminishing")

    flatness = features.get("spectral_flatness", 0.5)
    if flatness > 0.7: parts.append("noise-like broad spectrum")
    elif flatness < 0.2:parts.append("tonal harmonic pure tone")

    centroid = features.get("spectral_centroid", 0.5)
    if centroid > 0.6: parts.append("bright sharp high frequency")
    elif centroid < 0.3:parts.append("dark heavy low frequency")

    return " ".join(parts) or "signal undefined"


def encode(signal: np.ndarray, space: SemanticSpace,
           sr: int = 16000, label: str = "signal") -> NMeaning:
    features = _extract_features(signal, sr)
    description = _features_to_description(features)
    vector = space.embed(description)

    # named dims from signal features (mapped to semantic axes)
    dims = {
        "intensity":    float(np.clip(features["rms"] * 2 - 1, -1, 1)),
        "complexity":   float(np.clip(features["entropy"] * 2 - 1, -1, 1)),
        "brightness":   float(np.clip(features["spectral_centroid"] * 2 - 1, -1, 1)),
        "order":        float(np.clip(1 - features["spectral_flatness"] * 2, -1, 1)),
        "becoming":     float(np.clip(features["envelope_slope"] * 100, -1, 1)),
    }

    return NMeaning(
        vector=vector,
        dims=dims,
        source_text=description,
        source_modality="signal",
        label=label,
    )

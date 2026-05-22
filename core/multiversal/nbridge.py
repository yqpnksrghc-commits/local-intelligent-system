"""
N-dimensional Multiversal Bridge.

All meaning lives as a point in ℝⁿ.
All operations happen in that space.
Modality is only relevant at the boundary (input/output).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from core.multiversal.space import SemanticSpace
from core.multiversal.nmeaning import NMeaning
from core.multiversal import ngrammar
from core.multiversal.encoders import text as text_enc
from core.multiversal.encoders import math as math_enc
from core.multiversal.encoders import signal as signal_enc
from core.multiversal.decoders import text as text_dec
from core.multiversal.decoders import music as music_dec
from core.multiversal.decoders import math as math_dec
from core.multiversal.decoders import symbol as symbol_dec

OUTPUT_DIR = Path("data/multiversal")


class NBridge:
    def __init__(self, llm: ChatOllama, embed_model: str = "nomic-embed-text",
                 ollama_host: str = "http://localhost:11434"):
        self.llm = llm
        embeddings = OllamaEmbeddings(model=embed_model, base_url=ollama_host)
        self.space = SemanticSpace(embed_fn=embeddings.embed_query)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Encoding ───────────────────────────────────────────────────────

    def encode(self, source: str, modality: str = "text",
               language: str | None = None) -> NMeaning:
        """Extract full NMeaning from any source."""
        print(f"[nbridge] encoding ({modality}) …")
        if modality == "math":
            meaning = math_enc.encode(source, self.space, self.llm)
        else:
            meaning = text_enc.encode(source, self.space, language=language,
                                      modality=modality)
        # enrich with discovered dimensions
        return ngrammar.extract(source, self.space, self.llm, modality, language)

    def encode_signal(self, signal: np.ndarray, sr: int = 16000,
                      label: str = "signal") -> NMeaning:
        return signal_enc.encode(signal, self.space, sr=sr, label=label)

    # ── Decoding ───────────────────────────────────────────────────────

    def decode(self, meaning: NMeaning, target: str, session_id: str = "0") -> str:
        """Render a meaning into a target modality."""
        t = target.lower().strip()

        if t == "analysis":
            return meaning.summary()

        elif t == "math":
            from core.multiversal.meaning import Meaning as OldMeaning
            return math_dec.decode(_to_old_meaning(meaning), self.llm)

        elif t == "music":
            out = OUTPUT_DIR / f"meaning_{session_id}.mid"
            from core.multiversal.meaning import Meaning as OldMeaning
            path = music_dec.decode(_to_old_meaning(meaning), str(out))
            desc = music_dec.describe(_to_old_meaning(meaning))
            return f"{desc}\nMIDI: {path}"

        elif t == "symbol":
            out = OUTPUT_DIR / f"meaning_{session_id}.svg"
            from core.multiversal.meaning import Meaning as OldMeaning
            symbol_dec.save(_to_old_meaning(meaning), str(out))
            return str(out)

        else:
            return text_dec.decode(_to_old_meaning(meaning), target, self.llm)

    # ── Transduction ───────────────────────────────────────────────────

    def transduce(self, source: str, targets: list[str],
                  source_modality: str = "text",
                  source_language: str | None = None,
                  session_id: str = "0") -> dict:
        meaning = self.encode(source, source_modality, source_language)
        print(f"[nbridge] meaning (dim={meaning.dim}):\n  {meaning.summary()}\n")
        results = {"source": source, "modality": source_modality,
                   "meaning": meaning.summary(), "dim": meaning.dim}
        for target in targets:
            print(f"[nbridge] → {target} …")
            results[target] = self.decode(meaning, target, session_id)
        return results

    # ── Space operations ───────────────────────────────────────────────

    def interpolate(self, source_a: str, source_b: str, t: float = 0.5,
                    targets: list[str] | None = None,
                    session_id: str = "0") -> dict:
        """Find the meaning at position t between two inputs, then decode it."""
        m_a = self.encode(source_a)
        m_b = self.encode(source_b)
        midpoint = self.space.interpolate(m_a, m_b, t)
        result = {"t": t, "meaning": midpoint.summary()}
        if targets:
            for target in targets:
                result[target] = self.decode(midpoint, target, session_id)
        return result

    def analogy(self, a: str, b: str, c: str,
                targets: list[str] | None = None,
                session_id: str = "0") -> dict:
        """a:b::c:? — solve the analogy and decode the result."""
        m_a = self.encode(a)
        m_b = self.encode(b)
        m_c = self.encode(c)
        result_m = self.space.analogy(m_a, m_b, m_c)
        out = {"query": f"{a!r}:{b!r}::{c!r}:?", "meaning": result_m.summary()}
        if targets:
            for target in targets:
                out[target] = self.decode(result_m, target, session_id)
        return out

    def compose(self, sources: list[str], weights: list[float] | None = None,
                targets: list[str] | None = None,
                session_id: str = "0") -> dict:
        """Combine multiple meanings into one, then decode."""
        meanings = [self.encode(s) for s in sources]
        composed = self.space.compose(meanings, weights)
        out = {"composed_from": sources, "meaning": composed.summary()}
        if targets:
            for target in targets:
                out[target] = self.decode(composed, target, session_id)
        return out

    def measure(self, source: str, axes: list[str]) -> dict:
        """Measure where a meaning falls on arbitrary named axes."""
        meaning = self.encode(source)
        return {
            "source": source,
            "dim":    meaning.dim,
            "axes":   self.space.measure_dims(meaning, axes),
        }

    def project(self, source: str, axes: list[str],
                targets: list[str] | None = None,
                session_id: str = "0") -> dict:
        """Project meaning onto a semantic subspace, then decode."""
        meaning = self.encode(source)
        projected = self.space.project(meaning, axes)
        out = {"projected_onto": axes, "meaning": projected.summary()}
        if targets:
            for target in targets:
                out[target] = self.decode(projected, target, session_id)
        return out

    def compare(self, source_a: str, source_b: str,
                modality_a: str = "text", modality_b: str = "text") -> dict:
        m_a = self.encode(source_a, modality_a)
        m_b = self.encode(source_b, modality_b)
        result = ngrammar.compare(m_a, m_b, self.space)
        result["a"] = m_a.summary()
        result["b"] = m_b.summary()
        return result

    def nearest(self, source: str, k: int = 5) -> list[dict]:
        meaning = self.encode(source)
        hits = self.space.nearest(meaning, k)
        return [{"label": m.label, "source": m.source_text[:60],
                 "similarity": sim} for m, sim in hits]


# ── Adapter for decoders that still use the old Meaning type ──────────────

def _to_old_meaning(nm: NMeaning):
    from core.multiversal.meaning import Meaning
    dims = nm.dims
    return Meaning(
        predicate=nm.label or nm.source_text[:40],
        source_text=nm.source_text,
        source_modality=nm.source_modality,
        valence=float(dims.get("valence",
                  dims.get("positivity",
                  dims.get("joy", 0.0)))),
        arousal=float(dims.get("arousal",
                 dims.get("intensity",
                 dims.get("energy", 0.5)))),
        dominance=float(dims.get("dominance",
                   dims.get("agency",
                   dims.get("force", 0.5)))),
        force=float(dims.get("certainty",
               dims.get("confidence", 1.0))),
        tense=nm.source_text,  # best-effort
        speech_act="assertion",
        tags=list(dims.keys())[:10],
    )

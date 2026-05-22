"""
The Multiversal Bridge.

Accepts any input in any modality, extracts its invariant Meaning,
then renders that Meaning into any target modality.

Input modalities:  text (any language), math, music_desc, symbol_desc, emotion
Output modalities: text, math, music (MIDI), symbol (SVG), analysis
"""
from __future__ import annotations
import os
from pathlib import Path
from langchain_ollama import ChatOllama
from core.multiversal.meaning import Meaning
from core.multiversal import grammar
from core.multiversal.decoders import text as text_dec
from core.multiversal.decoders import music as music_dec
from core.multiversal.decoders import math as math_dec
from core.multiversal.decoders import symbol as symbol_dec

OUTPUT_DIR = Path("data/multiversal")


class Bridge:
    def __init__(self, llm: ChatOllama, output_dir: Path = OUTPUT_DIR):
        self.llm = llm
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, source: str, modality: str = "text",
                language: str | None = None) -> Meaning:
        return grammar.extract(source, self.llm, modality, language)

    def transduce(
        self,
        source: str,
        targets: list[str],
        source_modality: str = "text",
        source_language: str | None = None,
        session_id: str = "0",
    ) -> dict:
        """
        Extract meaning from source, render into all requested target modalities.

        targets: list of any combination of:
          - any language name ("english", "swahili", "mandarin", …)
          - "math"
          - "music"
          - "symbol"
          - "analysis"
        """
        print(f"[bridge] extracting meaning from {source_modality} …")
        meaning = self.extract(source, source_modality, source_language)
        print(f"[bridge] meaning:\n  {meaning.summary()}\n")

        results: dict[str, str] = {
            "source":   source,
            "modality": source_modality,
            "meaning":  meaning.summary(),
        }

        for target in targets:
            t = target.lower().strip()
            if t == "analysis":
                results["analysis"] = meaning.summary()

            elif t == "math":
                print(f"[bridge] → math …")
                results["math"] = math_dec.decode(meaning, self.llm)

            elif t == "music":
                print(f"[bridge] → music …")
                out = self.output_dir / f"meaning_{session_id}.mid"
                path = music_dec.decode(meaning, str(out))
                desc = music_dec.describe(meaning)
                results["music"] = f"{desc}\nMIDI: {path}"

            elif t == "symbol":
                print(f"[bridge] → symbol …")
                out = self.output_dir / f"meaning_{session_id}.svg"
                symbol_dec.save(meaning, str(out))
                results["symbol"] = str(out)

            else:
                print(f"[bridge] → {target} …")
                results[target] = text_dec.decode(meaning, target, self.llm)

        return results

    def compare(self, source_a: str, source_b: str,
                modality_a: str = "text", modality_b: str = "text") -> dict:
        """Compare the meanings of two inputs across any modalities."""
        m_a = self.extract(source_a, modality_a)
        m_b = self.extract(source_b, modality_b)
        comparison = grammar.compare(m_a, m_b, self.llm)
        return {
            "a": m_a.summary(),
            "b": m_b.summary(),
            "shared":    comparison.get("shared", []),
            "divergent": comparison.get("divergent", []),
            "distance":  comparison.get("distance", 0.5),
        }

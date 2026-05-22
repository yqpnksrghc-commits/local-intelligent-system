"""
Text encoder — any language, any script → ℝⁿ.
The embedding model handles all surface variation.
"""
from __future__ import annotations
from core.multiversal.nmeaning import NMeaning
from core.multiversal.space import SemanticSpace


def encode(text: str, space: SemanticSpace,
           language: str | None = None, modality: str = "text") -> NMeaning:
    vector = space.embed(text)
    return NMeaning(
        vector=vector,
        source_text=text,
        source_modality=modality,
        source_language=language,
        label=text[:40],
    )

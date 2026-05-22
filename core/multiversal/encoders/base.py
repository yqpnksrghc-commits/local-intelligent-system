"""
Encoder protocol — every modality implements this.
All encoders return an NMeaning positioned in the shared SemanticSpace.
"""
from __future__ import annotations
from typing import Protocol
from core.multiversal.nmeaning import NMeaning
from core.multiversal.space import SemanticSpace


class Encoder(Protocol):
    def encode(self, source: str | bytes, space: SemanticSpace, **kwargs) -> NMeaning:
        ...

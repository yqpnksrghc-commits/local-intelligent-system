"""
Message — a meaning in transit.

Not text. Not sound. A vector moving through semantic space
from one participant toward another, subject to channel
attenuation and receiver receptivity before it lands.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from core.multiversal.nmeaning import NMeaning


@dataclass
class Message:
    meaning:     NMeaning
    sender:      str
    recipients:  list[str]          # empty = broadcast
    timestamp:   datetime           = field(default_factory=datetime.utcnow)
    channel_id:  str | None         = None

    # Set after reception — what actually arrived at each receiver
    received_as: dict[str, NMeaning] = field(default_factory=dict)
    decoded_as:  dict[str, str]      = field(default_factory=dict)

    def summary(self) -> str:
        to = ", ".join(self.recipients) if self.recipients else "ALL"
        return (f"[{self.timestamp.strftime('%H:%M:%S')}] "
                f"{self.sender} → {to} | {self.meaning.label!r}")

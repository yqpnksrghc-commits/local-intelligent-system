"""
Meaning → music.

Maps the VAD affective model and semantic structure onto musical parameters,
then writes a MIDI file. No external music theory library required.

Mapping logic:
  valence   → mode (major = positive, minor = negative, modal = neutral)
  arousal   → tempo + dynamics (excited = fast/loud, calm = slow/soft)
  dominance → register + interval density (dominant = low/full, submissive = high/sparse)
  force     → harmonic stability (certainty = consonant, doubt = dissonant)
  tense     → melodic contour (future = ascending, past = descending, present = arch)
  speech_act→ phrase shape (question = unresolved, assertion = resolved cadence)
"""
from __future__ import annotations
import struct
import os
import tempfile
from core.multiversal.meaning import Meaning

# ── MIDI constants ─────────────────────────────────────────────────────────────

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
DORIAN      = [0, 2, 3, 5, 7, 9, 10]

TONIC = 60   # middle C


def _var_len(value: int) -> bytes:
    result = []
    result.append(value & 0x7F)
    value >>= 7
    while value:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(result))


def _midi_header(num_tracks: int = 1, ticks: int = 480) -> bytes:
    return b"MThd" + struct.pack(">IHHH", 6, 0, num_tracks, ticks)


def _note_event(delta: int, note: int, velocity: int, on: bool) -> bytes:
    status = 0x90 if on else 0x80
    return _var_len(delta) + bytes([status, max(0, min(127, note)), max(0, min(127, velocity))])


def _tempo_event(bpm: int) -> bytes:
    us = int(60_000_000 / bpm)
    return b"\x00\xff\x51\x03" + struct.pack(">I", us)[1:]


def _end_track() -> bytes:
    return b"\x00\xff\x2f\x00"


def _build_track(notes: list[tuple[int, int, int, int]]) -> bytes:
    """notes: list of (delta_ticks, pitch, velocity, duration_ticks)"""
    events = b""
    for delta, pitch, vel, dur in notes:
        events += _note_event(delta, pitch, vel, True)
        events += _note_event(dur, pitch, vel, False)
    events += _end_track()
    return b"MTrk" + struct.pack(">I", len(events)) + events


def decode(meaning: Meaning, output_path: str | None = None) -> str:
    """
    Generate a MIDI file from a Meaning. Returns the file path.
    """
    # ── parameter mapping ──────────────────────────────────────────────
    scale    = MAJOR_SCALE if meaning.valence > 0.15 else (MINOR_SCALE if meaning.valence < -0.15 else DORIAN)
    tempo    = int(60 + meaning.arousal * 120)          # 60–180 BPM
    velocity = int(50 + meaning.arousal * 60)           # 50–110
    register = int(TONIC - meaning.dominance * 12)      # dominant = lower register
    ticks    = 480
    dur      = int(ticks * (1.5 - meaning.arousal))     # slow when calm

    # dissonance: add tension notes when force is low (doubt)
    tension_offset = 0 if meaning.force > 0.6 else (6 if meaning.force < 0.3 else 1)

    # contour based on tense
    contour = {
        "future": [0, 1, 2, 3, 4, 5, 6, 7],       # ascending
        "past":   [7, 6, 5, 4, 3, 2, 1, 0],       # descending
        "present": [0, 2, 4, 6, 7, 6, 4, 2],      # arch
        "timeless": [0, 4, 7, 4, 0, 4, 7, 4],     # oscillating
    }.get(meaning.tense, [0, 2, 4, 5, 7, 5, 4, 2])

    # resolve or not based on speech_act
    if meaning.speech_act == "question":
        contour = contour[:-1] + [contour[-1] + 1]  # leave unresolved

    # ── build note sequence ────────────────────────────────────────────
    track_events: list[tuple[int, int, int, int]] = []
    for i, step in enumerate(contour):
        scale_idx = step % len(scale)
        octave    = step // len(scale)
        pitch     = register + scale[scale_idx] + octave * 12 + tension_offset
        delta     = 0 if i == 0 else 0
        track_events.append((0, pitch, velocity, dur))

    # ── write MIDI ─────────────────────────────────────────────────────
    header = _midi_header()
    track  = b"\x00" + _tempo_event(tempo) + b"".join(
        _note_event(0, p, v, True) + _note_event(d, p, v, False)
        for (_, p, v, d) in track_events
    ) + _end_track()
    track  = b"MTrk" + struct.pack(">I", len(track)) + track

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".mid")
        os.close(fd)

    with open(output_path, "wb") as f:
        f.write(header + track)

    return output_path


def describe(meaning: Meaning) -> str:
    """Human-readable musical description of the meaning."""
    mode  = "major" if meaning.valence > 0.15 else ("minor" if meaning.valence < -0.15 else "dorian")
    tempo = int(60 + meaning.arousal * 120)
    dyn   = "fff" if meaning.arousal > 0.8 else ("ff" if meaning.arousal > 0.6 else
            ("mf" if meaning.arousal > 0.4 else ("p" if meaning.arousal > 0.2 else "pp")))
    contour = {"future": "ascending", "past": "descending",
               "present": "arched", "timeless": "oscillating"}.get(meaning.tense, "arched")
    cadence = "unresolved (question)" if meaning.speech_act == "question" else "resolved"
    return (
        f"Mode: {mode}  |  Tempo: {tempo} BPM  |  Dynamics: {dyn}\n"
        f"Contour: {contour}  |  Cadence: {cadence}\n"
        f"Harmonic tension: {'high (doubt)' if meaning.force < 0.4 else 'low (certainty)'}"
    )

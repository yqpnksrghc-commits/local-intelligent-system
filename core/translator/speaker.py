"""
Text-to-speech output with runtime volume control.
Volume is a float 0.0 (silent) → 1.0 (full). Default: 0.8.
"""
import threading
import pyttsx3

_engine: pyttsx3.Engine | None = None
_lock = threading.Lock()
_volume: float = 0.8


def _get_engine() -> pyttsx3.Engine:
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 165)
        _engine.setProperty("volume", _volume)
    return _engine


def set_volume(level: float) -> None:
    """Set output volume. level must be between 0.0 and 1.0."""
    global _volume
    _volume = max(0.0, min(1.0, level))
    with _lock:
        eng = _get_engine()
        eng.setProperty("volume", _volume)
    print(f"[speaker] volume → {int(_volume * 100)}%")


def get_volume() -> float:
    return _volume


def speak(text: str) -> None:
    """Speak text aloud at the current volume. Blocks until done."""
    if not text:
        return
    with _lock:
        eng = _get_engine()
        eng.say(text)
        eng.runAndWait()


def set_rate(wpm: int) -> None:
    with _lock:
        _get_engine().setProperty("rate", wpm)

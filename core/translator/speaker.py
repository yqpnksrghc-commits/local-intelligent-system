"""
TTS output via edge-tts — 140+ languages, 400+ neural voices.
Volume is a float 0.0 (silent) → 1.0 (full). Default: 0.8.
Voice is auto-selected from the target language name.
"""
import asyncio
import tempfile
import os
import threading
import edge_tts
import pygame

pygame.mixer.init()

_volume: float = 0.8
_voice_cache: dict[str, str] = {}
_voices_loaded: list | None = None
_lock = threading.Lock()


async def _load_voices() -> list:
    global _voices_loaded
    if _voices_loaded is None:
        _voices_loaded = await edge_tts.list_voices()
    return _voices_loaded


async def _resolve_voice(language: str) -> str:
    """Pick the best available voice for a given language name or code."""
    if language in _voice_cache:
        return _voice_cache[language]

    voices = await _load_voices()
    lang_lower = language.lower()

    # try exact locale match (e.g. "en" → en-US, "zh" → zh-CN)
    # then fuzzy match on FriendlyName or Locale
    candidates = [
        v for v in voices
        if lang_lower in v["Locale"].lower()
        or lang_lower in v["FriendlyName"].lower()
    ]

    if not candidates:
        # fallback: ask for any voice that contains part of the lang string
        candidates = [v for v in voices if any(
            part in v["Locale"].lower() or part in v["FriendlyName"].lower()
            for part in lang_lower.split()
        )]

    if not candidates:
        # last resort: English
        candidates = [v for v in voices if v["Locale"].startswith("en-")]

    # prefer Neural voices
    neural = [v for v in candidates if "Neural" in v.get("VoiceTag", {}).get("VoicePersonalities", [])]
    chosen = (neural or candidates)[0]["ShortName"]
    _voice_cache[language] = chosen
    return chosen


async def _speak_async(text: str, language: str, volume: float) -> None:
    voice = await _resolve_voice(language)
    vol_str = f"{int((volume * 100) - 100):+d}%"   # edge-tts volume is relative to 0%

    communicate = edge_tts.Communicate(text, voice, volume=vol_str)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name

    await communicate.save(tmp_path)

    with _lock:
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(50)

    os.unlink(tmp_path)


def set_volume(level: float) -> None:
    global _volume
    _volume = max(0.0, min(1.0, level))
    with _lock:
        pygame.mixer.music.set_volume(_volume)
    print(f"[speaker] volume → {int(_volume * 100)}%")


def get_volume() -> float:
    return _volume


def speak(text: str, language: str = "english") -> None:
    """Speak text in the given language at the current volume. Blocks until done."""
    if not text:
        return
    asyncio.run(_speak_async(text, language, _volume))

"""
Universal translator pipeline.
Modes:
  - voice: mic → STT → translate → TTS
  - text:  stdin → translate → TTS + stdout
"""
from __future__ import annotations
import numpy as np
from langchain_ollama import ChatOllama
from core.translator import listener, translator, speaker

LANG_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French",
    "de": "German",  "zh": "Chinese", "ja": "Japanese",
    "ar": "Arabic",  "pt": "Portuguese", "ru": "Russian",
    "ko": "Korean",  "it": "Italian",  "hi": "Hindi",
}


class UniversalTranslator:
    def __init__(
        self,
        llm: ChatOllama,
        target_lang: str = "en",
        mode: str = "voice",         # "voice" | "text"
        whisper_model: str = "base",
    ):
        self.llm = llm
        self.target_lang = target_lang
        self.mode = mode
        self.whisper_model = whisper_model
        self._running = False

    # ------------------------------------------------------------------ #
    # Public controls                                                       #
    # ------------------------------------------------------------------ #

    def set_target(self, lang_code: str) -> None:
        self.target_lang = lang_code.lower()
        name = LANG_NAMES.get(self.target_lang, self.target_lang)
        print(f"[translator] target language → {name}")

    def set_volume(self, level: float) -> None:
        speaker.set_volume(level)

    def translate_text(self, text: str, source_lang: str | None = None) -> str:
        target_name = LANG_NAMES.get(self.target_lang, self.target_lang)
        result = translator.translate(text, target_name, source_lang, self.llm)
        return result

    def process_once(self) -> dict:
        """
        Run one translation cycle.
        Returns {"source": ..., "source_lang": ..., "translation": ...}
        """
        if self.mode == "voice":
            audio = listener.record()
            if audio.size == 0:
                return {}
            text, detected = listener.transcribe(audio, model_size=self.whisper_model)
        else:
            text = input("Input: ").strip()
            detected = None

        if not text:
            return {}

        source_name = LANG_NAMES.get(detected, detected) if detected else None
        print(f"[heard] ({source_name or '?'}) {text}")

        translation = self.translate_text(text, source_name)
        target_name = LANG_NAMES.get(self.target_lang, self.target_lang)
        print(f"[{target_name}] {translation}")
        speaker.speak(translation)

        return {"source": text, "source_lang": detected, "translation": translation}

    def run(self) -> None:
        """Continuous loop. Commands parsed from a parallel text channel."""
        self._running = True
        print(f"\nUniversal Translator active — target: {LANG_NAMES.get(self.target_lang, self.target_lang)}")
        print("Commands: 'lang <code>'  'vol <0-100>'  'mode voice|text'  'exit'\n")

        while self._running:
            try:
                self.process_once()
            except KeyboardInterrupt:
                break

        self._running = False
        print("[translator] stopped.")

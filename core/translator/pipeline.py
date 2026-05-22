"""
Universal translator pipeline — no hardcoded language list.
Whisper detects source (99 languages).
Ollama translates to any language.
edge-tts speaks output in 140+ languages with 400+ neural voices.

Modes:
  voice — mic → STT → translate → TTS
  text  — stdin → translate → TTS + stdout
"""
from __future__ import annotations
from langchain_ollama import ChatOllama
from core.translator import listener, translator, speaker


class UniversalTranslator:
    def __init__(
        self,
        llm: ChatOllama,
        target_lang: str = "english",
        mode: str = "voice",
        whisper_model: str = "base",
    ):
        self.llm = llm
        self.target_lang = target_lang
        self.mode = mode
        self.whisper_model = whisper_model
        self._running = False

    def set_target(self, lang: str) -> None:
        self.target_lang = lang.strip()
        print(f"[translator] target language → {self.target_lang}")

    def set_volume(self, level: float) -> None:
        speaker.set_volume(level)

    def translate_text(self, text: str, source_lang: str | None = None) -> str:
        return translator.translate(text, self.target_lang, source_lang, self.llm)

    def process_once(self) -> dict:
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

        print(f"[heard] ({detected or '?'}) {text}")

        translation = self.translate_text(text, detected)
        print(f"[{self.target_lang}] {translation}")
        speaker.speak(translation, language=self.target_lang)

        return {"source": text, "source_lang": detected, "translation": translation}

    def run(self) -> None:
        self._running = True
        print(f"\nUniversal Translator active")
        print(f"Target: {self.target_lang}  |  Mode: {self.mode}  |  Vol: {int(speaker.get_volume() * 100)}%")
        print("Commands: 'lang <name>'  'vol <0-100>'  'mode voice|text'  'exit'\n")

        while self._running:
            try:
                self.process_once()
            except KeyboardInterrupt:
                break

        self._running = False
        print("[translator] stopped.")

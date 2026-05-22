"""
Universal Translator — entry point.

Usage:
    python translate.py [--lang english] [--mode voice|text] [--vol 80] [--whisper base|small|medium|large]

    --lang accepts any language by name or code: english, spanish, mandarin,
           swahili, yoruba, hindi, arabic, japanese, welsh, zulu, tagalog …

Runtime commands (typed while running):
    lang <name>     change target language  (e.g. lang swahili)
    vol <0-100>     adjust output volume    (e.g. vol 65)
    mode voice      switch to microphone input
    mode text       switch to text input
    exit            quit
"""
import argparse
import threading
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from core.self.health import wait_for_ollama
from core.translator.pipeline import UniversalTranslator
from core.translator import speaker

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL       = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


def command_loop(t: UniversalTranslator) -> None:
    while t._running:
        try:
            cmd = input().strip()
        except (EOFError, KeyboardInterrupt):
            t._running = False
            break

        parts = cmd.split(maxsplit=1)
        if not parts:
            continue
        op = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if op == "exit":
            t._running = False

        elif op == "lang":
            if arg:
                t.set_target(arg)
            else:
                print("Usage: lang <language name>")

        elif op == "vol":
            try:
                t.set_volume(float(arg) / 100.0)
            except ValueError:
                print("vol expects a number 0-100")

        elif op == "mode":
            if arg in ("voice", "text"):
                t.mode = arg
                print(f"[translator] mode → {arg}")
            else:
                print("mode must be 'voice' or 'text'")


def main():
    parser = argparse.ArgumentParser(description="Universal Translator")
    parser.add_argument("--lang",    default="english", help="Target language (any name or code)")
    parser.add_argument("--mode",    default="voice",   choices=["voice", "text"])
    parser.add_argument("--vol",     default=80,        type=int, help="Volume 0-100")
    parser.add_argument("--whisper", default="base",    choices=["tiny", "base", "small", "medium", "large"])
    args = parser.parse_args()

    wait_for_ollama(OLLAMA_HOST)

    llm = ChatOllama(model=MODEL, temperature=0.1, base_url=OLLAMA_HOST)

    t = UniversalTranslator(
        llm=llm,
        target_lang=args.lang,
        mode=args.mode,
        whisper_model=args.whisper,
    )
    t.set_volume(args.vol / 100.0)

    cmd_thread = threading.Thread(target=command_loop, args=(t,), daemon=True)
    cmd_thread.start()

    t.run()


if __name__ == "__main__":
    main()

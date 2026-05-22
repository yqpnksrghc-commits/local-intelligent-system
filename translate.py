"""
Universal Translator — entry point.

Usage:
    python translate.py [--lang en] [--mode voice|text] [--vol 80] [--whisper base|small|medium]

Runtime commands (typed while running):
    lang <code>     change target language  (e.g. lang es)
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
    """Reads commands from stdin concurrently with the translation loop."""
    while t._running:
        try:
            cmd = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            t._running = False
            break

        parts = cmd.split()
        if not parts:
            continue

        if parts[0] == "exit":
            t._running = False

        elif parts[0] == "lang" and len(parts) > 1:
            t.set_target(parts[1])

        elif parts[0] == "vol" and len(parts) > 1:
            try:
                level = float(parts[1]) / 100.0
                t.set_volume(level)
            except ValueError:
                print("vol expects a number 0-100")

        elif parts[0] == "mode" and len(parts) > 1:
            if parts[1] in ("voice", "text"):
                t.mode = parts[1]
                print(f"[translator] mode → {t.mode}")
            else:
                print("mode must be 'voice' or 'text'")


def main():
    parser = argparse.ArgumentParser(description="Universal Translator")
    parser.add_argument("--lang",    default="en",    help="Target language code (default: en)")
    parser.add_argument("--mode",    default="voice", choices=["voice", "text"])
    parser.add_argument("--vol",     default=80,      type=int, help="Initial volume 0-100")
    parser.add_argument("--whisper", default="base",  choices=["tiny", "base", "small", "medium", "large"])
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

    # command listener runs alongside translation loop
    cmd_thread = threading.Thread(target=command_loop, args=(t,), daemon=True)
    cmd_thread.start()

    t.run()


if __name__ == "__main__":
    main()

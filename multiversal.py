"""
Multiversal Translator — entry point.

Translates meaning across any modality:
  language, mathematics, music, geometric symbol, emotional description.

Usage:
    python multiversal.py

Session commands:
    transduce <source text>
        → prompts for source modality and target modalities
    compare
        → compare the meanings of two inputs
    exit

Examples:
    > transduce
    Input: E = mc²
    Source modality [text]: math
    Targets (comma-separated) [english, symbol, music]: swahili, music, symbol, math

    > transduce
    Input: The wound heals, but slowly.
    Source modality [text]:
    Targets: music, symbol, math, japanese
"""
import os
import json
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from core.self.health import wait_for_ollama
from core.multiversal.bridge import Bridge

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL       = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


def print_results(results: dict) -> None:
    print()
    skip = {"source", "modality", "meaning"}
    print("── Meaning ─────────────────────────────────────")
    print(results.get("meaning", ""))
    print()
    for key, val in results.items():
        if key in skip:
            continue
        print(f"── {key.upper()} ─────────────────────────────────────")
        print(val)
        print()


def run_transduce(bridge: Bridge, session_id: str) -> None:
    source = input("Input: ").strip()
    if not source:
        return
    modality = input("Source modality [text]: ").strip() or "text"
    raw_targets = input("Targets (comma-separated) [english, symbol, music]: ").strip()
    targets = [t.strip() for t in raw_targets.split(",")] if raw_targets else ["english", "symbol", "music"]
    results = bridge.transduce(source, targets, source_modality=modality, session_id=session_id)
    print_results(results)


def run_compare(bridge: Bridge) -> None:
    print("── Input A ──────────────────────────────────────")
    a = input("Input A: ").strip()
    mod_a = input("Modality A [text]: ").strip() or "text"
    print("── Input B ──────────────────────────────────────")
    b = input("Input B: ").strip()
    mod_b = input("Modality B [text]: ").strip() or "text"
    result = bridge.compare(a, b, mod_a, mod_b)
    print(f"\nA: {result['a']}\n")
    print(f"B: {result['b']}\n")
    print(f"Shared:    {result['shared']}")
    print(f"Divergent: {result['divergent']}")
    print(f"Distance:  {result['distance']:.2f}  (0=identical meaning, 1=unrelated)\n")


def main():
    wait_for_ollama(OLLAMA_HOST)
    llm    = ChatOllama(model=MODEL, temperature=0.2, base_url=OLLAMA_HOST)
    bridge = Bridge(llm)

    print("\nMultiversal Translator")
    print("Meaning is the invariant. Modality is the surface.")
    print("Commands: transduce | compare | exit\n")

    session = 0
    while True:
        try:
            cmd = input("› ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd in ("exit", "quit"):
            break
        elif cmd == "transduce":
            run_transduce(bridge, str(session))
            session += 1
        elif cmd == "compare":
            run_compare(bridge)
        elif cmd == "":
            continue
        else:
            print("Commands: transduce | compare | exit")


if __name__ == "__main__":
    main()

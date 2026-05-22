"""
Meaning → any human language.
"""
from __future__ import annotations
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from core.multiversal.meaning import Meaning

_SYSTEM = SystemMessage(content=(
    "You are a universal meaning renderer. You receive a semantic meaning structure "
    "and render it as natural, idiomatic text in the requested language. "
    "Preserve the full meaning — predicate, roles, modality, affect, tense — "
    "using the natural conventions of the target language. Output only the rendered text."
))


def decode(meaning: Meaning, language: str, llm: ChatOllama) -> str:
    prompt = (
        f"Render the following meaning as natural {language} text:\n\n"
        f"{meaning.summary()}\n\n"
        f"Original source for reference only: {meaning.source_text!r}"
    )
    return llm.invoke([_SYSTEM, HumanMessage(content=prompt)]).content.strip()

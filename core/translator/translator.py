"""
LLM-based translation.
Uses the existing Ollama instance — no external API calls.
Auto-detects source language when not specified.
"""
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM = SystemMessage(content=(
    "You are a precise translation engine. "
    "When given text and a target language, output ONLY the translation — "
    "no explanation, no punctuation changes beyond what is natural, no extra words. "
    "If the source text is already in the target language, return it unchanged."
))


def translate(
    text: str,
    target_lang: str,
    source_lang: str | None = None,
    llm: ChatOllama | None = None,
) -> str:
    if not text:
        return ""

    source_clause = f"from {source_lang} " if source_lang else ""
    prompt = f"Translate the following {source_clause}to {target_lang}:\n\n{text}"

    response = llm.invoke([SYSTEM, HumanMessage(content=prompt)])
    return response.content.strip()


def detect_language(text: str, llm: ChatOllama) -> str:
    """Returns the ISO 639-1 language code (e.g. 'en', 'es', 'zh')."""
    prompt = (
        f"What language is this text written in? "
        f"Reply with only the ISO 639-1 two-letter code, nothing else.\n\n{text}"
    )
    result = llm.invoke([HumanMessage(content=prompt)])
    return result.content.strip().lower()[:5]

"""
Math encoder — formal expressions → ℝⁿ.

Converts the expression to its semantic description first,
then embeds that description. This grounds mathematical structure
in the same space as all other modalities.
"""
from __future__ import annotations
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from core.multiversal.nmeaning import NMeaning
from core.multiversal.space import SemanticSpace

_SYSTEM = SystemMessage(content=(
    "You are a mathematical semanticist. Translate this mathematical expression "
    "into a rich natural-language description of its MEANING — what it asserts "
    "about the world, what relationships it encodes, what it implies. "
    "Do not explain how to compute it. Describe what it means. 2-4 sentences."
))


def encode(expression: str, space: SemanticSpace, llm: ChatOllama) -> NMeaning:
    description = llm.invoke([_SYSTEM, HumanMessage(content=expression)]).content.strip()
    vector = space.embed(description)
    return NMeaning(
        vector=vector,
        source_text=expression,
        source_modality="math",
        label=expression[:40],
    )

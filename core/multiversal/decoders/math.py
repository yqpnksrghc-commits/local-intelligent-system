"""
Meaning → formal mathematical / logical notation.

Maps the predicate-argument structure onto:
  - First-order logic (FOL) for relational meanings
  - Set-theoretic notation for categorical meanings
  - Differential / functional form for processes
  - Probability notation for uncertain / modal meanings
"""
from __future__ import annotations
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from core.multiversal.meaning import Meaning

_SYSTEM = SystemMessage(content="""
You are a mathematical semanticist. You receive a universal meaning structure
and express it in the most appropriate formal notation:

- Use first-order logic (∀, ∃, →, ∧, ∨, ¬) for relational/propositional meaning
- Use set theory (∈, ⊆, ∩, ∪, ∅) for categorical meaning
- Use functions f: X → Y for process/transformation meaning
- Use probability P(X|Y) for uncertain or modal meaning
- Use differential equations for change-over-time meaning
- Use graph notation G=(V,E) for relational network meaning

Choose the form that captures the meaning most precisely.
Output the formal expression followed by a one-line plain-language gloss.
""")


def decode(meaning: Meaning, llm: ChatOllama) -> str:
    prompt = (
        f"Express the following meaning in formal mathematical/logical notation:\n\n"
        f"{meaning.summary()}"
    )
    return llm.invoke([_SYSTEM, HumanMessage(content=prompt)]).content.strip()

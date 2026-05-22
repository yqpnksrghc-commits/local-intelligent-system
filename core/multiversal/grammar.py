"""
Universal Grammar Engine.
Extracts a Meaning SIR from any surface-level input using the LLM
as a semantic parser — language-agnostic, modality-aware.
"""
from __future__ import annotations
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from core.multiversal.meaning import Meaning

_EXTRACT_SYSTEM = SystemMessage(content="""
You are a universal semantic parser. Your task is to extract the deep meaning
structure from any input — text in any language, mathematical notation, musical
description, symbolic description, emotional expression, or code.

You extract the INVARIANT MEANING beneath the surface form.

Return ONLY a valid JSON object with these fields (omit any you cannot determine):
{
  "predicate":  "<core action, state, or relation>",
  "agent":      "<who/what initiates — null if absent>",
  "patient":    "<who/what receives — null if absent>",
  "theme":      "<what is transferred or described — null if absent>",
  "location":   "<where — null if absent>",
  "time":       "<when — null if absent>",
  "manner":     "<how — null if absent>",
  "cause":      "<why — null if absent>",
  "speech_act": "<assertion|question|command|exclamation>",
  "modality":   "<factual|possible|necessary|conditional>",
  "force":      <0.0 to 1.0 — certainty>,
  "perspective": "<first|second|third|universal>",
  "valence":    <-1.0 to 1.0 — negative to positive>,
  "arousal":    <0.0 to 1.0 — calm to excited>,
  "dominance":  <0.0 to 1.0 — submissive to dominant>,
  "tense":      "<past|present|future|timeless>",
  "aspect":     "<simple|progressive|perfect|habitual>",
  "tags":       ["<semantic tag>", ...]
}
""")


def extract(text: str, llm: ChatOllama, source_modality: str = "text",
            source_language: str | None = None) -> Meaning:
    prompt = f"Input ({source_modality}):\n\n{text}"
    response = llm.invoke([_EXTRACT_SYSTEM, HumanMessage(content=prompt)])

    raw = response.content.strip()
    # strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    return Meaning(
        predicate=data.get("predicate", ""),
        agent=data.get("agent"),
        patient=data.get("patient"),
        theme=data.get("theme"),
        location=data.get("location"),
        time=data.get("time"),
        manner=data.get("manner"),
        cause=data.get("cause"),
        speech_act=data.get("speech_act", "assertion"),
        modality=data.get("modality", "factual"),
        force=float(data.get("force", 1.0)),
        perspective=data.get("perspective", "third"),
        valence=float(data.get("valence", 0.0)),
        arousal=float(data.get("arousal", 0.5)),
        dominance=float(data.get("dominance", 0.5)),
        tense=data.get("tense", "present"),
        aspect=data.get("aspect", "simple"),
        tags=data.get("tags", []),
        source_text=text,
        source_modality=source_modality,
        source_language=source_language,
    )


def compare(a: Meaning, b: Meaning, llm: ChatOllama) -> dict:
    """
    Compare two Meanings for semantic distance across any dimensions.
    Returns a dict: {shared, divergent, distance_estimate}.
    """
    prompt = (
        f"Compare these two universal meaning structures and identify what they share "
        f"and where they diverge:\n\nA: {a.summary()}\n\nB: {b.summary()}\n\n"
        f"Return JSON: {{\"shared\": [...], \"divergent\": [...], \"distance\": 0.0-1.0}}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json")
    try:
        return json.loads(raw)
    except Exception:
        return {"shared": [], "divergent": [], "distance": 0.5}

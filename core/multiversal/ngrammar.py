"""
N-dimensional grammar engine.

Extracts NMeaning from any input by:
1. Embedding the input directly into ℝⁿ via the semantic space
2. Using the LLM to discover the most salient named dimensions
3. Producing a label (nearest interpretable description)

No fixed slots. The space defines what's relevant.
"""
from __future__ import annotations
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from core.multiversal.nmeaning import NMeaning
from core.multiversal.space import SemanticSpace

_DIM_SYSTEM = SystemMessage(content="""
You are a semantic analyst. Given an input in any modality, identify the most
semantically significant dimensions of its meaning.

Return ONLY a JSON array of objects, each with:
  {"axis": "<dimension name>", "value": <float -1.0 to 1.0>, "rationale": "<one phrase>"}

Choose axes that are:
- Specific to this meaning (not generic)
- Orthogonal to each other (not redundant)
- Grounded in observable properties of the input

Examples of valid axes: temporality, agency, certainty, intensity, abstraction,
relationality, becoming, dissolution, recursion, symmetry, contradiction, emergence,
entropy, tension, resolution, scale, interiority, force, negation, becoming ...

Return 8-12 axes. No preamble.
""")

_LABEL_SYSTEM = SystemMessage(content=(
    "Given a meaning structure, produce a single concise label (3-7 words) that captures "
    "its essence across all modalities. No punctuation. No explanation."
))


def extract(
    source: str,
    space: SemanticSpace,
    llm: ChatOllama,
    modality: str = "text",
    language: str | None = None,
) -> NMeaning:
    # Step 1: embed into ℝⁿ
    vector = space.embed(source)

    # Step 2: discover named dimensions via LLM
    prompt = f"Input ({modality}):\n\n{source}"
    raw = llm.invoke([_DIM_SYSTEM, HumanMessage(content=prompt)]).content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json\n")
    try:
        dim_data = json.loads(raw)
        dims = {d["axis"]: float(d["value"]) for d in dim_data if "axis" in d and "value" in d}
    except Exception:
        dims = {}

    # verify dims against actual embedding geometry
    verified = {}
    for axis_name, llm_value in dims.items():
        geometric_value = space.axis_value(
            NMeaning(vector=vector, source_text=source, source_modality=modality),
            axis_name,
        )
        # blend: LLM semantic insight + geometric measurement
        verified[axis_name] = 0.5 * llm_value + 0.5 * geometric_value

    # Step 3: produce a label
    dim_summary = ", ".join(f"{k}:{v:+.2f}" for k, v in list(verified.items())[:6])
    label_prompt = f"Source: {source[:200]}\nKey dimensions: {dim_summary}"
    label = llm.invoke([_LABEL_SYSTEM, HumanMessage(content=label_prompt)]).content.strip()

    meaning = NMeaning(
        vector=vector,
        dims=verified,
        source_text=source,
        source_modality=modality,
        source_language=language,
        label=label,
    )

    space.add(meaning)
    return meaning


def compare(a: NMeaning, b: NMeaning, space: SemanticSpace) -> dict:
    """
    Full comparison: geometric distance + dimensional divergence.
    """
    dist = a.distance(b)
    sim  = a.cosine_similarity(b)

    # which dims exist in both?
    shared_axes = set(a.dims) & set(b.dims)
    dim_diff = {ax: b.dims[ax] - a.dims[ax] for ax in shared_axes}
    converging = {ax: d for ax, d in dim_diff.items() if abs(d) < 0.2}
    diverging  = {ax: d for ax, d in dim_diff.items() if abs(d) >= 0.2}

    # find the principal axis of difference
    if len(a.vector) == len(b.vector):
        diff_vec = b.vector - a.vector
        diff_norm = float(np.linalg.norm(diff_vec)) if hasattr(diff_vec, '__len__') else 0.0
    else:
        diff_norm = 0.0

    return {
        "cosine_similarity": sim,
        "angular_distance":  dist,
        "converging_dims":   converging,
        "diverging_dims":    diverging,
        "unique_to_a":       {k: v for k, v in a.dims.items() if k not in b.dims},
        "unique_to_b":       {k: v for k, v in b.dims.items() if k not in a.dims},
    }


# deferred import to avoid circular at module level
try:
    import numpy as np
except ImportError:
    pass

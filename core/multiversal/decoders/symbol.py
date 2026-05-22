"""
Meaning → geometric / symbolic SVG.

Geometric primitives carry universal cross-cultural meaning:
  Circle    — wholeness, continuity, unity
  Triangle  — direction, tension, force, conflict
  Square    — stability, structure, completion
  Spiral    — growth, recursion, becoming
  Wave      — change, rhythm, uncertainty
  Line      — connection, causality, time

Color (HSL):
  Hue       ← valence  (red = negative, green = positive, blue = neutral)
  Saturation← arousal  (grey = calm, vivid = excited)
  Lightness ← dominance (dark = dominant, light = submissive)

Composition:
  Number of elements ← complexity of predicate-argument structure
  Size of primary    ← force / certainty
  Orientation        ← tense (ascending = future, descending = past, level = present)
"""
from __future__ import annotations
import math
from core.multiversal.meaning import Meaning


def _hsl_to_hex(h: float, s: float, l: float) -> str:
    h = h % 360
    s /= 100; l /= 100
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2
    if   h < 60:  r, g, b = c, x, 0
    elif h < 120: r, g, b = x, c, 0
    elif h < 180: r, g, b = 0, c, x
    elif h < 240: r, g, b = 0, x, c
    elif h < 300: r, g, b = x, 0, c
    else:         r, g, b = c, 0, x
    return "#{:02x}{:02x}{:02x}".format(
        int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
    )


def _primary_shape(meaning: Meaning) -> str:
    """Select primary geometric form from semantic content."""
    act = meaning.speech_act
    pred = meaning.predicate.lower()
    if act == "question":
        return "wave"
    if act == "command":
        return "triangle"
    if any(w in pred for w in ("become", "grow", "evolve", "change", "transform")):
        return "spiral"
    if any(w in pred for w in ("connect", "relate", "cause", "link", "between")):
        return "line"
    if any(w in pred for w in ("be", "is", "exist", "state", "remain")):
        return "circle"
    if any(w in pred for w in ("contain", "include", "structure", "organize", "hold")):
        return "square"
    return "circle"


def _color(meaning: Meaning) -> str:
    # hue: negative (0°=red) → neutral (120°=green) … actually:
    # map valence -1..1 → hue 0..240 (red→green, skipping yellow midpoint at 60°)
    hue = (meaning.valence + 1) / 2 * 240   # 0° red … 240° blue-green
    sat = 20 + meaning.arousal * 80          # 20% (calm) … 100% (excited)
    lit = 30 + (1 - meaning.dominance) * 40  # 30% (dominant/dark) … 70% (submissive/light)
    return _hsl_to_hex(hue, sat, lit)


def _tilt(meaning: Meaning) -> float:
    """Rotation angle in degrees based on tense."""
    return {"future": -30, "past": 30, "present": 0, "timeless": 45}.get(meaning.tense, 0)


def decode(meaning: Meaning, size: int = 400) -> str:
    """Return a complete SVG string representing the Meaning."""
    cx = cy = size // 2
    color = _color(meaning)
    bg    = _hsl_to_hex(0, 0, 10)
    shape = _primary_shape(meaning)
    r     = int(cx * 0.35 * (0.5 + meaning.force * 0.5))
    tilt  = _tilt(meaning)
    stroke_w = max(1, int(meaning.dominance * 8))

    elements = []

    if shape == "circle":
        elements.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" '
            f'stroke="white" stroke-width="{stroke_w}" opacity="0.9"/>'
        )
        # inner ring if agent + patient both present (relation)
        if meaning.agent and meaning.patient:
            elements.append(
                f'<circle cx="{cx}" cy="{cy}" r="{int(r * 0.55)}" fill="none" '
                f'stroke="white" stroke-width="{max(1,stroke_w-2)}" opacity="0.5"/>'
            )

    elif shape == "triangle":
        h = int(r * 1.732)
        pts = f"{cx},{cy - r} {cx - r},{cy + h//2} {cx + r},{cy + h//2}"
        elements.append(
            f'<polygon points="{pts}" fill="{color}" stroke="white" '
            f'stroke-width="{stroke_w}" opacity="0.9" '
            f'transform="rotate({tilt},{cx},{cy})"/>'
        )

    elif shape == "square":
        s = int(r * 1.41)
        elements.append(
            f'<rect x="{cx-s}" y="{cy-s}" width="{2*s}" height="{2*s}" '
            f'fill="{color}" stroke="white" stroke-width="{stroke_w}" opacity="0.9" '
            f'transform="rotate({tilt},{cx},{cy})"/>'
        )

    elif shape == "spiral":
        pts = []
        for i in range(200):
            angle = i * 0.15
            rad   = i * (r / 200)
            x = cx + rad * math.cos(angle + math.radians(tilt))
            y = cy + rad * math.sin(angle + math.radians(tilt))
            pts.append(f"{x:.1f},{y:.1f}")
        elements.append(
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
            f'stroke-width="{stroke_w}" opacity="0.9"/>'
        )

    elif shape == "wave":
        pts = []
        amp   = r * 0.4
        freq  = 3
        for i in range(size):
            x = i
            y = cy + amp * math.sin(2 * math.pi * freq * i / size + math.radians(tilt))
            pts.append(f"{x:.1f},{y:.1f}")
        elements.append(
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
            f'stroke-width="{stroke_w}" opacity="0.9"/>'
        )

    elif shape == "line":
        angle_r = math.radians(tilt)
        x1 = cx - r * math.cos(angle_r)
        y1 = cy - r * math.sin(angle_r)
        x2 = cx + r * math.cos(angle_r)
        y2 = cy + r * math.sin(angle_r)
        elements.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{stroke_w}" '
            f'marker-end="url(#arrow)" opacity="0.9"/>'
        )

    # secondary marks for arousal (high = radiating lines)
    if meaning.arousal > 0.65:
        n = int(meaning.arousal * 8)
        for i in range(n):
            angle_r = math.radians(i * 360 / n)
            x1 = cx + int(r * 1.05 * math.cos(angle_r))
            y1 = cy + int(r * 1.05 * math.sin(angle_r))
            x2 = cx + int(r * 1.4  * math.cos(angle_r))
            y2 = cy + int(r * 1.4  * math.sin(angle_r))
            elements.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="{color}" stroke-width="2" opacity="0.5"/>'
            )

    defs = (
        '<defs><marker id="arrow" markerWidth="10" markerHeight="7" '
        'refX="10" refY="3.5" orient="auto">'
        f'<polygon points="0 0, 10 3.5, 0 7" fill="{color}"/>'
        '</marker></defs>'
    )

    label = meaning.predicate or meaning.source_text[:40]
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'style="background:{bg}">\n'
        f'{defs}\n'
        + "\n".join(elements)
        + f'\n<text x="{size//2}" y="{size-18}" text-anchor="middle" '
        f'font-family="monospace" font-size="12" fill="#aaaaaa">{label}</text>\n'
        f'</svg>'
    )
    return svg


def save(meaning: Meaning, path: str, size: int = 400) -> str:
    svg = decode(meaning, size)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    return path

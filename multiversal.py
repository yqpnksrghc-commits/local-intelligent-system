"""
Multiversal Translator — n-dimensional meaning space with multi-participant communication.

Every input encodes to a point in ℝⁿ.
Every output renders from that point.
Operations happen in the space itself.
Participants inhabit the space, communicate through it, and tune their own reception.

Solo commands:
  transduce       — encode any input, render to any modalities
  compare         — semantic distance between two inputs
  interpolate     — find meaning between two inputs at position t
  analogy         — a:b::c:? across any modalities
  compose         — combine multiple meanings into one
  measure         — measure meaning along any named axes
  project         — strip to a semantic subspace, then decode
  nearest         — find closest meanings seen this session

Multi-participant commands:
  join            — add a participant (name, modality, language, receptivity)
  send            — one participant transmits meaning to others
  channel         — open a channel with attenuation / warp / axis filter
  volume          — a participant adjusts their receptivity
  bandwidth       — a participant tunes to specific semantic axes
  consensus       — find the group's shared semantic position
  divergence      — show pairwise semantic distance between participants
  transcript      — print the communication log
  who             — show all participant states

  exit
"""
import os
import inspect
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from core.self.health import wait_for_ollama
from core.multiversal.nbridge import NBridge
from core.multiversal.participants.session import MultiSession

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL       = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def _ask(prompt: str, default: str = "") -> str:
    raw = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    return raw or default


def _targets() -> list[str]:
    raw = _ask("Targets (comma-separated)", "english, symbol, music")
    return [t.strip() for t in raw.split(",")]


def _print(results: dict) -> None:
    skip = {"source", "modality", "meaning", "dim"}
    print(f"\n── Meaning  (ℝ{results.get('dim', '?')}) " + "─" * 40)
    print(results.get("meaning", ""))
    for key, val in results.items():
        if key in skip:
            continue
        print(f"\n── {key.upper()} " + "─" * 40)
        print(val)
    print()


def cmd_transduce(bridge: NBridge, session_id: str) -> None:
    source   = _ask("Input")
    modality = _ask("Source modality", "text")
    targets  = _targets()
    _print(bridge.transduce(source, targets, source_modality=modality,
                            session_id=session_id))


def cmd_compare(bridge: NBridge) -> None:
    a     = _ask("Input A")
    mod_a = _ask("Modality A", "text")
    b     = _ask("Input B")
    mod_b = _ask("Modality B", "text")
    r = bridge.compare(a, b, mod_a, mod_b)
    print(f"\nA: {r['a']}\n")
    print(f"B: {r['b']}\n")
    print(f"Cosine similarity:  {r['cosine_similarity']:.4f}")
    print(f"Angular distance:   {r['angular_distance']:.4f}  (0=same, 1=opposite)")
    print(f"Converging dims:    {r['converging_dims']}")
    print(f"Diverging dims:     {r['diverging_dims']}")
    print(f"Unique to A:        {r['unique_to_a']}")
    print(f"Unique to B:        {r['unique_to_b']}\n")


def cmd_interpolate(bridge: NBridge, session_id: str) -> None:
    a = _ask("Input A")
    b = _ask("Input B")
    t = float(_ask("t (0=A, 1=B)", "0.5"))
    targets = _targets()
    _print(bridge.interpolate(a, b, t, targets, session_id))


def cmd_analogy(bridge: NBridge, session_id: str) -> None:
    print("Solve: A is to B as C is to ?")
    a = _ask("A")
    b = _ask("B")
    c = _ask("C")
    targets = _targets()
    _print(bridge.analogy(a, b, c, targets, session_id))


def cmd_compose(bridge: NBridge, session_id: str) -> None:
    raw = _ask("Inputs (comma-separated)")
    sources = [s.strip() for s in raw.split(",")]
    raw_w = _ask("Weights (comma-separated, optional)", "")
    weights = ([float(w) for w in raw_w.split(",") if w.strip()]
               if raw_w else None)
    targets = _targets()
    _print(bridge.compose(sources, weights, targets, session_id))


def cmd_measure(bridge: NBridge) -> None:
    source = _ask("Input")
    raw    = _ask("Axes (comma-separated)",
                  "certainty, temporality, agency, intensity, abstraction")
    axes   = [a.strip() for a in raw.split(",")]
    result = bridge.measure(source, axes)
    print(f"\nℝ{result['dim']} — {source!r}")
    for axis, val in result["axes"].items():
        bar_len = int((val + 1) / 2 * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {axis:<20} {bar}  {val:+.3f}")
    print()


def cmd_project(bridge: NBridge, session_id: str) -> None:
    source  = _ask("Input")
    raw     = _ask("Axes to project onto (comma-separated)")
    axes    = [a.strip() for a in raw.split(",")]
    targets = _targets()
    _print(bridge.project(source, axes, targets, session_id))


def cmd_nearest(bridge: NBridge) -> None:
    source = _ask("Input")
    k      = int(_ask("How many", "5"))
    hits   = bridge.nearest(source, k)
    if not hits:
        print("(no meanings in corpus yet — transduce some first)\n")
        return
    print()
    for i, h in enumerate(hits, 1):
        print(f"  {i}. [{h['similarity']:.4f}]  {h['label']!r}  —  {h['source']!r}")
    print()


# ── Multi-participant commands ─────────────────────────────────────────────


def cmd_join(ms: MultiSession) -> None:
    name       = _ask("Name")
    modality   = _ask("Modality", "text")
    language   = _ask("Language", "english")
    receptivity = float(_ask("Receptivity 0-100", "80")) / 100
    raw_bw     = _ask("Bandwidth axes (comma-separated, blank=all)", "")
    bandwidth  = [a.strip() for a in raw_bw.split(",") if a.strip()] or None
    ms.add(name, modality, language, receptivity, bandwidth)


def cmd_send(ms: MultiSession) -> None:
    sender    = _ask("Sender")
    source    = _ask("Message (any modality)")
    modality  = _ask("Source modality", "text")
    raw_to    = _ask("To (comma-separated, blank=broadcast)", "")
    recipients = [r.strip() for r in raw_to.split(",") if r.strip()] or None
    ms.send(sender, source, recipients, source_modality=modality)


def cmd_channel(ms: MultiSession) -> None:
    sender   = _ask("From")
    receiver = _ask("To")
    atten    = float(_ask("Attenuation 0-100 (signal loss %)", "0")) / 100
    raw_axes = _ask("Axes passed (comma-separated, blank=all)", "")
    axes     = [a.strip() for a in raw_axes.split(",") if a.strip()] or None
    warp     = _ask("Warp axis (blank=none)", "") or None
    ws       = float(_ask("Warp strength 0-100", "0")) / 100 if warp else 0.0
    ms.open_channel(sender, receiver, atten, axes, warp, ws)


def cmd_volume(ms: MultiSession) -> None:
    name  = _ask("Participant")
    level = float(_ask("Receptivity 0-100", "80")) / 100
    ms.get(name).set_volume(level)


def cmd_bandwidth(ms: MultiSession) -> None:
    name    = _ask("Participant")
    raw     = _ask("Axes (comma-separated, blank=omnidirectional)", "")
    axes    = [a.strip() for a in raw.split(",") if a.strip()] or None
    ms.get(name).set_bandwidth(axes)


def cmd_consensus(ms: MultiSession, bridge: NBridge, session_id: str) -> None:
    raw     = _ask("Participants (comma-separated, blank=all)", "")
    names   = [n.strip() for n in raw.split(",") if n.strip()] or None
    c       = ms.consensus(names)
    if c is None:
        print("No participant states yet — send some meaning first.\n")
        return
    print(f"\n── Consensus  ℝ{c.dim} " + "─" * 40)
    print(c.summary())
    raw_t   = _ask("Decode into (comma-separated, blank=skip)", "")
    if raw_t.strip():
        targets = [t.strip() for t in raw_t.split(",")]
        _print(bridge.transduce(c.source_text or c.label, targets,
                                session_id=session_id))


def cmd_divergence(ms: MultiSession) -> None:
    ms.print_state()


def cmd_transcript(ms: MultiSession) -> None:
    ms.print_transcript()


def cmd_who(ms: MultiSession) -> None:
    ms.print_state()


COMMANDS = {
    # solo
    "transduce":   cmd_transduce,
    "compare":     cmd_compare,
    "interpolate": cmd_interpolate,
    "analogy":     cmd_analogy,
    "compose":     cmd_compose,
    "measure":     cmd_measure,
    "project":     cmd_project,
    "nearest":     cmd_nearest,
    # multi-participant
    "join":        cmd_join,
    "send":        cmd_send,
    "channel":     cmd_channel,
    "volume":      cmd_volume,
    "bandwidth":   cmd_bandwidth,
    "consensus":   cmd_consensus,
    "divergence":  cmd_divergence,
    "transcript":  cmd_transcript,
    "who":         cmd_who,
}


def main():
    wait_for_ollama(OLLAMA_HOST)
    llm    = ChatOllama(model=MODEL, temperature=0.2, base_url=OLLAMA_HOST)
    bridge = NBridge(llm, embed_model=EMBED_MODEL, ollama_host=OLLAMA_HOST)
    ms     = MultiSession(bridge)

    solo   = {"transduce","compare","interpolate","analogy","compose",
               "measure","project","nearest"}
    multi  = {"join","send","channel","volume","bandwidth",
               "consensus","divergence","transcript","who"}

    print(f"\nMultiversal Translator  —  ℝⁿ semantic space")
    print(f"Meaning is the invariant. Modality is the surface.")
    print(f"Model: {MODEL}  |  Embeddings: {EMBED_MODEL}")
    print(f"\nSolo:  {' | '.join(sorted(solo))}")
    print(f"Multi: {' | '.join(sorted(multi))}\n")

    session = 0
    while True:
        try:
            cmd = input("› ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd in ("exit", "quit"):
            break
        if cmd not in COMMANDS:
            print(f"Solo:  {' | '.join(sorted(solo))}")
            print(f"Multi: {' | '.join(sorted(multi))}")
            continue

        fn  = COMMANDS[cmd]
        sig = inspect.signature(fn)
        sid = str(session)
        session += 1

        params = set(sig.parameters)
        if   "ms" in params and "bridge" in params and "session_id" in params:
            fn(ms, bridge, sid)
        elif "ms" in params and "bridge" in params:
            fn(ms, bridge)
        elif "ms" in params:
            fn(ms)
        elif "session_id" in params:
            fn(bridge, sid)
        else:
            fn(bridge)


if __name__ == "__main__":
    main()

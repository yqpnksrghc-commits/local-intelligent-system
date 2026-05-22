"""
MultiSession — the shared semantic space inhabited by all participants.

Manages:
  - participant registration and state (human and autonomous agents)
  - channel creation and routing
  - message transmission with agent auto-response
  - group operations: consensus, divergence, resonance mapping
  - trajectory tracking per participant
  - topological analysis: attractors, landscape, convergence field
  - diffusion: passive meaning spread through the network
  - the full communication transcript
"""
from __future__ import annotations
from datetime import datetime
from langchain_ollama import ChatOllama
from core.multiversal.space import SemanticSpace
from core.multiversal.nmeaning import NMeaning
from core.multiversal.nbridge import NBridge
from core.multiversal.participants.participant import Participant
from core.multiversal.participants.channel import Channel
from core.multiversal.participants.message import Message
from core.multiversal.dynamics import Trajectory, DiffusionModel
import core.multiversal.topology as topo


class MultiSession:
    def __init__(self, bridge: NBridge):
        self.bridge      = bridge
        self.space       = bridge.space
        self.llm         = bridge.llm
        self.participants: dict[str, Participant] = {}
        self.channels:     dict[str, Channel]     = {}
        self.transcript:   list[Message]          = []
        self.trajectories: dict[str, Trajectory]  = {}
        self.diffusion     = DiffusionModel()
        self._session_n    = 0
        # all meanings ever produced in this session
        self._all_meanings: list[NMeaning]        = []

    # ── Participant management ─────────────────────────────────────────

    def add(
        self,
        name: str,
        modality: str = "text",
        language: str = "english",
        receptivity: float = 0.8,
        bandwidth: list[str] | None = None,
    ) -> Participant:
        p = Participant(
            name=name,
            preferred_modality=modality,
            preferred_language=language,
            receptivity=receptivity,
            bandwidth=bandwidth,
        )
        self.participants[name] = p
        self.trajectories[name] = Trajectory(owner=name)
        print(f"[session] {name} joined  ({modality}/{language}  "
              f"receptivity={int(receptivity*100)}%)")
        return p

    def add_agent(
        self,
        name: str,
        policy: str = "explore",
        modality: str = "text",
        language: str = "english",
        receptivity: float = 0.8,
        bandwidth: list[str] | None = None,
    ) -> "AgentParticipant":
        from core.multiversal.participants.agent import AgentParticipant
        p = AgentParticipant(
            name=name, llm=self.llm, policy=policy,
            preferred_modality=modality, preferred_language=language,
            receptivity=receptivity, bandwidth=bandwidth,
        )
        self.participants[name] = p
        self.trajectories[name] = Trajectory(owner=name)
        print(f"[session] agent {name!r} joined  "
              f"(policy={policy}  {modality}/{language}  "
              f"receptivity={int(receptivity*100)}%)")
        return p

    def remove(self, name: str) -> None:
        self.participants.pop(name, None)
        print(f"[session] {name} left")

    def get(self, name: str) -> Participant:
        if name not in self.participants:
            raise KeyError(f"Unknown participant: {name!r}")
        return self.participants[name]

    # ── Channel management ────────────────────────────────────────────

    def open_channel(
        self,
        sender: str,
        receiver: str,
        attenuation: float = 0.0,
        axes_passed: list[str] | None = None,
        warp_axis: str | None = None,
        warp_strength: float = 0.0,
    ) -> Channel:
        cid = f"{sender}→{receiver}"
        ch = Channel(
            id=cid, sender=sender, receiver=receiver,
            attenuation=attenuation, axes_passed=axes_passed,
            warp_axis=warp_axis, warp_strength=warp_strength,
        )
        self.channels[cid] = ch
        print(f"[session] channel open: {ch.summary()}")
        return ch

    def _get_channel(self, sender: str, receiver: str) -> Channel | None:
        return self.channels.get(f"{sender}→{receiver}")

    # ── Transmission ──────────────────────────────────────────────────

    def send(
        self,
        sender_name: str,
        source: str,
        recipients: list[str] | None = None,   # None = broadcast
        source_modality: str = "text",
        source_language: str | None = None,
    ) -> Message:
        sender = self.get(sender_name)

        # encode
        meaning = self.bridge.encode(source, source_modality, source_language)
        meaning.label = meaning.label or source[:40]

        targets = recipients or [n for n in self.participants if n != sender_name]
        msg = Message(
            meaning=meaning,
            sender=sender_name,
            recipients=targets,
        )
        sender.outbox.append(msg)
        sender._update_state(meaning, self.space)

        print(f"\n{msg.summary()}")
        print(f"  ℝ{meaning.dim}  {meaning.summary()}\n")

        for name in targets:
            if name not in self.participants:
                continue
            receiver = self.participants[name]

            # channel processing
            ch = self._get_channel(sender_name, name)
            in_transit = ch.transmit(meaning, self.space) if ch else meaning

            # receiver applies their own receptivity and bandwidth
            received = receiver.receive(in_transit, self.space)
            if received is None:
                print(f"  → {name}: [below threshold — not received]")
                continue

            # decode into receiver's preferred modality
            decoded = self._decode_for(receiver, received)
            msg.received_as[name] = received
            msg.decoded_as[name]  = decoded
            receiver.inbox.append(msg)

            resonance = receiver.resonance_with(meaning)
            print(f"  → {name} [{receiver.preferred_modality}/"
                  f"receptivity={int(receiver.receptivity*100)}%/"
                  f"resonance={resonance:.2f}]:\n    {decoded}\n")

        # track trajectory for sender
        self._track(sender_name, meaning)
        self._all_meanings.append(meaning)

        # autonomous agent responses
        for name in targets:
            if name not in self.participants:
                continue
            receiver = self.participants[name]
            received_m = msg.received_as.get(name)
            if received_m is None:
                continue
            from core.multiversal.participants.agent import AgentParticipant
            if isinstance(receiver, AgentParticipant):
                response = receiver.respond(received_m, self.space,
                                            str(self._session_n))
                if response:
                    print(f"  [{name} auto-responds]:")
                    self.send(name, response, [sender_name],
                              source_modality="text",
                              source_language=receiver.preferred_language)

        self.transcript.append(msg)
        self._session_n += 1
        return msg

    def _track(self, name: str, meaning: NMeaning) -> None:
        if name not in self.trajectories:
            self.trajectories[name] = Trajectory(owner=name)
        self.trajectories[name].append(meaning, label=meaning.label)

    def _decode_for(self, receiver: Participant, meaning: NMeaning) -> str:
        target = receiver.preferred_language if receiver.preferred_modality == "text" \
                 else receiver.preferred_modality
        try:
            return self.bridge.decode(meaning, target, str(self._session_n))
        except Exception as e:
            return f"[decode error: {e}]"

    # ── Group operations ──────────────────────────────────────────────

    def consensus(self, participant_names: list[str] | None = None) -> NMeaning | None:
        """
        Find the consensus meaning across participant states.
        The centroid of all current states in ℝⁿ.
        """
        names = participant_names or list(self.participants)
        states = [self.participants[n].state for n in names
                  if self.participants[n].state is not None]
        if not states:
            return None
        consensus = self.space.compose(states)
        consensus.label = "group consensus"
        return consensus

    def divergence_map(self) -> dict[tuple[str, str], float]:
        """
        Pairwise angular distance between all participant states.
        High = participants are semantically far apart.
        """
        names = [n for n, p in self.participants.items() if p.state is not None]
        result = {}
        for i, a in enumerate(names):
            for b in names[i+1:]:
                dist = self.participants[a].state.distance(self.participants[b].state)
                result[(a, b)] = dist
        return result

    def resonance_map(self, meaning: NMeaning) -> dict[str, float]:
        """How much does each participant resonate with a given meaning?"""
        return {
            name: p.resonance_with(meaning)
            for name, p in self.participants.items()
        }

    def semantic_positions(self) -> dict[str, str]:
        """Current semantic state label for each participant."""
        return {
            name: (p.state.label if p.state else "uninitiated")
            for name, p in self.participants.items()
        }

    # ── Topology ──────────────────────────────────────────────────────

    def landscape(self, n_attractors: int = 3) -> dict:
        """Full topological analysis of all meanings produced this session."""
        return topo.landscape(self._all_meanings, self.space, n_attractors)

    def print_landscape(self, n_attractors: int = 3) -> None:
        if not self._all_meanings:
            print("(no meanings yet)\n")
            return
        L = self.landscape(n_attractors)
        print(f"\n── Semantic Landscape  ({L['n_meanings']} meanings) " + "─" * 30)
        print(f"  Spread:   {L['spread']:.3f}  (0=tight cluster, 1=maximally scattered)")
        print(f"  Density:  {L['density']:.3f}")
        c = L['centroid']
        print(f"  Centroid: {c.label!r}")
        print(f"\n  Attractors:")
        for a in L['attractors']:
            print(f"    · {a.label!r}")
        print(f"\n  Basin assignments:")
        for src, basin in L['assignments'].items():
            print(f"    {src[:30]:30} → {basin}")

        # convergence field
        states = [p.state for p in self.participants.values() if p.state is not None]
        if len(states) >= 2:
            cf = topo.convergence_field(states)
            print(f"\n  Convergence: {cf['status']}"
                  f"  cohesion={cf['cohesion']:.3f}"
                  f"  spread=[{cf['min_similarity']:.2f}–{cf['max_similarity']:.2f}]")
        print()

    def print_trajectories(self) -> None:
        print("\n── Trajectories " + "─" * 40)
        for name, traj in self.trajectories.items():
            print(f"  {traj.summary()}")
        print()

    def diffuse(self, steps: int = 1) -> None:
        """Run n diffusion steps — passive meaning spread through the network."""
        for i in range(steps):
            shifts = self.diffusion.step(self.participants, self.space)
            if shifts:
                desc = "  ".join(f"{n}:{v:.4f}" for n, v in shifts.items())
                print(f"[diffuse step {i+1}] {desc}")

    # ── Introspection ─────────────────────────────────────────────────

    def print_transcript(self) -> None:
        print("\n── Transcript " + "─" * 40)
        for msg in self.transcript:
            print(f"  {msg.summary()}")
        print()

    def print_state(self) -> None:
        print("\n── Session State " + "─" * 40)
        for name, p in self.participants.items():
            print(f"  {p.summary()}")
        divmap = self.divergence_map()
        if divmap:
            print("\n── Divergence Map " + "─" * 40)
            for (a, b), dist in sorted(divmap.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(dist * 20) + "░" * (20 - int(dist * 20))
                print(f"  {a:12} ↔ {b:12}  {bar}  {dist:.3f}")
        print()

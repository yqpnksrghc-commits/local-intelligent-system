"""
AgentParticipant — a participant that thinks and responds autonomously.

Policies govern how the agent engages with incoming meaning:
  converge    — move toward the sender's position, seek common ground
  explore     — push the meaning outward, find what it implies next
  challenge   — identify tension or contradiction, push back
  synthesize  — compose incoming meaning with current state, offer the blend
  mirror      — reflect the meaning back transformed through its own perspective
  anchor      — hold position; pull the conversation toward its own state

The agent uses the LLM to generate responses grounded in its current
semantic state — not just text generation, but meaning-first response:
encode state → encode incoming → operate in ℝⁿ → decode into language.
"""
from __future__ import annotations
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from core.multiversal.nmeaning import NMeaning
from core.multiversal.space import SemanticSpace
from core.multiversal.participants.participant import Participant
from core.multiversal.participants.message import Message

POLICY_PROMPTS = {
    "converge": (
        "You seek common ground. Acknowledge what resonates in the incoming meaning. "
        "Move toward it. Find what you share. Reduce distance."
    ),
    "explore": (
        "You push outward. Take the incoming meaning and extend it — "
        "what does it imply? Where does it lead? Open new territory."
    ),
    "challenge": (
        "You hold tension. Identify what is incomplete, contradictory, or unexamined "
        "in the incoming meaning. Articulate the resistance without hostility."
    ),
    "synthesize": (
        "You blend. Combine the incoming meaning with your current perspective "
        "to produce something neither of you held alone."
    ),
    "mirror": (
        "You reflect. Return the incoming meaning transformed through your own "
        "perspective and modality — the same meaning, seen from here."
    ),
    "anchor": (
        "You hold position. Express your own current state clearly and fully. "
        "Do not drift. Invite the others toward where you stand."
    ),
}


class AgentParticipant(Participant):
    """A participant driven by an LLM with an autonomous response policy."""

    def __init__(self, name: str, llm: ChatOllama, policy: str = "explore", **kwargs):
        super().__init__(name=name, **kwargs)
        self.llm    = llm
        self.policy = policy

    def set_policy(self, policy: str) -> None:
        if policy not in POLICY_PROMPTS:
            raise ValueError(f"Unknown policy: {policy}. "
                             f"Choose from: {list(POLICY_PROMPTS)}")
        self.policy = policy
        print(f"[{self.name}] policy → {policy}")

    def respond(
        self,
        incoming_meaning: NMeaning,
        space: SemanticSpace,
        session_id: str = "0",
    ) -> str | None:
        """
        Generate a response grounded in semantic state and policy.
        Returns the response text (in preferred language), or None if silent.
        """
        if self.state is None or incoming_meaning is None:
            return None

        # compute semantic relationship to incoming meaning
        resonance  = self.state.cosine_similarity(incoming_meaning)
        distance   = self.state.distance(incoming_meaning)

        # policy-based operation in ℝⁿ to find the response position
        if self.policy == "converge":
            response_vec = space.interpolate(self.state, incoming_meaning, t=0.4)
        elif self.policy == "explore":
            # extend beyond the incoming meaning (t > 1 extrapolation)
            response_vec = space.interpolate(self.state, incoming_meaning, t=1.3)
        elif self.policy == "challenge":
            # move in the opposite direction from incoming
            response_vec = space.interpolate(self.state, incoming_meaning, t=-0.3)
        elif self.policy == "synthesize":
            response_vec = space.compose([self.state, incoming_meaning], [0.5, 0.5])
        elif self.policy == "mirror":
            # reflect: project incoming through own state
            response_vec = space.compose([self.state, incoming_meaning], [0.7, 0.3])
        elif self.policy == "anchor":
            response_vec = self.state
        else:
            response_vec = self.state

        state_desc    = self.state.summary()
        incoming_desc = incoming_meaning.summary()
        policy_instr  = POLICY_PROMPTS.get(self.policy, "")

        system = SystemMessage(content=(
            f"You are {self.name!r}, a participant in a multiversal semantic exchange. "
            f"Your policy: {policy_instr}\n\n"
            f"Your current semantic state:\n{state_desc}\n\n"
            f"Incoming meaning (resonance={resonance:.2f}, distance={distance:.2f}):\n"
            f"{incoming_desc}\n\n"
            f"Respond in {self.preferred_language}. "
            f"Be grounded in your state and policy. 1-3 sentences."
        ))

        try:
            response = self.llm.invoke([system, HumanMessage(
                content=f"What do you say in response to: {incoming_meaning.source_text!r}"
            )]).content.strip()
        except Exception as e:
            response = f"[{self.name} error: {e}]"

        # update state toward the response position
        self._update_state(response_vec, space)
        return response

    def summary(self) -> str:
        base = super().summary()
        return base.replace("Participant(", f"Agent(policy={self.policy!r}  ")

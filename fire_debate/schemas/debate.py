from dataclasses import dataclass, field
from typing import List, Literal
from uuid import uuid4
from datetime import datetime

# Enums make the code safer than random strings
Stance = Literal["PRO", "CON"]
Phase = Literal["OPENING", "REBUTTAL", "CLOSING"]

@dataclass
class DebateTurn:
    """A single utterance by an agent."""
    turn_id: str
    agent_id: str
    stance: Stance
    phase: Phase
    text: str
    citations: List[str]  # Stores doc_ids used in this turn
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class DebateLog:
    """The complete artifact of a research experiment run."""
    debate_id: str
    claim_id: str
    claim_text: str
    ground_truth: bool
    turns: List[DebateTurn] = field(default_factory=list)

    def add_turn(self, turn: DebateTurn):
        self.turns.append(turn)
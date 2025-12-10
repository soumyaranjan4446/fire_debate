from dataclasses import dataclass, field
from typing import List, Literal, Optional
from uuid import uuid4
from datetime import datetime

# We add 'NEUTRAL' so the Moderator doesn't break the code if we log them
Stance = Literal["PRO", "CON", "NEUTRAL"]
Phase = Literal["OPENING", "REBUTTAL", "CLOSING", "MODERATION", "SYNTHESIS"]

@dataclass
class DebateTurn:
    """A single utterance by an agent."""
    turn_id: str
    agent_id: str
    stance: Stance
    phase: Phase
    text: str
    citations: List[str]
    
    # --- NEW: Stores the Agentic Search Query ---
    # This proves in your paper that the agent dynamically searched!
    search_query: Optional[str] = None
    
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class DebateLog:
    """The complete artifact of a research experiment run."""
    debate_id: str
    claim_id: str
    claim_text: str
    ground_truth: bool
    turns: List[DebateTurn] = field(default_factory=list)
    
    # --- NEW: Storage for the Final Results ---
    # The Insight folder calculates these, but we store them here to save to JSON
    summary: Optional[str] = None
    verdict_score: Optional[float] = None  # 0.0 to 1.0

    def add_turn(self, turn: DebateTurn):
        self.turns.append(turn)
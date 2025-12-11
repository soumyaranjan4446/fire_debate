from dataclasses import dataclass, field
from typing import List, Literal, Optional
from datetime import datetime

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
    
    # ✅ CRITICAL FIELD: Agentic Search Query storage
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
    
    # Results
    summary: Optional[str] = None
    verdict_score: Optional[float] = None

    def add_turn(self, turn: DebateTurn):
        self.turns.append(turn)
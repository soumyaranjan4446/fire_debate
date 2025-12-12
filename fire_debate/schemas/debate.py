from dataclasses import dataclass, field
from typing import List, Literal, Optional
from uuid import uuid4
from datetime import datetime, timezone # <--- Added timezone

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
    
    # FIX: Use timezone-aware UTC timestamp to avoid DeprecationWarning
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

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
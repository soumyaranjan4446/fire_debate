from dataclasses import dataclass, field
from typing import List, Optional, Any

@dataclass
class DebateTurn:
    """
    Represents a single speech or interaction by an agent.
    """
    turn_id: str
    agent_id: str
    agent_name: str
    stance: str  # "PRO", "CON", "NEUTRAL"
    text: str
    round: int
    
    # Fields with defaults must come AFTER required fields
    citations: List[str] = field(default_factory=list)
    search_query: str = ""
    phase: str = "ARGUMENT"  # E.g., "OPENING", "CROSS_EX_ASK"

@dataclass
class DebateLog:
    """
    The full transcript of a debate session.
    """
    debate_id: str
    claim_id: str
    claim_text: str
    ground_truth: Optional[bool]
    turns: List[DebateTurn]
    summary: str = ""

    def add_turn(self, turn: DebateTurn):
        self.turns.append(turn)
from typing import List, Optional
from fire_debate.agents.base import LLMClient
from fire_debate.schemas.debate import DebateTurn

class ModeratorAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.name = "Moderator"

    def monitor(self, history: List[DebateTurn]) -> Optional[str]:
        """
        Watches the debate. If it gets stuck/toxic, issues an instruction.
        Returns: None (if all good) or "Instruction string" (if intervention needed).
        """
        # Don't intervene in the very first round or if history is empty
        if not history or len(history) < 2:
            return None

        # Look at the last 2 turns
        last_turn = history[-1]
        prev_turn = history[-2]
        
        last_exchange = f"{prev_turn.agent_id}: {prev_turn.text[:200]}\n{last_turn.agent_id}: {last_turn.text[:200]}"
        
        prompt = f"""
        You are the Debate Moderator. Review this recent exchange:
        
        {last_exchange}
        
        Check for these issues:
        1. REPETITION: Are they just repeating the same point?
        2. CIRCULARITY: Are they ignoring each other's evidence?
        3. TOXICITY: Are they being rude instead of logical?
        
        If the debate is healthy, reply "PASS".
        If there is an issue, issue a short, direct instruction to the NEXT speaker to fix it (e.g., "Address the evidence about X directly").
        """
        
        response = self.llm.generate("You are a strict referee.", prompt)
        
        # If the LLM says PASS or gives a very short/empty response, do nothing
        if "PASS" in response.upper() or len(response) < 5:
            return None
        
        return response.strip()
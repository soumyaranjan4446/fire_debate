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
        
        # --- 1. HEURISTIC CHECKS (Fast & Cheap) ---
        
        # Tone Check: Stop rudeness immediately
        bad_words = ["stupid", "idiot", "liar", "nonsense", "delusional", "shut up"]
        if any(w in last_turn.text.lower() for w in bad_words):
            return "Maintain professional decorum. Attack the argument, not the person."

        # Lazy Citation Check: Stop empty evidence drops
        # If they cite something but write less than 60 chars, they are being lazy.
        if "[EVID:" in last_turn.text and len(last_turn.text) < 60:
             return "Please elaborate on your evidence. Do not just drop a citation without explanation."

        # --- 2. LLM LOGIC CHECK (Smart & Deep) ---
        
        last_exchange = f"{prev_turn.agent_id}: {prev_turn.text[:300]}\n{last_turn.agent_id}: {last_turn.text[:300]}"
        
        prompt = f"""
        You are the Debate Moderator. Review this recent exchange:
        
        {last_exchange}
        
        Check for these issues:
        1. REPETITION: Are they just repeating the same point?
        2. CIRCULARITY: Are they ignoring each other's evidence?
        
        If the debate is healthy and moving forward, reply "PASS".
        If there is an issue, issue a short, direct instruction to the NEXT speaker to fix it (e.g., "Address the evidence about X directly").
        """
        
        response = self.llm.generate("You are a strict referee.", prompt)
        
        # If the LLM says PASS or gives a very short/empty response, do nothing
        if "PASS" in response.upper() or len(response) < 5:
            return None
        
        return response.strip()
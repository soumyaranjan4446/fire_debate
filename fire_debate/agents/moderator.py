import re
from typing import List, Optional, Any
from fire_debate.schemas.debate import DebateTurn

class ModeratorAgent:
    def __init__(self, llm: Any):
        # We use Any for llm to avoid importing a base class that might not exist
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

        # Look at the last turn (The agent who just spoke)
        last_turn = history[-1]
        prev_turn = history[-2]
        
        # --- 1. HEURISTIC CHECKS (Fast & Cheap) ---
        
        # Tone Check: Stop rudeness immediately
        bad_words = ["stupid", "idiot", "liar", "nonsense", "delusional", "shut up", "dumb"]
        if any(w in last_turn.text.lower() for w in bad_words):
            return "Maintain professional decorum. Attack the argument, not the person."

        # Lazy Citation Check:
        # If they found citations but wrote a tiny 1-sentence response.
        if len(last_turn.citations) > 0 and len(last_turn.text) < 50:
             return "Please elaborate on your evidence. Do not just drop a citation without explanation."

        # --- 1.5 STRICT EVIDENCE CHECK (FIXED) ---
        # We check the DATA, not the text string.
        # If they are in a main phase but found ZERO sources, flag them.
        
        phase = getattr(last_turn, 'phase', "UNKNOWN")
        
        # We only enforce citations in research phases.
        # We do NOT enforce it in Cross-Ex (because search is disabled there).
        if phase in ["OPENING", "REBUTTAL", "CLOSING"]:
            if not last_turn.citations:
                return "Your claim lacks backing. You MUST use the search tool to find and cite evidence."

        # --- 2. LLM LOGIC CHECK (Smart & Deep) ---
        
        last_exchange = f"{prev_turn.agent_id}: {prev_turn.text[:300]}...\n{last_turn.agent_id}: {last_turn.text[:300]}..."
        
        # Combined Prompt for LocalHFClient
        full_prompt = (
            f"You are a strict Debate Moderator. Review this recent exchange:\n\n"
            f"{last_exchange}\n\n"
            f"Check for:\n"
            f"1. REPETITION: Are they repeating the same point?\n"
            f"2. CIRCULARITY: Are they ignoring evidence?\n\n"
            f"If the debate is healthy, reply 'PASS'.\n"
            f"If there is an issue, issue a 1-sentence instruction to the NEXT speaker to fix it."
        )
        
        # Generate response
        response = self.llm.generate(full_prompt, max_new_tokens=50)
        
        # FIX: Safer "PASS" check using Regex
        # Matches "PASS" only if it's a whole word, not inside "compassion" or "passive"
        if re.search(r"\bPASS\b", response, re.IGNORECASE):
            return None
            
        # If the response is empty or weirdly short, ignore it
        if len(response.strip()) < 5:
            return None
        
        return response.strip()
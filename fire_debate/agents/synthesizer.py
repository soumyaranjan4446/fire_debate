from typing import Any
from fire_debate.schemas.debate import DebateLog

class SynthesisAgent:
    def __init__(self, llm: Any):
        """
        The Synthesizer reads the debate history and produces a final summary.
        Args:
            llm: Instance of LocalHFClient (passed as Any to avoid circular imports)
        """
        self.llm = llm
        self.name = "Synthesizer"

    def synthesize(self, log: DebateLog) -> str:
        """
        Condenses the debate into a report + a one-sentence correction.
        """
        if not log.turns:
            return "No debate activity recorded."

        # 1. Build the Transcript
        # We format it like a script: "Alice (PRO): argument..."
        transcript = ""
        for turn in log.turns:
            # Safety: Truncate massive walls of text to save context tokens for the reasoning
            # We assume turn has 'agent_name' or 'agent_id' based on manager.py
            name = getattr(turn, 'agent_name', getattr(turn, 'agent_id', 'Debater'))
            
            clean_text = turn.text[:800] + "..." if len(turn.text) > 800 else turn.text
            transcript += f"[{name} ({turn.stance})]: {clean_text}\n\n"

        # 2. Construct the Analyst Prompt
        # We want the model to act as a neutral judge evaluating the evidence quality.
        prompt = (
            f"You are an impartial Fact-Checking Adjudicator. Review the following debate regarding the claim: '{log.claim_text}'.\n\n"
            f"=== DEBATE TRANSCRIPT ===\n"
            f"{transcript}"
            f"=========================\n\n"
            f"TASK:\n"
            f"1. Summarize the main arguments for both sides.\n"
            f"2. Evaluate who provided better EVIDENCE (citations and facts).\n"
            f"3. Identify if any logical fallacies were used.\n"
            f"4. Write a single, concise sentence starting with 'THE TRUTH:' that states the corrected fact based *only* on the evidence.\n\n"
            f"OUTPUT FORMAT:\n"
            f"Summary: [Concise summary]\n"
            f"Evidence Analysis: [Who had better proof?]\n"
            f"THE TRUTH: [Final Verdict: True/False and Why]"
        )

        # 3. Generate Report
        # We allow more tokens here (512) because the summary needs to be detailed
        # FIX: Single argument call
        report = self.llm.generate(prompt, max_new_tokens=512)
        
        return report.strip()
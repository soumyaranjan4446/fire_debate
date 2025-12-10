from fire_debate.agents.base import LLMClient
from fire_debate.schemas.debate import DebateLog

class SynthesisAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.name = "Synthesizer"

    def synthesize(self, log: DebateLog) -> str:
        """
        Condenses the entire debate into a structured report.
        """
        transcript = ""
        for turn in log.turns:
            # Skip Moderator turns in the final summary to keep it clean
            if turn.stance == "NEUTRAL":
                continue
            transcript += f"[{turn.agent_id}]: {turn.text}\n"

        prompt = f"""
        You are a Synthesis Agent. Your job is to summarize this debate for a Fact-Checking Judge.
        
        DEBATE TRANSCRIPT:
        {transcript[:3000]}  # Truncate to avoid context overflow if debate is huge
        
        Task:
        1. Identify the Main Claim.
        2. List the strongest evidence provided by the Proponent.
        3. List the strongest counter-evidence provided by the Opponent.
        4. Note any un-refuted points.
        
        Output a structured summary.
        """
        
        summary = self.llm.generate("You are an objective summarizer.", prompt)
        return summary
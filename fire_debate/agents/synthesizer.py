from fire_debate.agents.base import LLMClient
from fire_debate.schemas.debate import DebateLog

class SynthesisAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.name = "Synthesizer"

    def synthesize(self, log: DebateLog) -> str:
        """
        Condenses the debate into a report + a one-sentence correction.
        """
        transcript = ""
        for turn in log.turns:
            if turn.stance == "NEUTRAL": continue
            transcript += f"[{turn.agent_id}]: {turn.text}\n"

        prompt = f"""
        You are an Fact-Checking Adjudicator. Read this debate transcript:
        
        {transcript[:3500]}
        
        TASK:
        1. Summarize the main arguments for both sides.
        2. Based *only* on the evidence presented, determine the factual reality.
        3. Write a single, concise sentence starting with "THE TRUTH:" that states the corrected fact.
           - If the claim is True, reaffirm it.
           - If the claim is False, state what is actually true.
        
        FORMAT:
        Summary: [Your Summary Here]
        THE TRUTH: [Your One-Sentence Correction Here]
        """
        
        # We assume the "Proponent" LLM is smart enough to summarize (usually Qwen)
        summary = self.llm.generate("You are an objective judge.", prompt)
        return summary
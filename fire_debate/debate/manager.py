import json
from uuid import uuid4
from dataclasses import asdict
from fire_debate.schemas.debate import DebateLog, Phase, DebateTurn
from fire_debate.agents.debater import DebaterAgent
from fire_debate.agents.moderator import ModeratorAgent
from fire_debate.agents.synthesizer import SynthesisAgent
from fire_debate.rag.retriever import EvidenceRetriever

class DebateManager:
    def __init__(self, pro: DebaterAgent, con: DebaterAgent, retriever: EvidenceRetriever):
        self.pro = pro
        self.con = con
        self.retriever = retriever
        
        # Initialize Helper Agents (Share the LLM to save memory)
        self.moderator = ModeratorAgent(pro.llm)
        self.synthesizer = SynthesisAgent(pro.llm)

    def run_debate(self, claim: str, rounds: int = 2) -> DebateLog:
        print(f"\n🔥 DEBATE START: '{claim}'")
        log = DebateLog(
            debate_id=str(uuid4())[:8], claim_id="gen", claim_text=claim, ground_truth=False
        )
        
        # 1. Broad Initial Research (Populate Memory)
        print("🌍 Manager performing initial broad research...")
        docs = self.retriever.search_web(claim)
        self.retriever.index_documents(docs)

        # 2. Debate Loop
        phases = ["OPENING"] + ["REBUTTAL"] * (rounds - 1) + ["CLOSING"]
        agents = [self.pro, self.con]
        
        for phase in phases:
            print(f"\n--- {phase} ---")
            for agent in agents:
                # --- MODERATOR CHECK ---
                # Checks previous turns for repetition or toxicity
                instruction = self.moderator.monitor(log.turns)
                if instruction:
                    print(f"   👨‍⚖️ MODERATOR INTERVENES: {instruction}")
                    # We inject this instruction into the debate log as a special 'NEUTRAL' turn
                    # This lets the Agent see it in history without breaking the schema
                    mod_turn = DebateTurn(
                        turn_id=str(uuid4())[:8],
                        agent_id="Moderator",
                        stance="NEUTRAL",
                        phase="MODERATION",
                        text=f"[INSTRUCTION]: {instruction}",
                        citations=[],
                        search_query=None
                    )
                    log.add_turn(mod_turn)
                
                # --- AGENT TURN ---
                turn = agent.act(claim, log.turns, phase)
                log.add_turn(turn)
                print(f"🗣️ {turn.agent_id}: {turn.text[:100]}...")

        # 3. SYNTHESIS PHASE
        print("\n📝 Synthesizing Debate Report...")
        summary = self.synthesizer.synthesize(log)
        log.summary = summary 
        print(f"   Summary: {summary[:100]}...")

        self.retriever.clear_cache()
        return log

    def save_log(self, log, path):
        with open(path, 'w') as f:
            data = asdict(log)
            # Handle datetime serialization
            data['turns'] = [{**t, 'created_at': t['created_at'].isoformat()} for t in data['turns']]
            json.dump(data, f, indent=2)
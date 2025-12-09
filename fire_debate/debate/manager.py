import json
from uuid import uuid4
from typing import List
from dataclasses import asdict

from fire_debate.schemas.debate import DebateLog, DebateTurn, Phase
from fire_debate.agents.debater import DebaterAgent
from fire_debate.rag.retriever import EvidenceRetriever

class DebateManager:
    def __init__(
        self, 
        pro_agent: DebaterAgent, 
        con_agent: DebaterAgent,
        retriever: EvidenceRetriever
    ):
        self.pro = pro_agent
        self.con = con_agent
        self.retriever = retriever

    def run_debate(self, claim: str, rounds: int = 2) -> DebateLog:
        debate_id = str(uuid4())[:8]
        print(f"\n🔥 STARTING DEBATE {debate_id}: '{claim}'\n")

        # 1. Initialize Log
        log = DebateLog(
            debate_id=debate_id,
            claim_id="manual_run",
            claim_text=claim,
            ground_truth=False # Placeholder
        )

        # 2. Pre-load Knowledge (Populate Vector DB from Web)
        print("🌍 Scanning the web for context...")
        web_docs = self.retriever.search_web(claim)
        self.retriever.index_documents(web_docs)
        print(f"✅ Indexed {len(web_docs)} documents.")

        # 3. Debate Loop
        phases = ["OPENING"] + ["REBUTTAL"] * (rounds - 1) + ["CLOSING"]
        
        # Alternate turns
        agents = [self.pro, self.con]
        
        for i, phase in enumerate(phases):
            print(f"\n--- PHASE {i+1}: {phase} ---")
            
            for agent in agents:
                turn = agent.act(claim, log.turns, phase)
                log.add_turn(turn)
                print(f"\n🗣️ {turn.agent_id} ({turn.stance}):\n{turn.text}")
                if turn.citations:
                    print(f"   📎 Cited: {turn.citations}")

        # 4. Cleanup
        self.retriever.clear_cache()
        return log

    def save_log(self, log: DebateLog, path: str):
        with open(path, 'w') as f:
            # Convert dataclass to dict for JSON serialization
            data = asdict(log)
            # Handle datetime serialization
            data['turns'] = [
                {**t, 'created_at': t['created_at'].isoformat()} 
                for t in data['turns']
            ]
            json.dump(data, f, indent=2)
            print(f"\n💾 Debate saved to {path}")
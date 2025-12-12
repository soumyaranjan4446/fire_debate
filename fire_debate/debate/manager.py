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
        # Helper agents share the PRO agent's LLM to save memory
        self.moderator = ModeratorAgent(pro.llm)
        self.synthesizer = SynthesisAgent(pro.llm)

    def run_debate(self, claim: str, rounds: int = 2) -> DebateLog:
        # Display truncate
        print(f"\n🔥 DEBATE START: '{claim[:100]}...'")
        
        log = DebateLog(
            debate_id=str(uuid4())[:8], claim_id="gen", claim_text=claim, ground_truth=False
        )
        
        # 1. Initial Research (Safe Mode)
        # We search for the first 200 chars to seed the memory
        print("🌍 Manager seeding knowledge base...")
        try:
            safe_query = claim[:200]
            docs = self.retriever.search_web(safe_query)
            self.retriever.index_documents(docs)
        except Exception as e:
            print(f"   ⚠️ Initial search failed (continuing anyway): {e}")

        # 2. Debate Loop
        phases = ["OPENING"] + ["REBUTTAL"] * (rounds - 1) + ["CLOSING"]
        agents = [self.pro, self.con]
        
        # Variable to hold any pending instruction from the moderator
        pending_instruction = None

        for phase in phases:
            print(f"\n--- {phase} ---")
            for agent in agents:
                
                # --- A. Moderator Check ---
                # The moderator looks at the debate SO FAR to guide the NEXT speaker
                instruction = self.moderator.monitor(log.turns)
                
                if instruction:
                    print(f"   👨‍⚖️ MODERATOR: {instruction}")
                    # Log the intervention
                    mod_turn = DebateTurn(
                        turn_id=str(uuid4())[:8], agent_id="Moderator", stance="NEUTRAL",
                        phase="MODERATION", text=f"[INSTRUCTION]: {instruction}", citations=[]
                    )
                    log.add_turn(mod_turn)
                    pending_instruction = instruction
                else:
                    pending_instruction = None
                
                # --- B. Agent Turn ---
                # Pass the instruction directly to the agent
                turn = agent.act(claim, log.turns, phase, moderator_instruction=pending_instruction)
                
                log.add_turn(turn)
                print(f"🗣️ {turn.agent_id}: {turn.text[:100]}...")

        # 3. Synthesis
        print("\n📝 Synthesizing Report...")
        summary = self.synthesizer.synthesize(log)
        log.summary = summary 

        self.retriever.clear_cache()
        return log

    def save_log(self, log, path):
        with open(path, 'w') as f:
            data = asdict(log)
            # Handle datetime objects for JSON
            data['turns'] = [{**t, 'created_at': t['created_at'].isoformat()} for t in data['turns']]
            json.dump(data, f, indent=2)
import json
import time
import uuid
from typing import List

from fire_debate.schemas.debate import DebateLog, DebateTurn
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

    def _execute_turn(self, agent, claim, round_num, phase, context_arg, turns_list):
        """
        Helper to run a single agent turn, handle moderation, and log it.
        """
        # 1. Moderator Check
        # The moderator reviews the transcript so far to issue instructions
        instruction = self.moderator.monitor(turns_list)
        
        if instruction:
            # Log the intervention
            mod_turn = DebateTurn(
                turn_id=str(uuid.uuid4())[:8],
                agent_id="Moderator",
                agent_name="Moderator",
                stance="NEUTRAL",
                text=f"[INSTRUCTION]: {instruction}",
                round=round_num,
                citations=[],
                search_query=""
            )
            turns_list.append(mod_turn)
            
            # Append instruction to the context so the agent sees it
            if context_arg:
                context_arg = f"{context_arg}\n[MODERATOR]: {instruction}"
            else:
                context_arg = f"[MODERATOR]: {instruction}"

        # 2. Agent Act
        # Calls the LLM to generate the speech/answer
        text = agent.act(claim, round_num, phase, opponent_last_arg=context_arg)
        
        # 3. Log the Turn
        turn = DebateTurn(
            turn_id=str(uuid.uuid4())[:8],
            agent_id=agent.name,
            agent_name=agent.name,
            stance=agent.stance,
            text=text,
            round=round_num,
            citations=[],
            search_query="",
            phase=phase  # <--- CRITICAL: Save "OPENING", "CROSS_EX", etc.
        )
        turns_list.append(turn)
        
        return text

    def run_debate(self, claim: str, rounds: int = 1) -> DebateLog:
        # print(f"\n🔥 DEBATE START: '{claim[:100]}...'")
        
        # 1. Initialize Log
        debate_id = f"db_{int(time.time())}_{str(uuid.uuid4())[:4]}"
        turns: List[DebateTurn] = []
        
        # 2. Initial Research (Safe Mode)
        # Seed the knowledge base with general context about the claim
        try:
            self.retriever.clear_cache()
            # print("   🌍 Manager seeding knowledge base...")
            self.retriever.index_documents(self.retriever.search_web(claim[:200]))
        except Exception as e:
            print(f"   ⚠️ Initial search failed: {e}")

        # --- PHASE 1: OPENING STATEMENTS ---
        # print("\n   --- PHASE 1: OPENING ---")
        # Pro goes first, no context needed
        pro_open = self._execute_turn(self.pro, claim, 1, "OPENING", None, turns)
        # Con responds to Pro's opening
        con_open = self._execute_turn(self.con, claim, 1, "OPENING", pro_open, turns)

        # --- PHASE 2: REBUTTAL ---
        # print("\n   --- PHASE 2: REBUTTAL ---")
        # Pro rebuts Con's opening
        pro_reb = self._execute_turn(self.pro, claim, 2, "REBUTTAL", con_open, turns)
        # Con rebuts Pro's rebuttal
        con_reb = self._execute_turn(self.con, claim, 2, "REBUTTAL", pro_reb, turns)

        # --- PHASE 3: CROSS-EXAMINATION (The "Fire" Round) ---
        # print("\n   --- PHASE 3: CROSS-EXAMINATION ---")
        
        # Interaction A: Pro asks -> Con answers
        q_pro = self._execute_turn(self.pro, claim, 3, "CROSS_EX_ASK", con_reb, turns)
        a_con = self._execute_turn(self.con, claim, 3, "CROSS_EX_ANSWER", q_pro, turns)
        
        # Interaction B: Con asks -> Pro answers
        # Note: Con asks based on Pro's rebuttal or generic stance
        q_con = self._execute_turn(self.con, claim, 3, "CROSS_EX_ASK", pro_reb, turns) 
        a_pro = self._execute_turn(self.pro, claim, 3, "CROSS_EX_ANSWER", q_con, turns)

        # --- PHASE 4: CLOSING ---
        # print("\n   --- PHASE 4: CLOSING ---")
        # Context passed here is the result of their own cross-ex exchange
        context_for_pro = f"My Question: {q_pro}\nOpponent Answer: {a_con}"
        pro_close = self._execute_turn(self.pro, claim, 4, "CLOSING", context_for_pro, turns)
        
        context_for_con = f"My Question: {q_con}\nOpponent Answer: {a_pro}"
        con_close = self._execute_turn(self.con, claim, 4, "CLOSING", context_for_con, turns)

        # 4. Synthesis
        # print("\n   📝 Synthesizing Report...")
        log = DebateLog(
            debate_id=debate_id,
            claim_id="gen",
            claim_text=claim,
            ground_truth=None,
            turns=turns,
            summary=""
        )
        
        log.summary = self.synthesizer.synthesize(log) 

        return log

    def save_log(self, log: DebateLog, path: str):
        """
        Robust JSON saver. Now includes 'phase' and 'turn_id'.
        """
        try:
            with open(path, 'w', encoding='utf-8') as f:
                data = {
                    "debate_id": log.debate_id,
                    "claim_id": log.claim_id,
                    "claim_text": log.claim_text,
                    "ground_truth": log.ground_truth,
                    "summary": log.summary,
                    "turns": [
                        {
                            "turn_id": t.turn_id,   # <--- Added
                            "agent_id": t.agent_id,
                            "agent_name": getattr(t, 'agent_name', t.agent_id),
                            "stance": t.stance,
                            "text": t.text,
                            "round": t.round,
                            "phase": getattr(t, 'phase', "ARGUMENT"), # <--- Added (Critical)
                            "citations": t.citations
                        } 
                        for t in log.turns
                    ]
                }
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Failed to save log: {e}")
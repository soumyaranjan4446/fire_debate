import re
from uuid import uuid4
from typing import List
from dataclasses import dataclass

from fire_debate.schemas.debate import DebateTurn, Stance, Phase
from fire_debate.schemas.evidence import EvidenceDoc
from fire_debate.rag.retriever import EvidenceRetriever
from fire_debate.agents.librarian import Librarian
from fire_debate.agents.base import LLMClient

@dataclass
class AgentConfig:
    name: str
    stance: Stance
    style: str = "logical" 

class DebaterAgent:
    def __init__(
        self,
        config: AgentConfig,
        llm: LLMClient,
        retriever: EvidenceRetriever,
        librarian: Librarian
    ):
        self.cfg = config
        self.llm = llm
        self.retriever = retriever
        self.librarian = librarian

    def _generate_agentic_query(self, claim: str, history: List[DebateTurn]) -> str:
        """
        AGENTIC CAPABILITY: The agent analyzes the debate state and 
        formulates a targeted search query to find specific evidence.
        """
        # If it's the very first turn, just search the main topic
        if not history:
            return claim

        # Analyze the opponent's last point to find a weakness
        last_turn = history[-1]
        opponent_text = last_turn.text[:500] # Truncate for context window
        
        prompt = f"""
        You are an expert debater named {self.cfg.name}.
        Topic: "{claim}"
        
        Your Opponent ({last_turn.agent_id}) just argued: 
        "{opponent_text}"
        
        Your Goal: Refute them with specific facts.
        Task: Write a Google Search Query (max 8 words) to find evidence against their specific point.
        Output ONLY the query. Do not use quotes.
        """
        
        # Call LLM to generate the tool input
        query = self.llm.generate("You are a search query generator.", prompt)
        
        # Clean up the output
        query = query.strip().replace('"', '').replace("Search:", "").replace("Query:", "")
        return query

    def _format_evidence(self, docs: List[EvidenceDoc]) -> str:
        if not docs:
            return "No specific evidence found."
        
        text = "AVAILABLE EVIDENCE:\n"
        for d in docs:
            # Show reliability score so the agent knows what to trust
            text += f"[{d.doc_id}] (Trust: {d.reliability_score:.2f}): {d.snippet[:300]}...\n"
        return text

    def act(self, claim: str, history: List[DebateTurn], phase: Phase) -> DebateTurn:
        print(f"🤔 {self.cfg.name} is thinking...")

        # --- 1. AGENTIC THOUGHT & ACTION ---
        # The agent decides what to search for
        query = self._generate_agentic_query(claim, history)
        print(f"   🧠 Agent Decided to Search: '{query}'")

        # Execute Search: First look at the live web for this specific query
        web_docs = self.retriever.search_web(query)
        self.retriever.index_documents(web_docs) # Add to memory
        
        # Retrieve the best context from memory (Hybrid Search)
        raw_evidence = self.retriever.retrieve_context(query)
        
        # --- 2. QUALITY CONTROL (Librarian) ---
        # Filter out low-quality or irrelevant evidence
        trusted_evidence = self.librarian.filter_evidence(raw_evidence)
        
        # --- 3. ARGUMENT GENERATION ---
        # Construct the context for the LLM
        history_str = "\n".join(
            [f"{t.agent_id} ({t.stance}): {t.text}" for t in history[-3:]]
        )
        
        system_prompt = (
            f"You are {self.cfg.name}, arguing for {self.cfg.stance}. "
            f"Style: {self.cfg.style}.\n"
            "INSTRUCTIONS:\n"
            "1. Cite the provided EVIDENCE using [EVID:doc_id].\n"
            "2. Be direct and logical.\n"
            "3. If evidence is weak, attack the opponent's logic."
        )
        
        user_prompt = (
            f"Topic: {claim}\n"
            f"Current Phase: {phase}\n\n"
            f"DEBATE HISTORY:\n{history_str}\n\n"
            f"RETRIEVED KNOWLEDGE (Query: '{query}'):\n"
            f"{self._format_evidence(trusted_evidence)}\n\n"
            "Write your response (under 150 words):"
        )

        # Generate the argument
        response_text = self.llm.generate(system_prompt, user_prompt)
        
        # --- 4. RETURN ARTIFACT ---
        return DebateTurn(
            turn_id=str(uuid4())[:8],
            agent_id=self.cfg.name,
            stance=self.cfg.stance,
            phase=phase,
            text=response_text,
            citations=re.findall(r"\[EVID:(.*?)\]", response_text),
            
            # ✅ CRITICAL FIX: Save the dynamic search query so we can analyze it later
            search_query=query
        )
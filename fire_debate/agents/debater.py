import re
from uuid import uuid4
from typing import List, Dict
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
    style: str = "logical"  # logical, emotional, aggressive

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

    def _construct_system_prompt(self) -> str:
        """Defines the persona."""
        base = (
            f"You are {self.cfg.name}, a debater arguing for the {self.cfg.stance} side.\n"
            f"Style: {self.cfg.style}.\n"
            "INSTRUCTIONS:\n"
            "1. Use the provided EVIDENCE to support your claims.\n"
            "2. When you use a fact from evidence, cite it using [EVID:doc_id].\n"
            "3. If the evidence contradicts you, acknowledge it but argue around it.\n"
            "4. Keep your response under 150 words.\n"
            "5. Do NOT hallucinate sources."
        )
        return base

    def _format_evidence(self, docs: List[EvidenceDoc]) -> str:
        if not docs:
            return "No specific evidence found."
        
        text = "AVAILABLE EVIDENCE:\n"
        for d in docs:
            text += f"[{d.doc_id}] ({d.reliability_score:.2f} trust): {d.snippet[:200]}...\n"
        return text

    def act(self, claim: str, history: List[DebateTurn], phase: Phase) -> DebateTurn:
        print(f"🤔 {self.cfg.name} is thinking...")

        # 1. RAG Step: Retrieve Evidence based on claim
        # (Future improvement: Generate a specific search query based on history)
        raw_evidence = self.retriever.retrieve_context(claim)
        
        # 2. Librarian Step: Filter for quality
        trusted_evidence = self.librarian.filter_and_score(raw_evidence)
        
        # 3. Prompt Construction
        system_prompt = self._construct_system_prompt()
        
        # Format history for the LLM context
        history_text = "\n".join(
            [f"{t.agent_id} ({t.stance}): {t.text}" for t in history[-4:]] # Last 4 turns
        )
        
        user_prompt = (
            f"Topic: {claim}\n"
            f"Current Phase: {phase}\n\n"
            f"DEBATE HISTORY:\n{history_text}\n\n"
            f"{self._format_evidence(trusted_evidence)}\n\n"
            "Your Turn:"
        )

        # 4. LLM Generation
        response_text = self.llm.generate(system_prompt, user_prompt)

        # 5. Citation Extraction (Regex)
        cited_ids = re.findall(r"\[EVID:(.*?)\]", response_text)

        # 6. Return Typed Artifact
        return DebateTurn(
            turn_id=str(uuid4())[:8],
            agent_id=self.cfg.name,
            stance=self.cfg.stance,
            phase=phase,
            text=response_text,
            citations=cited_ids
        )
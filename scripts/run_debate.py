import yaml
import torch
from fire_debate.agents.local_client import LocalHFClient
from fire_debate.agents.debater import DebaterAgent, AgentConfig
from fire_debate.agents.librarian import Librarian
from fire_debate.rag.retriever import EvidenceRetriever
from fire_debate.debate.manager import DebateManager

def main():
    # 1. Load Config
    with open("configs/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Initialize Shared Resources
    # LLM (Brain) - Shared by both agents to save RAM
    llm = LocalHFClient(cfg)
    
    # RAG (Knowledge)
    retriever = EvidenceRetriever(cfg)
    librarian = Librarian()

    # 3. Create Agents
    pro_config = AgentConfig(name="Alice", stance="PRO", style="academic")
    con_config = AgentConfig(name="Bob", stance="CON", style="skeptical")

    alice = DebaterAgent(pro_config, llm, retriever, librarian)
    bob = DebaterAgent(con_config, llm, retriever, librarian)

    # 4. Create Manager
    manager = DebateManager(alice, bob, retriever)

    # 5. Run!
    topic = "Artificial Intelligence will eventually surpass human intelligence."
    log = manager.run_debate(topic, rounds=2)

    # 6. Save Artifact
    manager.save_log(log, "data/processed/debate_result.json")

if __name__ == "__main__":
    main()
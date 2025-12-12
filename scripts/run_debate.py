import sys
import os
from pathlib import Path

# --- 1. Path Setup (Critical for imports) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import yaml
import json
from fire_debate.agents.local_client import LocalHFClient
from fire_debate.agents.debater import DebaterAgent, AgentConfig
from fire_debate.agents.librarian import Librarian
from fire_debate.rag.retriever import EvidenceRetriever
from fire_debate.debate.manager import DebateManager

def main():
    print("🚀 Initializing Single Debate Test...")

    # 2. Load Config
    config_path = os.path.join(project_root, "configs", "base.yaml")
    if not os.path.exists(config_path):
        print(f"❌ Config not found at {config_path}")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 3. Initialize Shared Resources
    print(f"⚙️ Loading AI Brain ({cfg['llm']['model_id']})...")
    llm = LocalHFClient(cfg)
    
    print("📚 Connecting to Knowledge Base...")
    retriever = EvidenceRetriever(cfg)
    librarian = Librarian()

    # 4. Create Agents
    # We use "Rational" agents to see if they can effectively fact-check
    alice = DebaterAgent(AgentConfig("Alice", "PRO", "academic"), llm, retriever, librarian)
    bob = DebaterAgent(AgentConfig("Bob", "CON", "skeptical"), llm, retriever, librarian)
    
    # Manager handles Moderator & Synthesizer internally
    manager = DebateManager(alice, bob, retriever)

    # 5. Define a REAL WORLD Claim (Long text, similar to your dataset)
    # This tests if your 'Query Distillation' fix works (preventing Tavily crash)
    topic = (
        "Recent reports suggest that drinking 3 liters of coffee daily can reverse aging "
        "and cure balding, based on a viral study from the 'Institute of Caffeine Sciences'. "
        "Critics argue this is pseudoscience marketing by coffee distributors."
    )
    
    print(f"\n🔥 TOPIC: '{topic}'")
    print("------------------------------------------------")
    
    # 6. Run Debate
    # This triggers the full Agentic Loop: Think -> Query -> Search -> Argue -> Monitor
    log = manager.run_debate(topic, rounds=2)

    # 7. Save Artifact
    output_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "single_run_debug.json")
    
    manager.save_log(log, save_path)
    print(f"\n✅ Debate Log Saved: {save_path}")
    print("   (Check this file to see the 'search_query' fields!)")

if __name__ == "__main__":
    main()
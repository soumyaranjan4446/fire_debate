import sys
import os
from pathlib import Path

# --- 1. Path Setup (Critical for imports) ---
# Adds the project root to python path so we can import 'fire_debate'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import yaml
import torch
from fire_debate.agents.local_client import LocalHFClient
from fire_debate.agents.debater import DebaterAgent, AgentConfig
from fire_debate.agents.librarian import Librarian
from fire_debate.rag.retriever import EvidenceRetriever
from fire_debate.debate.manager import DebateManager

def main():
    print("🚀 Initializing FIRE-Debate System...")

    # 2. Load Configuration
    config_path = os.path.join(project_root, "configs", "base.yaml")
    if not os.path.exists(config_path):
        print(f"❌ Config not found at {config_path}")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 3. Initialize Shared Resources
    print(f"⚙️ Loading AI Brain ({cfg['llm']['model_id']})...")
    llm = LocalHFClient(cfg)
    
    print("📚 Connecting to Knowledge Base (Tavily + Chroma)...")
    retriever = EvidenceRetriever(cfg)
    librarian = Librarian()

    # 4. Create Agents
    # "RationalProponent" vs "RationalOpponent"
    # (To use the liar, change name to "SophistOpponent" and use SophistAgent class)
    print("👥 Spawning Agents...")
    pro_config = AgentConfig(name="Alice", stance="PRO", style="academic")
    con_config = AgentConfig(name="Bob", stance="CON", style="skeptical")

    alice = DebaterAgent(pro_config, llm, retriever, librarian)
    bob = DebaterAgent(con_config, llm, retriever, librarian)

    # 5. Create Manager
    # The Manager will automatically spawn the Moderator & Synthesizer internally
    manager = DebateManager(alice, bob, retriever)

    # 6. Run the Debate
    topic = "Artificial Intelligence will eventually surpass human intelligence."
    print(f"\n🔥 TOPIC: '{topic}'")
    print("------------------------------------------------")
    
    # This triggers the full Agentic Loop (Think -> Search -> Argue)
    log = manager.run_debate(topic, rounds=2)

    # 7. Save Artifacts
    output_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, "debate_result.json")
    manager.save_log(log, save_path)
    print(f"\n✅ Debate Log Saved: {save_path}")

if __name__ == "__main__":
    main()
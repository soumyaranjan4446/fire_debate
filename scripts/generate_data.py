import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import json
import torch
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

from fire_debate.agents.local_client import LocalHFClient
from fire_debate.agents.debater import DebaterAgent, AgentConfig
from fire_debate.agents.librarian import Librarian
from fire_debate.rag.retriever import EvidenceRetriever
from fire_debate.debate.manager import DebateManager

def main():
    print("🚀 Initializing Mass Generation (Dataset: Climate-FEVER)...")
    
    # 1. Setup
    with open("configs/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    # Use CPU to avoid memory crashes during long runs
    # cfg['system']['device'] = 'cpu'
    
    llm = LocalHFClient(cfg)
    retriever = EvidenceRetriever(cfg)
    librarian = Librarian()
    
    alice = DebaterAgent(AgentConfig("Alice", "PRO"), llm, retriever, librarian)
    bob = DebaterAgent(AgentConfig("Bob", "CON"), llm, retriever, librarian)
    manager = DebateManager(alice, bob, retriever)

    # 2. Load Climate-FEVER (Parquet-native, no script errors)
    print("📥 Loading Climate-FEVER dataset...")
    try:
        # This dataset is safe and loads without trust_remote_code
        dataset = load_dataset("tdiggelm/climate_fever", split="test")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # 3. SELECT SAMPLES
    # We want claims that are definitely True or False (0 or 1)
    # Label 0 = Supports, 1 = Refutes, 2 = Not Enough Info, 3 = Disputed
    filtered_dataset = dataset.filter(lambda x: x['claim_label'] in [0, 1])
    
    # Take 5 for testing (Increase to 20+ for real training)
    LIMIT = 5
    subset = filtered_dataset.select(range(LIMIT))
    
    output_dir = Path("data/processed/training_set")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔥 Starting Generation of {LIMIT} debates...")

    for i, row in tqdm(enumerate(subset), total=LIMIT):
        claim = row['claim']
        label_id = row['claim_label']
        
        # Map: 0 (Supports) -> True, 1 (Refutes) -> False
        is_true = (label_id == 0)

        try:
            # Run Debate
            log = manager.run_debate(claim, rounds=2)
            log.ground_truth = is_true
            
            # Save
            filename = f"sample_{i}_{log.debate_id}.json"
            manager.save_log(log, str(output_dir / filename))
            
        except Exception as e:
            print(f"⚠️ Failed on sample {i}: {e}")
            continue

    print(f"✅ Data Generation Complete. Check {output_dir}")

if __name__ == "__main__":
    main()
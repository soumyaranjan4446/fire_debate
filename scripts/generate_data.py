import sys
import os
# Fix imports so python can find fire_debate
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
    print("🚀 Initializing Mass Generation (Dataset: ARG-EN Local)...")
    
    # 1. Setup Configuration
    config_path = os.path.join(os.path.dirname(__file__), '../configs/base.yaml')
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Initialize Engines
    print(f"⚙️ Loading AI Brain: {cfg['llm']['model_id']}...")
    llm = LocalHFClient(cfg)
    
    print("📚 Connecting to Knowledge Base...")
    retriever = EvidenceRetriever(cfg)
    librarian = Librarian()
    
    # Initialize Agents
    alice = DebaterAgent(AgentConfig("Alice", "PRO"), llm, retriever, librarian)
    bob = DebaterAgent(AgentConfig("Bob", "CON"), llm, retriever, librarian)
    manager = DebateManager(alice, bob, retriever)

    # 2. Load Local JSON Files
    # NOTE: Ensure you moved your json files to data/raw/
    data_files = {
        "train": "data/raw/train.json",
        "validation": "data/raw/val.json", 
        "test": "data/raw/test.json"
    }
    
    # CHANGE THIS to "train" for training data, or "test" for evaluation data
    SPLIT_NAME = "train" 
    
    print(f"📥 Loading local dataset ({SPLIT_NAME} split)...")
    
    try:
        # Load the JSONs using Hugging Face's local loader
        dataset = load_dataset("json", data_files=data_files, split=SPLIT_NAME)
    except FileNotFoundError:
        print("❌ Error: Could not find JSON files in 'data/raw/'. Please move them there.")
        return
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # 3. Select Samples
    # Set to 20 or 50 for a good training batch. Set to 5 for a quick test.
    LIMIT = 5
    subset = dataset.select(range(LIMIT))
    
    # Output Directory
    output_dir = Path(f"data/processed/{SPLIT_NAME}_set")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔥 Starting Generation of {LIMIT} debates...")

    for i, row in tqdm(enumerate(subset), total=LIMIT):
        # Extract fields based on your JSON structure
        claim = row.get('content') 
        if not claim: continue

        label_id = row.get('label')
        
        # ARG-EN Label Mapping:
        # 0 = Real (True)
        # 1 = Fake (False)
        is_true = (label_id == 0)

        try:
            # Truncate very long news articles to first 300 chars to save context window
            short_claim = claim[:300] + "..." if len(claim) > 300 else claim
            
            # Run Debate
            log = manager.run_debate(short_claim, rounds=2)
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
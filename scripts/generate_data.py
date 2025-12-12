import sys
import os
# Fix imports so python can find fire_debate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import json
import torch
import random
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from pathlib import Path

from fire_debate.agents.local_client import LocalHFClient
from fire_debate.agents.debater import DebaterAgent, AgentConfig
from fire_debate.agents.librarian import Librarian
from fire_debate.rag.retriever import EvidenceRetriever
from fire_debate.debate.manager import DebateManager

def main():
    print("🚀 Initializing Balanced Mass Generation (Dataset: ARG-EN Local)...")
    
    # --- CONFIGURATION ---
    # Change this to "train" to generate training data
    # Change this to "test" to generate evaluation data
    SPLIT_NAME = "test" 
    
    # How many debates to generate per class?
    # 10 Real + 10 Fake = 20 Total Debates. 
    # Increase this to 50 or 100 for the final paper run.
    SAMPLES_PER_CLASS = 10 
    # ---------------------

    # 1. Setup Configuration
    config_path = os.path.join(os.path.dirname(__file__), '../configs/base.yaml')
    if not os.path.exists(config_path):
        print(f"❌ Config not found at {config_path}")
        return

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
    data_files = {
        "train": "data/raw/train.json",
        "validation": "data/raw/val.json", 
        "test": "data/raw/test.json"
    }
    
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

    # 3. Balance the Dataset (Critical for Research Quality)
    print("⚖️ Balancing classes (50% Real / 50% Fake)...")
    
    # Filter for Real (0) and Fake (1) based on ARG-EN labels
    # Label 0 = Real
    # Label 1 = Fake
    real_news = dataset.filter(lambda x: x['label'] == 0)
    fake_news = dataset.filter(lambda x: x['label'] == 1)
    
    print(f"   Found {len(real_news)} Real and {len(fake_news)} Fake items available.")
    
    # Select N samples from each class
    # We use shuffle to ensure we get different random articles each time
    real_subset = real_news.shuffle(seed=42).select(range(min(len(real_news), SAMPLES_PER_CLASS)))
    fake_subset = fake_news.shuffle(seed=42).select(range(min(len(fake_news), SAMPLES_PER_CLASS)))
    
    # Combine and Shuffle final list
    balanced_dataset = concatenate_datasets([real_subset, fake_subset])
    balanced_dataset = balanced_dataset.shuffle(seed=123)
    
    # Set up output directory
    # Note: We save to 'train_set' or 'test_set' to match the evaluation script logic
    output_dir = Path(f"data/processed/{SPLIT_NAME}_set")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔥 Starting Generation of {len(balanced_dataset)} debates...")

    for i, row in tqdm(enumerate(balanced_dataset), total=len(balanced_dataset)):
        # Extract fields from ARG-EN format
        claim = row.get('content') 
        if not claim: continue

        label_id = row.get('label')
        
        # Logic: Label 0 is Real (True), Label 1 is Fake (False)
        is_true = (label_id == 0)

        try:
            # Truncate very long news articles to first 400 chars to help the Agents focus
            short_claim = claim[:400] + "..." if len(claim) > 400 else claim
            
            # Run Debate
            log = manager.run_debate(short_claim, rounds=2)
            log.ground_truth = is_true
            
            # Save using unique ID to avoid overwrites
            filename = f"sample_{i}_{log.debate_id}.json"
            manager.save_log(log, str(output_dir / filename))
            
        except Exception as e:
            print(f"⚠️ Failed on sample {i}: {e}")
            continue

    print(f"✅ Data Generation Complete. Check {output_dir}")

if __name__ == "__main__":
    main()
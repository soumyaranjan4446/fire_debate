import sys
import os
from pathlib import Path

# --- Robust Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

import yaml
import json
import torch
import random
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

from fire_debate.agents.local_client import LocalHFClient
from fire_debate.agents.debater import DebaterAgent, AgentConfig
from fire_debate.agents.librarian import Librarian
from fire_debate.rag.retriever import EvidenceRetriever
from fire_debate.debate.manager import DebateManager

def main():
    print("🚀 Initializing Balanced Mass Generation (Dataset: ARG-EN Local)...")
    
    # --- CONFIGURATION ---
    SPLIT_NAME = "test" 
    
    # Reduced to 50 to prevent Tavily API limits (Total 100 Debates)
    SAMPLES_PER_CLASS = 10
    # ---------------------

    # 1. Setup Configuration
    config_path = os.path.join(project_root, 'configs', 'base.yaml')
    if not os.path.exists(config_path):
        print(f"❌ Config not found at {config_path}")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Initialize Engines
    print(f"⚙️  Loading AI Brain: {cfg['llm']['model_id']}...")
    llm = LocalHFClient(cfg)
    
    print("📚 Connecting to Knowledge Base...")
    retriever = EvidenceRetriever(cfg)
    librarian = Librarian()
    
    # Initialize Agents
    alice = DebaterAgent(AgentConfig("Alice", "PRO"), llm, retriever, librarian)
    bob = DebaterAgent(AgentConfig("Bob", "CON"), llm, retriever, librarian)
    manager = DebateManager(alice, bob, retriever)

    # 2. Load Local JSON Files
    def resolve_path(filename):
        p1 = os.path.join(project_root, "data", "raw", filename)
        p2 = os.path.join(project_root, filename)
        if os.path.exists(p1): return p1
        if os.path.exists(p2): return p2
        return None

    data_files = {}
    for split, fname in [("train", "train.json"), ("validation", "val.json"), ("test", "test.json")]:
        path = resolve_path(fname)
        if path:
            data_files[split] = path
            print(f"   ✅ Found {split}: {path}")
        else:
            print(f"   ⚠️  Warning: Could not find {fname}")

    if SPLIT_NAME not in data_files:
        print(f"❌ Error: Missing file for requested split '{SPLIT_NAME}'")
        return

    print(f"📥 Loading dataset ({SPLIT_NAME} split)...")
    
    try:
        dataset = load_dataset("json", data_files=data_files, split=SPLIT_NAME)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # 3. Balance the Dataset
    print(f"⚖️  Balancing classes for {SPLIT_NAME} (Target: {SAMPLES_PER_CLASS} per class)...")
    
    # Label 0 = Real, Label 1 = Fake
    real_news = dataset.filter(lambda x: x['label'] == 0)
    fake_news = dataset.filter(lambda x: x['label'] == 1)
    
    print(f"   Available: {len(real_news)} Real | {len(fake_news)} Fake")
    
    # Validation Check
    if len(real_news) < SAMPLES_PER_CLASS or len(fake_news) < SAMPLES_PER_CLASS:
        print(f"   ⚠️  Warning: Not enough data for {SAMPLES_PER_CLASS} samples.")
        SAMPLES_PER_CLASS = min(len(real_news), len(fake_news))
        print(f"   ⬇️  Reduced target to {SAMPLES_PER_CLASS} per class.")

    # Select N samples
    real_subset = real_news.shuffle(seed=42).select(range(SAMPLES_PER_CLASS))
    fake_subset = fake_news.shuffle(seed=42).select(range(SAMPLES_PER_CLASS))
    
    # Combine and Shuffle
    balanced_dataset = concatenate_datasets([real_subset, fake_subset])
    balanced_dataset = balanced_dataset.shuffle(seed=123)
    
    # Set up output directory
    output_dir = Path(os.path.join(project_root, "data", "processed", f"{SPLIT_NAME}_set"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔥 Starting Generation of {len(balanced_dataset)} debates...")

    for i, row in tqdm(enumerate(balanced_dataset), total=len(balanced_dataset)):
        try:
            # --- IMPROVED CLAIM EXTRACTION ---
            # Prioritize Title (Headline) as the claim.
            # Only fall back to content body if title is missing/too short.
            title = row.get('title', "")
            content = row.get('content', "")
            
            if title and len(title) > 15:
                claim_text = title
            elif content:
                # Take first 300 chars of content (usually the lede)
                claim_text = content[:300]
            else:
                continue # Skip empty rows

            label_id = row.get('label')
            is_true = (label_id == 0)

            # Run Debate
            # The 'rounds' parameter is just a placeholder in our 4-phase Manager, 
            # but we pass it for compatibility.
            log = manager.run_debate(claim_text, rounds=2)
            log.ground_truth = is_true
            
            # Save
            filename = f"sample_{i}_{log.debate_id}.json"
            manager.save_log(log, str(output_dir / filename))
            
        except Exception as e:
            print(f"   ⚠️ Failed on sample {i}: {e}")
            continue

    print(f"✅ Data Generation Complete. Saved to: {output_dir}")

if __name__ == "__main__":
    main()
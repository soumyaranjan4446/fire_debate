import sys
import os
import time
import yaml
import json
import random
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

# --- Robust Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

# --- Imports ---
from fire_debate.agents.openai_client import OpenAIClient
# from fire_debate.agents.azure_client import AzureClient 
from fire_debate.agents.debater import DebaterAgent, AgentConfig
from fire_debate.agents.librarian import Librarian
from fire_debate.rag.retriever import EvidenceRetriever
from fire_debate.debate.manager import DebateManager

def process_split(split_name, config, manager, rounds=2, samples_per_class=None):
    """
    Handles loading, balancing, and generating debates for a single split.
    """
    print(f"\n🔵 STARTING SPLIT: {split_name.upper()}")

    # 1. Resolve Data File
    def resolve_path(filename):
        p1 = os.path.join(project_root, "data", "raw", filename)
        p2 = os.path.join(project_root, filename)
        if os.path.exists(p1): return p1
        if os.path.exists(p2): return p2
        return None

    # Priority: "diverse_train.json" (if training) > "train.json"
    filename = "diverse_train.json" if split_name == "train" else f"{split_name}.json"
    path = resolve_path(filename)
    
    # Fallback for standard names
    if not path and split_name == "train": path = resolve_path("train.json")
    if not path and split_name == "test": path = resolve_path("test.json")

    if not path:
        print(f"   ⚠️ Skipping {split_name}: File not found.")
        return

    print(f"   📥 Loading from: {path}")

    try:
        dataset = load_dataset("json", data_files={split_name: path}, split=split_name)
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        return

    # 2. Balance Dataset (Real vs Fake)
    real_news = dataset.filter(lambda x: x['label'] == 0)
    fake_news = dataset.filter(lambda x: x['label'] == 1)
    
    print(f"   ⚖️  Available: {len(real_news)} Real | {len(fake_news)} Fake")

    if samples_per_class:
        target = samples_per_class
        if len(real_news) < target or len(fake_news) < target:
            target = min(len(real_news), len(fake_news))
            print(f"   ⚠️  Not enough data. Cap reduced to {target} per class.")
        
        real_subset = real_news.shuffle(seed=42).select(range(target))
        fake_subset = fake_news.shuffle(seed=42).select(range(target))
        final_dataset = concatenate_datasets([real_subset, fake_subset])
    else:
        final_dataset = dataset

    final_dataset = final_dataset.shuffle(seed=123)
    
    # 3. Output Directory
    output_dir = Path(os.path.join(project_root, "data", "processed", f"{split_name}_openai_set"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"   🔥 Generating {len(final_dataset)} debates...")
    print(f"   📂 Saving to: {output_dir}")

    # 4. Generation Loop
    success_count = 0
    
    for i, row in tqdm(enumerate(final_dataset), total=len(final_dataset), desc=f"{split_name}"):
        try:
            # Check for Resume: If file exists, skip
            existing_files = list(output_dir.glob(f"gen_{i}_*.json"))
            if existing_files:
                continue

            # Extract Claim
            title = row.get('title', "")
            content = row.get('content', "") or row.get('claim', "")
            
            if title and len(title) > 15:
                claim_text = title
            elif content:
                claim_text = content[:300]
            else:
                continue

            label_id = row.get('label')
            is_true = (label_id == 0)

            # Run Debate
            log = manager.run_debate(claim_text, rounds=rounds)
            log.ground_truth = is_true
            
            # Save
            save_name = f"gen_{i}_{log.debate_id}.json"
            manager.save_log(log, str(output_dir / save_name))
            
            success_count += 1
            
            # Rate Limit Protection
            time.sleep(1.2) 
            
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user. Exiting entire process.")
            sys.exit(0)
        except Exception as e:
            continue

    print(f"   ✅ Finished {split_name}. Generated: {success_count} new debates.")


def main():
    print("🚀 Initializing OpenAI Data Generation (Optimized for 4 Keys)...")
    
    # --- ⚙️ CONFIGURATION ---
    SPLITS_TO_PROCESS = ["train", "test"]
    
    LIMITS = {
        "train": 500,       # 500 per class = 1,000 Total Debates (Uses ~4k credits)
        "test": 50,         # 50 per class = 100 Total Debates (Safety buffer)
    }
    
    ROUNDS = 2
    # ------------------------

    # 1. Setup Config & Engines
    config_path = os.path.join(project_root, 'configs', 'base.yaml')
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    try:
        print(f"🧠 Connecting to OpenAI...")
        llm = OpenAIClient(cfg)
    except Exception as e:
        print(f"❌ Failed to load Client: {e}")
        return

    print("📚 Connecting to Knowledge Base...")
    retriever = EvidenceRetriever(cfg)
    librarian = Librarian()
    
    alice = DebaterAgent(AgentConfig("Alice", "PRO"), llm, retriever, librarian)
    bob = DebaterAgent(AgentConfig("Bob", "CON"), llm, retriever, librarian)
    manager = DebateManager(alice, bob, retriever)

    # 2. Run Train and Test Splits
    for split in SPLITS_TO_PROCESS:
        limit = LIMITS.get(split, 100) 
        process_split(split, cfg, manager, rounds=ROUNDS, samples_per_class=limit)

    print("\n🎉🎉 TRAIN & TEST DATA GENERATION COMPLETE! 🎉🎉")

if __name__ == "__main__":
    main()
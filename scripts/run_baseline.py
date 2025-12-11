import sys
import os

# --- 1. Path Setup (Critical for imports) ---
# Get the absolute path of the folder containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root (one level up)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# Add to python path
if project_root not in sys.path:
    sys.path.append(project_root)

import yaml
import json
import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from fire_debate.agents.local_client import LocalHFClient

def run_baseline():
    print("🤖 Initializing Baseline (Zero-Shot LLM)...")

    # 2. Robust Config Loading
    config_path = os.path.join(project_root, "configs", "base.yaml")
    if not os.path.exists(config_path):
        print(f"❌ Config not found at {config_path}")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Initialize the "Dumb" LLM (No RAG, No Agents)
    llm = LocalHFClient(cfg)
    
    # 3. Load the TEST Dataset
    # We must compare against the 'test_set' to be fair
    data_dir = os.path.join(project_root, "data", "processed", "test_set")
    files = glob.glob(os.path.join(data_dir, "*.json"))
    
    if not files:
        print(f"❌ No test data found in {data_dir}")
        print("💡 Run 'scripts/generate_data.py' with SPLIT_NAME='test' first.")
        return

    print(f"📂 Found {len(files)} samples for baseline comparison.")
    
    y_true = []
    y_pred = []
    
    print("🚀 Running Baseline Inference...")
    for fpath in tqdm(files):
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                
                claim = data['claim_text']
                ground_truth = 1 if data['ground_truth'] else 0
                y_true.append(ground_truth)
                
                # Simple Zero-Shot Prompt (The Control Group)
                prompt = (
                    f"Claim: {claim}\n"
                    "Is this claim True or False? Answer with a single word: TRUE or FALSE."
                )
                
                # The LLM tries to answer without tools/evidence
                response = llm.generate("You are a fact checker.", prompt)
                
                # Parse Response
                if "TRUE" in response.upper():
                    y_pred.append(1)
                else:
                    # If it says "False" or refuses to answer, we count it as False (or 0)
                    y_pred.append(0)
        except Exception as e:
            print(f"⚠️ Error reading {os.path.basename(fpath)}: {e}")
            continue
                
    # 4. Report Results
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "="*40)
    print(f"📉 BASELINE ACCURACY: {acc:.4f}")
    print("="*40)

if __name__ == "__main__":
    run_baseline()
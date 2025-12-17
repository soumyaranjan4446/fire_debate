import sys
import os

# --- 1. Path Setup (Critical for imports) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

import yaml
import json
import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from fire_debate.agents.local_client import LocalHFClient

def normalize_ground_truth(gt_raw):
    """
    Converts ground_truth from JSON into a safe integer label (0 or 1).
    """
    if isinstance(gt_raw, bool):
        return int(gt_raw)
    if isinstance(gt_raw, str):
        gt_raw = gt_raw.strip().lower()
        if gt_raw in ["true", "1", "yes"]: return 1
        elif gt_raw in ["false", "0", "no"]: return 0
    if isinstance(gt_raw, (int, float)):
        return int(gt_raw)
    return 0 # Default fallback

def run_baseline():
    print("🤖 Initializing Baseline (Zero-Shot LLM)...")

    # --- 2. Load Config ---
    config_path = os.path.join(project_root, "configs", "base.yaml")
    if not os.path.exists(config_path):
        print(f"❌ Config not found at {config_path}")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # --- 3. Initialize Baseline LLM ---
    llm = LocalHFClient(cfg)

    # --- 4. Load Test Dataset ---
    data_dir = os.path.join(project_root, "data", "processed", "test_set")
    files = glob.glob(os.path.join(data_dir, "*.json"))

    if not files:
        print(f"❌ No test data found in {data_dir}")
        return

    print(f"📂 Found {len(files)} samples for baseline comparison.")

    y_true = []
    y_pred = []

    print("🚀 Running Baseline Inference...")

    for fpath in tqdm(files):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # --- Extract Claim ---
            claim = data["claim_text"]
            
            # --- Normalize Ground Truth ---
            gt = normalize_ground_truth(data.get("ground_truth"))
            y_true.append(gt)

            # --- FIX: Construct Single Prompt String ---
            # We manually format the prompt since generate() likely expects one string.
            full_prompt = (
                "You are a strict fact checker. Determine if the following claim is True or False.\n\n"
                f"Claim: {claim}\n\n"
                "Task: Answer with exactly one word: TRUE or FALSE.\n"
                "Answer:"
            )

            # Generate (Limit tokens to avoid yapping)
            response = llm.generate(full_prompt, max_new_tokens=10)

            # --- Parse Prediction ---
            if "TRUE" in response.strip().upper():
                y_pred.append(1)
            else:
                y_pred.append(0)

        except Exception as e:
            # print(f"⚠️ Error reading {os.path.basename(fpath)}: {e}")
            pass

    # --- 5. Final Sanity Check ---
    if not y_pred:
        print("❌ No valid predictions made.")
        return

    # --- 6. Report Accuracy ---
    acc = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 40)
    print(f"📉 BASELINE ACCURACY: {acc:.4f}")
    print("=" * 40)

if __name__ == "__main__":
    run_baseline()
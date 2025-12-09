import yaml
import json
import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from fire_debate.agents.local_client import LocalHFClient

def run_baseline():
    # 1. Setup
    with open("configs/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    llm = LocalHFClient(cfg)
    
    # 2. Load the SAME dataset used for FIRE-Debate
    files = glob.glob("data/processed/training_set/*.json")
    y_true = []
    y_pred = []
    
    print(f"🤖 Running Baseline (Zero-Shot LLM) on {len(files)} samples...")
    
    for fpath in tqdm(files):
        with open(fpath, 'r') as f:
            data = json.load(f)
            claim = data['claim_text']
            ground_truth = 1 if data['ground_truth'] else 0
            y_true.append(ground_truth)
            
            # Simple Prompt
            prompt = (
                f"Claim: {claim}\n"
                "Is this claim True or False? Answer with a single word: TRUE or FALSE."
            )
            response = llm.generate("You are a fact checker.", prompt)
            
            # Parse Response
            if "TRUE" in response.upper():
                y_pred.append(1)
            else:
                y_pred.append(0) # Default to False if unsure/refusal
                
    # 3. Report
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "="*40)
    print(f"📉 BASELINE RESULTS: {acc:.4f}")
    print("="*40)

if __name__ == "__main__":
    run_baseline()
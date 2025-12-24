import sys
import os
import yaml
import json
import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 1. ROBUST PATH SETUP ---
# Determine the absolute path to the project root
# Logic: script is in /.../fire_debate/scripts/run_baseline_robust.py
script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(script_path)
project_root = os.path.dirname(scripts_dir)  # Go up one level from 'scripts'
package_dir = os.path.join(project_root, "fire_debate") # The inner package folder

print(f"🔧 Project Root set to: {project_root}")

# Add to path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Client
try:
    from fire_debate.agents.openai_client import OpenAIClient
    print("✅ Successfully imported 'fire_debate.agents'")
except ImportError:
    print("❌ Critical: Could not import fire_debate.agents. Check PYTHONPATH.")
    sys.exit(1)

# --- HELPER: Find Config Automatically ---
def find_config():
    """Searches for base.yaml in likely locations"""
    candidates = [
        os.path.join(project_root, "configs", "base.yaml"),
        os.path.join(package_dir, "configs", "base.yaml"),  # Check inside fire_debate/
        os.path.join(project_root, "base.yaml"),
        "configs/base.yaml"
    ]
    
    for path in candidates:
        if os.path.exists(path):
            print(f"✅ Found config at: {path}")
            return path
            
    return None

# --- HELPER: Data Parsing ---
def normalize_ground_truth(gt_raw):
    if isinstance(gt_raw, bool): return int(gt_raw)
    s = str(gt_raw).strip().lower()
    return 1 if s in ["true", "1", "yes"] else 0

def parse_prediction(text):
    if not text: return 0
    t = text.upper().replace("*", "").strip()
    if t.startswith("TRUE") or t == "TRUE": return 1
    if t.startswith("FALSE") or t == "FALSE": return 0
    if "TRUE" in t and "FALSE" not in t: return 1
    if "FALSE" in t and "TRUE" not in t: return 0
    return 0

# --- MAIN RUNNER ---
def run_baseline():
    print("\n🤖 Initializing Strong Baseline (GPT-4o-mini)...")

    # 1. Load Config
    config_path = find_config()
    if not config_path:
        print("❌ ERROR: 'base.yaml' not found in any standard folder.")
        print("   Please create 'configs/base.yaml' with your OPENAI_API_KEY.")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Init OpenAI
    try:
        llm = OpenAIClient(cfg)
    except Exception as e:
        print(f"❌ OpenAI Init Failed: {e}")
        return

    # 3. Load Data (Robust Search)
    data_dir = os.path.join(project_root, "data", "processed", "test_openai_set")
    
    # Fallback search if main path is empty
    if not os.path.exists(data_dir):
         data_dir = os.path.join(package_dir, "data", "processed", "test_openai_set")

    files = glob.glob(os.path.join(data_dir, "*.json"))
    
    if not files:
        print(f"❌ No test data found in: {data_dir}")
        print(f"   (Checked {project_root}/data/... and {package_dir}/data/...)")
        return

    print(f"📂 Found {len(files)} samples. Running Inference...")

    y_true = []
    y_pred = []

    # 4. Inference Loop
    for fpath in tqdm(files, desc="Fact Checking"):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle missing ground_truth key safely
            gt_val = data.get("ground_truth")
            if gt_val is None: gt_val = data.get("label", False)

            gt = normalize_ground_truth(gt_val)
            y_true.append(gt)

            prompt = (
                "You are an expert fact-checking AI.\n"
                f"Claim: \"{data.get('claim_text', data.get('claim'))}\"\n"
                "Is this claim True or False?\n"
                "Reply with exactly one word: TRUE or FALSE."
            )
            
            resp = llm.generate(prompt, max_new_tokens=10)
            y_pred.append(parse_prediction(resp))

        except Exception:
            continue

    # 5. Results
    acc = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*50)
    print(f"📉 BASELINE RESULT (Raw GPT-4o-mini)")
    print("="*50)
    print(f"Accuracy: {acc:.4f}")
    print("-" * 30)
    print(classification_report(y_true, y_pred, target_names=["FALSE", "TRUE"], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("="*50)

if __name__ == "__main__":
    run_baseline()
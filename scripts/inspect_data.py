import sys
import os
import glob
import json

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from fire_debate.insight.graph_builder import GraphBuilder

def inspect():
    # 1. Initialize the Neuro-Symbolic Engine
    print("🕵️ Loading Fallacy Detector (DeBERTa) for inspection...")
    # We use CPU for inspection to avoid messing with GPU memory if training is running
    builder = GraphBuilder(device="cpu")
    
    # 2. Find Data
    data_path = os.path.join(project_root, "data", "processed", "train_set", "*.json")
    files = glob.glob(data_path)
    
    if not files:
        print(f"❌ No data found in {data_path}")
        return

    print(f"📂 Found {len(files)} files. Analyzing samples...\n")

    # 3. Collect Samples
    true_samples = []
    false_samples = []

    for f in files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                # Store sample based on label
                if data['ground_truth'] and len(true_samples) < 3:
                    true_samples.append(data)
                elif not data['ground_truth'] and len(false_samples) < 3:
                    false_samples.append(data)
                
                # Stop once we have enough to inspect
                if len(true_samples) >= 3 and len(false_samples) >= 3:
                    break
        except:
            continue

    # 4. Helper Function to Print Analysis
    def analyze_sample(sample):
        claim = sample['claim_text']
        label = "TRUE (Real)" if sample['ground_truth'] else "FALSE (Fake)"
        
        # Get the first substantial argument (usually Turn 0 or 1)
        if not sample['turns']: return
        
        # Find the first argument that isn't empty
        arg_text = sample['turns'][0]['text']
        
        # --- THE CORE TEST ---
        # Run the text through the Neural Network right now
        scores = builder.fallacy_detector.detect(arg_text)
        logic_score = scores.get("logical reasoning", 0.0)
        
        print(f"📝 Claim: {claim[:80]}...")
        print(f"   Label: {label}")
        print(f"   Argument Snippet: {arg_text[:80]}...")
        print(f"   🧠 Computed Logic Score: {logic_score:.4f}") # 1.0 is Perfect Logic
        print("-" * 60)

    # 5. Print Results
    print("="*20 + " ANALYZING TRUE CLAIMS " + "="*20)
    for s in true_samples:
        analyze_sample(s)
        
    print("\n" + "="*20 + " ANALYZING FALSE CLAIMS " + "="*20)
    for s in false_samples:
        analyze_sample(s)

if __name__ == "__main__":
    inspect()
import sys
import os

# --- 1. Robust Path Setup (Fixes "Module Not Found" & Windows path issues) ---
# Get the absolute path of the folder containing this script (scripts/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root (fire_debate/)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# Add to python path
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import json
import glob
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from fire_debate.schemas.debate import DebateLog, DebateTurn
from fire_debate.insight.graph_builder import GraphBuilder
from fire_debate.insight.hgt_judge import HGTJudge

def load_test_data(data_dir):
    # Ensure we look at the absolute path
    abs_data_dir = os.path.abspath(data_dir)
    
    print(f"📂 Searching for data in:\n   -> {abs_data_dir}")
    
    # Search for json files
    search_path = os.path.join(abs_data_dir, "*.json")
    files = glob.glob(search_path)
    print(f"   Found {len(files)} test samples.")
    
    if len(files) == 0:
        return [], []

    # Use CPU for building graphs
    builder = GraphBuilder(device="cpu")
    graphs = []
    y_true = []
    
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                
                # Reconstruct DebateLog object
                turns = [DebateTurn(**t) for t in data['turns']]
                
                log = DebateLog(
                    debate_id=data['debate_id'],
                    claim_id=str(data['claim_id']),
                    claim_text=data['claim_text'],
                    ground_truth=data['ground_truth'],
                    turns=turns
                )
                
                # Build Neuro-Symbolic Graph
                graph = builder.build_graph(log)
                graphs.append(graph)
                y_true.append(1 if log.ground_truth else 0)
                
        except Exception as e:
            print(f"⚠️ Skipping broken file {os.path.basename(fpath)}: {e}")
            
    return graphs, y_true

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🧪 Evaluating on {device}...")
    
    # --- PATH FIX: Use absolute path to 'test_set' ---
    TEST_DIR = os.path.join(project_root, "data", "processed", "test_set")
    
    graphs, y_true = load_test_data(TEST_DIR)
    
    if not graphs:
        print(f"❌ No data found! Please check that this folder exists:")
        print(f"   {TEST_DIR}")
        print("💡 Hint: Did you run 'scripts/generate_data.py' with SPLIT_NAME='test'?")
        return

    # Load Model (Metadata needed for dimensions)
    metadata = graphs[0].metadata()
    model = HGTJudge(64, 1, 2, 2, metadata).to(device)
    
    # Use absolute path for model weights
    model_path = os.path.join(project_root, "data", "processed", "hgt_judge.pth")
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Trained model loaded from {os.path.basename(model_path)}")
    except FileNotFoundError:
        print(f"❌ Model weights not found at: {model_path}")
        print("   Run 'fire_debate/training/train_judge.py' first!")
        return

    # Inference
    model.eval()
    y_pred = []
    
    print("🚀 Running Inference Loop...")
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            
            # Forward Pass
            logits = model(graph.x_dict, graph.edge_index_dict)
            prob = logits.item()
            
            # Threshold at 0.5
            prediction = 1 if prob > 0.5 else 0
            y_pred.append(prediction)

    # Metrics
    print("\n" + "="*60)
    print("📊 FIRE-DEBATE RESULTS (Neuro-Symbolic Judge)")
    print("="*60)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("-" * 30)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["FALSE", "TRUE"], zero_division=0))
    print("-" * 30)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("="*60)

if __name__ == "__main__":
    evaluate()
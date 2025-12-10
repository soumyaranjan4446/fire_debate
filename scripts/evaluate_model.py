import sys
import os
# Fix imports so python can find fire_debate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import json
import glob
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from fire_debate.schemas.debate import DebateLog, DebateTurn
from fire_debate.insight.graph_builder import GraphBuilder
from fire_debate.insight.hgt_judge import HGTJudge

def load_test_data(data_dir):
    files = glob.glob(f"{data_dir}/*.json")
    print(f"📂 Loading {len(files)} test samples...")
    
    # Use CPU for building graphs (saving GPU for the model inference)
    builder = GraphBuilder(device="cpu")
    graphs = []
    y_true = []
    
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                
                # Reconstruct DebateLog object from JSON
                # The **t unpacking works even if we added 'search_query' to the schema
                turns = [DebateTurn(**t) for t in data['turns']]
                
                log = DebateLog(
                    debate_id=data['debate_id'],
                    claim_id=data['claim_id'],
                    claim_text=data['claim_text'],
                    ground_truth=data['ground_truth'],
                    turns=turns
                )
                
                # Build the Neuro-Symbolic Graph
                graph = builder.build_graph(log)
                graphs.append(graph)
                y_true.append(1 if log.ground_truth else 0)
                
        except Exception as e:
            print(f"⚠️ Skipping broken/incompatible file {fpath}: {e}")
            
    return graphs, y_true

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🧪 Evaluating on {device}...")
    
    # 1. Load Data
    # In a real paper, you would have a separate 'data/processed/test_set'
    # For now, we reuse training_set to verify the pipeline works
    graphs, y_true = load_test_data("data/processed/training_set")
    
    if not graphs:
        print("❌ No data found. Run 'scripts/generate_data.py' first.")
        return

    # 2. Load Model
    # Get metadata from the first graph to initialize dimensions correctly
    metadata = graphs[0].metadata()
    model = HGTJudge(64, 1, 2, 2, metadata).to(device)
    
    model_path = "data/processed/hgt_judge.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Trained model loaded from {model_path}")
    except FileNotFoundError:
        print("❌ Model weights not found. Run 'fire_debate/training/train_judge.py' first!")
        return

    # 3. Inference Loop
    model.eval()
    y_pred = []
    
    print("running Inference loop...")
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            
            # Forward Pass
            logits = model(graph.x_dict, graph.edge_index_dict)
            prob = logits.item()
            
            # Threshold at 0.5 (True > 0.5)
            prediction = 1 if prob > 0.5 else 0
            y_pred.append(prediction)

    # 4. Metrics Report
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
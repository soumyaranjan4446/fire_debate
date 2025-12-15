import sys
import os

# --- 1. Robust Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..')) # Go up 2 levels from fire_debate/training/
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import json
import glob
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from fire_debate.schemas.debate import DebateLog, DebateTurn
from fire_debate.insight.graph_builder import GraphBuilder
from fire_debate.insight.hgt_judge import HGTModel 

def load_test_data(data_dir):
    abs_data_dir = os.path.abspath(data_dir)
    print(f"📂 Searching for data in:\n   -> {abs_data_dir}")
    
    search_path = os.path.join(abs_data_dir, "*.json")
    files = glob.glob(search_path)
    print(f"   Found {len(files)} test samples.")
    
    if len(files) == 0:
        return [], []

    builder = GraphBuilder(device="cpu")
    graphs = []
    y_true = []
    
    for fpath in files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # --- FIX: Robust Turn Loading ---
                turns = []
                for idx, t in enumerate(data.get('turns', [])):
                    if isinstance(t, dict):
                        # 1. Inject missing ID
                        if 'turn_id' not in t: 
                            t['turn_id'] = str(idx)
                        # 2. Remove 'speaker' if present (Schema update)
                        if 'speaker' in t: 
                            del t['speaker']
                        # 3. Handle citation -> citations list
                        if 'citation' in t and 'citations' not in t:
                            t['citations'] = [t.pop('citation')]
                            
                        turns.append(DebateTurn(**t))
                    else:
                        turns.append(t)
                # --------------------------------
                
                log = DebateLog(
                    debate_id=data['debate_id'],
                    claim_id=str(data['claim_id']),
                    claim_text=data['claim_text'],
                    ground_truth=data['ground_truth'],
                    turns=turns
                )
                
                graph = builder.build_graph(log)
                
                # Basic validation
                if graph.num_edges == 0:
                    continue
                    
                graphs.append(graph)
                y_true.append(1 if log.ground_truth else 0)
                
        except Exception as e:
            # print(f"⚠️ Skipping broken file {os.path.basename(fpath)}: {e}")
            pass
            
    print(f"✅ Successfully loaded {len(graphs)} valid graphs.")
    return graphs, y_true

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🧪 Evaluating on {device}...")
    
    TEST_DIR = os.path.join(project_root, "data", "processed", "test_set")
    
    graphs, y_true = load_test_data(TEST_DIR)
    
    if not graphs:
        print(f"❌ No data found! Please check that this folder exists:")
        print(f"   {TEST_DIR}")
        return

    # --- FIX: Detect Feature Dimension ---
    sample_node_type = list(graphs[0].x_dict.keys())[0]
    in_channels = graphs[0].x_dict[sample_node_type].shape[1]
    print(f"🔍 Detected input feature dimension: {in_channels}")

    # Load Model
    # Use metadata from first graph (assumes all are same schema)
    if hasattr(graphs[0], 'metadata'):
        metadata = graphs[0].metadata()
    else:
        metadata = (list(graphs[0].x_dict.keys()), list(graphs[0].edge_index_dict.keys()))
    
    model = HGTModel(
        in_channels=in_channels, # Pass the detected dimension (771)
        hidden_channels=64, 
        out_channels=1, 
        num_heads=2, 
        num_layers=2, 
        metadata=metadata
    ).to(device)
    
    model_path = os.path.join(project_root, "data", "processed", "hgt_judge.pth")
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Trained model loaded from {os.path.basename(model_path)}")
    except FileNotFoundError:
        print(f"❌ Model weights not found at: {model_path}")
        print("   Run 'fire_debate/training/train_judge.py' first!")
        return
    except RuntimeError as e:
        print(f"❌ Shape Mismatch: {e}")
        print("   Hint: Did you retrain the model after changing input dimensions?")
        return

    # Inference
    model.eval()
    y_pred = []
    probs = []
    
    print("🚀 Running Inference Loop...")
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            
            # Forward Pass (Single item batch, batch_dict=None is fine)
            logits = model(graph.x_dict, graph.edge_index_dict, batch_dict=None)
            
            # Sigmoid for probability
            prob = torch.sigmoid(logits).item()
            probs.append(prob)
            
            # Threshold at 0.5
            prediction = 1 if prob > 0.5 else 0
            y_pred.append(prediction)

    # Metrics
    print("\n" + "="*60)
    print("📊 FIRE-DEBATE RESULTS (Neuro-Symbolic Judge)")
    print("="*60)
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("-" * 30)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["FALSE", "TRUE"], zero_division=0))
    print("-" * 30)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("="*60)

if __name__ == "__main__":
    evaluate()
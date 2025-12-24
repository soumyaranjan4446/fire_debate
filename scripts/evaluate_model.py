import sys
import os

# --- 1. Robust Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import json
import glob
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from fire_debate.schemas.debate import DebateLog, DebateTurn
from fire_debate.insight.graph_builder import GraphBuilder
from fire_debate.insight.hgt_judge import HGTModel


# ---------------------------------------------------------
# Load and build graphs from test data
# ---------------------------------------------------------
def load_test_data(data_dir):
    abs_data_dir = os.path.abspath(data_dir)
    print(f"📂 Searching for data in:\n   -> {abs_data_dir}")
    
    files = glob.glob(os.path.join(abs_data_dir, "*.json"))
    print(f"   Found {len(files)} test samples.")
    
    if not files:
        return [], []

    builder = GraphBuilder(device="cpu")
    graphs = []
    y_true = []

    print("🕸️  Building Graphs from Test Data...")
    for fpath in files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # ---- Robust Turn Loading ----
            turns = []
            for idx, t in enumerate(data.get('turns', [])):
                if isinstance(t, dict):
                    if 'turn_id' not in t:
                        t['turn_id'] = str(idx)

                    if 'speaker' in t:
                        del t['speaker']

                    if 'citation' in t and 'citations' not in t:
                        t['citations'] = [t.pop('citation')]

                    turns.append(DebateTurn(**t))
                else:
                    turns.append(t)

            # Handle Ground Truth key mismatch
            gt_value = data.get('ground_truth')
            if gt_value is None:
                gt_value = data.get('label', False)

            log = DebateLog(
                debate_id=str(data.get('debate_id', data.get('id', '0'))),
                claim_id=str(data.get('claim_id', 'claim_0')),
                claim_text=data.get('claim_text', data.get('claim', '')),
                ground_truth=gt_value,
                turns=turns
            )

            graph = builder.build_graph(log)

            # ---- VALIDATION (HeteroData-safe) ----
            if 'argument' not in graph.node_types:
                continue
            if graph['argument'].num_nodes == 0:
                continue

            graphs.append(graph)
            y_true.append(1 if log.ground_truth else 0)

        except Exception as e:
            # print(f"⚠️ Skipping broken file {os.path.basename(fpath)}: {e}")
            continue

    print(f"✅ Successfully loaded {len(graphs)} valid graphs.")
    return graphs, y_true


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🧪 Evaluating on {device}...")

    TEST_DIR = os.path.join(project_root, "data", "processed", "test_openai_set")
    
    if not os.path.exists(TEST_DIR):
        print(f"❌ Error: Test directory not found at {TEST_DIR}")
        print("   Run: python scripts/generate_openai_data.py --count 100 --out_dir data/processed/test_openai_set")
        return

    graphs, y_true = load_test_data(TEST_DIR)

    if not graphs:
        print("❌ No valid graphs found.")
        return

    # --- Detect feature dimension dynamically ---
    sample_node_type = list(graphs[0].x_dict.keys())[0]
    in_channels = graphs[0].x_dict[sample_node_type].shape[1]
    print(f"🔍 Detected input feature dimension: {in_channels}")

    # --- Metadata from graph schema ---
    metadata = graphs[0].metadata()

    # --- Load Model (UPDATED TO MATCH PRO-MODE TRAINING) ---
    model = HGTModel(
        in_channels=in_channels,
        hidden_channels=128,  # MATCH TRAINING: 64 -> 128
        out_channels=1,
        num_heads=8,          # MATCH TRAINING: 2 -> 8
        num_layers=3,         # MATCH TRAINING: 2 -> 3
        metadata=metadata
    ).to(device)

    model_path = os.path.join(project_root, "data", "processed", "hgt_judge.pth")

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Trained model loaded from {os.path.basename(model_path)}")
    except FileNotFoundError:
        print("❌ Model weights not found. Train the judge first.")
        return
    except RuntimeError as e:
        print(f"❌ Shape mismatch: {e}")
        return

    # --- Inference ---
    model.eval()
    y_pred = []
    probs = []

    print("🚀 Running Inference Loop...")
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            # Batch dict needs to be None for single graph inference
            logits = model(graph.x_dict, graph.edge_index_dict, batch_dict=None)
            prob = torch.sigmoid(logits).item()
            probs.append(prob)
            y_pred.append(1 if prob > 0.5 else 0)

    # --- Metrics ---
    print("\n" + "=" * 60)
    print("📊 FIRE-DEBATE RESULTS (Evidence-Aware HGT Judge)")
    print("=" * 60)
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("-" * 30)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["FALSE", "TRUE"], zero_division=0))
    print("-" * 30)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
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
    builder = GraphBuilder(device="cpu")
    graphs = []
    y_true = []
    
    for fpath in files:
        with open(fpath, 'r') as f:
            data = json.load(f)
            # Reconstruct Log
            turns = [DebateTurn(**t) for t in data['turns']]
            log = DebateLog(
                debate_id=data['debate_id'],
                claim_id=data['claim_id'],
                claim_text=data['claim_text'],
                ground_truth=data['ground_truth'],
                turns=turns
            )
            
            graph = builder.build_graph(log)
            graphs.append(graph)
            y_true.append(1 if log.ground_truth else 0)
            
    return graphs, y_true

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data (Ideally, generate a separate 'test_set' folder using generate_data.py)
    # For now, we reuse training_set just to test the code
    graphs, y_true = load_test_data("data/processed/training_set")
    
    if not graphs:
        print("❌ No data found.")
        return

    # 2. Load Model
    # We need to construct the model structure first, then load weights
    metadata = graphs[0].metadata()
    model = HGTJudge(64, 1, 2, 2, metadata).to(device)
    
    try:
        model.load_state_dict(torch.load("data/processed/hgt_judge.pth"))
        print("✅ Trained model loaded.")
    except FileNotFoundError:
        print("❌ Model weights not found. Run training first!")
        return

    # 3. Inference Loop
    model.eval()
    y_pred = []
    
    print("🧪 Running Inference...")
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            logits = model(graph.x_dict, graph.edge_index_dict)
            prob = logits.item()
            prediction = 1 if prob > 0.5 else 0
            y_pred.append(prediction)

    # 4. Metrics
    print("\n" + "="*40)
    print("📊 FIRE-DEBATE RESULTS")
    print("="*40)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["FALSE", "TRUE"]))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate()
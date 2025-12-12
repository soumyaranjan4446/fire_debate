import sys
import os
# Add the project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
from fire_debate.schemas.debate import DebateLog, DebateTurn
from fire_debate.insight.graph_builder import GraphBuilder
from fire_debate.insight.hgt_judge import HGTJudge

def load_log(path: str) -> DebateLog:
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Reconstruct objects
    turns = [DebateTurn(**t) for t in data['turns']]
    # Fix datetime string if necessary, or ignore for graph
    
    return DebateLog(
        debate_id=data['debate_id'],
        claim_id=data['claim_id'],
        claim_text=data['claim_text'],
        ground_truth=data['ground_truth'],
        turns=turns
    )

def main():
    print("--- 🔬 STARTING FIRE-DEBATE ANALYSIS ---")
    
    # 1. Load Data
    try:
        log = load_log("data/processed/debate_result.json")
    except FileNotFoundError:
        print("❌ No debate log found! Please run 'scripts/run_debate.py' first.")
        return

    # 2. Build Graph (Includes Fallacy Detection)
    # Note: First run downloads DeBERTa (may take time)
    builder = GraphBuilder(device="cuda" if torch.cuda.is_available() else "cpu")
    graph_data = builder.build_graph(log)
    
    print("\n✅ Graph Constructed Successfully:")
    print(graph_data)
    print(f"   Node Types: {graph_data.node_types}")
    print(f"   Edge Types: {graph_data.edge_types}")
    
    # 3. Initialize Judge Model
    # We infer metadata from the built graph
    metadata = graph_data.metadata()
    model = HGTJudge(
        hidden_channels=64, 
        out_channels=1, 
        num_heads=2, 
        num_layers=2, 
        metadata=metadata
    )
    
    # 4. Run Inference (Mock Prediction)
    # In a real scenario, you would load state_dict() from a trained model
    model.eval()
    with torch.no_grad():
        # HGT requires x_dict and edge_index_dict
        prediction = model(graph_data.x_dict, graph_data.edge_index_dict)
    
    print("\n--- ⚖️ JUDGEMENT ---")
    score = prediction.item()
    verdict = "TRUE" if score > 0.5 else "FALSE"
    print(f"Model Confidence: {score:.4f}")
    print(f"Predicted Verdict: {verdict}")
    
    # 5. Fallacy Report (Bonus)
    print("\n--- ⚠️ FALLACY REPORT ---")
    # We can peek at the fallacy features we computed
    # Recall: Features are [Embedding (384) | Fallacy (6)]
    # Let's just print the raw text analysis from the builder's detector (re-running for display)
    detector = builder.fallacy_detector
    for t in log.turns:
        scores = detector.analyze_turn(t.text)
        top = detector.get_top_fallacy(scores)
        if top:
            print(f"Agent {t.agent_id}: DETECTED '{top}'")
        else:
            print(f"Agent {t.agent_id}: Logical.")

if __name__ == "__main__":
    main()
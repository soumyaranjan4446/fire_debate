import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import glob
import json
import random
from fire_debate.schemas.debate import DebateLog, DebateTurn
from fire_debate.insight.graph_builder import GraphBuilder
from fire_debate.insight.hgt_judge import HGTJudge

def load_dataset(data_dir):
    abs_data_dir = os.path.abspath(data_dir)
    files = glob.glob(os.path.join(abs_data_dir, "*.json"))
    
    print(f"📂 Looking in: {abs_data_dir}")
    print(f"   Found {len(files)} debate logs.")
    
    if len(files) == 0: return []

    builder = GraphBuilder(device="cpu") 
    graphs = []
    
    print("🕸️  Building Graphs...")
    for i, fpath in enumerate(files):
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                turns = [DebateTurn(**t) for t in data['turns']]
                log = DebateLog(
                    debate_id=data['debate_id'],
                    claim_id=str(data['claim_id']),
                    claim_text=data['claim_text'],
                    ground_truth=data['ground_truth'],
                    turns=turns
                )
                
                graph = builder.build_graph(log)
                graph.y = torch.tensor([1.0 if log.ground_truth else 0.0], dtype=torch.float)
                
                # --- DEBUG: CHECK DATA HEALTH ---
                if i == 0:
                    print(f"   [Debug Sample] Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")
                    if graph.num_edges == 0:
                        print("   ⚠️ WARNING: Graph has 0 edges! The GNN cannot learn.")
                
                graphs.append(graph)
        except Exception as e:
            pass # Skip broken files silently
            
    return graphs

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🏋️ Training on {device}")
    
    TRAIN_DIR = os.path.join(project_root, "data", "processed", "train_set")
    graphs = load_dataset(TRAIN_DIR)
    
    if not graphs:
        print("❌ No data found.")
        return

    metadata = graphs[0].metadata()
    model = HGTJudge(64, 1, 2, 2, metadata).to(device)
    
    # FIX 1: Lower Learning Rate to prevent "shocking" the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # FIX 2: Remove weight temporarily to check if loss moves
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print("\n🚀 STARTING TRAINING LOOP (Debug Mode)...")
    model.train()
    
    for epoch in range(50): 
        total_loss = 0
        random.shuffle(graphs)
        
        # Track predictions to see if they change
        sample_pred = 0.0
        
        for i, graph in enumerate(graphs):
            graph = graph.to(device)
            optimizer.zero_grad()
            
            out = model(graph.x_dict, graph.edge_index_dict)
            loss = criterion(out.view(-1), graph.y)
            loss.backward()
            
            # Gradient Clipping (Prevents explosion)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            if i == 0: sample_pred = torch.sigmoid(out).item()
            
        avg_loss = total_loss / len(graphs)
        
        # Print Pred to show it's alive
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Sample Pred: {sample_pred:.4f}")

    out_path = os.path.join(project_root, "data", "processed", "hgt_judge.pth")
    torch.save(model.state_dict(), out_path)
    print(f"💾 Model saved.")

if __name__ == "__main__":
    train()
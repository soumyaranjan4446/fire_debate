import sys
import os
# Fix imports so python can find fire_debate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import glob
import json
import random
from torch_geometric.loader import DataLoader
from fire_debate.schemas.debate import DebateLog, DebateTurn
from fire_debate.insight.graph_builder import GraphBuilder
from fire_debate.insight.hgt_judge import HGTJudge

def load_dataset(data_dir):
    # Search for JSON files
    search_path = os.path.join(data_dir, "*.json")
    files = glob.glob(search_path)
    
    print(f"📂 Looking in: {os.path.abspath(data_dir)}")
    print(f"   Found {len(files)} debate logs.")
    
    if len(files) == 0:
        return []

    # Use CPU for graph building to save GPU memory for training
    builder = GraphBuilder(device="cpu") 
    graphs = []
    
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                
                # Reconstruct object
                turns = [DebateTurn(**t) for t in data['turns']]
                log = DebateLog(
                    debate_id=data['debate_id'],
                    claim_id=str(data['claim_id']), # Ensure ID is a string
                    claim_text=data['claim_text'],
                    ground_truth=data['ground_truth'],
                    turns=turns
                )
                
                # Build Neuro-Symbolic Graph
                graph = builder.build_graph(log)
                
                # Label (1.0 = True, 0.0 = False)
                graph.y = torch.tensor([1.0 if log.ground_truth else 0.0], dtype=torch.float)
                graphs.append(graph)
        except Exception as e:
            print(f"⚠️ Skipping broken file {os.path.basename(fpath)}: {e}")
            
    return graphs

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🏋️ Training on {device}")
    
    # 1. Load Data
    # FIX: Point to 'train_set' instead of 'training_set'
    TRAIN_DIR = "data/processed/train_set"
    
    graphs = load_dataset(TRAIN_DIR)
    
    if not graphs:
        print(f"❌ No data found in {TRAIN_DIR}")
        print("💡 Run 'scripts/generate_data.py' (with SPLIT_NAME='train') first.")
        return

    # 2. Setup Model
    # We get metadata (node types, edge types) from the first graph
    metadata = graphs[0].metadata()
    
    model = HGTJudge(
        hidden_channels=64, 
        out_channels=1, 
        num_heads=2, 
        num_layers=2, 
        metadata=metadata
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print("\n🚀 STARTING TRAINING LOOP...")
    model.train()
    
    # Train for 20 Epochs
    for epoch in range(20): 
        total_loss = 0
        random.shuffle(graphs)
        
        for graph in graphs:
            graph = graph.to(device)
            optimizer.zero_grad()
            
            # Forward Pass
            out = model(graph.x_dict, graph.edge_index_dict)
            
            loss = criterion(out.view(-1), graph.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(graphs)
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f}")

    # Save
    out_path = "data/processed/hgt_judge.pth"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"💾 Model saved to {out_path}")

if __name__ == "__main__":
    train()
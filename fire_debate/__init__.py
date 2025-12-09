import torch
import glob
import json
import random
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from fire_debate.schemas.debate import DebateLog, DebateTurn
from fire_debate.insight.graph_builder import GraphBuilder
from fire_debate.insight.hgt_judge import HGTJudge

def load_dataset(data_dir):
    files = glob.glob(f"{data_dir}/*.json")
    print(f"📂 Found {len(files)} debate logs.")
    
    builder = GraphBuilder(device="cpu") # Build graphs on CPU to save GPU for training
    graphs = []
    
    for fpath in files:
        with open(fpath, 'r') as f:
            data = json.load(f)
            # Reconstruct simplified object
            turns = [DebateTurn(**t) for t in data['turns']]
            log = DebateLog(
                debate_id=data['debate_id'],
                claim_id=data['claim_id'],
                claim_text=data['claim_text'],
                ground_truth=data['ground_truth'],
                turns=turns
            )
            
            # Convert to PyG Graph
            graph = builder.build_graph(log)
            
            # Attach label to graph object for training
            # 1.0 for True, 0.0 for False
            graph.y = torch.tensor([1.0 if log.ground_truth else 0.0], dtype=torch.float)
            graphs.append(graph)
            
    return graphs

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Prepare Data
    graphs = load_dataset("data/processed/training_set")
    if not graphs:
        print("❌ No data found. Run 'generate_data.py' first.")
        return

    # Use first graph to get metadata (node types, edge types)
    metadata = graphs[0].metadata()
    
    # 2. Setup Model
    model = HGTJudge(
        hidden_channels=64, 
        out_channels=1, 
        num_heads=2, 
        num_layers=2, 
        metadata=metadata
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print("\n🏋️ STARTING TRAINING...")
    model.train()
    
    for epoch in range(10): # Small epoch count for demo
        total_loss = 0
        random.shuffle(graphs)
        
        for graph in graphs:
            graph = graph.to(device)
            optimizer.zero_grad()
            
            # Forward Pass
            out = model(graph.x_dict, graph.edge_index_dict)
            
            # Calculate Loss
            loss = criterion(out.view(-1), graph.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(graphs)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

    # Save the trained brain
    torch.save(model.state_dict(), "data/processed/hgt_judge.pth")
    print("💾 Model saved to data/processed/hgt_judge.pth")

if __name__ == "__main__":
    train()
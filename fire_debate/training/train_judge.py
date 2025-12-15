import sys
import os

# --- Path Setup ---
# Assumes script is at fire_debate/training/train_judge.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import glob
import json
import random
from torch_geometric.loader import DataLoader  # Efficient batching for Graphs
from fire_debate.schemas.debate import DebateLog, DebateTurn
from fire_debate.insight.graph_builder import GraphBuilder
from fire_debate.insight.hgt_judge import HGTModel

def load_dataset(data_dir):
    """
    Loads JSON files, converts them to HeteroData graphs using GraphBuilder.
    Patches missing 'turn_id' in legacy data if necessary.
    """
    abs_data_dir = os.path.abspath(data_dir)
    files = glob.glob(os.path.join(abs_data_dir, "*.json"))
    
    print(f"📂 Looking in: {abs_data_dir}")
    print(f"   Found {len(files)} debate logs.")
    
    if len(files) == 0: return []

    # Initialize GraphBuilder
    try:
        builder = GraphBuilder(device="cpu") 
    except Exception as e:
        print(f"❌ Failed to initialize GraphBuilder: {e}")
        return []

    graphs = []
    print("🕸️  Building Graphs...")
    
    for i, fpath in enumerate(files):
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # --- FIX: Handle missing turn_id ---
                turns_data = data.get('turns', [])
                turns = []
                for idx, t in enumerate(turns_data):
                    if isinstance(t, dict):
                        # If 'turn_id' is missing from the JSON, inject it using the index
                        if 'turn_id' not in t:
                            t['turn_id'] = str(idx)
                        turns.append(DebateTurn(**t))
                    else:
                        turns.append(t)
                # -----------------------------------
                
                log = DebateLog(
                    debate_id=data['debate_id'],
                    claim_id=str(data['claim_id']),
                    claim_text=data['claim_text'],
                    ground_truth=data['ground_truth'],
                    turns=turns
                )
                
                graph = builder.build_graph(log)
                
                # Assign Label: 1.0 for True, 0.0 for False
                label = 1.0 if log.ground_truth else 0.0
                graph.y = torch.tensor([label], dtype=torch.float)
                
                # Basic validation
                if graph.num_edges == 0:
                    continue

                graphs.append(graph)
                
        except Exception as e:
            # Print the specific error to help debug, but keep going
            print(f"⚠️  Skipping broken file {os.path.basename(fpath)}: {e}") 
            
    print(f"✅ Successfully loaded {len(graphs)} graphs.")
    return graphs

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🏋️  Training on {device}")
    
    TRAIN_DIR = os.path.join(project_root, "data", "processed", "train_set")
    
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Training directory not found: {TRAIN_DIR}")
        return

    graphs = load_dataset(TRAIN_DIR)
    
    if not graphs:
        print("❌ No data found or all files were broken.")
        return

    # --- Feature Dimension Check ---
    # Check if x_dict exists and has keys
    if not graphs[0].x_dict:
        print("❌ Error: Graph has no node features (x_dict is empty). Check GraphBuilder.")
        return

    sample_node_type = list(graphs[0].x_dict.keys())[0]
    in_channels = graphs[0].x_dict[sample_node_type].shape[1]
    print(f"🔍 Detected input feature dimension: {in_channels}")

    # Use DataLoader for batching
    train_loader = DataLoader(graphs, batch_size=4, shuffle=True)

    # Get Metadata for HGT
    # Ensure metadata exists
    if hasattr(graphs[0], 'metadata'):
        metadata = graphs[0].metadata()
    else:
        # Fallback for older PyG versions
        metadata = (list(graphs[0].x_dict.keys()), list(graphs[0].edge_index_dict.keys()))
    
    # Initialize Model
    model = HGTModel(
        in_channels=in_channels,
        hidden_channels=64, 
        out_channels=1, 
        num_heads=2, 
        num_layers=2, 
        metadata=metadata
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print("\n🚀 STARTING TRAINING LOOP...")
    model.train()
    
    epochs = 50 
    
    for epoch in range(epochs): 
        total_loss = 0
        steps = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # --- FIX: Construct batch_dict manually ---
            # This extracts the batch vector (e.g., [0,0,1,1,2,2]) for each node type
            # so the model knows which nodes belong to which graph in the batch.
            batch_dict = {
                node_type: batch[node_type].batch 
                for node_type in batch.x_dict.keys()
            }
            
            # Forward pass with batch_dict
            out = model(batch.x_dict, batch.edge_index_dict, batch_dict)
            
            # Loss calculation
            # out.view(-1) aligns shape to [Batch_Size]
            loss = criterion(out.view(-1), batch.y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
        avg_loss = total_loss / steps if steps > 0 else 0
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:02d} | Avg Loss: {avg_loss:.4f}")

    # Save Model
    out_path = os.path.join(project_root, "data", "processed", "hgt_judge.pth")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    torch.save(model.state_dict(), out_path)
    print(f"💾 Model saved to: {out_path}")

if __name__ == "__main__":
    train()
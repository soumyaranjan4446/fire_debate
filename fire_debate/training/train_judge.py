import sys
import os

# --- Path Setup ---
# Fix: Go up 2 levels to find the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

if project_root not in sys.path:
    print(f"🔧 Adding project root to path: {project_root}")
    sys.path.insert(0, project_root)

import glob
import json
import torch
from torch_geometric.loader import DataLoader
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
                # Ground Truth is Boolean in JSON (True/False)
                label = 1.0 if log.ground_truth else 0.0
                graph.y = torch.tensor([label], dtype=torch.float)
                
                # Basic validation
                # HeteroData checks are slightly different than standard Data
                if 'argument' not in graph.node_types or graph['argument'].num_nodes == 0:
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
    
    # Updated to match where generate_data.py saves files
    TRAIN_DIR = os.path.join(project_root, "data", "processed", "train_set") 
    
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Training directory not found: {TRAIN_DIR}")
        print("   Did you run scripts/generate_data.py?")
        return

    graphs = load_dataset(TRAIN_DIR)
    
    if not graphs:
        print("❌ No data found or all files were broken.")
        return

    # Use DataLoader for batching
    train_loader = DataLoader(graphs, batch_size=4, shuffle=True)

    # Get Metadata for HGT (Node Types, Edge Types)
    metadata = graphs[0].metadata()
    
    print(f"🔍 Metadata detected: {metadata}")
    
    # Initialize Model
    model = HGTModel(
        in_channels=771,  # FIXED: Hardcoded to match GraphBuilder output
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
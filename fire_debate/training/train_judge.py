import sys
import os

# --- Path Setup ---
# Fix: Go up 2 levels to find the project root
# Level 0: .../fire_debate/training/train_judge.py
# Level 1: .../fire_debate/training
# Level 2: .../fire_debate (Package)
# Level 3: .../ (Project Root) -> WE NEED THIS ONE
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

if project_root not in sys.path:
    print(f"🔧 Adding project root to path: {project_root}")
    sys.path.insert(0, project_root)

# Verify import works immediately
try:
    import fire_debate
    print(f"✅ 'fire_debate' package found at: {os.path.dirname(fire_debate.__file__)}")
except ImportError:
    print("❌ CRITICAL: 'fire_debate' package still not found. Check directory structure.")
    sys.exit(1)

import glob
import json
import torch
from torch_geometric.loader import DataLoader
from fire_debate.schemas.debate import DebateLog, DebateTurn
from fire_debate.insight.graph_builder import GraphBuilder
from fire_debate.insight.hgt_judge import HGTModel

# ... (Rest of the file remains exactly the same) ...

def load_dataset(data_dir):
    """
    Loads JSON files from the specified directory and converts them to HeteroData graphs.
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
                if 'argument' not in graph.node_types or graph['argument'].num_nodes == 0:
                    continue

                graphs.append(graph)
                
        except Exception as e:
            # print(f"⚠️  Skipping broken file {os.path.basename(fpath)}: {e}") 
            continue
            
    print(f"✅ Successfully loaded {len(graphs)} graphs.")
    return graphs

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🏋️  Training on {device}")
    
    # --- CHANGED: Use the OpenAI Dataset Folder ---
    TRAIN_DIR = os.path.join(project_root, "data", "processed", "train_openai_set") 
    
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Training directory not found: {TRAIN_DIR}")
        print("   Did you run scripts/generate_openai_data.py?")
        return

    graphs = load_dataset(TRAIN_DIR)
    
    if not graphs:
        print("❌ No data found or all files were broken.")
        return

    # Use DataLoader for batching
    train_loader = DataLoader(graphs, batch_size=16, shuffle=True)

    # Get Metadata for HGT
    metadata = graphs[0].metadata()
    
    # --- CHANGED: Dynamic Input Size Detection ---
    # Automatically detects the embedding size (771, 384, etc.)
    sample_node_type = list(graphs[0].x_dict.keys())[0]
    input_dim = graphs[0].x_dict[sample_node_type].shape[1]
    
    print(f"🔍 Metadata detected: {metadata}")
    print(f"🔍 Input Dimension detected: {input_dim}")
    
    # Initialize Model
    model = HGTModel(
        in_channels=input_dim,  # Dynamic
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
        correct = 0
        total_samples = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Construct batch_dict manually for HGT
            batch_dict = {
                node_type: batch[node_type].batch 
                for node_type in batch.x_dict.keys()
            }
            
            # Forward pass
            out = model(batch.x_dict, batch.edge_index_dict, batch_dict)
            
            # Loss calculation
            loss = criterion(out.view(-1), batch.y)
            
            # Accuracy Metric
            preds = (torch.sigmoid(out.view(-1)) > 0.5).float()
            correct += (preds == batch.y).sum().item()
            total_samples += batch.y.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
        avg_loss = total_loss / steps if steps > 0 else 0
        accuracy = correct / total_samples if total_samples > 0 else 0
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2%}")

    # Save Model
    out_path = os.path.join(project_root, "data", "processed", "hgt_judge.pth")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    torch.save(model.state_dict(), out_path)
    print(f"💾 Model saved to: {out_path}")

if __name__ == "__main__":
    train()
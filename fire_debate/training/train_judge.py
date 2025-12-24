import sys
import os

# --- ROBUST PATH SETUP ---
script_path = os.path.abspath(__file__)
package_dir = os.path.dirname(os.path.dirname(script_path))
project_root = os.path.dirname(package_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import glob
import json
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter

from fire_debate.schemas.debate import DebateLog, DebateTurn
from fire_debate.insight.graph_builder import GraphBuilder
from fire_debate.insight.hgt_judge import HGTModel

def load_dataset(data_dir):
    abs_data_dir = os.path.abspath(data_dir)
    files = glob.glob(os.path.join(abs_data_dir, "*.json"))
    
    print(f"📂 Scanning directory: {abs_data_dir}")
    if len(files) == 0: return [], []

    logs = []
    labels = []

    for fpath in tqdm(files, desc="Parsing Files"):
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                item = json.load(f)
            
            gt_value = item.get('ground_truth')
            if gt_value is None:
                gt_value = item.get('label', False)
            
            turns = []
            for t in item.get('turns', []):
                if isinstance(t, dict):
                    if 'turn_id' not in t: t['turn_id'] = f"t_{len(turns)}"
                    turns.append(DebateTurn(**t))
                else:
                    turns.append(t)

            log = DebateLog(
                debate_id=str(item.get('debate_id', item.get('id', '0'))),
                claim_id=str(item.get('claim_id', 'claim_0')),
                claim_text=item.get('claim_text', item.get('claim', '')),
                ground_truth=gt_value,
                turns=turns
            )
            logs.append(log)
            labels.append(1.0 if gt_value else 0.0)

        except Exception:
            continue
            
    return logs, labels

def train():
    print("🚀 Starting PRO-MODE Training Sequence...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    TRAIN_DIR = os.path.join(project_root, "data", "processed", "train_openai_set")
    
    # 1. Load Data
    logs, labels = load_dataset(TRAIN_DIR)
    if not logs: return

    # Check Balance
    label_counts = Counter(labels)
    print(f"\n📊 Label Distribution: {dict(label_counts)}")

    # 2. Build Graphs
    print("\n🕸️  Building Neuro-Symbolic Graphs...")
    builder = GraphBuilder(device=str(device))
    graph_list = []
    
    for log, label in zip(tqdm(logs, desc="Graphing"), labels):
        try:
            g = builder.build_graph(log)
            g.y = torch.tensor([label], dtype=torch.float)
            if g['argument'].num_nodes > 0:
                graph_list.append(g)
        except Exception:
            continue

    if not graph_list: return

    # 3. Split & Loader
    train_graphs, val_graphs = train_test_split(graph_list, test_size=0.15, random_state=42) # More training data
    
    # Increased batch size for stability
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True) 
    val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
    
    print(f"   Training on {len(train_graphs)} graphs, Validating on {len(val_graphs)}.")

    # 4. Initialize LARGER Model
    sample_graph = graph_list[0]
    metadata = (list(sample_graph.x_dict.keys()), list(sample_graph.edge_index_dict.keys()))
    input_dim = sample_graph.x_dict[list(sample_graph.x_dict.keys())[0]].shape[1]

    model = HGTModel(
        in_channels=input_dim, 
        hidden_channels=128,  # UPGRADE: 64 -> 128
        out_channels=1,
        num_heads=8,          # UPGRADE: 4 -> 8
        num_layers=3,         # UPGRADE: 2 -> 3 (Deeper reasoning)
        metadata=metadata
    ).to(device)
    
    # 5. Optimized Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.005) # Lower LR, Lower Decay
    
    # Cosine Annealing is better for finding global minima
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.00001)
    
    criterion = nn.BCEWithLogitsLoss()

    # 6. Training Loop
    epochs = 50 # UPGRADE: Train longer
    best_acc = 0.0
    save_path = os.path.join(project_root, "data", "processed", "hgt_judge.pth")

    print("\n🏋️  Training Started...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        
        for batch in loop:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Reduce Noise slightly (0.05 -> 0.02) as data size increases
            for key in batch.x_dict:
                noise = torch.randn_like(batch.x_dict[key]) * 0.02
                batch.x_dict[key] += noise

            batch_dict = {
                node_type: batch[node_type].batch 
                for node_type in batch.x_dict.keys()
            }

            out = model(batch.x_dict, batch.edge_index_dict, batch_dict=batch_dict)
            loss = criterion(out.view(-1), batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
            loop.set_postfix(loss=loss.item())
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                batch_dict = {node_type: batch[node_type].batch for node_type in batch.x_dict.keys()}
                
                out = model(batch.x_dict, batch.edge_index_dict, batch_dict=batch_dict)
                preds = (torch.sigmoid(out) > 0.5).float().view(-1)
                correct += (preds == batch.y).sum().item()
                total += batch.y.size(0)
        
        val_acc = correct / total if total > 0 else 0
        print(f"   Val Acc: {val_acc:.2%}")

        scheduler.step()

        # Save Best Accuracy
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
    
    print(f"✅ Training Complete. Best Acc: {best_acc:.2%} Saved to: {save_path}")

if __name__ == "__main__":
    train()
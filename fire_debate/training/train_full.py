import sys
import os

# --- PATH SETUP ---
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
from fire_debate.schemas.debate import DebateLog, DebateTurn
from fire_debate.insight.graph_builder import GraphBuilder
from fire_debate.insight.hgt_judge import HGTModel

def load_all_data():
    """Scans BOTH train and test folders to get 100% of the data."""
    base_path = os.path.join(project_root, "data", "processed")
    
    # 1. Look in Train Folder
    train_files = glob.glob(os.path.join(base_path, "train_openai_set", "*.json"))
    train2_files = glob.glob(os.path.join(base_path, "train_set", "*.json"))
    # 2. Look in Test Folder
    test1_files = glob.glob(os.path.join(base_path, "test_set", "*.json"))
    
    all_files = train_files + test1_files + train2_files
    print(f"📂 Found {len(train_files)} Training files + {len(test1_files)} Test files.")
    print(f"🔥 TOTAL DATASET: {len(all_files)} samples.")
    
    if not all_files: return [], []

    logs = []
    labels = []

    for fpath in tqdm(all_files, desc="Loading Full Corpus"):
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                item = json.load(f)
            
            gt_value = item.get('ground_truth')
            if gt_value is None: gt_value = item.get('label', False)
            
            turns = []
            for t in item.get('turns', []):
                if isinstance(t, dict):
                    if 'turn_id' not in t: t['turn_id'] = f"t_{len(turns)}"
                    turns.append(DebateTurn(**t))
                else:
                    turns.append(t)

            log = DebateLog(
                debate_id=str(item.get('debate_id', '0')),
                claim_id="claim_0",
                claim_text=item.get('claim_text', ''),
                ground_truth=gt_value,
                turns=turns
            )
            logs.append(log)
            labels.append(1.0 if gt_value else 0.0)
        except: continue
            
    return logs, labels

def train_full():
    print("🚀 STARTING 'ALL-IN' TRAINING (100% DATA)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load EVERYTHING
    logs, labels = load_all_data()
    if not logs: return

    # 2. Build Graphs
    print("🕸️  Building Graphs...")
    builder = GraphBuilder(device=str(device))
    graph_list = []
    
    for log, label in zip(tqdm(logs, desc="Graphing"), labels):
        try:
            g = builder.build_graph(log)
            g.y = torch.tensor([label], dtype=torch.float)
            if g['argument'].num_nodes > 0:
                graph_list.append(g)
        except: continue

    # 3. No Split - Use Full Loader
    # Batch size 32 is stable
    full_loader = DataLoader(graph_list, batch_size=32, shuffle=True)
    
    # 4. Initialize Model (Pro Config)
    sample = graph_list[0]
    metadata = (list(sample.x_dict.keys()), list(sample.edge_index_dict.keys()))
    input_dim = sample.x_dict[list(sample.x_dict.keys())[0]].shape[1]

    model = HGTModel(
        in_channels=input_dim, 
        hidden_channels=128, 
        out_channels=1,
        num_heads=8,
        num_layers=3, 
        metadata=metadata
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    # We train for exactly 40 epochs. Since we have no validation set, 
    # we trust that 40 is the sweet spot based on previous runs.
    epochs = 40 
    save_path = os.path.join(project_root, "data", "processed", "hgt_judge.pth")

    print(f"\nTraining on {len(graph_list)} graphs for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(full_loader, desc=f"Ep {epoch+1}/{epochs}", leave=True)
        
        for batch in loop:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Light Noise
            for key in batch.x_dict:
                batch.x_dict[key] += torch.randn_like(batch.x_dict[key]) * 0.02

            batch_dict = {k: batch[k].batch for k in batch.x_dict.keys()}
            out = model(batch.x_dict, batch.edge_index_dict, batch_dict=batch_dict)
            
            loss = criterion(out.view(-1), batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        scheduler.step()

    # Save ONLY at the very end
    torch.save(model.state_dict(), save_path)
    print(f"✅ 'ALL-IN' Training Complete. Weights saved to: {save_path}")

if __name__ == "__main__":
    train_full()
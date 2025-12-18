import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear, global_mean_pool
from fire_debate.insight.graph_builder import GraphBuilder
from fire_debate.schemas.debate import DebateLog

class HGTModel(nn.Module):
    def __init__(self, hidden_channels=64, out_channels=1, num_heads=2, num_layers=2, metadata=None, in_channels=None):
        """
        TED-Style Interactive Attention GNN.
        Args:
            in_channels: Input feature dimension (Fixed to 771 by GraphBuilder).
        """
        super().__init__()
        
        # 1. Input Projections
        input_dim = in_channels if in_channels is not None else -1

        self.lin_dict = torch.nn.ModuleDict()
        if metadata:
            # Create a projection layer for EVERY node type defined in metadata
            # (argument, evidence, etc.)
            for node_type in metadata[0]:
                self.lin_dict[node_type] = Linear(input_dim, hidden_channels)

        # 2. HGT Layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)

        # 3. TED-Style "Interactive Attention"
        # We extract the Claim vector from the graph to perform cross-attention
        self.news_proj = nn.Linear(384, hidden_channels)
        self.debate_proj = nn.Linear(hidden_channels, hidden_channels)
        
        # Multi-Head Attention
        self.interaction_attn = nn.MultiheadAttention(embed_dim=hidden_channels, num_heads=2, batch_first=True)

        # 4. Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(64, 1) 
        )

    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        """
        Forward pass with correct Batch Handling.
        Safe for missing node types (e.g., if a debate has zero evidence).
        """
        # --- A. PRE-PROCESSING ---
        # Safety Check: We minimally need arguments to judge a debate
        if 'argument' not in x_dict:
             device = list(self.parameters())[0].device
             return torch.tensor([0.0], device=device, requires_grad=True)

        # 1. Handle Batch Indices
        # If batch_dict is None (inference mode), create zeros
        if batch_dict is None or 'argument' not in batch_dict:
            device = x_dict['argument'].device
            batch_indices = torch.zeros(x_dict['argument'].size(0), dtype=torch.long, device=device)
        else:
            batch_indices = batch_dict['argument']

        raw_x = x_dict['argument']
        
        # 2. Extract Claim (e_F) per graph in batch
        # We use global_mean_pool on the relevant slice to get [Batch_Size, 384]
        # GraphBuilder packs Claim at indices [384:768]
        if raw_x.shape[1] >= 768:
            claim_part = raw_x[:, 384:768]
            claim_raw = global_mean_pool(claim_part, batch_indices) # [Batch, 384]
        else:
            device = raw_x.device
            # Handle edge case of missing features
            num_graphs = batch_indices.max().item() + 1
            claim_raw = torch.zeros((num_graphs, 384), device=device)

        # --- B. GRAPH REASONING (HGT) ---
        # 1. Linear Projection (Input -> Hidden)
        # We iterate over whatever keys exist in x_dict.
        # If 'evidence' is missing in this specific batch, we just skip it safely.
        for node_type, x in x_dict.items():
            if node_type in self.lin_dict:
                x_dict[node_type] = self.lin_dict[node_type](x).relu()

        # 2. Convolution (Message Passing)
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        # --- C. POOLING (Get 'g') ---
        # We pool the ARGUMENT nodes to get the debate representation.
        # Evidence nodes have already passed their info to arguments via convolution.
        graph_embedding = global_mean_pool(x_dict['argument'], batch_indices)

        # --- D. INTERACTIVE ATTENTION (TED Method) ---
        # Project to hidden
        e_F = self.news_proj(claim_raw).unsqueeze(1) # [Batch, 1, Hidden]
        g = self.debate_proj(graph_embedding).unsqueeze(1) # [Batch, 1, Hidden]

        # Cross-Attention
        attn_out, _ = self.interaction_attn(query=e_F, key=g, value=g) # [Batch, 1, Hidden]

        # Remove sequence dim
        attn_out = attn_out.squeeze(1) # [Batch, Hidden]
        g = g.squeeze(1) # [Batch, Hidden]

        # Concatenate
        final_vec = torch.cat([g, attn_out], dim=1) # [Batch, Hidden*2]

        return self.classifier(final_vec)

class HGTJudge:
    def __init__(self, model_path=None, device="cuda"):
        print("⚖️  Initializing HGT GNN Judge (Evidence-Aware Mode)...")
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        
        self.builder = GraphBuilder(device=self.device)
        
        # --- METADATA DEFINITION ---
        # Defines the Schema of our Heterogeneous Graph
        self.metadata = (
            ['argument', 'evidence'], # Node Types
            [
                ('argument', 'follows', 'argument'),
                ('argument', 'supports', 'evidence'),
                ('argument', 'contradicts', 'evidence')
            ]
        )
        
        # Initialize Model
        # We force in_channels=771 because GraphBuilder guarantees padding
        self.model = HGTModel(
            hidden_channels=64, 
            out_channels=1, 
            num_heads=2, 
            num_layers=2, 
            metadata=self.metadata,
            in_channels=771 
        ).to(self.device)
        
        # Load Weights (Safely)
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"   ✅ Loaded trained weights from {model_path}")
            except Exception as e:
                print(f"   ⚠️ Architecture mismatch (Expected for Upgrade): {e}")
                print("   ⚠️ Running in UNTRAINED mode. Please run train_judge.py")
        else:
            print("   ⚠️ Running in UNTRAINED mode.")
        
        self.model.eval()

    def judge(self, log: DebateLog) -> dict:
        # 1. Build Heterogeneous Graph
        data = self.builder.build_graph(log)
        data = data.to(self.device)
        
        # 2. Run Inference
        with torch.no_grad():
            # Pass None for batch_dict in single-inference mode
            logit = self.model(data.x_dict, data.edge_index_dict, batch_dict=None)
            prob = torch.sigmoid(logit).item()

        verdict = prob > 0.5
        
        # Count stats for reason string
        n_args = data['argument'].num_nodes
        n_ev = data['evidence'].num_nodes if 'evidence' in data.node_types else 0
        
        return {
            "verdict": verdict,
            "confidence": prob if verdict else (1.0 - prob),
            "score": prob,
            "reason": f"Analyzed {n_args} arguments & {n_ev} evidence nodes."
        }
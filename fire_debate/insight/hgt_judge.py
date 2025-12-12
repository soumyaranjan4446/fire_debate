import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear

class HGTJudge(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super().__init__()
        
        # 1. Dynamic Input Projection
        # -1 allows it to accept ANY input size (e.g., your 386-dim vector)
        self.lin_dict = torch.nn.ModuleDict()
        node_types = metadata[0]
        
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        # 2. HGT Layers (Reasoning)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)

        # 3. Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(hidden_channels, 1) 
        )

    def forward(self, x_dict, edge_index_dict):
        # 1. Project to 64-dim
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu()

        # 2. Message Passing
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        # 3. Pooling (Focus on Arguments)
        if 'argument' in x_dict:
            node_embeddings = x_dict['argument']
        else:
            node_embeddings = list(x_dict.values())[0]
            
        graph_embedding = torch.mean(node_embeddings, dim=0, keepdim=True)

        # 4. Return Raw Logits (CRITICAL FIX)
        # We do NOT use sigmoid here because BCEWithLogitsLoss does it for us.
        return self.classifier(graph_embedding)
import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear

class HGTJudge(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super().__init__()
        
        # 1. Dynamic Input Projection
        # We use -1 (lazy initialization) so it auto-detects the input size (386)
        # on the first forward pass. This prevents mismatch errors.
        self.lin_dict = torch.nn.ModuleDict()
        node_types = metadata[0]
        
        for node_type in node_types:
            # Maps 386 -> hidden_channels (e.g., 64)
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        # 2. HGT Layers (The "Reasoning" Core)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # FIX: Removed group='sum' which causes crashes in newer PyG versions
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)

        # 3. Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2), # Added dropout to prevent overfitting
            nn.Linear(hidden_channels, 1) # Binary classification (True/False)
        )

    def forward(self, x_dict, edge_index_dict):
        # 1. Project features to same dimension (386 -> 64)
        # This handles the Neuro-Symbolic vectors we built
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu()

        # 2. Message Passing (HGT Layers)
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        # 3. Pooling (Readout)
        # We focus on "argument" nodes to determine the verdict.
        # If we added "evidence" nodes later, we would ignore them here 
        # and only look at the arguments they support.
        if 'argument' in x_dict:
            node_embeddings = x_dict['argument']
        else:
            # Fallback if graph structure changes (uses first available type)
            node_embeddings = list(x_dict.values())[0]
            
        # Global Mean Pooling: Summarize the whole debate into one vector
        graph_embedding = torch.mean(node_embeddings, dim=0, keepdim=True)

        # 4. Classification
        return torch.sigmoid(self.classifier(graph_embedding))
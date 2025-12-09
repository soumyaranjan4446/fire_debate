import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear

class HGTJudge(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super().__init__()
        
        # Linear projection to align feature dimensions of different node types
        # Argument features = 384 (MiniLM) + 6 (Fallacy) = 390
        self.lin_dict = torch.nn.ModuleDict()
        self.lin_dict['argument'] = Linear(-1, hidden_channels)
        self.lin_dict['evidence'] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads, group='sum')
            self.convs.append(conv)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1) # Binary classification (True/False)
        )

    def forward(self, x_dict, edge_index_dict):
        # 1. Project features to same dimension
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu()

        # 2. Message Passing (HGT Layers)
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        # 3. Pooling (Readout)
        # We perform Global Mean Pooling over all arguments to judge the whole debate
        # Alternatively: Pool only the "PRO" arguments
        argument_embeddings = x_dict['argument']
        graph_embedding = torch.mean(argument_embeddings, dim=0, keepdim=True)

        # 4. Classification
        return torch.sigmoid(self.classifier(graph_embedding))
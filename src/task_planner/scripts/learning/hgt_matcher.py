#!/usr/bin/env python3
"""HGT encoder and Ship--Task link scorer."""
from pyg_adapter import require_pyg


def build_model(metadata, hidden_channels=64, heads=2, layers=2):
    torch, _ = require_pyg()
    from torch import nn
    from torch_geometric.nn import HGTConv, Linear

    class HGTMatcher(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.ModuleDict({'ship': Linear(-1, hidden_channels), 'task': Linear(-1, hidden_channels), 'road': Linear(-1, hidden_channels)})
            self.layers = nn.ModuleList([HGTConv(hidden_channels, hidden_channels, metadata, heads=heads) for _ in range(layers)])
            self.edge_mlp = nn.Sequential(nn.LazyLinear(hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
            self.score = nn.Sequential(nn.Linear(hidden_channels * 3, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, 1))

        def forward(self, data, pair_index, pair_features):
            x = {k: self.input[k](v) for k, v in data.x_dict.items()}
            for conv in self.layers: x = conv(x, data.edge_index_dict)
            s, t = pair_index[:, 0], pair_index[:, 1]
            z = torch.cat([x['ship'][s], x['task'][t], self.edge_mlp(pair_features)], dim=-1)
            return self.score(z).squeeze(-1)
    return HGTMatcher()

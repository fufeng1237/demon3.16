#!/usr/bin/env python3
"""Convert the project HeteroGraph into PyG HeteroData."""
import numpy as np


def require_pyg():
    try:
        import torch
        from torch_geometric.data import HeteroData
        return torch, HeteroData
    except ImportError as exc:
        raise RuntimeError('Learning modules require torch and torch-geometric. '
                           'Install requirements-learning.txt first.') from exc


def to_heterodata(graph):
    torch, HeteroData = require_pyg()
    data = HeteroData()
    data['ship'].x = torch.tensor(graph.ship_x, dtype=torch.float)
    data['task'].x = torch.tensor(graph.task_x, dtype=torch.float)
    data['road'].x = torch.tensor(graph.road_x, dtype=torch.float)
    for key, src, rel, dst, edge_index, edge_attr in [
        ('rr', 'road', 'connects', 'road', graph.rr_edges, graph.rr_feat),
        ('sr', 'ship', 'at', 'road', graph.sr_edges, graph.sr_feat),
        ('tr', 'task', 'uses', 'road', graph.tr_edges, graph.tr_feat),
        ('st', 'ship', 'can_serve', 'task', graph.st_edges, graph.st_feat),
        ('tt', 'task', 'related', 'task', graph.tt_edges, graph.tt_feat),
    ]:
        store = data[(src, rel, dst)]
        store.edge_index = torch.tensor(edge_index, dtype=torch.long)
        store.edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        if src != dst:
            reverse = data[(dst, 'rev_' + rel, src)]
            reverse.edge_index = torch.tensor(edge_index[::-1].copy(), dtype=torch.long)
            reverse.edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return data


def ship_task_pairs(graph):
    """Return feasible (ship_index, task_index) pairs and their edge features."""
    return graph.st_edges.T.copy(), graph.st_feat.copy()

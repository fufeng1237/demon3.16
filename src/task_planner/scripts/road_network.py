#!/usr/bin/env python3
"""RoadNetwork data structures shared across task_planner modules."""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum


class NodeType(Enum):
    NODE = "node"
    PORT = "port"
    SHIP = "ship"
    GAS_STATION = "gas_station"


@dataclass
class RoadNode:
    id: int
    x: float          # world X (meters)
    y: float          # world Y (meters)
    node_type: NodeType = NodeType.NODE
    port_name: str = ""
    degree: int = 0

    @property
    def is_port(self):
        return self.node_type == NodeType.PORT

    @property
    def is_gas_station(self):
        return self.node_type == NodeType.GAS_STATION

    def to_dict(self):
        return {
            "id": int(self.id), "x": float(self.x), "y": float(self.y),
            "is_port": self.is_port, "is_gas_station": self.is_gas_station,
            "port_name": str(self.port_name), "degree": int(self.degree)
        }


@dataclass
class RoadEdge:
    from_id: int
    to_id: int
    weight: float

    def to_dict(self):
        return {"from": int(self.from_id), "to": int(self.to_id), "weight": float(self.weight)}


class RoadNetwork:
    def __init__(self):
        self.nodes: Dict[int, RoadNode] = {}
        self.edges: List[RoadEdge] = []
        self.adj: Dict[int, List[int]] = defaultdict(list)
        self.dist_matrix: np.ndarray = None

    def add_node(self, nid, x, y, ntype=NodeType.NODE, name=""):
        self.nodes[nid] = RoadNode(id=nid, x=x, y=y, node_type=ntype, port_name=name)

    def add_edge(self, u, v, w):
        if u == v:
            return
        for e in self.edges:
            if (e.from_id == u and e.to_id == v) or (e.from_id == v and e.to_id == u):
                if w < e.weight:
                    e.weight = w
                return
        self.edges.append(RoadEdge(u, v, w))
        self.adj[u].append(v)
        self.adj[v].append(u)
        self.nodes[u].degree += 1
        self.nodes[v].degree += 1

    def to_dict(self):
        dm = None
        if self.dist_matrix is not None:
            dm = [[float(v) if np.isfinite(v) else -1.0 for v in row]
                  for row in self.dist_matrix]
        return {
            "n_nodes": len(self.nodes), "n_edges": len(self.edges),
            "nodes": [n.to_dict() for n in sorted(self.nodes.values(), key=lambda x: x.id)],
            "edges": [e.to_dict() for e in self.edges],
            "distance_matrix": dm
        }

    @classmethod
    def from_dict(cls, data):
        net = cls()
        for nd in data["nodes"]:
            nt = NodeType.NODE
            if nd.get("is_port"): nt = NodeType.PORT
            elif nd.get("is_gas_station"): nt = NodeType.GAS_STATION
            net.add_node(nd["id"], nd["x"], nd["y"], nt, nd.get("port_name", ""))
            net.nodes[nd["id"]].degree = nd.get("degree", 0)
        for ed in data["edges"]:
            net.add_edge(ed["from"], ed["to"], ed.get("weight", ed.get("distance", 0)))
        if data.get("distance_matrix"):
            dm = np.array(data["distance_matrix"])
            dm[dm < 0] = np.inf
            net.dist_matrix = dm
        return net


def load_road_network(json_path: str, ports_config: str = None) -> RoadNetwork:
    """Load road network from C++ JSON output, converting pixel coords to world coords,
    and computing distance matrix if not present.

    Args:
        json_path: path to C++ JSON output
        ports_config: optional path to ports.txt (C++ format) or ports.yaml for naming
    """
    import json, yaml, os
    with open(json_path) as f:
        data = json.load(f)

    # Load port names from config if available
    port_names = {}
    gas_names = {}
    if ports_config and os.path.exists(ports_config):
        ext = os.path.splitext(ports_config)[1]
        if ext == '.yaml' or ext == '.yml':
            with open(ports_config) as f:
                cfg = yaml.safe_load(f)
            for p in cfg.get("ports", []):
                port_names[p["id"]] = p["name"]
            for g in cfg.get("gas_stations", []):
                gas_names[g["id"]] = g["name"]
        elif ext == '.txt':
            import re
            with open(ports_config) as f:
                for line in f:
                    m = re.match(r"Port\s+(\d+):", line)
                    if m:
                        pid = int(m.group(1))
                        port_names[pid] = f"Port_{chr(65+pid)}"  # Port_0→A, Port_1→B, ...

    # C++ outputs pixel coords with scale_factor=1.0
    PS = 2.0
    # Count port/gas nodes to assign names
    port_idx = 0
    gas_idx = 0
    for n in data["nodes"]:
        n["x"] = n["x"] * PS
        n["y"] = n["y"] * PS
        # Assign readable names
        if n.get("is_port"):
            if port_idx < len(port_names):
                n["port_name"] = list(port_names.values())[port_idx]
            else:
                n["port_name"] = f"Port_{chr(65+port_idx)}" if port_idx < 26 else f"Port_{port_idx}"
            port_idx += 1
        if n.get("is_gas_station"):
            if gas_idx < len(gas_names):
                n["port_name"] = list(gas_names.values())[gas_idx]
            else:
                n["port_name"] = f"Gas_{chr(65+gas_idx)}" if gas_idx < 26 else f"Gas_{gas_idx}"
            gas_idx += 1

    net = RoadNetwork.from_dict(data)

    # Reindex node IDs to be contiguous (C++ pruning leaves gaps)
    old_ids = sorted(net.nodes.keys())
    id_map = {old: new for new, old in enumerate(old_ids)}
    new_net = RoadNetwork()
    for old_id in old_ids:
        n = net.nodes[old_id]
        new_net.add_node(id_map[old_id], n.x, n.y, n.node_type, n.port_name)
    seen = set()
    for e in net.edges:
        nu, nv = id_map[e.from_id], id_map[e.to_id]
        if nu == nv: continue
        key = (min(nu, nv), max(nu, nv))
        if key not in seen:
            seen.add(key)
            new_net.add_edge(nu, nv, e.weight)
    net = new_net

    # Compute distance matrix if not present
    if net.dist_matrix is None:
        n = len(net.nodes)
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0.0)
        for e in net.edges:
            dist[e.from_id, e.to_id] = min(dist[e.from_id, e.to_id], e.weight)
            dist[e.to_id, e.from_id] = min(dist[e.to_id, e.from_id], e.weight)
        # Floyd-Warshall
        for k in range(n):
            dk = dist[k]
            for i in range(n):
                if dist[i, k] == np.inf: continue
                dist[i] = np.minimum(dist[i], dist[i, k] + dk)
        net.dist_matrix = dist
        print(f"  计算距离矩阵: {n}x{n}, "
              f"可达={np.sum((dist>0)&(dist<np.inf))}/{n**2}")

    return net

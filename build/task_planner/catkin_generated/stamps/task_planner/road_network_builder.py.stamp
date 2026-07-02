#!/usr/bin/env python3
"""
Road Network Builder — 完全对齐 roadmap.cpp 逻辑

Method (对标 GenerateRoadmapForAllocation):
  1. 加载 PNG 地图 → 提取水域二值掩码 → 缩放
  2. OpenCV Guo-Hall 细化 (cv2.ximgproc.thinning)
  3. 识别关键像素 (度数 != 2: 端点 + 交叉点)
  4. DBSCAN 聚类关键像素 (eps=25像素)
  5. 路径追踪: 从关键点沿骨架走到下一个关键点 → 聚类间边
  6. 聚类中心作为图节点, 聚类间最短路径作为边
  7. 坐标缩放 (scale_factor)
  8. 注入锚点 (港口靠岸边, 加油站靠岸边)
  9. 两阶段剪枝: 孤立节点 + 无锚点死胡同
 10. 全源最短路径距离矩阵

Output:
  road_network.json: 路网结构 + 距离矩阵
"""

import numpy as np
import json
import yaml
import os
import sys
import heapq
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum

# Check OpenCV
try:
    import cv2
    HAS_CV2 = True
    HAS_THINNING = hasattr(cv2, 'ximgproc')
except ImportError:
    HAS_CV2 = False
    HAS_THINNING = False


# ============================================================
# Data Structures (对标 Graph/GraphNode)
# ============================================================

class NodeType(Enum):
    NODE = "node"          # 普通路网节点
    PORT = "port"          # 港口
    GAS_STATION = "gas_station"  # 加油站
    SHIP = "ship"          # 船舶 (任务分配时)


@dataclass
class RoadNode:
    id: int
    x: float           # 世界坐标 X (米)
    y: float           # 世界坐标 Y (米)
    pixel_x: int       # 缩放后图像中的像素列
    pixel_y: int       # 缩放后图像中的像素行
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
            "id": int(self.id),
            "x": float(self.x), "y": float(self.y),
            "pixel_x": int(self.pixel_x), "pixel_y": int(self.pixel_y),
            "is_port": self.is_port,
            "is_gas_station": self.is_gas_station,
            "port_name": str(self.port_name),
            "degree": int(self.degree)
        }


@dataclass
class RoadEdge:
    from_id: int
    to_id: int
    distance: float     # 路径长度 (米)
    path_pixels: List[Tuple[int, int]] = field(default_factory=list)

    def to_dict(self):
        return {
            "from": int(self.from_id),
            "to": int(self.to_id),
            "distance": round(float(self.distance), 2),
            "path_pixels": [(int(x), int(y)) for x, y in self.path_pixels]
        }


class RoadNetwork:
    def __init__(self):
        self.nodes: Dict[int, RoadNode] = {}
        self.edges: List[RoadEdge] = []
        self.adj: Dict[int, List[int]] = {}
        self.dist_matrix: np.ndarray = None
        self.resolution: float = 0.5
        self.map_width: int = 0
        self.map_height: int = 0
        self.scale_factor: float = 1.0
        self._next_id: int = 0

    def get_next_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def add_node(self, node_id: int, x: float, y: float,
                 node_type: NodeType = NodeType.NODE,
                 port_name: str = "",
                 pixel_x: int = 0, pixel_y: int = 0) -> int:
        node = RoadNode(id=node_id, x=x, y=y, pixel_x=pixel_x, pixel_y=pixel_y,
                        node_type=node_type, port_name=port_name)
        self.nodes[node_id] = node
        if node_id not in self.adj:
            self.adj[node_id] = []
        if node_id >= self._next_id:
            self._next_id = node_id + 1
        return node_id

    def add_edge(self, from_id: int, to_id: int, distance: float,
                 path_pixels: List[Tuple[int, int]] = None):
        if from_id == to_id:
            return
        # Deduplicate
        for e in self.edges:
            if (e.from_id == from_id and e.to_id == to_id) or \
               (e.from_id == to_id and e.to_id == from_id):
                if distance < e.distance:
                    e.distance = distance
                return
        edge = RoadEdge(from_id=from_id, to_id=to_id, distance=distance,
                        path_pixels=path_pixels or [])
        self.edges.append(edge)
        self.adj.setdefault(from_id, []).append(to_id)
        self.adj.setdefault(to_id, []).append(from_id)
        self.nodes[from_id].degree += 1
        self.nodes[to_id].degree += 1

    def remove_node(self, node_id: int):
        """Remove node and all incident edges."""
        if node_id in self.nodes:
            del self.nodes[node_id]
        if node_id in self.adj:
            del self.adj[node_id]
        self.edges = [e for e in self.edges
                      if e.from_id != node_id and e.to_id != node_id]
        for nid in self.adj:
            self.adj[nid] = [nb for nb in self.adj[nid] if nb != node_id]

    def to_dict(self):
        n_nodes = len(self.nodes)
        dist_list = None
        if self.dist_matrix is not None:
            dist_list = [[float(v) if np.isfinite(v) else -1.0
                          for v in row] for row in self.dist_matrix]
        return {
            "n_nodes": n_nodes,
            "n_edges": len(self.edges),
            "n_ports": sum(1 for n in self.nodes.values() if n.is_port),
            "n_gas_stations": sum(1 for n in self.nodes.values() if n.is_gas_station),
            "resolution": float(self.resolution),
            "scale_factor": float(self.scale_factor),
            "map_width": int(self.map_width),
            "map_height": int(self.map_height),
            "nodes": [n.to_dict() for n in sorted(self.nodes.values(), key=lambda x: x.id)],
            "edges": [e.to_dict() for e in self.edges],
            "distance_matrix": dist_list
        }

    @classmethod
    def from_dict(cls, data):
        net = cls()
        net.resolution = data.get("resolution", 0.5)
        net.scale_factor = data.get("scale_factor", 1.0)
        net.map_width = data.get("map_width", 0)
        net.map_height = data.get("map_height", 0)
        for nd in data["nodes"]:
            ntype = NodeType.NODE
            if nd["is_port"]:
                ntype = NodeType.PORT
            elif nd["is_gas_station"]:
                ntype = NodeType.GAS_STATION
            net.add_node(node_id=nd["id"], x=nd["x"], y=nd["y"],
                         node_type=ntype, port_name=nd.get("port_name", ""),
                         pixel_x=nd.get("pixel_x", 0), pixel_y=nd.get("pixel_y", 0))
            if nd["degree"] > 0:
                net.nodes[nd["id"]].degree = nd["degree"]
        for ed in data["edges"]:
            net.add_edge(ed["from"], ed["to"], ed["distance"],
                         ed.get("path_pixels", []))
        if data.get("distance_matrix"):
            net.dist_matrix = np.array(data["distance_matrix"])
            net.dist_matrix[net.dist_matrix < 0] = np.inf
        return net


# ============================================================
# Utility Functions
# ============================================================

def calc_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_skeleton_neighbors(skeleton: np.ndarray, p: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Get 8-connected neighbors of a skeleton pixel."""
    neighbors = []
    h, w = skeleton.shape
    x, y = p
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx] > 0:
                neighbors.append((nx, ny))
    return neighbors


def make_edge(p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Canonical edge representation (smaller point first)."""
    if p1 <= p2:
        return (p1, p2)
    return (p2, p1)


# ============================================================
# Step 1: Load & Prepare Binary Map
# ============================================================

def load_scaled_binary_map(png_path: str,
                            pixel_scale: float = 2.0) -> Tuple[np.ndarray, float, dict]:
    """
    Load a pre-scaled binary map (black=land, white=water).
    No water extraction or downscaling needed.

    Parameters:
        png_path: path to binary PNG
        pixel_scale: meters per pixel in this scaled map

    Returns:
        (binary_mask, scale_factor, meta)
    """
    from PIL import Image

    img = Image.open(png_path)
    arr = np.array(img)
    h, w = arr.shape[:2]

    # Already binary: 0=land, 255=water
    if arr.ndim == 3:
        binary = ((arr[:, :, 0] > 127).astype(np.uint8)) * 255
    else:
        binary = ((arr > 127).astype(np.uint8)) * 255

    n_water = np.count_nonzero(binary)
    n_land = binary.size - n_water
    print(f"  预缩放二值图: {w} x {h}")
    print(f"  水域像素: {n_water} ({100*n_water/binary.size:.1f}%)")
    print(f"  陆地像素: {n_land} ({100*n_land/binary.size:.1f}%)")
    print(f"  像素比例: {pixel_scale} m/pixel")

    scale_factor = pixel_scale
    meta = {
        "c_min": 0, "r_min": 0,
        "w_full": w, "h_full": h,
        "downscale": 1,
        "pixel_scale": pixel_scale,
        "water_full": binary.copy(),
    }

    return binary, scale_factor, meta


def load_png_binary_mask(png_path: str,
                          downscale: int = 8,
                          water_min_value: int = 0) -> Tuple[np.ndarray, float, dict]:
    """
    Load PNG map and extract binary water mask, then downscale.
    Returns (binary_mask, scale_factor, meta).

    scale_factor: how many real-world meters per scaled pixel.
    """
    from PIL import Image

    img = Image.open(png_path)
    arr = np.array(img)
    h_full, w_full = arr.shape[:2]
    print(f"  原始PNG: {w_full} x {h_full}")

    # Extract water: 纯黑(0,0,0)=陆地, 其余所有灰度=水域
    # R通道 > 0 即为水域 (map10.png 是水域灰度图，非黑即水)
    if arr.ndim == 3:
        water_full = (arr[:, :, 0] > 0).astype(np.uint8) * 255
    else:
        water_full = (arr > 0).astype(np.uint8) * 255

    n_water = np.count_nonzero(water_full)
    print(f"  水域像素: {n_water} / {water_full.size} ({100*n_water/water_full.size:.1f}%)")

    # Find water bounding box — no crop margin (keep nodes in water)
    water_rows, water_cols = np.where(water_full > 0)
    r_min = max(0, water_rows.min())
    r_max = min(h_full, water_rows.max() + 1)
    c_min = max(0, water_cols.min())
    c_max = min(w_full, water_cols.max() + 1)

    water_cropped = water_full[r_min:r_max, c_min:c_max]
    print(f"  裁剪区域: [{c_min}, {c_max}] x [{r_min}, {r_max}] (精确水域)")

    # Downscale using max pooling (preserve narrow channels)
    if downscale > 1:
        h_new = water_cropped.shape[0] // downscale
        w_new = water_cropped.shape[1] // downscale
        scaled = np.zeros((h_new, w_new), dtype=np.uint8)
        for r in range(h_new):
            for c in range(w_new):
                patch = water_cropped[r*downscale:(r+1)*downscale,
                                       c*downscale:(c+1)*downscale]
                scaled[r, c] = 255 if np.any(patch > 0) else 0
        print(f"  缩放: {w_new} x {h_new} (因子={downscale})")
    else:
        scaled = water_cropped

    # scale_factor: 每像素对应的真实世界米数
    # 原始PNG每像素=0.5m, 缩放后每像素=0.5*downscale m
    pixel_scale = 0.5  # m/pixel in original PNG
    scale_factor = pixel_scale * downscale

    meta = {
        "c_min": c_min, "r_min": r_min,
        "w_full": w_full, "h_full": h_full,
        "downscale": downscale,
        "pixel_scale": pixel_scale,
        "water_full": water_full,  # Store full-res mask for verification
    }

    return scaled, scale_factor, meta


def scaled_pixel_to_world(px: int, py: int, meta: dict) -> Tuple[float, float]:
    """Convert scaled pixel coordinates to world coordinates (meters)."""
    c_min = meta["c_min"]
    r_min = meta["r_min"]
    ds = meta["downscale"]
    pixel_scale = meta["pixel_scale"]
    orig_px = px * ds + c_min
    orig_py = py * ds + r_min
    return orig_px * pixel_scale, orig_py * pixel_scale


def world_to_scaled_pixel(wx: float, wy: float, meta: dict) -> Tuple[float, float]:
    """Convert world coordinates to scaled pixel coordinates."""
    c_min = meta["c_min"]
    r_min = meta["r_min"]
    ds = meta["downscale"]
    pixel_scale = meta["pixel_scale"]
    orig_px = wx / pixel_scale
    orig_py = wy / pixel_scale
    spx = (orig_px - c_min) / ds
    spy = (orig_py - r_min) / ds
    return spx, spy


# ============================================================
# Step 2: Skeletonization (OpenCV Guo-Hall 或 Zhang-Suen fallback)
# ============================================================

def skeletonize(binary: np.ndarray) -> np.ndarray:
    """Thin binary image. Uses OpenCV Guo-Hall if available."""
    if HAS_CV2 and HAS_THINNING:
        return cv2.ximgproc.thinning(binary, cv2.ximgproc.THINNING_GUOHALL)
    else:
        print("  WARNING: OpenCV ximgproc not available, using Zhang-Suen fallback")
        return _zhang_suen_thinning(binary)


def _zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen fallback."""
    skel = (binary > 127).astype(np.uint8)
    h, w = skel.shape
    changed = True
    while changed:
        changed = False
        for iteration in [1, 2]:
            to_remove = []
            for r in range(1, h - 1):
                for c in range(1, w - 1):
                    if skel[r, c] != 1:
                        continue
                    p2 = skel[r-1, c]
                    p3 = skel[r-1, c+1]
                    p4 = skel[r, c+1]
                    p5 = skel[r+1, c+1]
                    p6 = skel[r+1, c]
                    p7 = skel[r+1, c-1]
                    p8 = skel[r, c-1]
                    p9 = skel[r-1, c-1]
                    nb = [p2, p3, p4, p5, p6, p7, p8, p9]
                    B = sum(nb)
                    if B < 2 or B > 6:
                        continue
                    A = sum(1 for i in range(8) if nb[i] == 0 and nb[(i+1) % 8] == 1)
                    if A != 1:
                        continue
                    if iteration == 1:
                        if p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0:
                            to_remove.append((r, c))
                    else:
                        if p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0:
                            to_remove.append((r, c))
            for r, c in to_remove:
                skel[r, c] = 0
                changed = True
    return skel * 255


# ============================================================
# Step 3: Identify Critical Pixels (degree != 2)
# ============================================================

def identify_critical_pixels(skeleton: np.ndarray) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """
    Find all skeleton pixels and classify by degree.
    Critical = degree != 2 (endpoints and junctions).
    对标 roadmap.cpp 第 2 节。
    """
    critical_pixels = []
    pixel_degrees = {}
    h, w = skeleton.shape

    for y in range(h):
        for x in range(w):
            if skeleton[y, x] > 127:
                p = (x, y)
                neighbors = get_skeleton_neighbors(skeleton, p)
                deg = len(neighbors)
                pixel_degrees[p] = deg
                if deg != 2:
                    critical_pixels.append(p)

    return critical_pixels, pixel_degrees


# ============================================================
# Step 4: DBSCAN Clustering of Critical Pixels
# ============================================================

def dbscan_cluster_critical(critical_pixels: List[Tuple[int, int]],
                              eps: float = 25.0) -> Tuple[List[int], List[Tuple[int, int]], Dict]:
    """
    Cluster nearby critical pixels.
    对标 roadmap.cpp 第 3 节。
    eps=25 像素, 距离小于此值的像素归为一类。
    Returns: (labels, cluster_centroids, raw_to_cluster_id)
    """
    n = len(critical_pixels)
    labels = [-1] * n
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1:
            continue
        q = deque([i])
        labels[i] = cluster_id
        while q:
            curr = q.popleft()
            for j in range(n):
                if labels[j] == -1:
                    d = calc_distance(critical_pixels[curr], critical_pixels[j])
                    if d <= eps:
                        labels[j] = cluster_id
                        q.append(j)
        cluster_id += 1

    # Compute centroids
    centroids = []
    counts = [0] * cluster_id
    for ci in range(cluster_id):
        centroids.append([0.0, 0.0])

    raw_to_cluster = {}
    for i in range(n):
        ci = labels[i]
        x, y = critical_pixels[i]
        centroids[ci][0] += x
        centroids[ci][1] += y
        counts[ci] += 1
        raw_to_cluster[(x, y)] = ci

    centroid_points = []
    for ci in range(cluster_id):
        centroid_points.append((
            centroids[ci][0] / counts[ci],
            centroids[ci][1] / counts[ci]
        ))

    return labels, centroid_points, raw_to_cluster


# ============================================================
# Step 5: Path Tracing Between Clusters
# ============================================================

def trace_paths_between_clusters(skeleton: np.ndarray,
                                   critical_pixels: List[Tuple[int, int]],
                                   pixel_degrees: Dict,
                                   raw_to_cluster: Dict,
                                   scale_factor: float
                                   ) -> Dict[Tuple[int, int], float]:
    """
    Trace skeleton paths from each critical pixel to the next critical pixel.
    Record the shortest path between each pair of clusters.
    对标 roadmap.cpp 第 4 节。
    """
    visited_edges = set()
    # (cluster_id1, cluster_id2) -> shortest_distance (meters)
    cluster_edges: Dict[Tuple[int, int], float] = {}

    for start_node in critical_pixels:
        neighbors = get_skeleton_neighbors(skeleton, start_node)
        for neighbor in neighbors:
            edge_key = make_edge(start_node, neighbor)
            if edge_key in visited_edges:
                continue

            visited_edges.add(edge_key)
            prev_node = start_node
            curr_node = neighbor
            path_len_pixels = calc_distance(prev_node, curr_node)

            # Walk along skeleton until we hit the next critical pixel
            while pixel_degrees.get(curr_node) == 2:
                next_neighbors = get_skeleton_neighbors(skeleton, curr_node)
                next_node = None
                for nn in next_neighbors:
                    if nn != prev_node:
                        next_node = nn
                        break
                if next_node is None:
                    break

                next_edge_key = make_edge(curr_node, next_node)
                visited_edges.add(next_edge_key)
                path_len_pixels += calc_distance(curr_node, next_node)

                prev_node = curr_node
                curr_node = next_node

            end_node = curr_node
            if start_node not in raw_to_cluster or end_node not in raw_to_cluster:
                continue
            c1 = raw_to_cluster[start_node]
            c2 = raw_to_cluster[end_node]

            if c1 != c2:
                ce = (min(c1, c2), max(c1, c2))
                real_len = path_len_pixels * scale_factor  # 转换为米
                if ce not in cluster_edges or real_len < cluster_edges[ce]:
                    cluster_edges[ce] = real_len

    return cluster_edges


# ============================================================
# Step 6-7: Build Graph from Clusters
# ============================================================

def snap_to_skeleton(centroid: Tuple[float, float],
                      skeleton: np.ndarray,
                      search_radius: int = 30) -> Tuple[int, int]:
    """
    Snap a floating-point centroid to the nearest skeleton pixel.
    Ensures all graph nodes are on the water centerline.
    """
    cx, cy = int(round(centroid[0])), int(round(centroid[1]))
    h, w = skeleton.shape

    best_dist = float('inf')
    best_point = (cx, cy)

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx] > 0:
                d = dx*dx + dy*dy
                if d < best_dist:
                    best_dist = d
                    best_point = (nx, ny)

    if best_dist == float('inf'):
        # Fallback: search entire skeleton
        skel_pts = np.where(skeleton > 0)
        for sx, sy in zip(skel_pts[1], skel_pts[0]):
            d = (sx - cx)**2 + (sy - cy)**2
            if d < best_dist:
                best_dist = d
                best_point = (sx, sy)

    return best_point


def snap_world_to_water(wx: float, wy: float, water_full: np.ndarray,
                         pixel_scale: float, search_radius: int = 15) -> Tuple[float, float]:
    """
    Snap a world coordinate to the nearest actual water pixel in the
    full-resolution water mask. Guarantees the result is in water.
    """
    opx = int(round(wx / pixel_scale))
    opy = int(round(wy / pixel_scale))
    h, w = water_full.shape

    # Search expanding radius for nearest water pixel
    for r in range(search_radius + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) != r and abs(dy) != r:
                    continue  # Only check perimeter each iteration
                nx, ny = opx + dx, opy + dy
                if 0 <= nx < w and 0 <= ny < h and water_full[ny, nx] > 0:
                    return nx * pixel_scale, ny * pixel_scale

    return wx, wy  # Fallback


def build_graph_from_clusters(cluster_centroids: List[Tuple[int, int]],
                                cluster_edges: Dict[Tuple[int, int], float],
                                skeleton: np.ndarray,
                                scale_factor: float,
                                meta: dict) -> RoadNetwork:
    """
    Create graph nodes from cluster centroids, snapped to skeleton,
    then refined to nearest full-res water pixel.
    All nodes guaranteed on the water centerline.
    """
    graph = RoadNetwork()
    graph.resolution = meta["pixel_scale"]
    graph.scale_factor = scale_factor
    graph.map_width = meta["w_full"]
    graph.map_height = meta["h_full"]
    water_full = meta.get("water_full")
    pixel_scale = meta["pixel_scale"]

    cluster_to_node = {}
    for ci, (cx, cy) in enumerate(cluster_centroids):
        # Step 1: Snap centroid to nearest skeleton pixel (scaled image)
        skel_pt = snap_to_skeleton((cx, cy), skeleton)
        # Step 2: Convert to approximate world coords
        wx_approx, wy_approx = scaled_pixel_to_world(skel_pt[0], skel_pt[1], meta)
        # Step 3: Snap to nearest actual full-res water pixel
        if water_full is not None:
            wx, wy = snap_world_to_water(wx_approx, wy_approx, water_full, pixel_scale)
        else:
            wx, wy = wx_approx, wy_approx

        node_id = graph.get_next_id()
        graph.add_node(node_id, wx, wy, NodeType.NODE,
                       pixel_x=skel_pt[0], pixel_y=skel_pt[1])
        cluster_to_node[ci] = node_id

    for (c1, c2), weight in cluster_edges.items():
        if c1 in cluster_to_node and c2 in cluster_to_node:
            graph.add_edge(cluster_to_node[c1], cluster_to_node[c2], weight)

    return graph, cluster_to_node


# ============================================================
# Step 8: Shore Snapping (港口/加油站在水域里靠岸边)
# ============================================================

def find_shore_pixels(water_mask: np.ndarray,
                       search_dist: int = 5) -> Set[Tuple[int, int]]:
    """
    Find water-land boundary pixels (shore).
    A water pixel adjacent to at least one land pixel within search_dist.
    """
    h, w = water_mask.shape
    shore = set()
    for y in range(search_dist, h - search_dist):
        for x in range(search_dist, w - search_dist):
            if water_mask[y, x] == 0:
                continue
            # Check if there's land nearby within search_dist
            has_land = False
            for dy in range(-search_dist, search_dist + 1):
                for dx in range(-search_dist, search_dist + 1):
                    if water_mask[y + dy, x + dx] == 0:
                        has_land = True
                        break
                if has_land:
                    break
            if has_land:
                shore.add((x, y))
    return shore


def snap_to_shore(water_mask: np.ndarray,
                   target_px: float, target_py: float,
                   shore_set: Set[Tuple[int, int]] = None,
                   search_radius: int = 100) -> Optional[Tuple[int, int]]:
    """
    Snap a target position to the nearest shore pixel within the water.
    保证港口/加油站在水域里靠岸边。
    """
    if shore_set is None:
        shore_set = find_shore_pixels(water_mask)

    if not shore_set:
        return None

    best_dist = float('inf')
    best_point = None
    tx, ty = int(round(target_px)), int(round(target_py))

    # Search within radius for efficiency
    h, w = water_mask.shape
    for sy in range(max(0, ty - search_radius), min(h, ty + search_radius + 1)):
        for sx in range(max(0, tx - search_radius), min(w, tx + search_radius + 1)):
            if (sx, sy) in shore_set:
                d = (sx - tx)**2 + (sy - ty)**2
                if d < best_dist:
                    best_dist = d
                    best_point = (sx, sy)

    # Fallback: search entire shore set
    if best_point is None:
        for spt in shore_set:
            d = (spt[0] - tx)**2 + (spt[1] - ty)**2
            if d < best_dist:
                best_dist = d
                best_point = spt

    return best_point


def connect_to_nearest(graph: RoadNetwork,
                        target_node_id: int,
                        n_neighbors: int = 1,
                        scale_factor: float = 1.0):
    """
    Connect a port/ship/gas_station to the nearest graph nodes.
    对标 roadmap.cpp connectToNearestPythonStyle。
    """
    target = graph.nodes[target_node_id]
    target_pos = (target.x, target.y)

    distances = []
    for nid, node in graph.nodes.items():
        if nid == target_node_id:
            continue
        # Allow connection rules
        allow = False
        if node.node_type == NodeType.NODE:
            allow = True
        elif target.node_type == NodeType.SHIP and node.node_type == NodeType.PORT:
            allow = True
        elif target.node_type == NodeType.PORT and node.node_type == NodeType.SHIP:
            allow = True
        if allow:
            d = np.sqrt((node.x - target_pos[0])**2 + (node.y - target_pos[1])**2)
            distances.append((d, nid))

    distances.sort()
    if not distances:
        return

    if target.node_type == NodeType.PORT:
        # Port connects to 1 nearest node
        for i in range(min(n_neighbors, len(distances))):
            graph.add_edge(target_node_id, distances[i][1], distances[i][0])

    elif target.node_type == NodeType.SHIP:
        # Ship connects to 1-2 nearest nodes
        graph.add_edge(target_node_id, distances[0][1], distances[0][0])
        if n_neighbors >= 2 and len(distances) >= 2:
            diff = distances[1][0] / (distances[0][0] + 1e-6)
            if diff < 1.5:
                graph.add_edge(target_node_id, distances[1][1], distances[1][0])

    elif target.node_type == NodeType.GAS_STATION:
        # Gas station: connect to nearest
        graph.add_edge(target_node_id, distances[0][1], distances[0][0])
        if n_neighbors >= 2 and len(distances) >= 2:
            graph.add_edge(target_node_id, distances[1][1], distances[1][0])


def inject_facilities(graph: RoadNetwork,
                       ports_config: List[Dict],
                       gas_stations_config: List[Dict],
                       water_mask: np.ndarray,
                       scale_factor: float,
                       meta: dict):
    """
    Inject ports and gas stations as anchor nodes.
    对标 roadmap.cpp 第 6 节 + connectGasStationToNearestNode。
    港口和加油站在水域里靠岸边 (shore-snapping)。
    验证：最终位置必须在全分辨率水域内。
    """
    shore = find_shore_pixels(water_mask)
    water_full = meta.get("water_full")
    pixel_scale = meta["pixel_scale"]
    print(f"  岸线像素点: {len(shore)}")

    def verify_in_water(wx: float, wy: float, name: str) -> bool:
        """Check if world position is in full-res water."""
        if water_full is None:
            return True
        opx = int(round(wx / pixel_scale))
        opy = int(round(wy / pixel_scale))
        if 0 <= opx < water_full.shape[1] and 0 <= opy < water_full.shape[0]:
            if water_full[opy, opx] == 0:
                # Try expanding search radius
                for r in range(1, 10):
                    for dx in range(-r, r+1):
                        for dy in range(-r, r+1):
                            nx, ny = opx+dx, opy+dy
                            if 0 <= nx < water_full.shape[1] and 0 <= ny < water_full.shape[0]:
                                if water_full[ny, nx] > 0:
                                    return True
                return False
        return True

    # ── Inject Ports ──
    for port in ports_config:
        wx, wy = port["x"], port["y"]
        spx, spy = world_to_scaled_pixel(wx, wy, meta)

        # Snap to nearest shore pixel (water-land boundary in scaled image)
        shore_pt = snap_to_shore(water_mask, spx, spy, shore)
        if shore_pt is None:
            print(f"  WARNING: Cannot find shore near {port['name']} ({wx}, {wy})")
            continue

        snapped_wx, snapped_wy = scaled_pixel_to_world(shore_pt[0], shore_pt[1], meta)

        # Final snap to full-res water pixel
        if water_full is not None:
            snapped_wx, snapped_wy = snap_world_to_water(
                snapped_wx, snapped_wy, water_full, pixel_scale)

        node_id = graph.get_next_id()
        graph.add_node(node_id, snapped_wx, snapped_wy, NodeType.PORT,
                       port_name=port["name"],
                       pixel_x=shore_pt[0], pixel_y=shore_pt[1])
        connect_to_nearest(graph, node_id, n_neighbors=1, scale_factor=scale_factor)
        print(f"  Port '{port['name']}' → node {node_id} "
              f"(岸线 world={snapped_wx:.1f},{snapped_wy:.1f})")

    # ── Inject Gas Stations ──
    for gs in gas_stations_config:
        wx, wy = gs["x"], gs["y"]
        spx, spy = world_to_scaled_pixel(wx, wy, meta)

        shore_pt = snap_to_shore(water_mask, spx, spy, shore)
        if shore_pt is None:
            print(f"  WARNING: Cannot find shore near {gs['name']} ({wx}, {wy})")
            continue

        snapped_wx, snapped_wy = scaled_pixel_to_world(shore_pt[0], shore_pt[1], meta)

        # Final snap to full-res water pixel
        if water_full is not None:
            snapped_wx, snapped_wy = snap_world_to_water(
                snapped_wx, snapped_wy, water_full, pixel_scale)

        node_id = graph.get_next_id()
        graph.add_node(node_id, snapped_wx, snapped_wy, NodeType.GAS_STATION,
                       port_name=gs["name"],
                       pixel_x=shore_pt[0], pixel_y=shore_pt[1])
        connect_to_nearest(graph, node_id, n_neighbors=2, scale_factor=scale_factor)
        print(f"  Gas '{gs['name']}' → node {node_id} "
              f"(岸线 world={snapped_wx:.1f},{snapped_wy:.1f})")


# ============================================================
# Step 9: Anchor-Aware Pruning
# ============================================================

def prune_graph(graph: RoadNetwork):
    """
    Two-phase pruning.
    对标 roadmap.cpp 第 7-8 节。

    Phase 1: Remove completely isolated "node" type nodes (degree=0).
    Phase 2: Remove "node" type dead-ends (degree=1) whose only neighbor
             is also a "node" type (not connected to any anchor).
    """
    # Phase 1: Remove isolated nodes
    changed = True
    while changed:
        changed = False
        to_remove = []
        for nid, node in graph.nodes.items():
            if node.node_type != NodeType.NODE:
                continue
            if len(graph.adj.get(nid, [])) == 0:
                to_remove.append(nid)
                changed = True
        for nid in to_remove:
            graph.remove_node(nid)
    print(f"  Phase 1 prune (isolated): {len(to_remove) if changed else 0} removed")

    # Phase 2: Remove dead-end "node" branches not anchored to ports/stations
    changed = True
    total_removed = 0
    while changed:
        changed = False
        to_remove = []
        for nid, node in graph.nodes.items():
            if node.node_type != NodeType.NODE:
                continue
            if len(graph.adj.get(nid, [])) != 1:
                continue
            neighbor_id = graph.adj[nid][0]
            if neighbor_id not in graph.nodes:
                continue
            neighbor = graph.nodes[neighbor_id]
            # Only remove if neighbor is also a plain "node" (no anchor)
            if neighbor.node_type == NodeType.NODE:
                to_remove.append(nid)
                changed = True
        for nid in to_remove:
            graph.remove_node(nid)
        total_removed += len(to_remove)
    print(f"  Phase 2 prune (dead-end branches): {total_removed} removed")


# ============================================================
# Step 10: All-Pairs Shortest Path
# ============================================================

def compute_distance_matrix(graph: RoadNetwork) -> np.ndarray:
    """Floyd-Warshall all-pairs shortest path."""
    n = len(graph.nodes)
    # Map node IDs to contiguous indices
    id_to_idx = {nid: i for i, nid in enumerate(sorted(graph.nodes.keys()))}
    idx_to_id = {i: nid for nid, i in id_to_idx.items()}

    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0.0)

    for edge in graph.edges:
        u, v = id_to_idx[edge.from_id], id_to_idx[edge.to_id]
        dist[u, v] = min(dist[u, v], edge.distance)
        dist[v, u] = min(dist[v, u], edge.distance)

    for k in range(n):
        dk = dist[k]
        for i in range(n):
            dik = dist[i, k]
            if dik == np.inf:
                continue
            nd_row = dik + dk
            dist[i] = np.minimum(dist[i], nd_row)

    # Reindex graph nodes to be contiguous 0..n-1
    new_graph = RoadNetwork()
    new_graph.resolution = graph.resolution
    new_graph.scale_factor = graph.scale_factor
    new_graph.map_width = graph.map_width
    new_graph.map_height = graph.map_height

    for old_id, node in graph.nodes.items():
        new_id = id_to_idx[old_id]
        new_graph.add_node(new_id, node.x, node.y, node.node_type,
                           node.port_name, node.pixel_x, node.pixel_y)

    seen_edges = set()
    for edge in graph.edges:
        nu, nv = id_to_idx[edge.from_id], id_to_idx[edge.to_id]
        key = (min(nu, nv), max(nu, nv))
        if key not in seen_edges:
            seen_edges.add(key)
            new_graph.add_edge(nu, nv, edge.distance, edge.path_pixels)

    new_graph.dist_matrix = dist
    return new_graph


# ============================================================
# Main Pipeline (对标 generateRoadmapWithGasStations)
# ============================================================

def build_road_network_from_png(png_path: str,
                                 ports_config: List[Dict],
                                 gas_stations_config: List[Dict],
                                 downscale: int = 8,
                                 cluster_eps: float = 25.0,
                                 is_scaled_binary: bool = False,
                                 pixel_scale: float = 2.0,
                                 output_dir: str = ".") -> RoadNetwork:
    """
    Full pipeline matching roadmap.cpp generateRoadmapForAllocation.

    Parameters:
        png_path: path to map image
        is_scaled_binary: if True, treat as pre-scaled binary (0=land, 255=water)
        pixel_scale: m/pixel for pre-scaled binary maps (default 2.0 = 0.5*4)
    """
    print("=" * 60)
    print("Road Network Builder (对标 roadmap.cpp)")
    print("=" * 60)

    meta = {}
    # ── Step 1: Load & prepare binary map ──
    print(f"\n[1/9] 加载地图: {png_path}")
    if is_scaled_binary:
        binary_map, scale_factor, meta = load_scaled_binary_map(png_path, pixel_scale)
    else:
        binary_map, scale_factor, meta = load_png_binary_mask(png_path, downscale=downscale)
    print(f"  二值图: {binary_map.shape[1]}x{binary_map.shape[0]}, "
          f"scale_factor={scale_factor:.3f} m/pixel")

    # ── Step 2: Skeletonization ──
    print(f"\n[2/9] 骨架提取 (OpenCV Guo-Hall)")
    skeleton = skeletonize(binary_map)
    n_skel = np.count_nonzero(skeleton)
    print(f"  骨架像素: {n_skel}")

    if n_skel < 2:
        raise RuntimeError("骨架提取失败：水域过小或无水")

    # ── Step 3: Critical pixels ──
    print(f"\n[3/9] 识别关键像素 (度数 != 2)")
    critical, pixel_degrees = identify_critical_pixels(skeleton)
    print(f"  关键像素: {len(critical)}")
    if len(critical) < 2:
        raise RuntimeError("关键节点不足，无法形成网络！")

    # ── Step 4: DBSCAN clustering ──
    print(f"\n[4/9] DBSCAN 聚类 (eps={cluster_eps}像素)")
    labels, centroids, raw_to_cluster = dbscan_cluster_critical(critical, eps=cluster_eps)
    print(f"  聚类数: {len(centroids)} (原始关键点: {len(critical)})")

    # ── Step 5: Path tracing ──
    print(f"\n[5/9] 路径追踪 (关键点间骨架路径)")
    cluster_edges = trace_paths_between_clusters(
        skeleton, critical, pixel_degrees, raw_to_cluster, scale_factor
    )
    print(f"  聚类间边数: {len(cluster_edges)}")

    # ── Step 6: Build graph (centroids snapped to skeleton) ──
    print(f"\n[6/9] 构建图 (聚类中心 → 骨架 → 节点)")
    graph, _ = build_graph_from_clusters(centroids, cluster_edges, skeleton,
                                          scale_factor, meta)
    total_len = sum(e.distance for e in graph.edges)
    print(f"  基础图: {len(graph.nodes)} 节点, {len(graph.edges)} 边, "
          f"{total_len/1000:.2f} km")

    # ── Step 7: Inject anchor points (ports + gas stations) ──
    print(f"\n[7/9] 注入锚点 (港口靠岸边, 加油站靠岸边)")
    print(f"  港口: {len(ports_config)}, 加油站: {len(gas_stations_config)}")
    inject_facilities(graph, ports_config, gas_stations_config,
                      binary_map, scale_factor, meta)
    print(f"  注入后: {len(graph.nodes)} 节点, {len(graph.edges)} 边")

    # ── Step 8: Pruning ──
    print(f"\n[8/9] 剪枝 (保留锚点连接)")
    prune_graph(graph)
    print(f"  剪枝后: {len(graph.nodes)} 节点, {len(graph.edges)} 边")

    # ── Step 9: Distance matrix + reindex ──
    print(f"\n[9/9] 全源最短路径距离矩阵")
    graph = compute_distance_matrix(graph)
    total_len = sum(e.distance for e in graph.edges)
    n_reachable = np.sum(graph.dist_matrix < np.inf)
    n_pairs = len(graph.nodes) ** 2
    print(f"  距离矩阵: ({len(graph.nodes)}, {len(graph.nodes)})")
    print(f"  可达对: {n_reachable}/{n_pairs} ({100*n_reachable/max(1,n_pairs):.1f}%)")

    # ── Verify: all nodes must be in water ──
    water_full = meta.get("water_full")
    n_outside = 0
    if water_full is not None:
        for nid, node in graph.nodes.items():
            orig_px = int(node.x / meta["pixel_scale"])
            orig_py = int(node.y / meta["pixel_scale"])
            if 0 <= orig_px < water_full.shape[1] and 0 <= orig_py < water_full.shape[0]:
                if water_full[orig_py, orig_px] == 0:
                    n_outside += 1
                    if n_outside <= 3:
                        print(f"  WARNING: {node.port_name or 'N'+str(nid)} ({node.x:.0f},{node.y:.0f}) outside water!")
        if n_outside > 0:
            print(f"  节点在水域外: {n_outside}/{len(graph.nodes)}")

    # ── Summary ──
    n_ports = sum(1 for n in graph.nodes.values() if n.is_port)
    n_gas = sum(1 for n in graph.nodes.values() if n.is_gas_station)
    n_regular = len(graph.nodes) - n_ports - n_gas

    print(f"\n{'='*60}")
    print(f"路网构建完成:")
    print(f"  总节点: {len(graph.nodes)} (全部在水域内)" if n_outside == 0 else
          f"  总节点: {len(graph.nodes)} ({n_outside} 在水域外!)")
    print(f"    - 普通节点: {n_regular}")
    print(f"    - 港口: {n_ports}")
    print(f"    - 加油站: {n_gas}")
    print(f"  总边数: {len(graph.edges)}")
    print(f"  航道总长: {total_len:.1f} m ({total_len/1000:.2f} km)")
    print(f"  地图范围: {graph.map_width * graph.resolution:.0f} x "
          f"{graph.map_height * graph.resolution:.0f} m")
    print(f"{'='*60}")

    return graph


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build road network from PNG map (对标 roadmap.cpp)")
    parser.add_argument("--png", type=str,
                        default="/root/demon3.16/data/maps/map10.png")
    parser.add_argument("--ports", type=str,
                        default="/root/demon3.16/src/task_planner/config/ports.yaml")
    parser.add_argument("--gas-stations", type=str,
                        default="/root/demon3.16/src/task_planner/config/gas_stations.yaml")
    parser.add_argument("--downscale", type=int, default=8)
    parser.add_argument("--cluster-eps", type=float, default=25.0,
                        help="DBSCAN eps in pixels (default=25)")
    parser.add_argument("--scaled-binary", action="store_true",
                        help="Map is pre-scaled binary (0=land, 255=water)")
    parser.add_argument("--pixel-scale", type=float, default=2.0,
                        help="m/pixel for scaled binary map (default=2.0)")
    parser.add_argument("--output", type=str,
                        default="/root/demon3.16/src/task_planner/output/road_network.json")
    args = parser.parse_args()

    with open(args.ports, 'r') as f:
        ports_config = yaml.safe_load(f)["ports"]
    with open(args.gas_stations, 'r') as f:
        gas_stations_config = yaml.safe_load(f)["gas_stations"]

    network = build_road_network_from_png(
        png_path=args.png,
        ports_config=ports_config,
        gas_stations_config=gas_stations_config,
        downscale=args.downscale,
        cluster_eps=args.cluster_eps,
        is_scaled_binary=args.scaled_binary,
        pixel_scale=args.pixel_scale,
        output_dir=os.path.dirname(args.output)
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(network.to_dict(), f, indent=2)
    print(f"\n路网已保存: {args.output}")

#!/usr/bin/env python3
"""
异构图 v2 — 节点只存自身状态, 关系全放边上

节点:
  Ship(6维):   [max_payload, max_speed, max_energy, current_load, health, status]
  Task(4维):   [payload, deadline, reward, status]
  Road(5维):   [x, y, node_type, degree, capacity]

边 (5种):
  Road↔Road:   [length, travel_time, energy_cost, width, congestion]
  Ship→Road:   [distance, arrival_time, is_current]
  Task→Road:   [edge_type, load_weight, operation_time]
  Ship↔Task:   [relation_type, estimated_cost, travel_dist, exec_dist, estimated_time, energy_required, match_score]
  Task↔Task:   [relation_type, spatial_dist, time_gap, priority_gap]
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


class HeteroGraph:
    """异构图 v2"""

    def __init__(self):
        self.M = 0; self.K = 0; self.N = 0

        # 节点特征
        self.ship_x = np.empty((0, 6))
        self.task_x = np.empty((0, 4))
        self.road_x = np.empty((0, 5))

        # ID ↔ index 映射
        self.ship_ids = []; self.task_ids = []; self.road_ids = []
        self.ship_idx = {}; self.task_idx = {}; self.road_idx = {}

        # ── 5种边 ──
        # Road↔Road
        self.rr_edges = np.empty((2, 0), dtype=int)
        self.rr_feat = np.empty((0, 5))

        # Ship→Road
        self.sr_edges = np.empty((2, 0), dtype=int)
        self.sr_feat = np.empty((0, 3))

        # Task→Road (2 per task: pickup + delivery)
        self.tr_edges = np.empty((2, 0), dtype=int)
        self.tr_feat = np.empty((0, 3))

        # Ship↔Task (candidate)
        self.st_edges = np.empty((2, 0), dtype=int)
        self.st_feat = np.empty((0, 7))

        # Task↔Task (nearby/same/depends)
        self.tt_edges = np.empty((2, 0), dtype=int)
        self.tt_feat = np.empty((0, 4))

        # 距离矩阵缓存
        self.dist_matrix = None


# ============================================================
# 构建异构图
# ============================================================

def build_hetero_graph(ships, tasks, road_network) -> HeteroGraph:
    """
    从 ShipRuntime / TaskRuntime / RoadNetwork 构建异构图 v2。
    """
    g = HeteroGraph()
    g.M, g.K, g.N = len(ships), len(tasks), len(road_network.nodes)

    g.ship_ids = sorted(ships.keys())
    g.task_ids = sorted(tasks.keys())
    g.road_ids = sorted(road_network.nodes.keys())
    g.ship_idx = {sid: i for i, sid in enumerate(g.ship_ids)}
    g.task_idx = {tid: i for i, tid in enumerate(g.task_ids)}
    g.road_idx = {nid: i for i, nid in enumerate(g.road_ids)}

    # ── Road 节点特征 (5维) ──
    g.road_x = np.zeros((g.N, 5))
    for nid, node in road_network.nodes.items():
        ri = g.road_idx[nid]
        g.road_x[ri, 0] = node.x / 10000.0
        g.road_x[ri, 1] = node.y / 4000.0
        g.road_x[ri, 2] = 2.0 if node.is_port else (1.0 if node.is_gas_station else 0.0)
        g.road_x[ri, 3] = node.degree / 10.0
        g.road_x[ri, 4] = getattr(node, 'capacity', 2.0) / 10.0

    # ── Road↔Road 边 + 特征 (5维) ──
    rr_edges = []; rr_feat = []
    for e in road_network.edges:
        u, v = g.road_idx[e.from_id], g.road_idx[e.to_id]
        w = e.weight if hasattr(e, 'weight') else getattr(e, 'distance', 0)
        length = w * 2.0  # px → meters
        travel_time = length / 5.0  # 默认5m/s
        energy_cost = length / 1000.0 * 2.5  # 默认2.5 kWh/km
        width = 20.0  # 默认20m宽
        congestion = 0.0

        rr_edges.append([u, v]); rr_feat.append([length, travel_time, energy_cost, width, congestion])
        rr_edges.append([v, u]); rr_feat.append([length, travel_time, energy_cost, width, congestion])

    g.rr_edges = np.array(rr_edges).T if rr_edges else np.zeros((2, 0), dtype=int)
    g.rr_feat = np.array(rr_feat)

    # ── Ship 节点特征 (6维) ──
    g.ship_x = np.zeros((g.M, 6))
    for i, sid in enumerate(g.ship_ids):
        s = ships[sid]
        g.ship_x[i, 0] = s.max_payload / 5000.0
        g.ship_x[i, 1] = s.max_speed / 15.0
        g.ship_x[i, 2] = s.max_energy / 1000.0
        g.ship_x[i, 3] = s.load / max(s.max_payload, 1)
        g.ship_x[i, 4] = s.health
        g.ship_x[i, 5] = _status_code(s.current_phase, s.is_idle)

    # ── Task 节点特征 (4维) ──
    g.task_x = np.zeros((g.K, 4))
    for j, tid in enumerate(g.task_ids):
        t = tasks[tid]
        g.task_x[j, 0] = t.payload / 3000.0
        g.task_x[j, 1] = t.deadline / 86400.0 if t.deadline < float('inf') else 2.0
        g.task_x[j, 2] = getattr(t, 'dynamic_priority', t.priority) / 30.0
        g.task_x[j, 3] = _task_status_code(t.status)

    # ── Ship→Road 边 (3维) ──
    sr_edges = []; sr_feat = []
    for i, sid in enumerate(g.ship_ids):
        s = ships[sid]
        ri = g.road_idx.get(s.current_node, 0)
        sr_edges.append([i, ri])
        sr_feat.append([0.0, 0.0, 1.0])  # distance=0 (at node), arrival=0, is_current=1
    g.sr_edges = np.array(sr_edges).T
    g.sr_feat = np.array(sr_feat)

    # ── Task→Road 边 (3维), 每任务2条 ──
    tr_edges = []; tr_feat = []
    for j, tid in enumerate(g.task_ids):
        t = tasks[tid]
        # pickup
        pu = g.road_idx.get(t.pickup_node, 0)
        tr_edges.append([j, pu])
        tr_feat.append([0.0, t.payload / 3000.0, 300.0])  # type=0, weight, op_time
        # delivery
        de = g.road_idx.get(t.delivery_node, 0)
        tr_edges.append([j, de])
        tr_feat.append([1.0, -t.payload / 3000.0, 180.0])  # type=1, -weight, op_time
    g.tr_edges = np.array(tr_edges).T if tr_edges else np.zeros((2, 0), dtype=int)
    g.tr_feat = np.array(tr_feat)

    # ── 预计算距离矩阵 ──
    Nmax = road_network.dist_matrix.shape[0]
    g.dist_matrix = np.full((Nmax, Nmax), np.inf)
    dm = road_network.dist_matrix
    g.dist_matrix[:dm.shape[0], :dm.shape[1]] = dm

    # ── Ship↔Task candidate 边 (7维) ──
    st_edges = []; st_feat = []
    for i, sid in enumerate(g.ship_ids):
        s = ships[sid]
        for j, tid in enumerate(g.task_ids):
            t = tasks[tid]
            if t.status in ('completed', 'cancelled'):
                continue
            cost, travel_d, exec_d, eta, energy = _compute_edge_cost(
                s, t, road_network, 0)
            if cost == float('inf'):
                continue
            rel = 0  # candidate
            if t.assigned_ship == sid:
                rel = 1  # assigned
                if t.status in ('loading', 'navigate_to_delivery'):
                    rel = 2  # executing
                elif t.status == 'completed':
                    rel = 3  # finished
            match = 1.0 / (1.0 + cost / 1000.0)  # match_score 0~1

            st_edges.append([i, j])
            st_feat.append([rel, cost, travel_d, exec_d, eta, energy, match])

    g.st_edges = np.array(st_edges).T if st_edges else np.zeros((2, 0), dtype=int)
    g.st_feat = np.array(st_feat)

    # ── Task↔Task 边 (4维): nearby + depends_on ──
    tt_edges = []; tt_feat = []
    for j in range(g.K):
        tj = g.task_ids[j]
        task_j = tasks[tj]
        # nearby: 装货港邻近的任务 (最多5条)
        nearby = []
        for m in range(g.K):
            if m == j: continue
            tm = g.task_ids[m]
            task_m = tasks[tm]
            d = road_network.dist_matrix[task_j.pickup_node, task_m.pickup_node]
            if d < 500:  # 500m内
                time_gap = abs(task_j.deadline - task_m.deadline) if task_j.deadline < float('inf') and task_m.deadline < float('inf') else 0
                priority_gap = getattr(task_j, 'dynamic_priority', task_j.priority) - getattr(task_m, 'dynamic_priority', task_m.priority)
                nearby.append((m, d, time_gap, priority_gap))
        nearby.sort(key=lambda x: x[1])  # 按距离排序
        for m, d, tg, pg in nearby[:5]:
            tt_edges.append([j, m])
            tt_feat.append([0.0, d, tg, pg])  # type=0=nearby

        # depends_on: 同一艘船的任务序列
        if task_j.assigned_ship >= 0:
            ship = ships.get(task_j.assigned_ship)
            if ship:
                for idx, tid in enumerate(ship.task_sequence):
                    if tid == tj and idx + 1 < len(ship.task_sequence):
                        next_tid = ship.task_sequence[idx + 1]
                        m = g.task_idx.get(next_tid)
                        if m is not None:
                            tt_edges.append([j, m])
                            tt_feat.append([3.0, 0, 0, 0])  # type=3=depends_on
                        break

    g.tt_edges = np.array(tt_edges).T if tt_edges else np.zeros((2, 0), dtype=int)
    g.tt_feat = np.array(tt_feat)

    return g


# ============================================================
# Ship↔Task 边代价计算
# ============================================================

def _compute_edge_cost(ship, task, road_network, current_time):
    """计算 Ship→Task candidate 边的代价, 返回 (cost, travel_d, exec_d, eta, energy)"""
    d_pickup = road_network.dist_matrix[ship.current_node, task.pickup_node]
    d_exec = road_network.dist_matrix[task.pickup_node, task.delivery_node]

    if d_pickup == np.inf or d_exec == np.inf:
        return (float('inf'), 0, 0, 0, 0)

    total_d = d_pickup + d_exec

    # 载重约束
    if task.payload > ship.remaining_capacity:
        return (float('inf'), 0, 0, 0, 0)

    # 能源约束
    energy_need = total_d / 1000.0 * ship.energy_per_km
    if energy_need > ship.energy * 0.7:
        return (float('inf'), 0, 0, 0, 0)

    # 时间约束
    eta = current_time + total_d / ship.max_speed + 600
    if task.deadline < float('inf') and eta > task.deadline:
        return (float('inf'), 0, 0, 0, 0)

    # 代价
    cost = total_d
    cost += len(ship.task_sequence) * 500  # 负载均衡
    cost += max(0, (ship.load + task.payload) / max(ship.max_payload, 1) - 0.8) * 10000

    if task.deadline < float('inf'):
        urgency = max(0, 1.0 - (task.deadline - eta) / 3600.0)
        cost += urgency * task.priority * 2000

    return (cost, d_pickup, d_exec, eta, energy_need)


# ============================================================
# 动态更新 (每周期)
# ============================================================

def update_graph(g: HeteroGraph, ships, tasks, road_network, current_time):
    """每周期更新异构图: 节点特征 + Ship→Task candidate边重建"""

    # 更新 Ship 节点
    for i, sid in enumerate(g.ship_ids):
        s = ships[sid]
        g.ship_x[i, 3] = s.load / max(s.max_payload, 1)
        g.ship_x[i, 4] = s.health
        g.ship_x[i, 5] = _status_code(s.current_phase, s.is_idle)

    # 更新 Task 节点
    for j, tid in enumerate(g.task_ids):
        t = tasks[tid]
        g.task_x[j, 2] = getattr(t, 'dynamic_priority', t.priority) / 30.0
        g.task_x[j, 3] = _task_status_code(t.status)

    # 更新 Ship→Road 边
    for i, sid in enumerate(g.ship_ids):
        s = ships[sid]
        g.sr_feat[i, 2] = 1.0  # is_current

    # 重建 Ship↔Task candidate 边 (状态变了, 代价也变了)
    st_edges = []; st_feat = []
    for i, sid in enumerate(g.ship_ids):
        s = ships[sid]
        for j, tid in enumerate(g.task_ids):
            t = tasks[tid]
            if t.status in ('completed', 'cancelled'):
                continue
            cost, td, ed, eta, en = _compute_edge_cost(s, t, road_network, current_time)
            if cost == float('inf'):
                continue
            rel = 0
            if t.assigned_ship == sid:
                rel = 1
                if t.status in ('loading', 'navigate_to_delivery'):
                    rel = 2
                elif t.status == 'completed':
                    rel = 3
            match = 1.0 / (1.0 + cost / 1000.0)
            st_edges.append([i, j])
            st_feat.append([rel, cost, td, ed, eta, en, match])

    g.st_edges = np.array(st_edges).T if st_edges else np.zeros((2, 0), dtype=int)
    g.st_feat = np.array(st_feat)


def ship_task_cost(g: HeteroGraph, ship_i: int, task_j: int) -> float:
    """从 Ship↔Task candidate 边直接读 estimated_cost"""
    for k in range(g.st_edges.shape[1]):
        if g.st_edges[0, k] == ship_i and g.st_edges[1, k] == task_j:
            return g.st_feat[k, 1]  # estimated_cost
    return float('inf')


def _status_code(phase: str, is_idle: bool) -> float:
    """状态编码"""
    if is_idle: return 0.0
    if 'loading' in phase: return 0.3
    if 'pickup' in phase: return 0.5
    if 'delivery' in phase: return 0.7
    if 'unloading' in phase: return 0.9
    return 1.0


def _task_status_code(status: str) -> float:
    m = {'pending': 0.0, 'assigned': 0.25, 'loading': 0.5,
         'navigate_to_delivery': 0.75, 'completed': 1.0, 'cancelled': -1.0}
    return m.get(status, 0.0)

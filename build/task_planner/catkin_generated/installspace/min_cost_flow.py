#!/usr/bin/env python3
"""
最小费用流任务分配 — 全局最优

流网络: Source → Ships → Tasks → Sink
  Source→Ship: 容量=船能执行的任务数, 费用=0
  Ship→Task:   容量=1, 费用=ship_task_cost(ship, task)
  Task→Sink:   容量=1, 费用=0

算法: Successive Shortest Path (SPFA找增广路)
"""

import numpy as np
from collections import deque, defaultdict
from typing import List, Dict, Tuple


def min_cost_flow_allocate(ships, tasks, hetero_graph,
                            current_time: float = 0):
    """
    最小费用流全局分配。

    Args:
        ships: Dict[int, ShipRuntime]
        tasks: Dict[int, TaskRuntime]
        hetero_graph: HeteroGraph 实例
        current_time: 当前时间

    Returns:
        assignments: List[(ship_id, task_id, cost)]
        total_cost: float
        flow: int (分配的任务数)
    """
    g = hetero_graph
    M, K = g.M, g.K
    # 建立 Ship→Task 边索引: (ship_i, task_j) → cost
    st_cost = {}
    for k in range(g.st_edges.shape[1]):
        si, tj = g.st_edges[0, k], g.st_edges[1, k]
        st_cost[(si, tj)] = g.st_feat[k, 1]  # estimated_cost

    if K == 0:
        return [], 0.0, 0

    # 节点编号
    SOURCE = 0
    SINK = M + K + 1
    N_NODES = M + K + 2

    # 邻接表: u -> [(v, cap, cost, rev_index)]
    graph = defaultdict(list)

    def add_edge(u, v, cap, cost):
        graph[u].append([v, cap, cost, len(graph[v])])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1])

    # Source → Ships
    for i in range(M):
        ship = ships[g.ship_ids[i]]
        # 每艘船最多执行的任务数 = 能源 / 每个任务平均能耗
        avg_energy = 50  # kWh per task
        max_tasks = min(10, max(1, int(ship.energy / avg_energy)))
        add_edge(SOURCE, 1 + i, max_tasks, 0)

    # Ships → Tasks
    for i in range(M):
        for j in range(K):
            cost = st_cost.get((i, j), float('inf'))
            if cost < float('inf'):
                add_edge(1 + i, 1 + M + j, 1, int(cost))

    # Tasks → Sink
    for j in range(K):
        add_edge(1 + M + j, SINK, 1, 0)

    # ======= SSSP (SPFA) 找增广路 =======
    flow = 0
    total_cost = 0.0

    while flow < K:
        dist = [float('inf')] * N_NODES
        prev_v = [-1] * N_NODES
        prev_e = [-1] * N_NODES
        in_q = [False] * N_NODES

        dist[SOURCE] = 0
        q = deque([SOURCE])
        in_q[SOURCE] = True

        while q:
            u = q.popleft()
            in_q[u] = False
            for ei, (v, cap, w, _) in enumerate(graph[u]):
                if cap > 0 and dist[v] > dist[u] + w:
                    dist[v] = dist[u] + w
                    prev_v[v] = u
                    prev_e[v] = ei
                    if not in_q[v]:
                        q.append(v)
                        in_q[v] = True

        if dist[SINK] == float('inf'):
            break

        # 沿最短路径推流
        d = 1
        v = SINK
        while v != SOURCE:
            u = prev_v[v]
            ei = prev_e[v]
            graph[u][ei][1] -= d
            rev = graph[u][ei][3]
            graph[v][rev][1] += d
            v = u

        flow += d
        total_cost += dist[SINK] * d

    # ======= 提取分配结果 =======
    assignments = []
    for i in range(M):
        sid = g.ship_ids[i]
        for ei, (v, cap, w, _) in enumerate(graph[1 + i]):
            if M < v <= M + K and cap == 0:
                tj = v - M - 1
                tid = g.task_ids[tj]
                assignments.append((sid, tid, w))

    # 应用分配
    for sid, tid, cost in assignments:
        tasks[tid].assigned_ship = sid
        tasks[tid].status = "assigned"
        if tid not in ships[sid].task_sequence:
            ships[sid].task_sequence.append(tid)
        ships[sid].load += tasks[tid].payload

    return assignments, total_cost, flow


def optimize_sequence(ship, tasks, dist_matrix):
    """2-opt 优化单艘船的任务执行顺序"""
    tids = list(ship.task_sequence)
    if len(tids) < 2:
        return

    def seq_cost(seq):
        cur = ship.current_node
        total = 0.0
        for tid in seq:
            t = tasks.get(tid)
            if not t: continue
            total += dist_matrix[cur, t.pickup_node]
            total += dist_matrix[t.pickup_node, t.delivery_node]
            cur = t.delivery_node
        return total

    improved = True
    it = 0
    while improved and it < 100:
        improved = False
        it += 1
        best = seq_cost(tids)
        for i in range(len(tids)):
            for j in range(i + 2, len(tids)):
                new_seq = tids[:i] + list(reversed(tids[i:j+1])) + tids[j+1:]
                c = seq_cost(new_seq)
                if c < best - 1e-6:
                    tids = new_seq
                    best = c
                    improved = True
    ship.execution_order = tids

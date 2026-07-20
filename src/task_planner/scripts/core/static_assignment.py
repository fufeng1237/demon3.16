#!/usr/bin/env python3
"""静态初始任务分配的场景与求解入口。

此模块刻意不依赖 ``RealTimeScheduler``：不推进时间、不更新动态优先级、
不执行任务，也不触发故障或滚动重分配。它是静态实验和初始方案的唯一入口。
"""
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List
import re
import numpy as np

from alns_scheduler import ALNSScheduler, RouteNode
from graph_evaluator import GraphEvaluator


@dataclass
class StaticShip:
    ship_id: int
    name: str
    max_payload: float
    max_energy: float
    max_speed: float
    energy_per_km: float
    current_node: int
    energy: float
    load: float = 0.0
    task_sequence: List[int] = field(default_factory=list)
    health: float = 1.0
    current_phase: str = 'idle'

    @property
    def remaining_capacity(self):
        return self.max_payload - self.load

    @property
    def energy_ratio(self):
        return self.energy / self.max_energy if self.max_energy else 0.0

    @property
    def is_idle(self):
        return self.current_phase == 'idle'


@dataclass
class StaticTask:
    task_id: int
    pickup_node: int
    delivery_node: int
    payload: float
    priority: int
    deadline: float = float('inf')
    status: str = 'pending'
    assigned_ship: int = -1


def nearest_node(rn, x, y):
    return min(rn.nodes, key=lambda nid: (rn.nodes[nid].x-x)**2 + (rn.nodes[nid].y-y)**2)


def read_configs(usvs_path, tasks_path):
    ships, tasks = [], []
    with open(usvs_path, encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.replace('USV:', '').strip().split(',')
            if len(p) >= 8:
                ships.append((int(p[0]), f'S{int(p[0])}', int(p[4]), float(p[6]),
                              float(p[3]), int(p[1]), int(p[2])))
    pattern = (r'Task\s+(\d+):\s*pickup\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'
               r'\s*delivery\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)')
    with open(tasks_path, encoding='utf-8') as f:
        for line in f:
            m = re.match(pattern, line)
            if m:
                tasks.append(tuple(map(int, m.groups())))
    return ships, tasks


def build_static_scene(rn, usvs_path, tasks_path, seed=42):
    """从配置生成静态快照；种子只决定实验中的离散载荷和优先级。"""
    ship_cfg, task_cfg = read_configs(usvs_path, tasks_path)
    rng = np.random.default_rng(seed)
    ships = {}
    for sid, name, cap, energy, speed, px, py in ship_cfg:
        ships[sid] = StaticShip(sid, name, cap, energy, speed, 2.5,
                                nearest_node(rn, px * 2, py * 2), energy * 0.9)
    tasks = {}
    for tid, ppx, ppy, dpx, dpy in task_cfg:
        pickup, delivery = nearest_node(rn, ppx * 2, ppy * 2), nearest_node(rn, dpx * 2, dpy * 2)
        if np.isinf(rn.dist_matrix[pickup, delivery]):
            continue
        tasks[tid] = StaticTask(tid, pickup, delivery,
                                float(rng.choice([300, 500, 800, 1000, 1500])),
                                int(rng.choice([1, 2, 3])))
    return SimpleNamespace(ships=ships, tasks=tasks)


def solve_initial_assignment(rn, ships, tasks, node_names=None, alns_config=None,
                             initial_routes=None, optimize=True):
    """只求一次初始分配；不会调用任何执行、事件或重分配代码。"""
    alns = ALNSScheduler(GraphEvaluator(rn, node_names, tasks), tasks, rn, node_names,
                         config=alns_config)
    routes = initial_routes if initial_routes is not None else alns.build_initial_routes(ships)
    return alns.optimize(ships, routes) if optimize else routes


def greedy_initial_routes(rn, ships, tasks):
    """静态贪心基线，按完整取送对追加，保持容量、能量和可达性约束。"""
    routes = {sid: [] for sid in ships}
    for tid in sorted(tasks, key=lambda t: -tasks[t].payload * tasks[t].priority):
        task, best = tasks[tid], None
        for sid, ship in ships.items():
            if task.payload > ship.max_payload:
                continue
            cur = routes[sid][-1].node_id if routes[sid] else ship.current_node
            d1, d2 = rn.dist_matrix[cur, task.pickup_node], rn.dist_matrix[task.pickup_node, task.delivery_node]
            if np.isinf(d1) or np.isinf(d2) or (d1+d2)/1000*ship.energy_per_km > ship.energy*.7:
                continue
            score = d1 + d2 + len(routes[sid]) * 150
            if best is None or score < best[0]:
                best = (score, sid)
        if best is not None:
            sid = best[1]
            routes[sid] += [RouteNode(task.pickup_node, 'PICKUP', tid),
                            RouteNode(task.delivery_node, 'DELIVERY', tid)]
            task.assigned_ship = sid
    return routes

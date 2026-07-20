#!/usr/bin/env python3
"""Route Executor — 船沿路网最短路径逐步移动"""

import numpy as np
from typing import Dict, List, Tuple


class RouteExecutor:
    def __init__(self, road_network, tasks: Dict, node_names: Dict = None):
        self.rn = road_network
        self.tasks = tasks
        self.node_names = node_names or {}
        # 缓存: (from_node, to_node) → 最短路径节点列表
        self._path_cache: Dict[Tuple[int, int], List[int]] = {}

    def _shortest_path(self, from_node: int, to_node: int) -> List[int]:
        """Dijkstra 找 from→to 的最短路径节点序列"""
        key = (from_node, to_node)
        if key in self._path_cache:
            return self._path_cache[key]

        n = self.rn.dist_matrix.shape[0]
        dist = np.full(n, np.inf)
        prev = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        dist[from_node] = 0

        for _ in range(n):
            u = np.argmin(np.where(visited, np.inf, dist))
            if dist[u] == np.inf:
                break
            visited[u] = True
            if u == to_node:
                break
            for v in self.rn.adj.get(u, []):
                w = self.rn.dist_matrix[u, v]
                if w < np.inf and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u

        # 重建路径
        path = []
        cur = to_node
        while cur >= 0:
            path.append(cur)
            if cur == from_node:
                break
            cur = prev[cur]
        path.reverse()

        if path[0] != from_node:
            path = [from_node, to_node]  # fallback

        self._path_cache[key] = path
        return path

    def execute_step(self, ships: Dict, routes: Dict, dt: float, current_time: float):
        """执行一个时间步"""
        for sid, ship in ships.items():
            route = routes.get(sid, [])
            if not route:
                ship.current_phase = 'idle'
                continue
            self._advance_ship(ship, route, dt, current_time)

    def _advance_ship(self, ship, route: List, dt: float, current_time: float):
        """沿路网逐步推进船"""
        if not route:
            return

        next_rn = route[0]

        # 已经在目标节点
        if ship.current_node == next_rn.node_id:
            self._handle_arrival(ship, route)
            return

        # 找最短路径
        path = self._shortest_path(ship.current_node, next_rn.node_id)

        # 沿途移动
        remaining_dt = dt
        while remaining_dt > 0 and len(path) > 1:
            u, v = path[0], path[1]
            seg_dist = self.rn.dist_matrix[u, v]
            if seg_dist == np.inf:
                break

            travel_dist = ship.max_speed * remaining_dt
            if travel_dist >= seg_dist:
                # 到达 v
                ship.current_node = v
                ship.total_distance += seg_dist
                ship.energy -= seg_dist / 1000.0 * ship.energy_per_km
                ship.total_time += seg_dist / ship.max_speed
                remaining_dt -= seg_dist / ship.max_speed
                path.pop(0)
                ship.current_phase = 'sailing'
            else:
                # 在边上走了一段, 没到 v
                ship.total_distance += travel_dist
                ship.energy -= travel_dist / 1000.0 * ship.energy_per_km
                ship.total_time += remaining_dt
                ship.current_phase = 'sailing'
                remaining_dt = 0

        # 检查是否到达目标
        if ship.current_node == next_rn.node_id:
            self._handle_arrival(ship, route)

    def _handle_arrival(self, ship, route: List):
        """到达 RouteNode, 处理 PICKUP/DELIVERY"""
        rn = route.pop(0)

        if rn.action == "PICKUP":
            ship.current_phase = 'loading'
            task = self.tasks.get(rn.task_id)
            if task:
                ship.load += task.payload
                task.status = 'loaded'
                task.assigned_ship = ship.ship_id

        elif rn.action == "DELIVERY":
            ship.current_phase = 'unloading'
            task = self.tasks.get(rn.task_id)
            if task:
                ship.load -= task.payload
                task.status = 'completed'
                if rn.task_id not in ship.completed_tasks:
                    ship.completed_tasks.append(rn.task_id)

        elif rn.action == "PASS":
            ship.current_phase = 'sailing'

#!/usr/bin/env python3
"""
共享基类 — 所有对比算法复用: RouteNode, 约束检查, 代价计算, 贪心初始解
"""
import numpy as np
from copy import deepcopy
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class RouteNode:
    """Route 中的一个节点"""
    node_id: int
    action: str        # "PICKUP" | "DELIVERY" | "PASS"
    task_id: int = -1

    def __repr__(self):
        if self.action == "PASS":
            return f"N{self.node_id}"
        return f"{self.action[:4]}(T{self.task_id}@{self.node_id})"


class BaseScheduler:
    """所有调度算法的共享基类"""

    def __init__(self, road_network, ships: Dict, tasks: Dict):
        self.rn = road_network
        self.ships = ships
        self.tasks = tasks
        self.loading_time = 300
        self.unloading_time = 180
        self.load_penalty_factor = 0.5

    # ================================================================
    #  路线代价计算
    # ================================================================

    def _route_distance(self, ship, route: List[RouteNode]) -> float:
        """单船总航行距离 (m)"""
        total = 0.0; cur = ship.current_node
        for rn in route:
            total += self.rn.dist_matrix[cur][rn.node_id]
            cur = rn.node_id
        return total

    def _ship_time(self, ship, route: List[RouteNode]) -> float:
        """单船完工时间 (s)"""
        if not route:
            return 0.0
        t = 0.0; cur = ship.current_node
        for rn in route:
            d = self.rn.dist_matrix[cur][rn.node_id]
            t += d / max(ship.max_speed, 1.0)
            if rn.action == 'PICKUP':
                t += self.loading_time
            elif rn.action == 'DELIVERY':
                t += self.unloading_time
            cur = rn.node_id
        return t

    def makespan(self, routes: Dict) -> float:
        """舰队完工时间 (最晚船的完工时间)"""
        return max(self._ship_time(self.ships[sid], seq)
                   for sid, seq in routes.items())

    def total_distance(self, routes: Dict) -> float:
        """舰队总航行距离 (m)"""
        return sum(self._route_distance(self.ships[sid], seq)
                   for sid, seq in routes.items())

    def total_energy(self, routes: Dict) -> float:
        """舰队总能耗 (kWh), 载重敏感"""
        total = 0.0
        for sid, seq in routes.items():
            ship = self.ships[sid]
            cur = ship.current_node; load = ship.load
            for rn in seq:
                d = self.rn.dist_matrix[cur][rn.node_id]
                ratio = 1.0 + self.load_penalty_factor * load / max(ship.max_payload, 1)
                total += (d / 1000.0) * ship.energy_per_km * ratio
                if rn.action == 'PICKUP':
                    t = self.tasks.get(rn.task_id)
                    if t: load += t.payload
                elif rn.action == 'DELIVERY':
                    t = self.tasks.get(rn.task_id)
                    if t: load -= t.payload
                cur = rn.node_id
        return total

    def load_std(self, routes: Dict) -> float:
        """各船任务负载 (吨) 的标准差"""
        loads = []
        for sid, seq in routes.items():
            ld = sum(self.tasks[rn.task_id].payload
                     for rn in seq if rn.action == 'PICKUP')
            loads.append(ld)
        return float(np.std(loads)) if loads else 0.0

    def fleet_cost(self, routes: Dict) -> float:
        """综合代价 (用于算法内部优化) = 0.70×M_norm + 0.15×D_norm + 0.08×E_norm + 0.07×B_norm"""
        # 归一化基准: 第一次调用时自动设定
        if not hasattr(self, '_norm_base'):
            raw = {'M': max(self.makespan(routes), 1),
                   'D': max(self.total_distance(routes), 1),
                   'E': max(self.total_energy(routes), 1),
                   'B': max(self.load_std(routes), 1)}
            self._norm_base = raw

        base = self._norm_base
        M_norm = self.makespan(routes) / base['M']
        D_norm = self.total_distance(routes) / base['D']
        E_norm = self.total_energy(routes) / base['E']
        B_norm = self.load_std(routes) / base['B']
        return 0.70 * M_norm + 0.15 * D_norm + 0.08 * E_norm + 0.07 * B_norm

    # ================================================================
    #  约束检查
    # ================================================================

    def _check_route(self, ship, route: List[RouteNode]) -> bool:
        """验证单条 Route 满足所有硬约束"""
        load = ship.load; cur = ship.current_node
        in_progress = set()
        for rn in route:
            if self.rn.dist_matrix[cur][rn.node_id] == np.inf:
                return False
            cur = rn.node_id
            if rn.action == "PICKUP":
                if rn.task_id in in_progress:
                    return False
                t = self.tasks.get(rn.task_id)
                if not t: return False
                load += t.payload
                if load > ship.max_payload:
                    return False
                in_progress.add(rn.task_id)
            elif rn.action == "DELIVERY":
                if rn.task_id not in in_progress:
                    return False
                t = self.tasks.get(rn.task_id)
                if not t: return False
                load -= t.payload
                in_progress.remove(rn.task_id)
        return len(in_progress) == 0

    def _validate_all(self, routes: Dict) -> bool:
        return all(self._check_route(self.ships[sid], seq)
                   for sid, seq in routes.items())

    # ================================================================
    #  贪心初始解 (所有算法共用)
    # ================================================================

    def greedy_init(self) -> Dict[int, List[RouteNode]]:
        """贪心构造初始解: 每步选 score = delta_time + 0.5×ship_time 最小的船"""
        routes = {sid: [] for sid in self.ships}
        task_list = sorted(self.tasks.keys(),
                           key=lambda tid: -self.tasks[tid].payload)

        for tid in task_list:
            task = self.tasks[tid]
            best_sid, best_pu, best_de, best_score = None, -1, -1, float('inf')
            for sid in self.ships:
                ship = self.ships[sid]
                pu, de, ok = self._find_insertion(ship, routes[sid], task)
                if not ok: continue
                delta_t = self._insertion_time_delta(ship, routes[sid], pu, de, task)
                ship_t = self._ship_time(ship, routes[sid])
                score = delta_t + 0.5 * ship_t  # 负载均衡偏置
                if score < best_score:
                    best_score = score; best_sid = sid
                    best_pu, best_de = pu, de
            if best_sid is not None:
                r = routes[best_sid]
                r.insert(best_pu, RouteNode(task.pickup_node, "PICKUP", tid))
                r.insert(best_de + 1, RouteNode(task.delivery_node, "DELIVERY", tid))
        return routes

    # ================================================================
    #  插入工具
    # ================================================================

    def _find_insertion(self, ship, route: List[RouteNode], task
                        ) -> Tuple[int, int, bool]:
        """找最佳插入位置 (O(n²)), 返回 (pu_pos, de_pos, feasible)"""
        best_pu, best_de, best_delta = 0, 0, float('inf')
        n = len(route)
        pu_node = task.pickup_node
        de_node = task.delivery_node
        payload = task.payload
        capacity = ship.max_payload
        rn_dist = self.rn.dist_matrix

        nodes = [ship.current_node] + [rn.node_id for rn in route]
        edge_d = [rn_dist[nodes[i]][nodes[i+1]] for i in range(n)]

        cum_load = [ship.load]
        cur_load = ship.load
        for rn in route:
            if rn.action == 'PICKUP':
                t = self.tasks.get(rn.task_id)
                if t: cur_load += t.payload
            elif rn.action == 'DELIVERY':
                t = self.tasks.get(rn.task_id)
                if t: cur_load -= t.payload
            cum_load.append(cur_load)

        for pu_pos in range(n + 1):
            prev_pu = nodes[pu_pos]
            nxt_pu = nodes[pu_pos + 1] if pu_pos < n else None
            if rn_dist[prev_pu][pu_node] == np.inf: continue
            if nxt_pu is not None and rn_dist[pu_node][nxt_pu] == np.inf: continue

            if pu_pos < n:
                d_pu = rn_dist[prev_pu][pu_node] + rn_dist[pu_node][nxt_pu] - edge_d[pu_pos]
            else:
                d_pu = rn_dist[prev_pu][pu_node]

            load_after = cum_load[pu_pos] + payload
            if load_after > capacity: continue

            max_load = load_after
            for de_pos in range(pu_pos, n + 1):
                if de_pos > pu_pos:
                    max_load = max(max_load, cum_load[de_pos] + payload)
                if max_load > capacity: break

                prev_de = pu_node if de_pos == pu_pos else nodes[de_pos]
                nxt_de = nodes[de_pos + 1] if de_pos < n else None
                if rn_dist[prev_de][de_node] == np.inf: continue
                if nxt_de is not None and rn_dist[de_node][nxt_de] == np.inf: continue

                if de_pos == pu_pos == n:
                    d_de = rn_dist[pu_node][de_node]
                elif de_pos == pu_pos:
                    d_de = (rn_dist[pu_node][de_node] + rn_dist[de_node][nxt_pu]
                            - rn_dist[pu_node][nxt_pu])
                elif de_pos < n:
                    d_de = rn_dist[prev_de][de_node] + rn_dist[de_node][nxt_de] - edge_d[de_pos]
                else:
                    d_de = rn_dist[prev_de][de_node]

                delta = d_pu + d_de
                if delta < best_delta:
                    best_delta = delta; best_pu = pu_pos; best_de = de_pos

        return best_pu, best_de, best_delta < float('inf')

    def _insertion_time_delta(self, ship, route, pu_pos, de_pos, task) -> float:
        """计算插入后该船完工时间的增加量 (s)"""
        # 简化: 距离增量 / speed + 装卸固定时间
        n = len(route)
        nodes = [ship.current_node] + [rn.node_id for rn in route]
        rn_dist = self.rn.dist_matrix

        # 距离增量 (复现简化版)
        d_pu = rn_dist[nodes[pu_pos]][task.pickup_node]
        if pu_pos < n:
            d_pu += rn_dist[task.pickup_node][nodes[pu_pos + 1]]
            d_pu -= rn_dist[nodes[pu_pos]][nodes[pu_pos + 1]]

        prev_de = task.pickup_node if de_pos == pu_pos else nodes[de_pos]
        d_de = rn_dist[prev_de][task.delivery_node]
        if de_pos < n:
            nxt_de = nodes[de_pos + 1]
            d_de += rn_dist[task.delivery_node][nxt_de]
            d_de -= rn_dist[prev_de][nxt_de]

        delta_d = d_pu + d_de
        speed = max(ship.max_speed, 1.0)
        return delta_d / speed + self.loading_time + self.unloading_time

    def _do_insert(self, routes: Dict, sid: int, tid: int,
                   pu_pos: int, de_pos: int, task):
        """在 routes[sid] 的指定位置插入 PICKUP+DELIVERY"""
        r = routes[sid]
        r.insert(pu_pos, RouteNode(task.pickup_node, "PICKUP", tid))
        r.insert(de_pos + 1, RouteNode(task.delivery_node, "DELIVERY", tid))

    def _remove_task(self, routes: Dict, sid: int, tid: int):
        """从 routes[sid] 删除指定任务的两个 RouteNode"""
        routes[sid] = [rn for rn in routes[sid] if rn.task_id != tid]

    def _collect_all_tasks(self, routes: Dict) -> List[Tuple[int, int]]:
        """收集所有已分配任务: [(ship_id, task_id), ...]"""
        result = []
        for sid, seq in routes.items():
            seen = set()
            for rn in seq:
                if rn.action == "PICKUP" and rn.task_id not in seen:
                    seen.add(rn.task_id)
                    result.append((sid, rn.task_id))
        return result

    def _copy_routes(self, routes: Dict) -> Dict:
        return {sid: list(seq) for sid, seq in routes.items()}

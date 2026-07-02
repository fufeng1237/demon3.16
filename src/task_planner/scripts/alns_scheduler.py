#!/usr/bin/env python3
"""
Graph-guided ALNS — 优化 RoadNode 访问序列 (Route)

核心:
  Route = List[RouteNode]  每个 RouteNode = (node_id, action, task_id)
  action ∈ {PICKUP, DELIVERY, PASS}

  插入任务 = 在 Route 中插入一对 (PICKUP + DELIVERY) 节点
  约束: PICKUP 必须在 DELIVERY 之前, 容量不超, 能源不超
"""

import numpy as np
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


@dataclass
class RouteNode:
    """Route 中的一个节点"""
    node_id: int
    action: str        # "PICKUP" | "DELIVERY" | "PASS"
    task_id: int = -1  # 关联任务 (PASS 时为 -1)

    def __repr__(self):
        if self.action == "PASS":
            return f"N{self.node_id}"
        return f"{self.action[:4]}(T{self.task_id}@{self.node_id})"


class ALNSScheduler:
    """Graph-guided ALNS — 优化 RoadNode 访问序列"""

    def __init__(self, evaluator, tasks: Dict, road_network,
                 node_names: Dict = None):
        self.evaluator = evaluator
        self.tasks = tasks
        self.rn = road_network
        self.node_names = node_names or {}

        self.max_iter = 300
        self.T0 = 200.0; self.T_min = 0.1; self.alpha = 0.98
        self.K_candidates = 5

        self.destroy_weights = {'random': 1.0, 'worst': 1.0, 'shaw': 1.0, 'energy': 1.0}
        self.repair_weights = {'greedy': 1.0, 'regret2': 1.0}
        self.destroy_scores = defaultdict(float); self.repair_scores = defaultdict(float)
        self.destroy_count = defaultdict(int); self.repair_count = defaultdict(int)

    # ================================================================
    # 公开接口
    # ================================================================

    def build_initial_routes(self, ships: Dict) -> Dict[int, List[RouteNode]]:
        """
        Graph-guided Cheapest Insertion 构建初始 Route。
        平衡策略: 贪心分配 + 每轮强制选当前任务最少的可行船。
        """
        ships_copy = {sid: deepcopy(s) for sid, s in ships.items()}
        routes = {sid: [] for sid in ships_copy}

        task_list = [tid for tid in self.tasks
                     if self.tasks[tid].status not in ('completed', 'cancelled')]
        def sort_key(tid):
            t = self.tasks[tid]
            u = t.dynamic_priority if hasattr(t, 'dynamic_priority') else t.priority
            return -t.payload * u
        task_list = sorted(task_list, key=sort_key)

        # Round-robin across ships for balance: sort ships by load each round
        ship_order = sorted(ships_copy.keys())
        ship_idx = 0

        for tid in task_list:
            task = self.tasks[tid]
            # Try each ship in round-robin order
            best_sid = None; best_pu = -1; best_de = -1; best_delta = float('inf')
            for _ in range(len(ship_order)):
                sid = ship_order[ship_idx % len(ship_order)]
                ship_idx += 1
                ship = ships_copy[sid]
                c = self.evaluator.evaluate(ship, task)
                if not c.is_feasible(): continue
                pu, de, delta = self._best_insert_pair(ship, routes[sid], task)
                if delta < best_delta:
                    best_delta = delta; best_sid = sid; best_pu = pu; best_de = de

            if best_sid is not None:
                r = routes[best_sid]
                r.insert(best_pu, RouteNode(task.pickup_node, "PICKUP", tid))
                r.insert(best_de + 1, RouteNode(task.delivery_node, "DELIVERY", tid))
                self.tasks[tid].assigned_ship = best_sid
            # 更新船顺序: 当前最空闲的排前面
            ship_order.sort(key=lambda sid: (len(routes[sid]), np.random.random()))

        return routes

    def optimize(self, ships: Dict, current_routes: Dict[int, List[RouteNode]] = None,
                  frozen_count: int = 0) -> Dict[int, List[RouteNode]]:
        """ALNS 优化 Route (SA)"""
        if current_routes is None:
            current_routes = {sid: list(self._get_route(ships[sid]))
                              for sid in ships}

        current_cost = self._fleet_cost(current_routes, ships)
        best_routes = deepcopy(current_routes); best_cost = current_cost
        T = self.T0; it = 0; no_improve = 0

        while T > self.T_min and it < self.max_iter:
            d_op = self._select_destroy(); r_op = self._select_repair()

            destroyed, removed_tasks = self._apply_destroy(
                d_op, current_routes, ships, frozen_count)
            if not removed_tasks:
                it += 1; T *= self.alpha; continue

            repaired = self._apply_repair(r_op, destroyed, ships, removed_tasks)
            if not self._validate_all(repaired, ships):
                it += 1; T *= self.alpha; continue

            new_cost = self._fleet_cost(repaired, ships)
            delta = new_cost - current_cost

            if delta < 0 or random.random() < np.exp(-delta / T):
                current_routes = repaired; current_cost = new_cost
                self.destroy_scores[d_op] += 1; self.repair_scores[r_op] += 1
                if new_cost < best_cost:
                    best_routes = deepcopy(repaired); best_cost = new_cost
                    self.destroy_scores[d_op] += 3; self.repair_scores[r_op] += 3
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            self.destroy_count[d_op] += 1; self.repair_count[r_op] += 1
            if it % 50 == 0: self._update_weights()
            if no_improve > 100: break
            it += 1; T *= self.alpha

        return best_routes

    def route_to_node_list(self, route: List[RouteNode]) -> List[int]:
        """Route → 纯节点 ID 列表 (给路径规划器)"""
        return [rn.node_id for rn in route]

    def format_route(self, ship, route: List[RouteNode]) -> str:
        """格式化 Route 输出"""
        parts = []
        cur = ship.current_node; total = 0
        for rn in route:
            d = self.rn.dist_matrix[cur, rn.node_id]
            total += d; cur = rn.node_id
            name = self.node_names.get(rn.node_id, f'N{rn.node_id}')
            if rn.action == "PICKUP":
                t = self.tasks.get(rn.task_id)
                parts.append(f"{name}(+{t.payload}t)" if t else f"{name}(PICKUP)")
            elif rn.action == "DELIVERY":
                t = self.tasks.get(rn.task_id)
                parts.append(f"{name}(-{t.payload}t)" if t else f"{name}(DELIVERY)")
            else:
                parts.append(name)
        return f"[{' → '.join(parts)}] {total/1000:.1f}km"

    # ================================================================
    # 代价计算
    # ================================================================

    def _route_cost(self, ship, route: List[RouteNode]) -> float:
        total = 0.0; cur = ship.current_node
        for rn in route:
            total += self.rn.dist_matrix[cur, rn.node_id]
            cur = rn.node_id
        return total

    def _fleet_cost(self, routes: Dict, ships: Dict) -> float:
        total = sum(self._route_cost(ships[sid], seq) for sid, seq in routes.items())
        # load balance penalty
        lens = [len(seq) for seq in routes.values()]
        total += sum(1 for l in lens if l == 0) * 5000
        if lens: total += np.var(lens) * 300
        return total

    # ================================================================
    # 插入逻辑 (PICKUP + DELIVERY 成对)
    # ================================================================

    def _best_insert_pair(self, ship, route: List[RouteNode], task
                           ) -> Tuple[int, int, float]:
        """
        找插入 PICKUP+DELIVERY 对的最佳位置。
        返回 (pickup_pos, delivery_pos, delta_cost)。
        """
        best_pu = 0; best_de = 0; best_delta = float('inf')
        n = len(route)

        for pu_pos in range(n + 1):
            for de_pos in range(pu_pos, n + 1):
                # 构造临时序列
                pu_rn = RouteNode(task.pickup_node, "PICKUP", task.task_id)
                de_rn = RouteNode(task.delivery_node, "DELIVERY", task.task_id)
                new_route = list(route)
                new_route.insert(pu_pos, pu_rn)
                new_route.insert(de_pos + 1, de_rn)  # +1 因为 pu 已插入

                # 约束检查
                if not self._check_single_route(ship, new_route):
                    continue

                delta = self._route_cost(ship, new_route) - self._route_cost(ship, route)
                if delta < best_delta:
                    best_delta = delta; best_pu = pu_pos; best_de = de_pos

        return best_pu, best_de, best_delta

    def _best_insert_positions(self, ship, route: List[RouteNode], task
                                ) -> Tuple[int, int, float]:
        """与 _best_insert_pair 相同, 但接受外部 route 参数"""
        return self._best_insert_pair(ship, route, task)

    # ================================================================
    # 约束验证
    # ================================================================

    def _check_single_route(self, ship, route: List[RouteNode]) -> bool:
        """验证单条 Route 满足所有约束"""
        load = ship.load; cur = ship.current_node
        in_progress = set()  # 已 PICKUP 但未 DELIVERY 的任务

        for rn in route:
            if self.rn.dist_matrix[cur, rn.node_id] == np.inf:
                return False
            cur = rn.node_id

            if rn.action == "PICKUP":
                if rn.task_id in in_progress:
                    return False  # 重复 PICKUP
                t = self.tasks.get(rn.task_id)
                if not t: return False
                load += t.payload
                if load > ship.max_payload:
                    return False
                in_progress.add(rn.task_id)

            elif rn.action == "DELIVERY":
                if rn.task_id not in in_progress:
                    return False  # DELIVERY 前必须 PICKUP
                t = self.tasks.get(rn.task_id)
                if not t: return False
                load -= t.payload
                in_progress.remove(rn.task_id)

        return len(in_progress) == 0  # 所有 PICKUP 都有对应 DELIVERY

    def _validate_all(self, routes: Dict, ships: Dict) -> bool:
        return all(self._check_single_route(ships[sid], seq)
                   for sid, seq in routes.items())

    # ================================================================
    # Destroy / Repair 算子
    # ================================================================

    def _select_destroy(self) -> str:
        total = sum(self.destroy_weights.values())
        r = random.random() * total; cumulative = 0
        for op, w in self.destroy_weights.items():
            cumulative += w
            if r <= cumulative: return op
        return 'random'

    def _select_repair(self) -> str:
        total = sum(self.repair_weights.values())
        r = random.random() * total; cumulative = 0
        for op, w in self.repair_weights.items():
            cumulative += w
            if r <= cumulative: return op
        return 'greedy'

    def _apply_destroy(self, op: str, routes: Dict, ships: Dict, frozen_count: int
                        ) -> Tuple[Dict, List[int]]:
        """Destroy: 移除 PICKUP+DELIVERY 任务对"""
        new_routes = {sid: list(seq) for sid, seq in routes.items()}

        # 收集可移除的任务 (不在冻结区的)
        removable = []
        for sid, seq in new_routes.items():
            count = 0
            for idx, rn in enumerate(seq):
                count += 1
                if rn.action == "PICKUP" and count > frozen_count:
                    removable.append((sid, rn.task_id))

        if not removable:
            return new_routes, []

        if op == 'random':
            n = max(1, int(len(removable) * 0.2))
            chosen = random.sample(removable, min(n, len(removable)))
        elif op == 'worst':
            scored = []
            for sid, tid in removable:
                ship = ships[sid]; task = self.tasks[tid]
                d = (self.rn.dist_matrix[ship.current_node, task.pickup_node] +
                     self.rn.dist_matrix[task.pickup_node, task.delivery_node])
                scored.append((d, sid, tid))
            scored.sort(key=lambda x: -x[0])
            n = max(1, int(len(scored) * 0.2))
            chosen = [(s, t) for _, s, t in scored[:n]]
        elif op == 'shaw':
            if removable:
                ref = random.choice(removable)
                ref_pu = self.tasks[ref[1]].pickup_node
                similar = [(s, t) for s, t in removable
                          if self.tasks[t].pickup_node == ref_pu]
                n = max(1, int(len(similar) * 0.5))
                chosen = similar[:n]
            else:
                chosen = []
        elif op == 'energy':
            scored = [(ships[sid].energy_ratio, sid, tid) for sid, tid in removable]
            scored.sort(); n = max(1, int(len(scored) * 0.2))
            chosen = [(s, t) for _, s, t in scored[:n]]
        else:
            chosen = []

        removed_tasks = []
        for sid, tid in chosen:
            # 删除 PICKUP 和 DELIVERY 两个 RouteNode
            new_routes[sid] = [rn for rn in new_routes[sid] if rn.task_id != tid]
            removed_tasks.append(tid)

        return new_routes, removed_tasks

    def _apply_repair(self, op: str, routes: Dict, ships: Dict,
                       removed: List[int]) -> Dict:
        """Repair: 重新插入 PICKUP+DELIVERY 对"""
        new_routes = {sid: list(seq) for sid, seq in routes.items()}

        if op == 'greedy':
            for tid in removed:
                task = self.tasks[tid]
                best_sid = None; best_pu = -1; best_de = -1; best_delta = float('inf')
                for sid, seq in new_routes.items():
                    pu, de, delta = self._best_insert_pair(ships[sid], seq, task)
                    if delta < best_delta:
                        best_delta = delta; best_sid = sid
                        best_pu = pu; best_de = de
                if best_sid is not None:
                    r = new_routes[best_sid]
                    r.insert(best_pu, RouteNode(task.pickup_node, "PICKUP", tid))
                    r.insert(best_de + 1, RouteNode(task.delivery_node, "DELIVERY", tid))

        elif op == 'regret2':
            remaining = list(removed)
            while remaining:
                best_tid = None; best_sid = None; best_pu = -1; best_de = -1
                best_regret = -1
                for tid in remaining:
                    task = self.tasks[tid]
                    costs = []
                    for sid, seq in new_routes.items():
                        pu, de, delta = self._best_insert_pair(ships[sid], seq, task)
                        costs.append((delta, sid, pu, de))
                    costs.sort()
                    if len(costs) >= 2:
                        regret = costs[1][0] - costs[0][0]
                    elif len(costs) == 1:
                        regret = costs[0][0]
                    else:
                        continue
                    if regret > best_regret:
                        best_regret = regret; best_tid = tid
                        best_sid = costs[0][1]; best_pu = costs[0][2]; best_de = costs[0][3]
                if best_tid is None: break
                task = self.tasks[best_tid]
                r = new_routes[best_sid]
                r.insert(best_pu, RouteNode(task.pickup_node, "PICKUP", best_tid))
                r.insert(best_de + 1, RouteNode(task.delivery_node, "DELIVERY", best_tid))
                remaining.remove(best_tid)

        return new_routes

    def _get_route(self, ship) -> List[RouteNode]:
        """从 ship.task_sequence (Task ID 列表) 转换为 RouteNode 列表"""
        route = []
        for tid in ship.task_sequence:
            t = self.tasks.get(tid)
            if not t: continue
            route.append(RouteNode(t.pickup_node, "PICKUP", tid))
            route.append(RouteNode(t.delivery_node, "DELIVERY", tid))
        return route

    def _update_weights(self):
        for op in self.destroy_weights:
            if self.destroy_count[op] > 0:
                s = self.destroy_scores[op] / self.destroy_count[op]
                self.destroy_weights[op] = max(0.1, self.destroy_weights[op] * 0.7 + s * 0.3)
        for op in self.repair_weights:
            if self.repair_count[op] > 0:
                s = self.repair_scores[op] / self.repair_count[op]
                self.repair_weights[op] = max(0.1, self.repair_weights[op] * 0.7 + s * 0.3)
        self.destroy_scores.clear(); self.repair_scores.clear()
        self.destroy_count.clear(); self.repair_count.clear()

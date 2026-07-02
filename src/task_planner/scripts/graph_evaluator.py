#!/usr/bin/env python3
"""
Graph Evaluator — Ship-Task 匹配评分器

职责: 给定异构图和当前状态, 计算每艘船执行每个任务的匹配评分。
输出 Top-K 候选船给 ALNS。

评分维度:
  1. InsertCost   — 插入该任务到当前队列的增量代价
  2. Feasibility  — 容量/能源/路网可达性
  3. Urgency      — 截止时间紧急度
  4. MatchScore   — 综合 (可后续由 HGT 替代)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class TaskCandidate:
    """一个 Ship-Task 候选对"""
    ship_id: int
    task_id: int
    insert_cost: float       # 插入代价 (米)
    insert_position: int      # 最佳插入位置
    capacity_ok: bool
    energy_ok: bool
    reachable: bool
    urgency: float            # 紧急度
    final_score: float        # 综合评分 (0~1)

    def is_feasible(self) -> bool:
        return self.capacity_ok and self.energy_ok and self.reachable


class GraphEvaluator:
    """规则版 Evaluator, 后续可替换为 HGT"""

    def __init__(self, road_network, node_names: Dict[int, str] = None):
        self.rn = road_network
        self.node_names = node_names or {}

    # ================================================================
    # 公开接口
    # ================================================================

    def evaluate(self, ship, task, current_time: float = 0) -> TaskCandidate:
        """评估单艘船对单个任务"""
        c = TaskCandidate(
            ship_id=ship.ship_id, task_id=task.task_id,
            insert_cost=float('inf'), insert_position=-1,
            capacity_ok=False, energy_ok=False, reachable=False,
            urgency=0.0, final_score=0.0
        )

        # 1. 路网可达性
        d_pickup = self.rn.dist_matrix[ship.current_node, task.pickup_node]
        d_exec = self.rn.dist_matrix[task.pickup_node, task.delivery_node]
        if d_pickup == np.inf or d_exec == np.inf:
            return c
        c.reachable = True

        # 2. 容量
        if task.payload <= ship.remaining_capacity:
            c.capacity_ok = True

        # 3. 能源
        total_dist = d_pickup + d_exec
        energy_need = total_dist / 1000.0 * ship.energy_per_km
        if energy_need <= ship.energy * 0.7:
            c.energy_ok = True

        if not c.is_feasible():
            return c

        # 4. 插入代价: 找最佳插入位置
        best_pos, best_cost = self._best_insert_position(ship, task)
        c.insert_position = best_pos
        c.insert_cost = best_cost

        # 5. 紧急度
        if task.deadline < float('inf'):
            eta = current_time + total_dist / ship.max_speed + 600
            c.urgency = max(0, 1.0 - (task.deadline - eta) / 3600.0)
        else:
            c.urgency = 0.0

        # 6. 综合评分 (规则版)
        cost_score = 1.0 / (1.0 + c.insert_cost / 1000.0)
        # 强负载均衡: 已有任务的船对新任务大幅降低评分, 空船大幅提高
        n = len(ship.task_sequence)
        if n == 0:
            balance_bonus = 0.3  # 空船加分
        else:
            balance_bonus = -0.15 * n  # 每多一个任务扣分
        c.final_score = max(0.0, min(1.0, cost_score + balance_bonus - c.urgency * 0.2))
        c.final_score = max(0.0, min(1.0, c.final_score))

        return c

    def evaluate_all(self, ships, tasks, current_time: float = 0
                      ) -> Dict[Tuple[int, int], TaskCandidate]:
        """评估所有 Ship-Task 对"""
        results = {}
        for sid, ship in ships.items():
            for tid, task in tasks.items():
                if task.status in ('completed', 'cancelled'):
                    continue
                c = self.evaluate(ship, task, current_time)
                if c.is_feasible():
                    results[(sid, tid)] = c
        return results

    def get_top_k(self, task, ships, K: int = 5, current_time: float = 0
                   ) -> List[TaskCandidate]:
        """返回最适合执行此任务的 Top-K 艘船"""
        candidates = []
        for sid, ship in ships.items():
            c = self.evaluate(ship, task, current_time)
            if c.is_feasible():
                candidates.append(c)
        candidates.sort(key=lambda x: -x.final_score)
        return candidates[:K]

    def get_top_k_for_all(self, ships, tasks, K: int = 5, current_time: float = 0
                           ) -> Dict[int, List[TaskCandidate]]:
        """返回每个任务的 Top-K 候选船"""
        result = {}
        for tid in tasks:
            result[tid] = self.get_top_k(tasks[tid], ships, K, current_time)
        return result

    # ================================================================
    # 内部方法
    # ================================================================

    def _best_insert_position(self, ship, task) -> Tuple[int, float]:
        """找任务在船队列中的最佳插入位置, 返回 (位置, 增量代价)"""
        if not ship.task_sequence:
            # 空队列: 直接追加
            d_total = (self.rn.dist_matrix[ship.current_node, task.pickup_node] +
                       self.rn.dist_matrix[task.pickup_node, task.delivery_node])
            return 0, d_total

        best_pos = len(ship.task_sequence)
        # 当前路径代价 (不含此任务)
        current_cost = self._route_cost(ship, ship.task_sequence)
        best_cost = float('inf')

        # 尝试每个插入位置
        for pos in range(len(ship.task_sequence) + 1):
            new_seq = list(ship.task_sequence)
            new_seq.insert(pos, task.task_id)
            new_cost = self._route_cost(ship, new_seq)
            delta = new_cost - current_cost
            if delta < best_cost:
                best_cost = delta
                best_pos = pos

        return best_pos, best_cost

    def _route_cost(self, ship, task_id_list: List[int]) -> float:
        """计算执行一组任务的总航程"""
        total = 0.0
        cur = ship.current_node
        for tid in task_id_list:
            # 需要 task 对象来获取 pickup/delivery
            # 这里简化: 假设 task 可以从外部查询
            total += self.rn.dist_matrix[cur, cur]  # placeholder
        return total

    def _route_cost_with_tasks(self, ship, task_id_list: List[int],
                                 tasks: Dict) -> float:
        """带任务字典的路径代价"""
        total = 0.0
        cur = ship.current_node
        for tid in task_id_list:
            t = tasks.get(tid)
            if not t:
                continue
            total += self.rn.dist_matrix[cur, t.pickup_node]
            total += self.rn.dist_matrix[t.pickup_node, t.delivery_node]
            cur = t.delivery_node
        return total

    def route_cost(self, ship, tasks: Dict) -> float:
        """计算船当前队列的总航程"""
        return self._route_cost_with_tasks(ship, ship.task_sequence, tasks)

    def check_route_constraints(self, ship, task_id_list: List[int],
                                  tasks: Dict) -> Dict:
        """检查约束: 容量/能源/可达性"""
        result = {'capacity_ok': True, 'energy_ok': True, 'reachable': True,
                  'cumulative_load': 0, 'total_energy': 0}
        cur = ship.current_node
        cumulative_load = ship.load

        for tid in task_id_list:
            t = tasks.get(tid)
            if not t:
                continue

            d1 = self.rn.dist_matrix[cur, t.pickup_node]
            d2 = self.rn.dist_matrix[t.pickup_node, t.delivery_node]
            if d1 == np.inf or d2 == np.inf:
                result['reachable'] = False
                break

            result['total_energy'] += (d1 + d2) / 1000.0 * ship.energy_per_km

            # 装货后检查容量
            cumulative_load += t.payload
            if cumulative_load > ship.max_payload:
                result['capacity_ok'] = False

            # 卸货
            cumulative_load -= t.payload

            cur = t.delivery_node

        if result['total_energy'] > ship.energy * 0.8:
            result['energy_ok'] = False

        result['cumulative_load'] = cumulative_load
        return result

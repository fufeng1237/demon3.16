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
                 node_names: Dict = None, config: Optional[Dict] = None):
        self.evaluator = evaluator
        self.tasks = tasks
        self.rn = road_network
        self.node_names = node_names or {}

        # SA 参数 (匹配归一化代价 J≈1.0 的尺度)
        self.max_iter = 1000
        self.T0 = 2.0; self.T_min = 0.001; self.alpha = 0.995
        self.no_improve_limit = 300
        self.K_candidates = 5

        # 消融实验开关。默认值保持既有 Graph + ALNS 行为；这些参数只影响
        # 候选船筛选、权重学习、SA 接受准则及算子集合。
        config = config or {}
        self.use_graph_candidates = config.get('use_graph_candidates', True)
        self.use_adaptive_weights = config.get('use_adaptive_weights', True)
        self.use_sa = config.get('use_sa', True)
        self.destroy_ops = config.get('destroy_ops')
        self.repair_ops = config.get('repair_ops')
        self.graph_candidate_map = config.get('graph_candidate_map', {})
        self.task_order_scores = config.get('task_order_scores', {})
        self.learned_bottleneck = config.get('learned_bottleneck', False)
        # A deterministic local-improvement pass complements the stochastic
        # destroy/repair loop.  It is deliberately small and only accepts a
        # strict makespan improvement, so it cannot trade the primary metric
        # for a cosmetic aggregate-cost gain.
        self.post_optimize_moves = int(config.get('post_optimize_moves', 12))
        self.education_task_limit = int(config.get('education_task_limit', 24))
        self.max_iter = int(config.get('max_iter', self.max_iter))
        self.no_improve_limit = int(config.get('no_improve_limit', self.no_improve_limit))
        self.K_candidates = int(config.get('k_candidates', self.K_candidates))

        # 算子权重
        self.destroy_weights = {'random': 1.0, 'worst': 1.0, 'shaw': 1.0,
                               'energy': 1.0, 'bottleneck': 1.5}
        if self.learned_bottleneck:
            self.destroy_weights['learned_bottleneck'] = 1.8
        self.repair_weights = {'greedy': 1.0, 'regret2': 1.0}
        if self.destroy_ops:
            self.destroy_weights = {op: self.destroy_weights[op] for op in self.destroy_ops}
        if self.repair_ops:
            self.repair_weights = {op: self.repair_weights[op] for op in self.repair_ops}
        self.destroy_scores = defaultdict(float); self.repair_scores = defaultdict(float)
        self.destroy_count = defaultdict(int); self.repair_count = defaultdict(int)

        # ── 多目标代价函数权重 ──
        self.w_distance  = 0.15   # D: 总航行距离
        self.w_makespan  = 0.70   # M: Makespan (第一目标)
        self.w_energy    = 0.08   # E: 总能耗 (载重敏感)
        self.w_balance   = 0.05   # B: 完成时间方差
        self.w_stability = 0.02   # S: Route 稳定性

        # 负载均衡偏置: score = delta + λ × current_ship_time
        self.balance_lambda = 0.50

        # 能耗参数
        self.load_penalty_factor = 0.5     # 满载时能耗 +50%
        self.loading_time   = 300          # 装载时间 (s)
        self.unloading_time = 180          # 卸载时间 (s)

        # ── 归一化基准 & 稳定性状态 ──
        self._norm_base = None       # 固定归一化基准 (首次求解时设定)
        self._prev_routes = None     # 上一轮 Route (Rolling Horizon 稳定性用)

    # ================================================================
    # 公开接口
    # ================================================================

    def build_initial_routes(self, ships: Dict) -> Dict[int, List[RouteNode]]:
        """
        多目标贪心初始解: 每步对所有船比选，score = delta + λ×ship_time。
        负载均衡偏置确保任务不会集中到同一艘船，从源头上控制 makespan。
        """
        ships_copy = {sid: deepcopy(s) for sid, s in ships.items()}
        routes = {sid: [] for sid in ships_copy}

        task_list = [tid for tid in self.tasks
                     if self.tasks[tid].status not in ('completed', 'cancelled')]
        def sort_key(tid):
            t = self.tasks[tid]
            u = t.dynamic_priority if hasattr(t, 'dynamic_priority') else t.priority
            # Learned confidence is a soft ordering hint, never a feasibility rule.
            return -t.payload * u * (1.0 + 0.15 * self.task_order_scores.get(tid, 0.0))
        task_list = sorted(task_list, key=sort_key)

        for tid in task_list:
            task = self.tasks[tid]
            best_sid = None; best_pu = -1; best_de = -1; best_score = float('inf')

            candidate_ids = self._candidate_ship_ids(ships_copy, routes, task)
            for sid in candidate_ids:
                ship = ships_copy[sid]
                c = self.evaluator.evaluate(ship, task)
                if not c.is_feasible():
                    continue
                pu, de, delta = self._best_insert_pair(ship, routes[sid], task)
                if delta == float('inf'):
                    continue
                # score = 插入代价 + 负载均衡偏置 × 当前船完工时间
                ship_time = self._calc_single_ship_time(ship, routes[sid])
                score = delta + self.balance_lambda * ship_time
                if score < best_score:
                    best_score = score; best_sid = sid
                    best_pu = pu; best_de = de

            if best_sid is not None:
                r = routes[best_sid]
                r.insert(best_pu, RouteNode(task.pickup_node, "PICKUP", tid))
                r.insert(best_de + 1, RouteNode(task.delivery_node, "DELIVERY", tid))
                self.tasks[tid].assigned_ship = best_sid

        return routes

    def build_learned_initial_routes(self, ships: Dict, ranked_ships: Dict[int, list],
                                     confidence: Dict[int, float]) -> Dict[int, List[RouteNode]]:
        """Construct a feasible initial solution biased by learned Ship--Task rankings.

        The model chooses an order and preferred first ship; exact insertion,
        precedence, capacity and energy are still enforced by ALNS utilities.
        """
        routes = {sid: [] for sid in ships}
        pending = [tid for tid, task in self.tasks.items()
                   if task.status not in ('completed', 'cancelled')]
        pending.sort(key=lambda tid: (-confidence.get(tid, 0.0),
                                      -self.tasks[tid].payload * self.tasks[tid].priority))
        for tid in pending:
            task = self.tasks[tid]
            preferred = [sid for sid, _ in ranked_ships.get(tid, []) if sid in ships]
            fallback = [sid for sid in ships if sid not in preferred]
            best = None
            # Prefer model ranks; within each candidate use exact route insertion cost.
            for rank, sid in enumerate(preferred + fallback):
                pu, de, delta = self._best_insert_pair(ships[sid], routes[sid], task)
                if delta == float('inf'):
                    continue
                rank_penalty = 0.02 * rank * max(delta, 1.0)
                score = delta + self.balance_lambda * self._calc_single_ship_time(ships[sid], routes[sid]) + rank_penalty
                if best is None or score < best[0]: best = (score, sid, pu, de)
            if best:
                _, sid, pu, de = best
                routes[sid].insert(pu, RouteNode(task.pickup_node, 'PICKUP', tid))
                routes[sid].insert(de + 1, RouteNode(task.delivery_node, 'DELIVERY', tid))
                task.assigned_ship = sid
        return routes

    def optimize(self, ships: Dict, current_routes: Dict[int, List[RouteNode]] = None,
                  frozen_count: int = 0) -> Dict[int, List[RouteNode]]:
        """ALNS 优化 Route (SA) — 层级优化: makespan 优先, 距离/能耗为辅"""
        if current_routes is None:
            current_routes = {sid: list(self._get_route(ships[sid]))
                              for sid in ships}

        # 保存初始解 (用于回退)
        initial_routes = deepcopy(current_routes)
        initial_raw = self._fleet_cost_raw(initial_routes, ships)
        initial_makespan = initial_raw['M']

        # 固定归一化基准: 基于本次优化的初始解
        self._norm_base = {k: max(v, 1) for k, v in initial_raw.items()}

        current_cost = self._fleet_cost(current_routes, ships)
        best_routes = deepcopy(current_routes); best_cost = current_cost
        best_raw = self._fleet_cost_raw(current_routes, ships)
        best_makespan = best_raw['M']
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
            new_raw = self._fleet_cost_raw(repaired, ships)
            delta = new_cost - current_cost

            # Makespan guard: 不接受 makespan 严重退化的解 (主目标保护)
            if new_raw['M'] > best_makespan * 1.03:
                it += 1; T *= self.alpha; continue

            accept = delta < 0
            if not accept and self.use_sa:
                accept = random.random() < np.exp(-delta / T)
            if accept:
                current_routes = repaired; current_cost = new_cost
                self.destroy_scores[d_op] += 1; self.repair_scores[r_op] += 1
                if new_raw['M'] < best_makespan - 1.0:
                    # makespan 改善 → 无条件接受为 best
                    best_routes = deepcopy(repaired); best_cost = new_cost
                    best_raw = new_raw; best_makespan = new_raw['M']
                    self.destroy_scores[d_op] += 3; self.repair_scores[r_op] += 3
                    no_improve = 0
                elif (new_raw['M'] <= best_makespan * 1.01
                      and new_cost < best_cost):
                    # makespan 基本持平 + J 更优 → 接受
                    best_routes = deepcopy(repaired); best_cost = new_cost
                    best_raw = new_raw
                    self.destroy_scores[d_op] += 2; self.repair_scores[r_op] += 2
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            self.destroy_count[d_op] += 1; self.repair_count[r_op] += 1
            if self.use_adaptive_weights and it % 50 == 0:
                self._update_weights()
            if no_improve > self.no_improve_limit: break
            it += 1; T *= self.alpha

        # ── 层级回退: 如果 SA 把 makespan 改差了, 退回初始解 ──
        final_raw = self._fleet_cost_raw(best_routes, ships)
        if final_raw['M'] > initial_makespan + 1.0:
            # makespan 退化了 → 回退到初始解
            chosen_routes = initial_routes
        elif (final_raw['M'] >= initial_makespan - 1.0
              and best_raw['M'] >= initial_makespan - 1.0):
            # makespan 没显著改善 → 比较综合代价, 选更优者
            initial_cost = self._fleet_cost(initial_routes, ships)
            if initial_cost <= best_cost:
                chosen_routes = initial_routes
            else:
                chosen_routes = best_routes
        else:
            chosen_routes = best_routes
        # The educational neighbourhood is a static-improvement stage.  A
        # rolling caller with a frozen execution prefix keeps that prefix
        # untouched and uses the regular prefix-aware destroy/repair loop.
        if frozen_count > 0:
            return chosen_routes
        return self._polish_makespan(chosen_routes, ships)

    def _polish_makespan(self, routes: Dict, ships: Dict) -> Dict:
        """Memetic-style education after ALNS.

        It searches exact task-pair reinsertion, bottleneck-to-other-ship
        relocation and cross-ship exchange.  Moves are accepted
        lexicographically (makespan, distance, energy) and are fully checked,
        and is therefore safe for the initial static allocation stage.
        """
        best = deepcopy(routes)

        def limited_task_ids(seq):
            ids = [rn.task_id for rn in seq if rn.action == 'PICKUP']
            # Full neighbourhood on small instances; bounded deterministic
            # neighbourhood on large instances keeps 80+ task planning usable.
            if len(ids) > self.education_task_limit:
                ids.sort(key=lambda tid: self.tasks[tid].payload * self.tasks[tid].priority,
                         reverse=True)
                return ids[:self.education_task_limit]
            return ids

        def static_key(candidate):
            raw = self._fleet_cost_raw(candidate, ships)
            return raw['M'], raw['D'], raw['E']

        for _ in range(max(self.post_optimize_moves, 0)):
            before_key = static_key(best)
            source_sid = max(best, key=lambda sid: self._calc_single_ship_time(ships[sid], best[sid]))
            task_ids = limited_task_ids(best[source_sid])
            best_move = None
            best_key = before_key

            # Intra-route task-pair reinsertion is the inexpensive "education"
            # operator missing from the former ALNS tail.  It can improve the
            # route ordering even when changing vessel is not beneficial.
            for sid, seq in best.items():
                for tid in limited_task_ids(seq):
                    reduced = {s: list(route) for s, route in best.items()}
                    reduced[sid] = [rn for rn in reduced[sid] if rn.task_id != tid]
                    task = self.tasks[tid]
                    pu, de, delta = self._best_insert_pair(ships[sid], reduced[sid], task)
                    if delta == float('inf'):
                        continue
                    trial = {s: list(route) for s, route in reduced.items()}
                    trial[sid].insert(pu, RouteNode(task.pickup_node, 'PICKUP', tid))
                    trial[sid].insert(de + 1, RouteNode(task.delivery_node, 'DELIVERY', tid))
                    candidate_key = static_key(trial)
                    if candidate_key < best_key:
                        best_key, best_move = candidate_key, trial

            for tid in task_ids:
                reduced = {sid: list(seq) for sid, seq in best.items()}
                reduced[source_sid] = [rn for rn in reduced[source_sid] if rn.task_id != tid]
                task = self.tasks[tid]
                for target_sid, target_route in reduced.items():
                    if target_sid == source_sid:
                        continue
                    pu, de, delta = self._best_insert_pair(ships[target_sid], target_route, task)
                    if delta == float('inf'):
                        continue
                    trial = {sid: list(seq) for sid, seq in reduced.items()}
                    trial[target_sid].insert(pu, RouteNode(task.pickup_node, 'PICKUP', tid))
                    trial[target_sid].insert(de + 1, RouteNode(task.delivery_node, 'DELIVERY', tid))
                    if not self._validate_all(trial, ships):
                        continue
                    candidate_key = static_key(trial)
                    if candidate_key < best_key:
                        best_key, best_move = candidate_key, trial

            # A pure relocation may be blocked by capacity or by the receiving
            # ship becoming the new bottleneck.  Explore a paired exchange in
            # that case: remove one task from each ship, then reinsert both at
            # their exact best feasible positions.
            for tid_a in task_ids:
                for target_sid, target_seq in best.items():
                    if target_sid == source_sid:
                        continue
                    target_tasks = limited_task_ids(target_seq)
                    for tid_b in target_tasks:
                        reduced = {sid: list(seq) for sid, seq in best.items()}
                        reduced[source_sid] = [rn for rn in reduced[source_sid] if rn.task_id != tid_a]
                        reduced[target_sid] = [rn for rn in reduced[target_sid] if rn.task_id != tid_b]
                        task_a, task_b = self.tasks[tid_a], self.tasks[tid_b]
                        pu_b, de_b, cost_b = self._best_insert_pair(
                            ships[source_sid], reduced[source_sid], task_b)
                        if cost_b == float('inf'):
                            continue
                        trial = {sid: list(seq) for sid, seq in reduced.items()}
                        trial[source_sid].insert(pu_b, RouteNode(task_b.pickup_node, 'PICKUP', tid_b))
                        trial[source_sid].insert(de_b + 1, RouteNode(task_b.delivery_node, 'DELIVERY', tid_b))
                        pu_a, de_a, cost_a = self._best_insert_pair(
                            ships[target_sid], trial[target_sid], task_a)
                        if cost_a == float('inf'):
                            continue
                        trial[target_sid].insert(pu_a, RouteNode(task_a.pickup_node, 'PICKUP', tid_a))
                        trial[target_sid].insert(de_a + 1, RouteNode(task_a.delivery_node, 'DELIVERY', tid_a))
                        if not self._validate_all(trial, ships):
                            continue
                        candidate_key = static_key(trial)
                        if candidate_key < best_key:
                            best_key, best_move = candidate_key, trial
            if best_move is None:
                break
            best = best_move
        return best

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
    # 代价计算 — 多目标: D(距离) + M(Makespan) + E(能耗) + B(均衡) + S(稳定性)
    # ================================================================

    def _route_cost(self, ship, route: List[RouteNode]) -> float:
        """单船距离 (被 _best_insert_pair 用作快速 delta 估算)"""
        total = 0.0; cur = ship.current_node
        for rn in route:
            total += self.rn.dist_matrix[cur, rn.node_id]
            cur = rn.node_id
        return total

    # ── 五项原始指标 ──

    def _calc_single_ship_time(self, ship, route: List[RouteNode]) -> float:
        """单船完工时间 (s). 空船返回 0."""
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

    def _calc_distance(self, routes: Dict, ships: Dict) -> float:
        """D: 总航行距离 (m)"""
        return sum(self._route_cost(ships[sid], seq) for sid, seq in routes.items())

    def _calc_makespan(self, routes: Dict, ships: Dict) -> float:
        """M: 最晚完成时间 (s). 空船不计."""
        max_time = 0.0
        for sid, seq in routes.items():
            ship = ships[sid]
            if not seq:
                continue
            t = 0.0; cur = ship.current_node
            for rn in seq:
                d = self.rn.dist_matrix[cur][rn.node_id]
                t += d / max(ship.max_speed, 1.0)
                if rn.action == 'PICKUP':
                    t += self.loading_time
                elif rn.action == 'DELIVERY':
                    t += self.unloading_time
                cur = rn.node_id
            max_time = max(max_time, t)
        return max_time

    def _calc_energy(self, routes: Dict, ships: Dict) -> float:
        """E: 总能耗 (kWh), 载重越高单位距离耗能越多"""
        total = 0.0
        for sid, seq in routes.items():
            ship = ships[sid]
            cur = ship.current_node; load = ship.load
            for rn in seq:
                d = self.rn.dist_matrix[cur][rn.node_id]
                # 能耗 = 距离(km) × 单位能耗 × (1 + α×载重率)
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

    def _calc_balance(self, routes: Dict, ships: Dict) -> float:
        """B: 各船完成时间的方差 (s²). 越小越均衡."""
        times = []
        for sid, seq in routes.items():
            ship = ships[sid]
            if not seq:
                times.append(0.0)
                continue
            t = 0.0; cur = ship.current_node
            for rn in seq:
                d = self.rn.dist_matrix[cur][rn.node_id]
                t += d / max(ship.max_speed, 1.0)
                if rn.action == 'PICKUP':
                    t += self.loading_time
                elif rn.action == 'DELIVERY':
                    t += self.unloading_time
                cur = rn.node_id
            times.append(t)
        if not times:
            return 0.0
        return float(np.var(times))

    def _calc_stability(self, routes: Dict, ships: Dict) -> float:
        """S: Route 变化率 (0~1). 仅在 Rolling Horizon 时非零."""
        if self._prev_routes is None:
            return 0.0
        changes = 0; total = 0
        for sid, seq in routes.items():
            prev = self._prev_routes.get(sid, [])
            new_tids = {rn.task_id for rn in seq if rn.action == 'PICKUP'}
            old_tids = {rn.task_id for rn in prev if rn.action == 'PICKUP'}
            total += max(len(new_tids | old_tids), 1)
            changes += len(new_tids ^ old_tids)  # 换了船或新增/移除
        if total == 0:
            return 0.0
        return changes / total

    # ── 原始指标字典 ──

    def _fleet_cost_raw(self, routes: Dict, ships: Dict) -> Dict[str, float]:
        """返回五项原始指标 (用于初始化归一化基准)"""
        return {
            'D': self._calc_distance(routes, ships),
            'M': self._calc_makespan(routes, ships),
            'E': self._calc_energy(routes, ships),
            'B': self._calc_balance(routes, ships),
            'S': self._calc_stability(routes, ships),
        }

    # ── 归一化综合代价 (ALNS 优化目标) ──

    def _fleet_cost(self, routes: Dict, ships: Dict) -> float:
        """
        综合代价 J = w1*D_norm + w2*M_norm + w3*E_norm + w4*B_norm + w5*S_norm + idle_penalty
        归一化基准固定为初始解的指标值, 保证 SA 过程中 cost 可比.
        """
        raw = self._fleet_cost_raw(routes, ships)
        base = self._norm_base if self._norm_base else {k: max(v, 1) for k, v in raw.items()}

        D_norm = raw['D'] / base['D']
        M_norm = raw['M'] / max(base['M'], 1e-6)
        E_norm = raw['E'] / max(base['E'], 1e-6)
        B_norm = raw['B'] / max(base['B'], 1e-6)
        S_norm = raw['S']  # 已是比例

        J = (self.w_distance  * D_norm +
             self.w_makespan  * M_norm +
             self.w_energy    * E_norm +
             self.w_balance   * B_norm +
             self.w_stability * S_norm)

        # 空闲船重罚: 每艘 +0.5 (J 基准≈1.0, 防止 ALNS 把任务集中到少数船)
        idle_count = sum(1 for seq in routes.values() if len(seq) == 0)
        J += idle_count * 0.5

        return J

    def get_cost_breakdown(self, routes: Dict, ships: Dict) -> Dict:
        """调试用: 返回各项指标的原始值和归一化值"""
        raw = self._fleet_cost_raw(routes, ships)
        base = self._norm_base if self._norm_base else {k: max(v, 1) for k, v in raw.items()}
        idle_count = sum(1 for seq in routes.values() if len(seq) == 0)
        return {
            'D_raw': raw['D'], 'D_norm': raw['D'] / base['D'],
            'M_raw': raw['M'], 'M_norm': raw['M'] / max(base['M'], 1e-6),
            'E_raw': raw['E'], 'E_norm': raw['E'] / max(base['E'], 1e-6),
            'B_raw': raw['B'], 'B_norm': raw['B'] / max(base['B'], 1e-6),
            'S_raw': raw['S'],
            'idle_count': idle_count, 'idle_penalty': idle_count * 0.5,
            'J_total': self._fleet_cost(routes, ships),
        }

    # ================================================================
    # 插入逻辑 (PICKUP + DELIVERY 成对)
    # ================================================================

    def _best_insert_pair(self, ship, route: List[RouteNode], task
                           ) -> Tuple[int, int, float]:
        """
        O(n²) 多目标插入: 同时考虑距离、完工时间和能耗。
        返回 (pickup_pos, delivery_pos, weighted_delta)。

        delta = w_d × Δtravel_time + w_m × Δtotal_time + w_e × Δenergy_time_equiv
        所有分量统一为"等效时间 (秒)"，使权重直接控制 trade-off。
        """
        best_pu = 0; best_de = 0; best_delta = float('inf')
        n = len(route)
        pu_node = task.pickup_node
        de_node = task.delivery_node
        payload = task.payload
        capacity = ship.max_payload
        speed = max(ship.max_speed, 1.0)
        epk = ship.energy_per_km
        rn_dist = self.rn.dist_matrix

        # ── 预计算 ──
        nodes = [ship.current_node] + [rn.node_id for rn in route]
        edge_d = [rn_dist[nodes[i]][nodes[i+1]] for i in range(n)]

        cum_load = [ship.load]
        cur_load = ship.load
        for rnode in route:
            if rnode.action == 'PICKUP':
                t = self.tasks.get(rnode.task_id)
                if t: cur_load += t.payload
            elif rnode.action == 'DELIVERY':
                t = self.tasks.get(rnode.task_id)
                if t: cur_load -= t.payload
            cum_load.append(cur_load)

        for pu_pos in range(n + 1):
            prev_pu = nodes[pu_pos]
            nxt_pu = nodes[pu_pos + 1] if pu_pos < n else None
            if rn_dist[prev_pu][pu_node] == np.inf:
                continue
            if nxt_pu is not None and rn_dist[pu_node][nxt_pu] == np.inf:
                continue

            # ── PICKUP 距离增量 O(1) ──
            if pu_pos < n:
                delta_d_pu = (rn_dist[prev_pu][pu_node]
                              + rn_dist[pu_node][nxt_pu] - edge_d[pu_pos])
            else:
                delta_d_pu = rn_dist[prev_pu][pu_node]

            load_after_pu = cum_load[pu_pos] + payload
            if load_after_pu > capacity:
                continue

            max_load = load_after_pu
            for de_pos in range(pu_pos, n + 1):
                if de_pos > pu_pos:
                    max_load = max(max_load, cum_load[de_pos] + payload)
                if max_load > capacity:
                    break  # 容量超额, 继续右移只会更大

                # ── DELIVERY 距离增量 O(1) ──
                prev_de = pu_node if de_pos == pu_pos else nodes[de_pos]
                nxt_de = nodes[de_pos + 1] if de_pos < n else None
                if rn_dist[prev_de][de_node] == np.inf:
                    continue
                if nxt_de is not None and rn_dist[de_node][nxt_de] == np.inf:
                    continue

                if de_pos == pu_pos == n:
                    delta_d_de = rn_dist[pu_node][de_node]
                elif de_pos == pu_pos:
                    delta_d_de = (rn_dist[pu_node][de_node]
                                  + rn_dist[de_node][nxt_pu]
                                  - rn_dist[pu_node][nxt_pu])
                elif de_pos < n:
                    delta_d_de = (rn_dist[prev_de][de_node]
                                  + rn_dist[de_node][nxt_de] - edge_d[de_pos])
                else:
                    delta_d_de = rn_dist[prev_de][de_node]

                delta_d = delta_d_pu + delta_d_de

                # ── 多目标 delta (统一为"等效时间"秒) ──
                travel_time = delta_d / speed
                total_time = travel_time + self.loading_time + self.unloading_time

                avg_load = cum_load[pu_pos] + payload / 2.0
                load_ratio = 1.0 + self.load_penalty_factor * avg_load / max(capacity, 1)
                energy_kwh = (delta_d / 1000.0) * epk * load_ratio
                # 1 kWh → 船可航行 1000/(epk×speed) 秒 → 等效时间
                energy_time_equiv = energy_kwh * 1000.0 / (epk * speed) if epk > 0 else 0.0

                delta = (self.w_distance  * travel_time +
                         self.w_makespan  * total_time +
                         self.w_energy    * energy_time_equiv)

                # Full-route energy feasibility: an individually feasible task
                # must not make the accumulated route exceed the safety reserve.
                candidate = list(route)
                candidate.insert(pu_pos, RouteNode(task.pickup_node, "PICKUP", task.task_id))
                candidate.insert(de_pos + 1, RouteNode(task.delivery_node, "DELIVERY", task.task_id))
                if not self._energy_feasible(ship, candidate):
                    continue

                if delta < best_delta:
                    best_delta = delta
                    best_pu = pu_pos
                    best_de = de_pos

        return best_pu, best_de, best_delta

    def _energy_feasible(self, ship, route: List[RouteNode]) -> bool:
        """Check cumulative load-sensitive energy with a 10% emergency reserve."""
        used, load, cur = 0.0, ship.load, ship.current_node
        for rn in route:
            d = self.rn.dist_matrix[cur, rn.node_id]
            if np.isinf(d):
                return False
            ratio = 1.0 + self.load_penalty_factor * load / max(ship.max_payload, 1)
            used += d / 1000.0 * ship.energy_per_km * ratio
            if rn.action == 'PICKUP':
                t = self.tasks.get(rn.task_id)
                if t: load += t.payload
            elif rn.action == 'DELIVERY':
                t = self.tasks.get(rn.task_id)
                if t: load -= t.payload
            cur = rn.node_id
        return used <= ship.energy * 0.90 + 1e-6

    def _best_insert_positions(self, ship, route: List[RouteNode], task
                                ) -> Tuple[int, int, float]:
        """与 _best_insert_pair 相同, 但接受外部 route 参数"""
        return self._best_insert_pair(ship, route, task)

    def _candidate_ship_ids(self, ships: Dict, routes: Dict,
                            task) -> List[int]:
        """以实际路网最短路构造 Ship-Task Top-K 候选集。

        分数使用“当前路线末端→取货点→送货点”的路网距离，先排除不可达和
        单任务容量不满足的船。关闭开关时返回全部船，作为 plain-ALNS 消融项。
        """
        graph_candidates = self.graph_candidate_map.get(task.task_id)
        if self.use_graph_candidates and graph_candidates:
            return [sid for sid in graph_candidates if sid in ships]
        candidates = []
        for sid, ship in ships.items():
            if task.payload > ship.max_payload:
                continue
            seq = routes.get(sid, [])
            start = seq[-1].node_id if seq else ship.current_node
            d1 = self.rn.dist_matrix[start, task.pickup_node]
            d2 = self.rn.dist_matrix[task.pickup_node, task.delivery_node]
            if np.isinf(d1) or np.isinf(d2):
                continue
            candidates.append((float(d1 + d2), sid))
        candidates.sort()
        if self.use_graph_candidates:
            return [sid for _, sid in candidates[:self.K_candidates]]
        return [sid for _, sid in candidates]

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

        return len(in_progress) == 0 and self._energy_feasible(ship, route)

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
        elif op == 'bottleneck':
            # 瞄准 makespan 最大的瓶颈船, 移除其任务
            ship_times = {}
            for sid, seq in new_routes.items():
                if not seq:
                    ship_times[sid] = 0.0
                    continue
                ship = ships[sid]
                t = 0.0; cur = ship.current_node
                for rn in seq:
                    d = self.rn.dist_matrix[cur][rn.node_id]
                    t += d / max(ship.max_speed, 1.0)
                    if rn.action == 'PICKUP': t += self.loading_time
                    elif rn.action == 'DELIVERY': t += self.unloading_time
                    cur = rn.node_id
                ship_times[sid] = t
            bottleneck_sid = max(ship_times, key=ship_times.get)
            # 只从瓶颈船上移除任务
            bottleneck_removable = [(s, t) for s, t in removable if s == bottleneck_sid]
            if bottleneck_removable:
                n = max(1, int(len(bottleneck_removable) * 0.3))
                chosen = random.sample(bottleneck_removable, min(n, len(bottleneck_removable)))
            else:
                chosen = []
        elif op == 'learned_bottleneck':
            # Learn-guided counterpart of bottleneck destroy: retain confident
            # assignments and release low-confidence tasks from the slowest ship.
            ship_times = {sid: self._calc_single_ship_time(ships[sid], seq)
                          for sid, seq in new_routes.items()}
            bottleneck_sid = max(ship_times, key=ship_times.get)
            candidates = [(self.task_order_scores.get(tid, 0.0), bottleneck_sid, tid)
                          for sid, tid in removable if sid == bottleneck_sid]
            candidates.sort(key=lambda x: x[0])
            n = max(1, int(len(candidates) * 0.3))
            chosen = [(sid, tid) for _, sid, tid in candidates[:n]]
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
                best_sid = None; best_pu = -1; best_de = -1; best_score = float('inf')
                for sid in self._candidate_ship_ids(ships, new_routes, task):
                    seq = new_routes[sid]
                    pu, de, delta = self._best_insert_pair(ships[sid], seq, task)
                    if delta == float('inf'):
                        continue
                    ship_time = self._calc_single_ship_time(ships[sid], seq)
                    score = delta + self.balance_lambda * ship_time
                    if score < best_score:
                        best_score = score; best_sid = sid
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
                    for sid in self._candidate_ship_ids(ships, new_routes, task):
                        seq = new_routes[sid]
                        pu, de, delta = self._best_insert_pair(ships[sid], seq, task)
                        if delta == float('inf'):
                            continue
                        ship_time = self._calc_single_ship_time(ships[sid], seq)
                        score = delta + self.balance_lambda * ship_time
                        costs.append((score, sid, pu, de))
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

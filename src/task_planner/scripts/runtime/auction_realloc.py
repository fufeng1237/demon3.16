#!/usr/bin/env python3
"""
拍卖法动态重分配

原理:
  - 任务 = 物品 (每个物品有当前价格)
  - 船 = 竞拍者 (选择收益最高的任务出价)
  - profit(ship, task) = -cost(ship, task) - price(task)
  - 每轮: 船选最佳任务, 出价 = (best - second_best) + ε
  - 收敛: 价格稳定, 分配不变

优势:
  - 增量: 只重拍受影响的任务 (不是全部)
  - 自动转移: 价格机制自动处理任务在船间移动
  - 无需显式"释放": 价格高了自然会被其他船竞得
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict


def auction_reallocate(ships, tasks, hetero_graph,
                        fixed_ships: Set[int] = None,
                        fixed_tasks: Set[int] = None,
                        epsilon: float = 0.01,
                        max_iterations: int = 50,
                        current_time: float = 0) -> Tuple[List, Dict, int]:
    """
    拍卖法重分配。

    Args:
        ships: Dict[int, ShipRuntime]
        tasks: Dict[int, TaskRuntime]
        hetero_graph: HeteroGraph
        fixed_ships: 不参与竞拍的船 ID (如正在装卸)
        fixed_tasks: 不可重分配的任务 ID (如在途)
        epsilon: 近似精度
        max_iterations: 最大迭代
        current_time: 当前时间

    Returns:
        new_assignments: [(ship_id, task_id, bid), ...]
        final_prices: Dict[int, float]
        rounds: 收敛需要的轮数
    """
    g = hetero_graph
    M, K = g.M, g.K

    if K == 0:
        return [], {}, 0

    fixed_ships = fixed_ships or set()
    fixed_tasks = fixed_tasks or set()

    active_ships = [i for i in range(M) if g.ship_ids[i] not in fixed_ships]
    active_tasks = [j for j in range(K) if g.task_ids[j] not in fixed_tasks]

    if not active_ships or not active_tasks:
        return [], {}, 0

    prices = np.zeros(K)
    for j in range(K):
        tid = g.task_ids[j]
        task = tasks[tid]
        if task.assigned_ship >= 0 and task.assigned_ship not in fixed_ships:
            prices[j] = 0.1

    assigned_to = {}

    # 从 Ship→Task 边直接读代价
    cost_cache = np.full((M, K), np.inf)
    for k in range(g.st_edges.shape[1]):
        si, tj = g.st_edges[0, k], g.st_edges[1, k]
        cost_cache[si, tj] = g.st_feat[k, 1]  # estimated_cost

    def compute_profits(ship_i):
        """返回 (best_task_j, best_profit, second_best_profit)"""
        best_j = -1
        best_p = -float('inf')
        second_p = -float('inf')
        for j in active_tasks:
            c = cost_cache[ship_i, j]
            if c == float('inf'):
                continue
            p = -c - prices[j]
            if p > best_p:
                second_p = best_p
                best_p = p
                best_j = j
            elif p > second_p:
                second_p = p
        return best_j, best_p, second_p

    # ======= 拍卖迭代 =======
    rounds = 0
    for rounds in range(1, max_iterations + 1):
        changed = False

        for i in active_ships:
            # 找出最佳和次佳任务
            best_j, best_p, second_p = compute_profits(i)

            if best_j < 0:
                continue

            # 计算出价
            if second_p == -float('inf'):
                bid = epsilon
            else:
                bid = max(epsilon, best_p - second_p + epsilon)

            # 旧所有者
            prev_i = assigned_to.get(best_j)
            if prev_i is not None and prev_i == i:
                continue  # 已经归我

            # 更新价格
            prices[best_j] += bid

            # 转让
            assigned_to[best_j] = i
            changed = True

        if not changed:
            break

    # ======= 构建结果 =======
    new_assignments = []
    for j, i in assigned_to.items():
        sid = g.ship_ids[i]
        tid = g.task_ids[j]
        new_assignments.append((sid, tid, prices[j]))

    return new_assignments, {g.task_ids[j]: p for j, p in enumerate(prices)}, rounds


def apply_auction_result(ships, tasks, new_assignments,
                          fixed_tasks: Set[int] = None):
    """
    应用拍卖结果: 更新 ships.task_sequence 和 tasks.assigned_ship。
    返回变更列表。
    """
    fixed_tasks = fixed_tasks or set()
    changes = []
    task_map = {t.task_id: t for t in tasks.values()}

    # 构建新分配映射
    new_owner = {}
    for sid, tid, _ in new_assignments:
        new_owner[tid] = sid

    # 从旧船移除被转移的任务
    for tid, sid in new_owner.items():
        task = tasks.get(tid)
        if not task or tid in fixed_tasks:
            continue

        old_ship_id = task.assigned_ship
        if old_ship_id >= 0 and old_ship_id != sid:
            old_ship = ships.get(old_ship_id)
            if old_ship and tid in old_ship.task_sequence:
                old_ship.task_sequence.remove(tid)
                old_ship.load = max(0, old_ship.load - task.payload)
                changes.append({
                    "type": "transfer",
                    "task": tid, "from": old_ship_id, "to": sid,
                    "reason": "auction"
                })

        # 分配给新船
        task.assigned_ship = sid
        task.status = "assigned"
        ship = ships.get(sid)
        if ship and tid not in ship.task_sequence:
            ship.task_sequence.append(tid)
            ship.load += task.payload

    return changes


def build_fixed_sets(ships, tasks) -> Tuple[Set[int], Set[int]]:
    """
    构建不可重分配的集合:
      fixed_ships: 正在装卸/在途的船
      fixed_tasks: 正在执行中的任务
    """
    fixed_ships = set()
    fixed_tasks = set()

    for sid, ship in ships.items():
        if ship.current_phase in ("loading", "unloading", "navigate_to_delivery"):
            fixed_ships.add(sid)
            if ship.current_task_id >= 0:
                fixed_tasks.add(ship.current_task_id)

    return fixed_ships, fixed_tasks


def compute_reallocation_summary(old_assignments, new_assignments, ships):
    """重分配总结: 哪些任务转移了"""
    old_map = {tid: sid for sid, tid, _ in old_assignments}
    new_map = {tid: sid for sid, tid, _ in new_assignments}

    transferred = []
    for tid in new_map:
        if tid in old_map and old_map[tid] != new_map[tid]:
            transferred.append({
                "task": tid,
                "from": old_map[tid],
                "to": new_map[tid]
            })

    return {
        "total_tasks": len(new_assignments),
        "transferred": len(transferred),
        "details": transferred
    }

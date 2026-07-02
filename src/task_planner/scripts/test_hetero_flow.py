#!/usr/bin/env python3
"""
异构路网 + 最小费用流 + 拍卖重分配 — 完整测试
"""

import sys, os, numpy as np
from collections import defaultdict
from copy import deepcopy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from road_network import load_road_network
from hetero_graph import build_hetero_graph
from min_cost_flow import min_cost_flow_allocate, optimize_sequence
from auction_realloc import (auction_reallocate, apply_auction_result,
                               build_fixed_sets, compute_reallocation_summary)
from real_time_scheduler import (RealTimeScheduler, ShipRuntime, TaskRuntime)


def main():
    # ── 1. 加载路网 ──
    rn = load_road_network(
        os.path.join(os.path.dirname(__file__), "../output/road_network.json"),
        ports_config=os.path.join(os.path.dirname(__file__), "../config/ports.yaml"))

    port_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_port])
    gas_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_gas_station])
    node_names = {nid: (n.port_name if n.port_name else f"N{nid}")
                  for nid, n in rn.nodes.items()}

    print(f"路网: {len(rn.nodes)}节点, {len(rn.edges)}边")
    print(f"港口: {[node_names[p] for p in port_ids]}")

    # ── 2. 创建船和任务 ──
    sched = RealTimeScheduler(rn, port_ids, gas_ids, node_names)
    sched.add_ship(0, "Ship_0", 2000, 500, 8.0, port_ids[0])
    sched.add_ship(1, "Ship_1", 1500, 400, 7.5, port_ids[5])
    sched.add_ship(2, "Ship_2", 2500, 600, 7.0, port_ids[9])
    sched.add_ship(3, "Ship_3", 1800, 450, 8.5, port_ids[3])

    tasks_added = 0
    np.random.seed(42)
    for i, pu in enumerate(port_ids):
        for j, de in enumerate(port_ids):
            if i == j: continue
            d = rn.dist_matrix[pu, de]
            if d <= 0 or d == np.inf: continue
            dl = float('inf')
            if tasks_added < 5: dl = 3600 * 2  # 前5个任务2小时截止
            sched.add_task(tasks_added, pu, de,
                           float(np.random.choice([300, 500, 800, 1000, 1500])),
                           int(np.random.choice([1, 2, 3], p=[0.2, 0.4, 0.4])),
                           dl)
            tasks_added += 1
            if tasks_added >= 20: break
        if tasks_added >= 20: break

    print(f"船: {len(sched.ships)}, 任务: {len(sched.tasks)}")

    # ── 3. 构建异构图 ──
    g = build_hetero_graph(sched.ships, sched.tasks, rn)
    print(f"\n异构图: Ship={g.M}, Task={g.K}, Road={g.N}")
    print(f"  Road edges: {g.road_edges.shape[1]}")
    print(f"  预计算距离矩阵: {g.roadmap_dist.shape}")

    # ── 4. 最小费用流分配 ──
    print(f"\n{'='*60}")
    print("  Phase 1: 最小费用流初始分配")
    print(f"{'='*60}")

    assignments, total_cost, flow = min_cost_flow_allocate(
        sched.ships, sched.tasks, g, current_time=sched.current_time)

    # 优化每艘船的执行顺序
    for ship in sched.ships.values():
        optimize_sequence(ship, sched.tasks, rn.dist_matrix)

    print(f"  结果: 分配 {flow}/{g.K} 任务, 总代价 {total_cost:.0f}")
    print(f"  含义: 总代价越小 = 分配越好 (航行距离 + 时间惩罚的综合)\n")

    for ship in sorted(sched.ships.values(), key=lambda s: s.ship_id):
        if ship.task_sequence:
            tasks_str = " → ".join(f"T{tid}" for tid in ship.task_sequence)
            total_d = sum(rn.dist_matrix[sched.tasks[tid].pickup_node,
                                          sched.tasks[tid].delivery_node]
                         for tid in ship.task_sequence if tid in sched.tasks)
            n = node_names.get(ship.current_node, "?")
            print(f"  {ship.name} @{n}: [{tasks_str}]  {total_d/1000:.1f}km")

    # ── 5. 对比: 贪心 vs 最小费用流 ──
    print(f"\n{'='*60}")
    print("  对比: 贪心 vs 最小费用流")
    print(f"{'='*60}")

    # 重置然后用贪心分配
    for ship in sched.ships.values():
        ship.task_sequence.clear()
        ship.load = 0
    for task in sched.tasks.values():
        task.assigned_ship = -1
        task.status = "pending"

    sched.initial_allocate()

    greedy_cost = 0
    for ship in sched.ships.values():
        for tid in ship.task_sequence:
            t = sched.tasks[tid]
            greedy_cost += (rn.dist_matrix[ship.current_node, t.pickup_node] +
                            rn.dist_matrix[t.pickup_node, t.delivery_node])

    print(f"  贪心代价: {greedy_cost:.0f}   |  最小费用流代价: {total_cost:.0f}")
    if total_cost < greedy_cost:
        print(f"  最小费用流优化: {(1 - total_cost/greedy_cost)*100:.1f}% 更好")
    else:
        print(f"  贪心更好 (小规模时差异不大)")

    # ── 用回最小费用流结果继续测试 ──
    for ship in sched.ships.values():
        ship.task_sequence.clear()
        ship.load = 0
    for task in sched.tasks.values():
        task.assigned_ship = -1
        task.status = "pending"
    min_cost_flow_allocate(sched.ships, sched.tasks, g)
    for ship in sched.ships.values():
        optimize_sequence(ship, sched.tasks, rn.dist_matrix)

    print(f"\n  负载均衡后的分配:")
    for ship in sorted(sched.ships.values(), key=lambda s: s.ship_id):
        total_d = sum(rn.dist_matrix[sched.tasks[tid].pickup_node,
                                      sched.tasks[tid].delivery_node]
                     for tid in ship.task_sequence if tid in sched.tasks)
        print(f"  {ship.name}: {len(ship.task_sequence)}任务, {total_d/1000:.1f}km")

    # ── 6. 拍卖重分配: 不运行时间, 直接模拟场景 ──
    print(f"\n{'='*60}")
    print("  Phase 2: 拍卖重分配 (Ship_0 能源紧急)")
    print(f"{'='*60}")

    # 场景: Ship_0 energy=6%, Ship_1 空闲, Ship_2/Ship_3 正常
    sched.ships[0].energy = sched.ships[0].max_energy * 0.06
    sched.ships[1].current_phase = "pending"  # 空闲
    sched.ships[1].task_sequence.clear()

    before_tasks = {sid: list(s.task_sequence) for sid, s in sched.ships.items()}

    # Ship_0 所有任务都设为 pending (还没开始) — 让拍卖可以转移
    for tid in sched.ships[0].task_sequence:
        if tid in sched.tasks:
            sched.tasks[tid].status = "pending"

    # 重建异构图
    g2 = build_hetero_graph(sched.ships, sched.tasks, rn)

    # 固定集合: 无 (所有任务都是 pending, 可以重分配)
    fixed_ships = set()
    fixed_tasks = set()

    new_assignments, prices, rounds = auction_reallocate(
        sched.ships, sched.tasks, g2,
        fixed_ships=fixed_ships, fixed_tasks=fixed_tasks,
        epsilon=1.0, current_time=sched.current_time)

    apply_auction_result(sched.ships, sched.tasks, new_assignments, fixed_tasks)
    after_tasks = {sid: list(s.task_sequence) for sid, s in sched.ships.items()}

    print(f"\n  拍卖结果 (ε=1.0, {rounds}轮收敛):")
    print(f"  前: { {k: v for k, v in before_tasks.items()} }")
    print(f"  后: { {k: v for k, v in after_tasks.items()} }")

    # 找出转移的任务
    transferred = []
    for tid in sched.tasks:
        old_ship = None
        for sid, tasks_before in before_tasks.items():
            if tid in tasks_before:
                old_ship = sid
                break
        new_ship = sched.tasks[tid].assigned_ship
        if old_ship is not None and new_ship != old_ship:
            transferred.append((tid, old_ship, new_ship))

    if transferred:
        print(f"\n  转移了 {len(transferred)} 个任务:")
        for tid, old_s, new_s in transferred:
            t = sched.tasks[tid]
            pn = node_names.get(t.pickup_node, "?")
            dn = node_names.get(t.delivery_node, "?")
            print(f"    T{tid} ({pn}→{dn}): "
                  f"{sched.ships[old_s].name} → {sched.ships[new_s].name}")
    else:
        print(f"\n  无任务转移 (拍卖价格未收敛到转移阈值)")

    print(f"\n{'='*60}")
    print("  测试完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

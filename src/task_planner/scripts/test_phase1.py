#!/usr/bin/env python3
"""
Phase 1 测试: 异构图 + Graph Evaluator + ALNS
"""

import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from road_network import load_road_network
from hetero_graph import build_hetero_graph
from graph_evaluator import GraphEvaluator, TaskCandidate
from alns_scheduler import ALNSScheduler
from real_time_scheduler import RealTimeScheduler

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    print("=" * 65)
    print("  Phase 1: 异构图 + Graph Evaluator + ALNS")
    print("=" * 65)

    # ── 加载 ──
    rn = load_road_network(f'{BASE}/../output/road_network.json',
                           f'{BASE}/../config/ports.yaml')
    port_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_port])
    gas_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_gas_station])
    node_names = {nid: (n.port_name if n.port_name else f'N{nid}')
                  for nid, n in rn.nodes.items()}

    sched = RealTimeScheduler(rn, port_ids, gas_ids, node_names)

    # 4艘船分布在不同港口
    sched.add_ship(0, "Ship_0", 2000, 500, 8.0, port_ids[0])   # Port_J
    sched.add_ship(1, "Ship_1", 1500, 400, 7.5, port_ids[5])   # Port_E
    sched.add_ship(2, "Ship_2", 2500, 600, 7.0, port_ids[9])   # Port_A
    sched.add_ship(3, "Ship_3", 1800, 450, 8.5, port_ids[3])   # Port_G

    # 20个任务
    np.random.seed(42); tid = 0
    for i, pu in enumerate(port_ids):
        for j, de in enumerate(port_ids):
            if i == j: continue
            d = rn.dist_matrix[pu, de]
            if d <= 0 or d == np.inf: continue
            dl = 3600 * 2 if tid < 5 else float('inf')
            sched.add_task(tid, pu, de,
                          float(np.random.choice([300,500,800,1000,1500])),
                          int(np.random.choice([1,2,3])), dl)
            tid += 1
            if tid >= 20: break
        if tid >= 20: break

    # ===== Step 1: 异构图 =====
    print(f"\n{'─'*65}")
    print("  Step 1: 构建异构图")
    print(f"{'─'*65}")
    g = build_hetero_graph(sched.ships, sched.tasks, rn)
    print(f"  Ship={g.M} Task={g.K} Road={g.N}")
    print(f"  Road↔Road: {g.rr_edges.shape[1]}条")
    print(f"  Ship→Road: {g.sr_edges.shape[1]}条")
    print(f"  Task→Road: {g.tr_edges.shape[1]}条")
    print(f"  Ship↔Task: {g.st_edges.shape[1]}条")
    print(f"  Task↔Task: {g.tt_edges.shape[1]}条")

    # ===== Step 2: Graph Evaluator =====
    print(f"\n{'─'*65}")
    print("  Step 2: Graph Evaluator — Ship-Task 匹配评分")
    print(f"{'─'*65}")

    evaluator = GraphEvaluator(rn, node_names)

    # 展示几个评分示例
    sample_task = sched.tasks[0]
    print(f"\n  示例: T0 ({node_names[sample_task.pickup_node]}→{node_names[sample_task.delivery_node]}, "
          f"{sample_task.payload}t)")
    candidates = evaluator.get_top_k(sample_task, sched.ships, K=4)
    print(f"  Top-K 候选船:")
    for c in candidates:
        print(f"    Ship_{c.ship_id}: insert_cost={c.insert_cost/1000:.1f}km "
              f"pos={c.insert_position} cap={c.capacity_ok} energy={c.energy_ok} "
              f"score={c.final_score:.3f}")

    sample_task2 = sched.tasks[10]
    print(f"\n  示例: T10 ({node_names[sample_task2.pickup_node]}→{node_names[sample_task2.delivery_node]}, "
          f"{sample_task2.payload}t)")
    candidates2 = evaluator.get_top_k(sample_task2, sched.ships, K=4)
    for c in candidates2:
        print(f"    Ship_{c.ship_id}: insert_cost={c.insert_cost/1000:.1f}km "
              f"pos={c.insert_position} cap={c.capacity_ok} energy={c.energy_ok} "
              f"score={c.final_score:.3f}")

    # ===== Step 3: ALNS 初始调度 =====
    print(f"\n{'─'*65}")
    print("  Step 3: ALNS 生成初始 Route")
    print(f"{'─'*65}")

    alns = ALNSScheduler(evaluator, sched.tasks, rn, node_names)

    # Cheapest Insertion 初始解
    print(f"\n  3a: Cheapest Insertion 初始解")
    init_routes = alns.build_initial_routes(sched.ships)

    init_cost = sum(alns._route_cost(sched.ships[sid], seq)
                    for sid, seq in init_routes.items())
    print(f"  初始代价: {init_cost/1000:.1f}km")
    for sid in sorted(init_routes.keys()):
        seq = init_routes[sid]
        ship = sched.ships[sid]
        detail = alns.get_route_detail(ship, {sid: seq})
        tasks_str = " → ".join(f"T{t}" for t in seq)
        print(f"  {ship.name} ({len(seq)}任务, {detail['total_dist']/1000:.1f}km): {tasks_str}")

    # 应用 Route 到船上
    for sid, seq in init_routes.items():
        sched.ships[sid].task_sequence = seq

    # ===== Step 4: ALNS 优化 =====
    print(f"\n  3b: ALNS 优化 (SA, {alns.max_iter}轮)")

    optimized = alns.optimize(sched.ships)

    opt_cost = sum(alns._route_cost(sched.ships[sid], seq)
                   for sid, seq in optimized.items())
    improvement = (init_cost - opt_cost) / init_cost * 100 if init_cost > 0 else 0
    print(f"  优化后代价: {opt_cost/1000:.1f}km ({improvement:+.1f}%)")

    for sid in sorted(optimized.keys()):
        seq = optimized[sid]
        ship = sched.ships[sid]
        route_info = alns.get_route_detail(ship, {sid: seq})

        print(f"\n  {ship.name} @{node_names.get(ship.current_node, '?')}:")
        print(f"  {'─'*55}")
        load = 0
        for step in route_info['steps']:
            load += step['payload']
            print(f"  T{step['task']}: {step['pickup']}(+{step['payload']}t,load={load}t)"
                  f" → {step['delivery']}(-{step['payload']}t)"
                  f"  [{step['dist_to_pickup']/1000:.1f}+{step['dist_exec']/1000:.1f}km]")
            load -= step['payload']
        print(f"  {'─'*55}")
        print(f"  总计: {len(route_info['steps'])}任务, {route_info['total_dist']/1000:.1f}km")

    # ===== Step 5: 约束验证 =====
    print(f"\n{'─'*65}")
    print("  Step 4: 约束验证")
    print(f"{'─'*65}")
    all_ok = alns._validate_all(optimized, sched.ships)
    print(f"  全局约束: {'✓ 通过' if all_ok else '✗ 失败'}")

    for sid in sorted(optimized.keys()):
        ship = sched.ships[sid]
        seq = optimized[sid]
        ok = alns._check_single_route(ship, seq)
        total_d = alns._route_cost(ship, seq)
        print(f"  {ship.name}: {len(seq)}任务 {total_d/1000:.1f}km {'✓' if ok else '✗'}")

    # ===== 总结 =====
    fleet_dist = sum(alns._route_cost(sched.ships[sid], optimized[sid])
                     for sid in optimized)
    min_dist = sum(len(seq) for seq in optimized.values())
    print(f"\n{'='*65}")
    print(f"  Phase 1 完成: {min_dist}任务分配, 船队总航程 {fleet_dist/1000:.1f}km")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""v2 系统集成测试: 异构图v2 + 状态驱动调度 + MCF + Auction"""

import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from road_network import load_road_network
from hetero_graph import build_hetero_graph, update_graph
from state_evaluator import evaluate_fleet
from min_cost_flow import min_cost_flow_allocate, optimize_sequence
from auction_realloc import auction_reallocate, apply_auction_result
from real_time_scheduler import RealTimeScheduler


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    rn = load_road_network(f'{base}/../../output/road_network.json',
                           f'{base}/../../config/ports.yaml')
    port_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_port])
    gas_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_gas_station])
    node_names = {nid: (n.port_name if n.port_name else f'N{nid}')
                  for nid, n in rn.nodes.items()}

    sched = RealTimeScheduler(rn, port_ids, gas_ids, node_names)
    sched.add_ship(0, "Ship_0", 2000, 500, 8.0, port_ids[0])
    sched.add_ship(1, "Ship_1", 1500, 400, 7.5, port_ids[5])
    sched.add_ship(2, "Ship_2", 2500, 600, 7.0, port_ids[9])
    sched.add_ship(3, "Ship_3", 1800, 450, 8.5, port_ids[3])

    np.random.seed(42)
    tid = 0
    for i, pu in enumerate(port_ids):
        for j, de in enumerate(port_ids):
            if i == j: continue
            d = rn.dist_matrix[pu, de]
            if d <= 0 or d == np.inf: continue
            dl = 3600 * 2 if tid < 5 else float('inf')
            sched.add_task(tid, pu, de, float(np.random.choice([300, 500, 800, 1000, 1500])),
                          int(np.random.choice([1, 2, 3])), dl)
            tid += 1
            if tid >= 20: break
        if tid >= 20: break

    print("=" * 60)
    print("  v2 系统测试: 异构图 + 状态驱动调度")
    print("=" * 60)

    # ===== Phase 1: 异构图 + 最小费用流 =====
    print(f"\n{'─'*60}")
    print(" Phase 1: 构建异构图 v2 + 最小费用流分配")
    print(f"{'─'*60}")

    g = build_hetero_graph(sched.ships, sched.tasks, rn)
    print(f"  Ship={g.M} Road={g.N} Task={g.K}")
    print(f"  Road↔Road: {g.rr_edges.shape[1]}条")
    print(f"  Ship→Road: {g.sr_edges.shape[1]}条")
    print(f"  Task→Road: {g.tr_edges.shape[1]}条")
    print(f"  Ship↔Task: {g.st_edges.shape[1]}条 (candidate)")
    print(f"  Task↔Task: {g.tt_edges.shape[1]}条 (nearby+depends)")

    assignments, total_cost, flow = min_cost_flow_allocate(
        sched.ships, sched.tasks, g)

    for ship in sched.ships.values():
        optimize_sequence(ship, sched.tasks, rn.dist_matrix)

    print(f"\n  MCF结果: {flow}/{g.K}任务, 总代价={total_cost:.0f}")
    for ship in sorted(sched.ships.values(), key=lambda s: s.ship_id):
        tasks_str = ",".join(f"T{t}" for t in ship.task_sequence)
        total_d = sum(rn.dist_matrix[sched.tasks[tid].pickup_node,
                                      sched.tasks[tid].delivery_node]
                     for tid in ship.task_sequence if tid in sched.tasks)
        n = node_names.get(ship.current_node, "?")
        print(f"  {ship.name} @{n}: [{tasks_str}] {total_d/1000:.1f}km")

    # ===== Phase 2: 状态驱动评估 =====
    print(f"\n{'─'*60}")
    print(" Phase 2: 状态驱动评估 (每艘船独立判断)")
    print(f"{'─'*60}")

    decisions, release_pool, idle_ships = evaluate_fleet(
        sched.ships, sched.tasks, rn, sched.current_time)

    for sid, (action, reason, state) in decisions.items():
        icon = {'normal': '✓', 'release_some': '⚠', 'release_all': '✗', 'take_tasks': '+'}
        print(f"  {icon.get(action,'?')} {sched.ships[sid].name}: {action} — {reason}")
        if action != 'normal':
            print(f"     energy={state['energy_score']:.2f} load={state['load_ratio']:.2f} "
                  f"health={state['health']:.2f} idle={state['idle']:.0f} "
                  f"pos={state['position_score']:.2f} eta={state['eta_reliability']:.2f}")

    print(f"\n  汇总: {len(release_pool)}任务释放, {len(idle_ships)}船空闲")

    # ===== Phase 3: 模拟异常 + 状态驱动重分配 =====
    print(f"\n{'─'*60}")
    print(" Phase 3: 模拟异常 → 状态驱动重分配")
    print(f"{'─'*60}")

    # 场景: Ship_0 能源降到 8%, Ship_1 空闲
    sched.ships[0].energy = sched.ships[0].max_energy * 0.08
    sched.ships[1].task_sequence.clear()
    sched.ships[1].load = 0

    before = {sid: list(s.task_sequence) for sid, s in sched.ships.items()}
    print(f"  前: { {k:v for k,v in before.items()} }")

    # 状态评估
    decisions, release_pool, idle_ships = evaluate_fleet(
        sched.ships, sched.tasks, rn, sched.current_time)

    for sid, (action, reason, state) in decisions.items():
        icon = '⚠' if action != 'normal' else '✓'
        print(f"  {icon} {sched.ships[sid].name}: {action}")
        if action != 'normal':
            print(f"     energy_score={state['energy_score']:.2f} "
                  f"({state.get('energy_detail','')})")

    # Auction 重分配
    if release_pool:
        update_graph(g, sched.ships, sched.tasks, rn, sched.current_time)

        fixed_tasks = set()
        for tid, task in sched.tasks.items():
            if task.status in ('loading', 'navigate_to_delivery', 'unloading'):
                fixed_tasks.add(tid)

        new_assignments, prices, rounds = auction_reallocate(
            sched.ships, sched.tasks, g,
            fixed_tasks=fixed_tasks, epsilon=1.0)

        apply_auction_result(sched.ships, sched.tasks, new_assignments, fixed_tasks)

        after = {sid: list(s.task_sequence) for sid, s in sched.ships.items()}
        print(f"\n  后: { {k:v for k,v in after.items()} }")

        transferred = []
        for tid in sched.tasks:
            old = None
            for sid, tasks_before in before.items():
                if tid in tasks_before: old = sid; break
            new = sched.tasks[tid].assigned_ship
            if old is not None and new != old and new >= 0:
                transferred.append((tid, old, new))

        if transferred:
            print(f"\n  转移了 {len(transferred)} 个任务:")
            for tid, old_s, new_s in transferred:
                t = sched.tasks[tid]
                pn = node_names.get(t.pickup_node, "?")
                dn = node_names.get(t.delivery_node, "?")
                print(f"    T{tid} ({pn}→{dn}): "
                      f"{sched.ships[old_s].name} → {sched.ships[new_s].name}")
        else:
            print(f"  无任务转移")
    else:
        print(f"  无需重分配")

    # ===== Phase 4: 最终状态 =====
    print(f"\n{'─'*60}")
    print(" Phase 4: 最终状态")
    print(f"{'─'*60}")

    for ship in sorted(sched.ships.values(), key=lambda s: s.ship_id):
        decisions, release_pool, idle_ships = evaluate_fleet(
            {ship.ship_id: ship}, sched.tasks, rn, sched.current_time)
        _, reason, state = decisions[ship.ship_id]
        e = state['energy_score']
        bar = '█' * min(20, int(e * 10)) + '░' * max(0, 20 - int(e * 10))
        print(f"  {ship.name}: |{bar}| e={e:.2f} load={state['load_ratio']:.2f} "
              f"h={state['health']:.2f} eta={state['eta_reliability']:.2f} "
              f"tasks={len(ship.task_sequence)} {reason}")

    print(f"\n  测试完成")

if __name__ == '__main__':
    main()

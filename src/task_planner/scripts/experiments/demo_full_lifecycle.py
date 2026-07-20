#!/usr/bin/env python3
"""
完整生命周期演示: 初始分配 → 时间推进 → 装卸 → 异常 → 重分配
"""

import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from road_network import load_road_network
from hetero_graph import build_hetero_graph, update_graph
from state_evaluator import evaluate_fleet
from min_cost_flow import min_cost_flow_allocate, optimize_sequence
from auction_realloc import auction_reallocate, apply_auction_result, build_fixed_sets
from real_time_scheduler import RealTimeScheduler

BASE = os.path.dirname(os.path.abspath(__file__))


def print_status(sched, node_names, title=""):
    """打印当前状态"""
    if title:
        print(f"\n{'─'*65}")
        print(f"  {title}  t={sched.current_time/60:.0f}min")
        print(f"{'─'*65}")
    for ship in sorted(sched.ships.values(), key=lambda s: s.ship_id):
        node = node_names.get(ship.current_node, '?')
        phase_icons = {
            'pending': '⏸', 'navigate_to_pickup': '➡️',
            'loading': '📦', 'navigate_to_delivery': '🚢', 'unloading': '📤'
        }
        icon = phase_icons.get(ship.current_phase, '?')
        e_pct = int(ship.energy_ratio * 20)
        e_bar = '█' * e_pct + '░' * (20 - e_pct)

        # 任务序列
        task_desc = ""
        for tid in ship.task_sequence:
            t = sched.tasks.get(tid)
            if not t: continue
            pn = node_names.get(t.pickup_node, '?')
            dn = node_names.get(t.delivery_node, '?')
            mark = '←' if tid == ship.current_task_id else ''
            task_desc += f" {mark}T{tid}({pn}→{dn})"

        print(f"  {icon} {ship.name} @{node} |{e_bar}| {ship.energy_ratio*100:.0f}%"
              f"  load={ship.load:.0f}/{ship.max_payload:.0f}t"
              f"  hp={ship.health*100:.0f}%"
              f"  done={len(ship.completed_tasks)}"
              f"{task_desc}")


def main():
    print("=" * 65)
    print("  内河船舶任务调度 — 完整生命周期演示")
    print("=" * 65)

    rn = load_road_network(f'{BASE}/../../output/road_network.json',
                           f'{BASE}/../../config/ports.yaml')
    port_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_port])
    gas_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_gas_station])
    node_names = {nid: (n.port_name if n.port_name else f'N{nid}')
                  for nid, n in rn.nodes.items()}

    sched = RealTimeScheduler(rn, port_ids, gas_ids, node_names)

    # ── 4艘船, 分布在不同港口 ──
    sched.add_ship(0, "Ship_0", 2000, 500, 8.0, port_ids[0])   # Port_J (下游)
    sched.add_ship(1, "Ship_1", 1500, 400, 7.5, port_ids[5])   # Port_E (中游)
    sched.add_ship(2, "Ship_2", 2500, 600, 7.0, port_ids[9])   # Port_A (上游)
    sched.add_ship(3, "Ship_3", 1800, 450, 8.5, port_ids[3])   # Port_G

    # ── 8个运输任务 ──
    task_data = [
        # id, pickup, delivery, payload, pri, deadline(秒)
        (0, port_ids[0], port_ids[9], 1000, 3, float('inf')),   # Port_J→Port_A
        (1, port_ids[0], port_ids[5], 800, 2, float('inf')),    # Port_J→Port_E
        (2, port_ids[5], port_ids[0], 500, 1, float('inf')),    # Port_E→Port_J
        (3, port_ids[9], port_ids[5], 1500, 3, 3600),           # Port_A→Port_E (1h截止!)
        (4, port_ids[3], port_ids[0], 800, 2, float('inf')),    # Port_G→Port_J
        (5, port_ids[5], port_ids[9], 600, 2, float('inf')),    # Port_E→Port_A
        (6, port_ids[0], port_ids[3], 1000, 1, float('inf')),   # Port_J→Port_G
        (7, port_ids[9], port_ids[3], 300, 3, 7200),            # Port_A→Port_G (2h截止)
    ]
    for tid, pu, de, payload, pri, dl in task_data:
        sched.add_task(tid, pu, de, payload, pri, dl)

    # ===== Step 1: 初始分配 (MCF) =====
    print(f"\n{'='*65}")
    print("  Step 1: 最小费用流初始分配")
    print(f"{'='*65}")

    g = build_hetero_graph(sched.ships, sched.tasks, rn)
    assignments, cost, flow = min_cost_flow_allocate(sched.ships, sched.tasks, g)
    for s in sched.ships.values():
        optimize_sequence(s, sched.tasks, rn.dist_matrix)

    for ship in sorted(sched.ships.values(), key=lambda s: s.ship_id):
        for tid in ship.task_sequence:
            t = sched.tasks[tid]
            pn = node_names.get(t.pickup_node, '?')
            dn = node_names.get(t.delivery_node, '?')
            d = rn.dist_matrix[ship.current_node, t.pickup_node] + rn.dist_matrix[t.pickup_node, t.delivery_node]
            print(f"  {ship.name}: T{tid} {pn}→{dn} {t.payload}t {d/1000:.1f}km")

    print_status(sched, node_names, "初始状态")

    # ===== Step 2: 推进时间, 观察船移动 =====
    print(f"\n{'='*65}")
    print("  Step 2: 时间推进 — 船开始航行")
    print(f"{'='*65}")

    # 推进30分钟 (1800秒) — 船从港口出发, 前往装货港
    for _ in range(6):
        sched.scheduling_loop(300)

    print_status(sched, node_names, "30分钟后")
    print(f"\n  观察: 船开始移动, 到达装货港后进入装货阶段")
    print(f"  Ship_0 在 Port_J 直接装货 (就在装货港), 其他船需要航行")

    # ===== Step 3: 继续推进, 观测装货→运输→卸货 =====
    print(f"\n{'='*65}")
    print("  Step 3: 继续推进 — 装货完成, 开始运输")
    print(f"{'='*65}")

    for _ in range(12):  # 再推进1小时
        sched.scheduling_loop(300)

    print_status(sched, node_names, "1.5小时后")
    print(f"\n  观察: 船进入运输阶段, 能源开始消耗")

    completed = sum(1 for t in sched.tasks.values() if t.status == 'completed')
    print(f"  已完成任务: {completed}/8")

    # ===== Step 4: 模拟异常 — Ship_0 能源紧急 =====
    print(f"\n{'='*65}")
    print("  Step 4: 异常 — Ship_0 能源紧急 (强制降至 6%)")
    print(f"{'='*65}")

    sched.ships[0].energy = sched.ships[0].max_energy * 0.06

    decisions, release_pool, idle_ships = evaluate_fleet(
        sched.ships, sched.tasks, rn, sched.current_time)

    before_realloc = {sid: (list(s.task_sequence), s.energy)
                      for sid, s in sched.ships.items()}

    for sid, (action, reason, state) in decisions.items():
        if action != 'normal':
            print(f"  {sched.ships[sid].name}: {action}")
            print(f"    energy_score={state['energy_score']:.2f} ({state.get('energy_detail','')})")
            print(f"    → {reason}")

    # 拍卖重分配
    if release_pool:
        update_graph(g, sched.ships, sched.tasks, rn, sched.current_time)
        fixed_s, fixed_t = build_fixed_sets(sched.ships, sched.tasks)
        new_assignments, prices, rounds = auction_reallocate(
            sched.ships, sched.tasks, g, fixed_ships=fixed_s,
            fixed_tasks=fixed_t, epsilon=1.0)
        apply_auction_result(sched.ships, sched.tasks, new_assignments, fixed_t)

        after_realloc = {sid: list(s.task_sequence) for sid, s in sched.ships.items()}
        print(f"\n  重分配结果 ({rounds}轮):")
        for sid in sched.ships:
            before_tasks = before_realloc[sid][0]
            after_tasks = after_realloc[sid]
            if before_tasks != after_tasks:
                print(f"    {sched.ships[sid].name}: {before_tasks} → {after_tasks}")

        # 显示转移
        for tid in sched.tasks:
            t = sched.tasks[tid]
            if t.assigned_ship >= 0:
                old_ship = None
                for sid, (tasks_before, _) in before_realloc.items():
                    if tid in tasks_before: old_ship = sid; break
                if old_ship is not None and old_ship != t.assigned_ship:
                    pn = node_names.get(t.pickup_node, '?')
                    dn = node_names.get(t.delivery_node, '?')
                    print(f"    ⚡ T{tid} ({pn}→{dn}): Ship_{old_ship} → Ship_{t.assigned_ship}")

    # ===== Step 5: 最终推进到完成 =====
    print(f"\n{'='*65}")
    print("  Step 5: 继续推进直到任务完成")
    print(f"{'='*65}")

    for _ in range(200):
        all_done = all(t.status in ('completed', 'cancelled')
                      for t in sched.tasks.values())
        if all_done: break
        sched.scheduling_loop(300)

    print_status(sched, node_names, "最终状态")

    # ===== 总结 =====
    print(f"\n{'='*65}")
    print("  调度总结")
    print(f"{'='*65}")
    total_dist = sum(s.total_distance for s in sched.ships.values())
    total_energy = sum(s.max_energy - s.energy for s in sched.ships.values())
    completed = sum(1 for t in sched.tasks.values() if t.status == 'completed')
    realloc_events = sum(1 for e in sched.event_log if e.get('changed'))

    for ship in sorted(sched.ships.values(), key=lambda s: s.ship_id):
        done_names = [f"T{t}" for t in ship.completed_tasks]
        print(f"  {ship.name}: 完成{len(ship.completed_tasks)}任务 {done_names}"
              f"  航程{ship.total_distance/1000:.1f}km"
              f"  能耗{ship.max_energy-ship.energy:.0f}kWh"
              f"  时间{ship.total_time/60:.0f}min")

    print(f"\n  总航程: {total_dist/1000:.1f}km")
    print(f"  总能耗: {total_energy:.0f}kWh")
    print(f"  完成: {completed}/{len(sched.tasks)}")
    print(f"  重分配生效: {realloc_events}次")
    print(f"\n  运行: PYTHONPATH=src/task_planner python3 -m scripts.experiments.demo_full_lifecycle")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()

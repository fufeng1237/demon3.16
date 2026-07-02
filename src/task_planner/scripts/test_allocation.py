#!/usr/bin/env python3
"""任务分配与重分配测试"""

import sys, os, json, numpy as np
from copy import deepcopy
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from road_network import load_road_network
from task_assigner import (
    Ship, Task, ShipStatus, TaskStatus,
    allocate, format_allocation_result
)
from event_monitor import (
    EventMonitor, Reallocator, format_reallocation_output
)


def main():
    # ── 1. 加载路网 ──
    rn = load_road_network(
        os.path.join(os.path.dirname(__file__), "../output/road_network.json"),
        ports_config=os.path.join(os.path.dirname(__file__), "../config/ports.yaml")
    )
    print(f"路网: {len(rn.nodes)} 节点, {len(rn.edges)} 边")

    # 找到港口节点
    port_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_port])
    gas_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_gas_station])
    node_names = {}
    for nid, n in rn.nodes.items():
        if n.is_port: node_names[nid] = n.port_name
        elif n.is_gas_station: node_names[nid] = f"GS_{n.port_name}"
        else: node_names[nid] = f"N{nid}"

    print(f"港口: {[node_names[p] for p in port_ids]}")
    print(f"加油站: {[node_names[g] for g in gas_ids]}")

    # ── 2. 创建船舶 ──
    ships = [
        Ship(id=0, name="Ship_0", max_payload=2000, max_energy=500, max_speed=8,
             position_x=rn.nodes[port_ids[0]].x, position_y=rn.nodes[port_ids[0]].y,
             current_node=port_ids[0], energy=450, energy_per_km=2.5),
        Ship(id=1, name="Ship_1", max_payload=1500, max_energy=400, max_speed=7.5,
             position_x=rn.nodes[port_ids[5]].x, position_y=rn.nodes[port_ids[5]].y,
             current_node=port_ids[5], energy=360, energy_per_km=2.2),
        Ship(id=2, name="Ship_2", max_payload=2500, max_energy=600, max_speed=7,
             position_x=rn.nodes[port_ids[9]].x, position_y=rn.nodes[port_ids[9]].y,
             current_node=port_ids[9], energy=540, energy_per_km=3.0),
        Ship(id=3, name="Ship_3", max_payload=1800, max_energy=450, max_speed=8.5,
             position_x=rn.nodes[port_ids[3]].x, position_y=rn.nodes[port_ids[3]].y,
             current_node=port_ids[3], energy=405, energy_per_km=2.8),
    ]

    # ── 3. 创建任务 (港口对之间的运输) ──
    tasks = []
    tid = 0
    for i, pickup in enumerate(port_ids):
        for j, delivery in enumerate(port_ids):
            if i == j: continue
            d = rn.dist_matrix[pickup, delivery]
            if d <= 0 or d == np.inf: continue
            tasks.append(Task(
                id=tid, pickup_node=pickup, delivery_node=delivery,
                payload=float(np.random.choice([300, 500, 800, 1000, 1500])),
                start_time=0, deadline=3600 * 8,
                priority=int(np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3]))
            ))
            tid += 1
            if tid >= 30: break
        if tid >= 30: break

    print(f"\n任务: {len(tasks)} 个")
    for t in tasks[:5]:
        print(f"  T{t.id}: {node_names[t.pickup_node]} → {node_names[t.delivery_node]}, "
              f"{t.payload}t, pri={t.priority}")
    if len(tasks) > 5: print(f"  ... 等 {len(tasks)-5} 个")

    # ── 4. 初始分配 ──
    print("\n" + "=" * 60)
    print("  初始任务分配")
    print("=" * 60)

    result = allocate(deepcopy(ships), deepcopy(tasks), rn.dist_matrix,
                      enable_2opt=True, enable_transfer=True, node_id_to_name=node_names)
    print(format_allocation_result(result, rn.dist_matrix, node_names, "INITIAL ALLOCATION"))

    # ── 5. 重分配场景 1: 能源不足 ──
    print("\n" + "=" * 60)
    print("  重分配场景 1: Ship_0 能源不足")
    print("=" * 60)

    monitor = EventMonitor(energy_threshold=0.2)
    reallocator = Reallocator()

    s1 = deepcopy(result.ships)
    t1 = deepcopy(result.tasks)
    for s in s1:
        if s.id == 0:
            s.energy = s.max_energy * 0.12
            print(f"  触发: {s.name} 能源={s.energy:.0f}/{s.max_energy:.0f} kWh\n")

    events = monitor.detect_events(s1, t1)
    for evt in events:
        ships_before = deepcopy(s1)
        decision = reallocator.decide(evt, s1, t1, rn.dist_matrix, gas_ids, port_ids, [])

        print(f"  事件: {evt.type.value}")
        print(f"  决策: {decision.action}")
        print(f"  理由: {decision.rationale}\n")

        if decision.triggered:
            r2 = allocate(s1, t1, rn.dist_matrix, True, True, node_id_to_name=node_names)
            print(format_reallocation_output(decision, ships_before, r2.ships, t1))
            s1, t1 = r2.ships, r2.tasks

    # ── 6. 重分配场景 2: 船舶故障 ──
    print("\n" + "=" * 60)
    print("  重分配场景 2: Ship_1 故障")
    print("=" * 60)

    for s in s1:
        if s.id == 1:
            s.health = 0.15
            print(f"  触发: {s.name} 健康度={s.health}\n")

    events2 = monitor.detect_events(s1, t1)
    for evt in events2:
        ships_before2 = deepcopy(s1)
        decision2 = reallocator.decide(evt, s1, t1, rn.dist_matrix, gas_ids, port_ids, [])

        print(f"  事件: {evt.type.value}")
        print(f"  决策: {decision2.action}")
        print(f"  理由: {decision2.rationale}\n")

        if decision2.triggered:
            r3 = allocate(s1, t1, rn.dist_matrix, True, True, node_id_to_name=node_names)
            print(format_reallocation_output(decision2, ships_before2, r3.ships, t1))

    print("\n测试完成。")


if __name__ == "__main__":
    main()

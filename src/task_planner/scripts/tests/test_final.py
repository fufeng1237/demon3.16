#!/usr/bin/env python3
"""
最终系统测试: 异构图 + ALNS Route 生成 + 执行 + 事件 + Route Repair
"""

import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from road_network import load_road_network
from real_time_scheduler import RealTimeScheduler
from scheduler import Scheduler

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    print("=" * 65)
    print("  基于异构图与Graph-guided ALNS的多无人船动态协同运输调度系统")
    print("=" * 65)

    rn = load_road_network(f'{BASE}/../../output/road_network.json',
                           f'{BASE}/../../config/ports.yaml')
    port_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_port])
    gas_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_gas_station])
    node_names = {nid: (n.port_name if n.port_name else f'N{nid}')
                  for nid, n in rn.nodes.items()}

    sched_helper = RealTimeScheduler(rn, port_ids, gas_ids, node_names)
    sched_helper.add_ship(0, "Ship_0", 2000, 500, 8.0, port_ids[0])
    sched_helper.add_ship(1, "Ship_1", 1500, 400, 7.5, port_ids[5])
    sched_helper.add_ship(2, "Ship_2", 2500, 600, 7.0, port_ids[9])
    sched_helper.add_ship(3, "Ship_3", 1800, 450, 8.5, port_ids[3])

    np.random.seed(42); tid = 0
    for i, pu in enumerate(port_ids):
        for j, de in enumerate(port_ids):
            if i == j: continue
            d = rn.dist_matrix[pu, de]
            if d <= 0 or d == np.inf: continue
            dl = 3600 * 2 if tid < 3 else float('inf')
            sched_helper.add_task(tid, pu, de,
                                 float(np.random.choice([300,500,800,1000,1500])),
                                 int(np.random.choice([1,2,3])), dl)
            tid += 1
            if tid >= 12: break
        if tid >= 12: break

    scheduler = Scheduler(rn, sched_helper.ships, sched_helper.tasks,
                          port_ids, gas_ids, node_names)

    # ===== 1. 初始调度 =====
    routes = scheduler.initialize()

    # ===== 2. 输出 Route (RoadNode 序列) =====
    print(f"\n{'='*60}")
    print("  System Output: RoadNode Sequences for Path Planner")
    print(f"{'='*60}")
    node_seqs = scheduler.get_node_sequences()
    for sid in sorted(node_seqs.keys()):
        seq = node_seqs[sid]
        ship = scheduler.ships[sid]
        print(f"  {ship.name}: {seq}  ({len(seq)} nodes)")

    # ===== 3. 模拟运行 =====
    print(f"\n{'='*60}")
    print("  Simulation: 执行 + 事件处理")
    print(f"{'='*60}")

    # 运行30分钟
    scheduler.run(1800, dt=300)
    scheduler._print_status()

    # 模拟异常: Ship_0 能源紧急
    print(f"\n  >>> Event: Ship_0 energy LOW ({scheduler.ships[0].energy_ratio*100:.0f}%)")
    scheduler.ships[0].energy = scheduler.ships[0].max_energy * 0.06
    scheduler.step(60)
    node_seqs = scheduler.get_node_sequences()
    for sid in sorted(node_seqs.keys()):
        print(f"  {scheduler.ships[sid].name}: {node_seqs[sid]}")

    # 查看事件日志
    if scheduler.event_log:
        print(f"\n  Route Repair 记录:")
        for ev in scheduler.event_log[-3:]:
            print(f"  [{ev['time']:.0f}s] {ev['event']['type']}")
            print(f"    before: {ev['before']}")
            print(f"    after:  {ev['after']}")

    # ===== 4. 最终状态 =====
    print(f"\n{'='*60}")
    print("  Final State")
    print(f"{'='*60}")
    scheduler._print_status()

    completed = sum(1 for t in scheduler.tasks.values() if t.status == 'completed')
    total_dist = sum(s.total_distance for s in scheduler.ships.values())
    repairs = len([e for e in scheduler.event_log if e['before'] != e['after']])
    print(f"\n  Completed: {completed}/{len(scheduler.tasks)}")
    print(f"  Fleet distance: {total_dist/1000:.1f}km")
    print(f"  Route Repairs: {repairs}")
    print(f"\n{'='*65}")
    print(f"  测试完成")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()

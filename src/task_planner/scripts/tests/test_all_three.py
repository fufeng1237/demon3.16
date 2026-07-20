#!/usr/bin/env python3
"""测试: 动态优先级 + 失败回滚 + 可视化"""

import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from road_network import load_road_network
from real_time_scheduler import RealTimeScheduler


def main():
    rn = load_road_network(
        os.path.join(os.path.dirname(__file__), "../../output/road_network.json"),
        ports_config=os.path.join(os.path.dirname(__file__), "../../config/ports.yaml")
    )

    port_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_port])
    gas_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_gas_station])
    node_names = {nid: (n.port_name if n.port_name else f"N{nid}")
                  for nid, n in rn.nodes.items()}

    sched = RealTimeScheduler(rn, port_ids, gas_ids, node_names, use_ros=False)

    # 4艘船
    sched.add_ship(0, "Ship_0", 2000, 500, 8.0, port_ids[0])
    sched.add_ship(1, "Ship_1", 1500, 400, 7.5, port_ids[5])
    sched.add_ship(2, "Ship_2", 2500, 600, 7.0, port_ids[9])
    sched.add_ship(3, "Ship_3", 1800, 450, 8.5, port_ids[3])

    # 15个任务, 各种优先级和截止时间
    task_specs = []
    tid = 0
    for i, pu in enumerate(port_ids):
        for j, de in enumerate(port_ids):
            if i == j: continue
            d = rn.dist_matrix[pu, de]
            if d <= 0 or d == np.inf: continue
            # 混合截止时间: 一些紧, 一些松
            dl = (3600 * 1 + np.random.randint(0, 3600)) if tid < 5 else float('inf')
            task_specs.append((pu, de, float(np.random.choice([300,500,800,1000,1500])),
                              np.random.choice([1,2,3]), dl))
            tid += 1
            if tid >= 15: break
        if tid >= 15: break

    for tid, (pu, de, payload, pri, dl) in enumerate(task_specs):
        sched.add_task(tid, pu, de, payload, pri, dl)

    # ── 初始分配 ──
    sched.initial_allocate()
    print(sched.format_status())

    # ── 显示任务链 ──
    print("\n任务链依赖:")
    for task in sorted(sched.tasks.values(), key=lambda t: t.task_id):
        if task.downstream_tasks:
            ds = ", ".join(f"T{t}" for t in task.downstream_tasks)
            print(f"  T{task.task_id} (p={task.dynamic_priority:.1f}) → 下游: [{ds}]")

    # ── 模拟运行1小时 ──
    print(f"\n{'='*65}")
    print("  推进 1 小时...")
    print(f"{'='*65}")
    for _ in range(12):
        sched.scheduling_loop(300)
    print(sched.format_status())

    # ── 场景1: 能源紧急 → 测试失败回滚 ──
    print(f"\n{'='*65}")
    print("  场景1: Ship_0 能源紧急 + 失败回滚 (在途货物)")
    print(f"{'='*65}")

    # 让 Ship_0 进入 navigate_to_delivery 阶段
    sched.ships[0].current_phase = "navigate_to_delivery"
    sched.ships[0].load = 800  # 模拟已装货
    sched.ships[0].energy = sched.ships[0].max_energy * 0.05

    sched.scheduling_loop(60)
    print(sched.format_status())

    # 看回滚结果
    print("\n失败回滚记录:")
    for ev in sched.event_log[-5:]:
        if "relay" in str(ev.get("event", {}).get("type", "")) or "release" in str(ev):
            print(f"  [{ev['time']:.0f}s] {ev.get('event',{}).get('type','?')}")
            m = ev.get('message','')
            print(f"    {m}")

    # ── 场景2: 优先级动态变化 ──
    print(f"\n{'='*65}")
    print("  场景2: 任务优先级动态变化")
    print(f"{'='*65}")

    # 强制一个快过期的任务
    for task in sched.tasks.values():
        if task.status not in ("completed", "cancelled"):
            task.deadline = sched.current_time + 1800  # 30分钟后过期
            task.update_priority(sched.current_time, 1.0, 1.0)
            print(f"  T{task.task_id}: p{task.base_priority}→{task.dynamic_priority:.1f} "
                  f"(30min到期, 紧急度飙升)")
            break

    sched.scheduling_loop(60)
    print(sched.format_status())

    # ── 场景3: 故障回滚 (在途货物接力) ──
    print(f"\n{'='*65}")
    print("  场景3: 故障 + 在途货物接力")
    print(f"{'='*65}")

    # 找一艘在 navigate_to_delivery 的船
    for s in sched.ships.values():
        if s.task_sequence and s.current_phase != "navigate_to_delivery":
            s.current_phase = "navigate_to_delivery"
            s.load = 1200
            s.health = 0.15
            print(f"  {s.name} 故障, 在途货物 {s.load:.0f}t")
            break

    sched.scheduling_loop(60)
    print(sched.format_status())

    # 回滚结果
    print("\n跟力/重分配记录:")
    for ev in sched.event_log[-8:]:
        msg = ev.get('message', '')
        if msg and ('relay' in msg.lower() or 'release' in msg.lower() or '故障' in msg):
            ch = "✓" if ev.get('changed') else "—"
            print(f"  [{ev['time']:.0f}s] [{ch}] {msg}")

    # ── 最终总结 ──
    print(f"\n{'='*65}")
    print("  最终状态")
    print(f"{'='*65}")
    print(sched.format_status())

    # 重分配前后对比
    print("\n重分配变更记录:")
    print(sched.format_reallocation_comparison())

    # 统计
    for ship in sorted(sched.ships.values(), key=lambda s: s.ship_id):
        print(f"  {ship.name}: {len(ship.completed_tasks)}完成, "
              f"{ship.total_distance/1000:.1f}km, {ship.energy:.0f}kWh")
    completed = sum(1 for t in sched.tasks.values() if t.status == "completed")
    failed = sum(1 for t in sched.tasks.values() if t.status in ("cancelled", "released"))
    print(f"  完成: {completed}, 失败: {failed}, 生效重分配: "
          f"{sum(1 for e in sched.event_log if e.get('changed'))}")


if __name__ == "__main__":
    main()

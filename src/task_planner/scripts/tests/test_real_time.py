#!/usr/bin/env python3
"""实时自适应调度测试 — 展示重分配真正生效"""

import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from road_network import load_road_network
from real_time_scheduler import RealTimeScheduler


def main():
    # ── 加载路网 ──
    rn = load_road_network(
        os.path.join(os.path.dirname(__file__), "../../output/road_network.json"),
        ports_config=os.path.join(os.path.dirname(__file__), "../../config/ports.yaml")
    )

    port_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_port])
    gas_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_gas_station])
    node_names = {nid: (n.port_name if n.port_name else f"N{nid}")
                  for nid, n in rn.nodes.items()}

    print(f"路网: {len(rn.nodes)} 节点, {len(rn.edges)} 边")
    print(f"港口: {[node_names[p] for p in port_ids]}")

    # ── 创建调度器 ──
    sched = RealTimeScheduler(rn, port_ids, gas_ids, node_names)

    # 4艘船
    sched.add_ship(0, "Ship_0", 2000, 500, 8.0, port_ids[0])
    sched.add_ship(1, "Ship_1", 1500, 400, 7.5, port_ids[5])
    sched.add_ship(2, "Ship_2", 2500, 600, 7.0, port_ids[9])
    sched.add_ship(3, "Ship_3", 1800, 450, 8.5, port_ids[3])

    # 15个运输任务
    task_specs = []
    tid = 0
    for i, pu in enumerate(port_ids):
        for j, de in enumerate(port_ids):
            if i == j: continue
            d = rn.dist_matrix[pu, de]
            if d <= 0 or d == np.inf: continue
            task_specs.append((pu, de, float(np.random.choice([300,500,800,1000,1500])),
                               np.random.choice([1,2,3], p=[0.2,0.4,0.4]),
                               3600*4 + np.random.randint(0, 3600*8)))  # deadline: 4-12小时
            tid += 1
            if tid >= 15: break
        if tid >= 15: break

    for tid, (pu, de, payload, pri, dl) in enumerate(task_specs):
        ok = sched.add_task(tid, pu, de, payload, pri, dl)
        if not ok:
            print(f"  T{tid}: 不可达, 跳过")

    print(f"任务: {len(sched.tasks)} 个")

    # ── 初始分配 ──
    sched.initial_allocate()
    print(sched.format_status())

    # ── 模拟运行: 推进 30 分钟 ──
    print(f"\n{'='*65}")
    print(f"  模拟运行 30 分钟...")
    print(f"{'='*65}")
    sched.scheduling_loop(1800)

    print(sched.format_status())

    # ── 场景1: Ship_0 能源耗尽 ──
    print(f"\n{'='*65}")
    print(f"  场景: Ship_0 能源紧急 (强制设为 8%)")
    print(f"{'='*65}")
    sched.ships[0].energy = sched.ships[0].max_energy * 0.08

    sched.scheduling_loop(60)
    print(sched.format_status())
    print(sched.format_reallocation_comparison())

    # ── 场景2: Ship_1 故障 ──
    print(f"\n{'='*65}")
    print(f"  场景: Ship_1 故障 (健康度=0.2)")
    print(f"{'='*65}")
    sched.ships[1].health = 0.2

    sched.scheduling_loop(60)
    print(sched.format_status())
    print(sched.format_reallocation_comparison())

    # ── 场景3: 新任务到达 ──
    print(f"\n{'='*65}")
    print(f"  场景: 新任务到达")
    print(f"{'='*65}")
    # 找一个在空闲船附近的港口对
    new_pu, new_de = port_ids[2], port_ids[7]
    sched.add_task(100, new_pu, new_de, 800, 3, sched.current_time + 3600*2)
    print(f"  新任务 T100: {node_names[new_pu]} → {node_names[new_de]}, 800t, pri=3")

    sched.scheduling_loop(60)
    print(sched.format_status())

    # ── 最终: 推进到所有任务完成 ──
    print(f"\n{'='*65}")
    print(f"  推进时间直到所有任务完成...")
    print(f"{'='*65}")

    for _ in range(200):
        all_done = all(
            t.status in ("completed", "cancelled", "released")
            for t in sched.tasks.values()
        )
        if all_done:
            break
        sched.scheduling_loop(600)  # 每10分钟检查一次

        # 每2小时打印一次状态
        if int(sched.current_time) % 7200 < 600:
            print(sched.format_status())

    print(sched.format_status())
    print(sched.format_event_log())

    # ── 总结 ──
    print(f"\n{'='*65}")
    print(f"  调度总结")
    print(f"{'='*65}")
    for ship in sorted(sched.ships.values(), key=lambda s: s.ship_id):
        print(f"  {ship.name}: 完成 {len(ship.completed_tasks)} 任务, "
              f"航程 {ship.total_distance/1000:.1f}km, "
              f"剩余能源 {ship.energy:.0f}kWh")

    completed = sum(1 for t in sched.tasks.values() if t.status == "completed")
    print(f"  总完成: {completed}/{len(sched.tasks)} 任务")
    print(f"  重分配事件: {len([e for e in sched.event_log if e.get('changed')])} 次生效")
    print(f"  事件总数: {len(sched.event_log)}")


if __name__ == "__main__":
    main()

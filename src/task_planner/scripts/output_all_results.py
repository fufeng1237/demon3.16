#!/usr/bin/env python3
"""
完整输出: 异构图可视化 + 分配结果 + 最小费用流 + 拍卖法演示
"""

import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from road_network import load_road_network
from hetero_graph import build_hetero_graph, update_graph
from state_evaluator import evaluate_fleet
from min_cost_flow import min_cost_flow_allocate, optimize_sequence
from auction_realloc import auction_reallocate, apply_auction_result, build_fixed_sets
from real_time_scheduler import RealTimeScheduler
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))

def main():
    rn = load_road_network(f'{BASE}/../output/road_network.json',
                           f'{BASE}/../config/ports.yaml')
    port_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_port])
    gas_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_gas_station])
    node_names = {nid: (n.port_name if n.port_name else f'N{nid}')
                  for nid, n in rn.nodes.items()}

    sched = RealTimeScheduler(rn, port_ids, gas_ids, node_names)
    sched.add_ship(0, "Ship_0", 2000, 500, 8.0, port_ids[0])
    sched.add_ship(1, "Ship_1", 1500, 400, 7.5, port_ids[5])
    sched.add_ship(2, "Ship_2", 2500, 600, 7.0, port_ids[9])
    sched.add_ship(3, "Ship_3", 1800, 450, 8.5, port_ids[3])

    np.random.seed(42); tid = 0
    for i, pu in enumerate(port_ids):
        for j, de in enumerate(port_ids):
            if i == j: continue
            d = rn.dist_matrix[pu, de]
            if d <= 0 or d == np.inf: continue
            sched.add_task(tid, pu, de, float(np.random.choice([300,500,800,1000,1500])),
                          int(np.random.choice([1,2,3])), float('inf'))
            tid += 1
            if tid >= 12: break
        if tid >= 12: break

    g = build_hetero_graph(sched.ships, sched.tasks, rn)

    # ================================================================
    # 1. 异构图可视化
    # ================================================================
    print("=" * 70)
    print("  一、异构图结构")
    print("=" * 70)

    print(f"""
节点 (3种):
  Ship  ({g.M}个, 6维)  —  [max_payload, max_speed, max_energy, load, health, status]
  Task  ({g.K}个, 4维)  —  [payload, deadline, priority, status]
  Road  ({g.N}个, 5维)  —  [x, y, node_type, degree, capacity]

边 (5种):
  Road↔Road  {g.rr_edges.shape[1]:>4}条  — [length, travel_time, energy_cost, width, congestion]
  Ship→Road  {g.sr_edges.shape[1]:>4}条  — [distance, arrival_time, is_current]
  Task→Road  {g.tr_edges.shape[1]:>4}条  — [edge_type, load_weight, operation_time]
  Ship↔Task  {g.st_edges.shape[1]:>4}条  — [relation, cost, travel_d, exec_d, eta, energy, match_score]
  Task↔Task  {g.tt_edges.shape[1]:>4}条  — [relation, spatial_dist, time_gap, priority_gap]
""")

    # 打印具体节点属性
    print("── Ship 节点属性 ──")
    for i, sid in enumerate(g.ship_ids):
        s = sched.ships[sid]
        print(f"  Ship_{sid}: max_payload={s.max_payload}t max_speed={s.max_speed}m/s "
              f"max_energy={s.max_energy}kWh load={s.load:.0f}t health={s.health:.1f} "
              f"status={s.current_phase}")

    print("\n── Task 节点属性 ──")
    for j, tid in enumerate(g.task_ids[:6]):
        t = sched.tasks[tid]
        pn = node_names.get(t.pickup_node, '?')
        dn = node_names.get(t.delivery_node, '?')
        print(f"  T{tid}: {pn}→{dn} {t.payload}t pri={t.priority} status={t.status}")

    print("\n── Road↔Road 边属性 (前5条) ──")
    for k in range(min(5, g.rr_edges.shape[1])):
        u, v = g.rr_edges[0, k], g.rr_edges[1, k]
        f = g.rr_feat[k]
        print(f"  N{u}↔N{v}: length={f[0]:.0f}m time={f[1]:.0f}s energy={f[2]:.2f}kWh width={f[3]:.0f}m cong={f[4]:.1f}")

    print("\n── Ship↔Task 边属性 (前5条) ──")
    for k in range(min(5, g.st_edges.shape[1])):
        si, tj = g.st_edges[0, k], g.st_edges[1, k]
        f = g.st_feat[k]
        rel_names = ['candidate', 'assigned', 'executing', 'finished']
        rel = rel_names[int(f[0])] if int(f[0]) < 4 else '?'
        print(f"  Ship_{si}↔T{tj}: {rel} cost={f[1]:.0f} travel={f[2]:.0f}m "
              f"exec={f[3]:.0f}m eta={f[4]:.0f}s energy={f[5]:.2f}kWh match={f[6]:.2f}")

    # ================================================================
    # 2. 异构网络可视化图
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(26, 16))

    # 2a: 路网 + 船位
    ax = axes[0, 0]
    img = np.array(Image.open('/root/demon3.16/data/maps/binary_map_scaled.png'))
    if img.ndim == 3: img = img[:,:,0]
    bg = np.flipud(img); h, w = bg.shape; ps = 2.0; wh = h*ps
    ax.imshow(bg, extent=[0, w*ps, 0, wh], origin='lower', cmap='gray')
    for e in rn.edges:
        n1, n2 = rn.nodes[e.from_id], rn.nodes[e.to_id]
        ax.plot([n1.x, n2.x], [wh-n1.y, wh-n2.y], 'cyan', lw=0.6, alpha=0.3, zorder=1)
    for n in rn.nodes.values():
        if n.is_port:
            ax.scatter(n.x, wh-n.y, c='red', s=40, marker='s', zorder=4)
            ax.annotate(n.port_name, (n.x, wh-n.y+15), fontsize=6, color='white', ha='center')
        elif n.is_gas_station:
            ax.scatter(n.x, wh-n.y, c='lime', s=30, marker='^', zorder=4)
    colors = ['#ff3333','#3388ff','#33cc33','#ff9933']
    for i, sid in enumerate(g.ship_ids):
        node = rn.nodes.get(sched.ships[sid].current_node)
        if node:
            ax.scatter(node.x, wh-node.y, c=colors[i], s=200, marker='*', edgecolors='white', lw=2, zorder=6)
    wr, wc = np.where(bg>127); m=30
    ax.set_xlim(max(0,wc.min()-m)*ps, min(w,wc.max()+m)*ps)
    ax.set_ylim(max(0,wr.min()-m)*ps, min(h,wr.max()+m)*ps)
    ax.set_title('Road Network + Ships', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')

    # 2b: Ship↔Task 代价热力图
    ax = axes[0, 1]
    cm = np.full((g.M, g.K), np.nan)
    for k in range(g.st_edges.shape[1]):
        si, tj = g.st_edges[0, k], g.st_edges[1, k]
        cm[si, tj] = g.st_feat[k, 1] / 1000.0  # km
    im = ax.imshow(cm, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(g.K))
    ax.set_xticklabels([f'T{j}' for j in range(g.K)], fontsize=7, rotation=90)
    ax.set_yticks(range(g.M))
    ax.set_yticklabels([f'Ship_{i}' for i in range(g.M)], fontsize=9)
    ax.set_title('Ship→Task Cost Matrix (km)\nGreen=Cheap Red=Expensive White=Infeasible', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 2c: Road↔Road 边属性分布
    ax = axes[0, 2]
    ax.hist(g.rr_feat[:, 0], bins=30, color='steelblue', alpha=0.7)
    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Count')
    ax.set_title('Road Edge Length Distribution', fontsize=10, fontweight='bold')

    # 2d: Ship 节点特征
    ax = axes[1, 0]
    im = ax.imshow(g.ship_x, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(6))
    ax.set_xticklabels(['max_payload','max_speed','max_energy','load','health','status'], fontsize=7, rotation=45)
    ax.set_yticks(range(g.M))
    ax.set_yticklabels([f'Ship_{i}' for i in range(g.M)], fontsize=9)
    ax.set_title('Ship Node Features', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 2e: Task 节点特征
    ax = axes[1, 1]
    im = ax.imshow(g.task_x, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['payload','deadline','priority','status'], fontsize=7, rotation=45)
    ax.set_yticks(range(g.K))
    ax.set_yticklabels([f'T{j}' for j in range(g.K)], fontsize=7)
    ax.set_title('Task Node Features', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 2f: 边类型统计
    ax = axes[1, 2]
    edge_counts = [g.rr_edges.shape[1], g.sr_edges.shape[1], g.tr_edges.shape[1],
                   g.st_edges.shape[1], g.tt_edges.shape[1]]
    edge_names = ['Road↔Road', 'Ship→Road', 'Task→Road', 'Ship↔Task', 'Task↔Task']
    bars = ax.bar(edge_names, edge_counts, color=['#5599dd','#dd9944','#44bb44','#dd4444','#9944dd'])
    for b, v in zip(bars, edge_counts):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+2, str(v), ha='center', fontsize=9)
    ax.set_title('Edge Type Distribution', fontsize=10, fontweight='bold')
    ax.set_ylabel('Count')

    plt.tight_layout()
    viz_path = f'{BASE}/../output/hetero_graph_detail.png'
    plt.savefig(viz_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n异构图可视化: {viz_path}")

    # ================================================================
    # 3. 最小费用流分配
    # ================================================================
    print(f"\n{'='*70}")
    print("  二、最小费用流分配 (Min-Cost Flow)")
    print("=" * 70)
    print("""
算法: Successive Shortest Path (SPFA)

流网络:
  Source ──[容量=船可执行任务数, 费用=0]──→ Ship_i
  Ship_i ──[容量=1, 费用=cost(ship_i,task_j)]──→ Task_j
  Task_j ──[容量=1, 费用=0]──→ Sink

cost(ship, task) = d_pickup + d_exec + load_balance + energy_risk + urgency
  读取自异构图 Ship↔Task 边的 estimated_cost 字段

每次迭代用 SPFA 找最短增广路, 沿路径推流 1 单位。
迭代 ≤ K 次 (每次分配一个任务), 总复杂度 O(M×K²)。
""")

    assignments, total_cost, flow = min_cost_flow_allocate(sched.ships, sched.tasks, g)
    for s in sched.ships.values():
        optimize_sequence(s, sched.tasks, rn.dist_matrix)

    print(f"结果: {flow}/{g.K} 任务分配, 总代价={total_cost:.0f}")
    for ship in sorted(sched.ships.values(), key=lambda s: s.ship_id):
        tasks_str = " → ".join(f"T{t}" for t in ship.task_sequence)
        total_d = sum(rn.dist_matrix[sched.tasks[tid].pickup_node,
                                      sched.tasks[tid].delivery_node]
                     for tid in ship.task_sequence if tid in sched.tasks)
        print(f"  {ship.name}: [{tasks_str}] {total_d/1000:.1f}km")
        for tid in ship.task_sequence:
            t = sched.tasks[tid]
            pn = node_names.get(t.pickup_node, '?')
            dn = node_names.get(t.delivery_node, '?')
            c = rn.dist_matrix[ship.current_node, t.pickup_node] + rn.dist_matrix[t.pickup_node, t.delivery_node]
            print(f"    T{tid}: {pn}→{dn} {t.payload}t pri={t.priority} cost={c/1000:.1f}km")

    # ================================================================
    # 4. 拍卖法重分配
    # ================================================================
    print(f"\n{'='*70}")
    print("  三、拍卖法动态重分配 (Auction Algorithm)")
    print("=" * 70)
    print("""
原理:
  任务 = 物品 (每个有当前价格 p_j)
  船 = 竞拍者 (选择收益最高的任务出价)
  profit(ship, task) = -cost(ship, task) - price(task)

每轮:
  1. 每艘船找出收益最高和次高的任务
  2. 出价 = (best - second_best) + ε
  3. 更新价格, 出价最高的船获得任务
  4. 价格收敛 → 分配稳定

固定规则: 正在装卸/在途的任务不参与拍卖 (防止中断)
""")

    # 场景1: 能源紧急
    print("── 场景1: Ship_0 能源紧急 ──")
    sched.ships[0].energy = sched.ships[0].max_energy * 0.02
    sched.ships[1].task_sequence.clear(); sched.ships[1].load = 0

    before = {sid: list(s.task_sequence) for sid, s in sched.ships.items()}
    decisions, release_pool, idle_ships = evaluate_fleet(sched.ships, sched.tasks, rn, 0)

    for sid, (action, reason, state) in decisions.items():
        print(f"  {sched.ships[sid].name}: {action} — {reason}")
        if action == 'release_all':
            print(f"    energy_score={state['energy_score']:.2f} ({state.get('energy_detail','')})")

    if release_pool:
        update_graph(g, sched.ships, sched.tasks, rn, 0)
        fixed_ships, fixed_tasks = build_fixed_sets(sched.ships, sched.tasks)
        print(f"  固定(不参与拍卖): Ships={fixed_ships}, Tasks={fixed_tasks}")

        new_assignments, prices, rounds = auction_reallocate(
            sched.ships, sched.tasks, g, fixed_ships=fixed_ships,
            fixed_tasks=fixed_tasks, epsilon=1.0)
        apply_auction_result(sched.ships, sched.tasks, new_assignments, fixed_tasks)
        print(f"  拍卖: {rounds}轮收敛, {len(new_assignments)}个新分配")

        after = {sid: list(s.task_sequence) for sid, s in sched.ships.items()}
        print(f"  前: { {k:v for k,v in before.items()} }")
        print(f"  后: { {k:v for k,v in after.items()} }")

        transferred = 0
        for tid in sched.tasks:
            old = None
            for sid, tasks_before in before.items():
                if tid in tasks_before: old = sid; break
            new = sched.tasks[tid].assigned_ship
            if old is not None and new != old and new >= 0:
                t = sched.tasks[tid]
                pn = node_names.get(t.pickup_node,'?'); dn = node_names.get(t.delivery_node,'?')
                print(f"  T{tid} ({pn}→{dn}): {sched.ships[old].name} → {sched.ships[new].name}")
                transferred += 1
        if transferred == 0:
            print(f"  (任务保持原分配或未分配)")

    # 场景2: 故障 + 在途货物
    print(f"\n── 场景2: 故障 + 在途货物 ──")
    sched.ships[1].current_phase = 'navigate_to_delivery'
    sched.ships[1].current_task_id = list(sched.ships[1].task_sequence)[0] if sched.ships[1].task_sequence else -1
    sched.ships[1].health = 0.15
    sched.ships[1].energy = sched.ships[1].max_energy * 0.4

    # Re-evaluate after scene change
    decisions2, pool2, idle2 = evaluate_fleet(sched.ships, sched.tasks, rn, 0)
    for sid in sched.ships:
        action, reason, state = decisions2[sid]
        if action != 'normal':
            print(f"  {sched.ships[sid].name}: {action} — {reason}")

    fixed_ships2, fixed_tasks2 = build_fixed_sets(sched.ships, sched.tasks)
    print(f"  固定: Ships={fixed_ships2} (在途/装卸), Tasks={fixed_tasks2}")
    print(f"  Ship_1 当前阶段={sched.ships[1].current_phase} → {'固定不参与' if 1 in fixed_ships2 else '可参与'}拍卖")
    print(f"  Ship_1 当前任务={sched.ships[1].current_task_id} → {'固定不重分配' if sched.ships[1].current_task_id in fixed_tasks2 else '可重分配'}")

    print(f"\n{'='*70}")
    print("  运行: python3 src/task_planner/scripts/output_all_results.py")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

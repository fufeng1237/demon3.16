#!/usr/bin/env python3
"""异构路网可视化 — 完整展示: 路网 + 异构图(Ship/Task/Road) + 约束 + 代价"""

import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from road_network import load_road_network
from hetero_graph import build_hetero_graph, ship_task_cost
from real_time_scheduler import RealTimeScheduler
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    rn = load_road_network(f'{base}/../../output/road_network.json',
                           f'{base}/../../config/ports.yaml')
    port_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_port])
    gas_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_gas_station])
    node_names = {nid: (n.port_name if n.port_name else f'N{nid}')
                  for nid, n in rn.nodes.items()}

    sched = RealTimeScheduler(rn, port_ids, gas_ids, node_names)
    sched.add_ship(0, 'Ship_0', 2000, 500, 8.0, port_ids[0])   # Port_J
    sched.add_ship(1, 'Ship_1', 1500, 400, 7.5, port_ids[5])   # Port_E
    sched.add_ship(2, 'Ship_2', 2500, 600, 7.0, port_ids[9])   # Port_A
    sched.add_ship(3, 'Ship_3', 1800, 450, 8.5, port_ids[3])   # Port_G

    np.random.seed(42)
    tid = 0
    for i, pu in enumerate(port_ids):
        for j, de in enumerate(port_ids):
            if i == j: continue
            d = rn.dist_matrix[pu, de]
            if d <= 0 or d == np.inf: continue
            sched.add_task(tid, pu, de, float(np.random.choice([300,500,800,1000,1500])),
                          int(np.random.choice([1,2,3])), float('inf'))
            tid += 1
            if tid >= 15: break
        if tid >= 15: break

    g = build_hetero_graph(sched.ships, sched.tasks, rn)

    # ======== 大图: 4个子图 ========
    fig = plt.figure(figsize=(30, 22))

    # ── 子图1: 路网 + Ship/Task 标注 ──
    ax1 = fig.add_subplot(2, 3, (1, 2))
    img = np.array(Image.open('/root/demon3.16/data/maps/binary_map_scaled.png'))
    if img.ndim == 3: img = img[:,:,0]
    bg = np.flipud(img)
    h, w = img.shape; ps = 2.0; wh = h*ps; ww = w*ps
    ax1.imshow(bg, extent=[0,ww,0,wh], origin='lower', cmap='gray')

    # 路网边 (青色)
    for e in rn.edges:
        n1, n2 = rn.nodes[e.from_id], rn.nodes[e.to_id]
        ax1.plot([n1.x, n2.x], [wh-n1.y, wh-n2.y], 'cyan', lw=0.8, alpha=0.4, zorder=1)

    # 路网节点 (黄色小点)
    for n in rn.nodes.values():
        if not n.is_port and not n.is_gas_station:
            ax1.scatter(n.x, wh-n.y, c='yellow', s=3, alpha=0.5, zorder=2)

    # 港口 (红色方块), 加油站 (绿色三角)
    for n in rn.nodes.values():
        if n.is_port:
            ax1.scatter(n.x, wh-n.y, c='red', s=80, marker='s', edgecolors='white', lw=1.5, zorder=4)
            ax1.annotate(n.port_name, (n.x, wh-n.y+20), fontsize=7, color='white',
                        fontweight='bold', ha='center', zorder=5)
        elif n.is_gas_station:
            ax1.scatter(n.x, wh-n.y, c='lime', s=60, marker='^', edgecolors='white', lw=1, zorder=4)

    # Ship 位置 (大星号)
    ship_colors = ['#ff3333', '#3388ff', '#33cc33', '#ff9933']
    for i, sid in enumerate(g.ship_ids):
        s = sched.ships[sid]
        node = rn.nodes.get(s.current_node)
        if node:
            ax1.scatter(node.x, wh-node.y, c=ship_colors[i], s=300, marker='*',
                       edgecolors='white', lw=2, zorder=7)
            ax1.annotate(s.name, (node.x+40, wh-node.y-20), fontsize=9,
                        color=ship_colors[i], fontweight='bold', zorder=8)

    # 水区域裁剪
    wr, wc = np.where(bg>127); m=30
    ax1.set_xlim(max(0,wc.min()-m)*ps, min(w,wc.max()+m)*ps)
    ax1.set_ylim(max(0,wr.min()-m)*ps, min(h,wr.max()+m)*ps)
    ax1.set_title('Road Network + Ships + Ports', fontsize=13, fontweight='bold')
    ax1.set_aspect('equal')

    # ── 子图2: 异构图结构 (网络图) ──
    ax2 = fig.add_subplot(2, 3, 3)
    ax2.set_xlim(-1, 15); ax2.set_ylim(-1, 17)
    ax2.axis('off')
    ax2.set_title('Heterogeneous Graph Structure', fontsize=13, fontweight='bold')

    # 绘制概念图
    # Ship 层
    for i in range(4):
        ax2.scatter(2, 14-i*3, c=ship_colors[i], s=200, marker='*', edgecolors='white', lw=2, zorder=5)
        ax2.text(2.5, 14-i*3, f'Ship_{i}', fontsize=9, color=ship_colors[i], fontweight='bold', va='center')
        ax2.text(4.5, 14-i*3-0.3, f'energy={sched.ships[i].energy_ratio*100:.0f}%\nload={sched.ships[i].load:.0f}t\nspeed={sched.ships[i].max_speed}m/s', fontsize=6, color='gray')

    # Road 层
    ax2.scatter([7]*6, range(5,11), c='yellow', s=30, zorder=3)
    ax2.text(7.5, 8, 'Road Nodes (123)', fontsize=9, color='yellow', va='center')
    ax2.text(7.5, 7.3, f'Range: X=[{min(n.x for n in rn.nodes.values()):.0f},{max(n.x for n in rn.nodes.values()):.0f}]m\nPorts: 10, Gas: 4\nEdges: 149, Reachable: 99.2%', fontsize=7, color='gray')

    # Task 层
    for j in range(min(8, g.K)):
        ax2.scatter(12, 15-j*1.5, c='white', s=30, zorder=3)
    ax2.text(12.5, 8, f'Tasks ({g.K})', fontsize=9, color='white', va='center')
    ax2.text(12.5, 7.3, f'4 ship×15 tasks\nweight: 300-1500t\npriority: 1-3', fontsize=7, color='gray')

    # 边标注
    ax2.annotate('', xy=(7,14), xytext=(3,14), arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    ax2.text(5, 14.3, 'Ship→Road (located_at)', fontsize=7, color='gray', ha='center')
    ax2.annotate('', xy=(12,13), xytext=(8,13), arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    ax2.text(10, 13.3, 'Task→Road (pickup/delivery)', fontsize=7, color='gray', ha='center')
    ax2.annotate('', xy=(12,10), xytext=(3,10), arrowprops=dict(arrowstyle='<->', color='#ff4444', lw=1.5))
    ax2.text(7.5, 10.5, 'Ship↔Task (cost edge)', fontsize=7, color='#ff4444', ha='center')

    # ── 子图3: 代价矩阵热力图 ──
    ax3 = fig.add_subplot(2, 3, 4)
    cost_mat = np.zeros((g.M, g.K))
    for i in range(g.M):
        for j in range(g.K):
            c = ship_task_cost(g, i, j, sched.ships, sched.tasks, 0)
            cost_mat[i, j] = c if c < float('inf') else np.nan

    im = ax3.imshow(cost_mat / 1000.0, cmap='RdYlGn_r', aspect='auto', vmax=np.nanmax(cost_mat)/1000)
    ax3.set_xticks(range(g.K))
    ax3.set_xticklabels([f'T{j}' for j in range(g.K)], fontsize=6, rotation=90)
    ax3.set_yticks(range(g.M))
    ax3.set_yticklabels([f'Ship_{i}' for i in range(g.M)], fontsize=8)
    ax3.set_title('Ship→Task Cost Matrix (km)\nGreen=Cheap  Red=Expensive  White=Infeasible', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Cost (km)')

    # 标注不可行的原因
    infeasible = np.sum(np.isnan(cost_mat))
    ax3.text(0.5, -0.15, f'Infeasible pairs: {infeasible}/{g.M*g.K} (overload/energy/distance)',
            transform=ax3.transAxes, fontsize=8, color='gray', ha='center')

    # ── 子图4: Ship 特征 ──
    ax4 = fig.add_subplot(2, 3, 5)
    feature_names = ['x', 'y', 'energy_ratio', 'load_ratio', 'health', 'speed', 'is_idle', 'task_cnt']
    im4 = ax4.imshow(g.ship_x, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
    ax4.set_xticks(range(8))
    ax4.set_xticklabels(feature_names, fontsize=7, rotation=45)
    ax4.set_yticks(range(g.M))
    ax4.set_yticklabels([f'Ship_{i}' for i in range(g.M)], fontsize=8)
    ax4.set_title('Ship Node Features (8-dim)', fontsize=11, fontweight='bold')
    plt.colorbar(im4, ax=ax4)

    # ── 子图5: 约束检查 ──
    ax5 = fig.add_subplot(2, 3, 6)
    ax5.axis('off')
    ax5.set_xlim(0, 10); ax5.set_ylim(0, 12)
    ax5.set_title('Constraints & Cost Function', fontsize=13, fontweight='bold')

    constraints = [
        ('✓ Pass', 'd_pickup + d_exec < ∞', 'Reachable via road network', '#33cc33'),
        ('✓ Pass', 'task.payload ≤ ship.remaining_capacity', 'Load constraint', '#33cc33'),
        ('✓ Pass', 'energy_need ≤ ship.energy × 0.7', 'Energy margin 30%', '#33cc33'),
        ('✓ Pass', 'eta ≤ task.deadline', 'Deadline feasible', '#33cc33'),
        ('✗ Fail', 'Any constraint fails', 'Edge does not exist (cost=∞)', '#ff3333'),
    ]
    for idx, (status, cond, desc, color) in enumerate(constraints):
        y = 11 - idx * 1.4
        ax5.text(0.3, y, f'{status}', fontsize=10, color=color, fontweight='bold')
        ax5.text(1.5, y+0.15, f'{cond}', fontsize=9, color='white', fontweight='bold')
        ax5.text(1.5, y-0.3, f'{desc}', fontsize=7, color='gray')

    y = 11 - len(constraints) * 1.4 - 0.5
    ax5.text(0.3, y, 'Cost(ship, task) =', fontsize=10, color='#ffcc00', fontweight='bold')
    formula = [
        '  d_pickup + d_exec                   ← base travel distance',
        '+ n_existing_tasks × 500              ← load balancing penalty',
        '+ max(0, load_ratio - 0.8) × 10000    ← overload penalty',
        '+ urgency × priority × 2000           ← deadline urgency',
        '+ max(0, energy_need - 0.3E) × 10     ← energy risk',
    ]
    for idx, line in enumerate(formula):
        ax5.text(0.5, y-0.5-idx*0.6, line, fontsize=7, color='#aaccff', family='monospace')

    plt.tight_layout()
    out = 'src/task_planner/output/hetero_graph_viz.png'
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='#111111')
    plt.close()
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()

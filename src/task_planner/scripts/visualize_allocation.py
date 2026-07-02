#!/usr/bin/env python3
"""分配结果可视化 — 路网 + 船 + 任务连线"""

import sys, os, json, numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from road_network import load_road_network
from task_assigner import Ship, Task, allocate


def visualize(road_network_json, map_png, output_png,
              ports_config=None, allocation_json=None):
    rn = load_road_network(road_network_json, ports_config)

    # 加载地图 (B版: flip + origin='lower')
    img = np.array(Image.open(map_png))
    if img.ndim == 3: img = img[:, :, 0]
    h, w = img.shape; ps = 2.0; wh = h * ps; ww = w * ps

    bg = np.flipud(img)
    water = bg > 127; wr, wc = np.where(water); m = 30
    r0 = max(0, wr.min() - m) * ps; r1 = min(h, wr.max() + m) * ps
    c0 = max(0, wc.min() - m) * ps; c1 = min(w, wc.max() + m) * ps

    # 找港口/加油站的命名
    port_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_port])
    node_names = {}
    for nid, n in rn.nodes.items():
        if n.is_port: node_names[nid] = n.port_name
        elif n.is_gas_station: node_names[nid] = f'GS_{n.port_name}'
        else: node_names[nid] = f'N{nid}'

    # ── 如果没有分配结果, 运行分配 ──
    if allocation_json and os.path.exists(allocation_json):
        with open(allocation_json) as f:
            data = json.load(f)
        ships = []
        for sd in data['ships']:
            s = Ship(sd['id'], sd['name'], sd['max_payload'], sd['max_energy'],
                     sd['max_speed'], sd['position_x'], sd['position_y'],
                     sd['current_node'], sd['energy'], sd.get('load', 0),
                     energy_per_km=sd.get('energy_per_km', 2.5))
            s.execution_order = sd.get('execution_order', [])
            s.task_queue = sd.get('task_queue', [])
            ships.append(s)
        tasks = [Task.from_dict(td) for td in data.get('tasks', [])]
    else:
        # 自动生成测试数据
        ships = [
            Ship(0, 'Ship_0', 2000, 500, 8, rn.nodes[port_ids[0]].x, rn.nodes[port_ids[0]].y, port_ids[0], 450, energy_per_km=2.5),
            Ship(1, 'Ship_1', 1500, 400, 7.5, rn.nodes[port_ids[5]].x, rn.nodes[port_ids[5]].y, port_ids[5], 360, energy_per_km=2.2),
            Ship(2, 'Ship_2', 2500, 600, 7, rn.nodes[port_ids[9]].x, rn.nodes[port_ids[9]].y, port_ids[9], 540, energy_per_km=3.0),
            Ship(3, 'Ship_3', 1800, 450, 8.5, rn.nodes[port_ids[3]].x, rn.nodes[port_ids[3]].y, port_ids[3], 405, energy_per_km=2.8),
        ]
        tasks = []
        tid = 0
        for i, pu in enumerate(port_ids):
            for j, de in enumerate(port_ids):
                if i == j: continue
                d = rn.dist_matrix[pu, de]
                if d <= 0 or d == np.inf: continue
                tasks.append(Task(tid, pu, de, float(np.random.choice([300,500,800,1000,1500])), 0, 3600*8, int(np.random.choice([1,2,3]))))
                tid += 1
                if tid >= 30: break
            if tid >= 30: break
        from task_assigner import allocate as do_allocate
        allocate(ships, tasks, rn.dist_matrix, True, True, node_id_to_name=node_names)

    # ── 创建图 ──
    fig, axes = plt.subplots(1, 2, figsize=(28, 10))

    ship_colors = ['#ff3333', '#3388ff', '#33cc33', '#ff9933']
    ship_light = ['#ffaaaa', '#aaccff', '#aaffaa', '#ffddaa']

    for ax_idx, ax in enumerate(axes):
        ax.imshow(bg, extent=[0, ww, 0, wh], origin='lower', cmap='gray')

        # 路网 (淡淡的)
        for e in rn.edges:
            n1, n2 = rn.nodes[e.from_id], rn.nodes[e.to_id]
            ax.plot([n1.x, n2.x], [wh - n1.y, wh - n2.y],
                    'white', lw=0.3, alpha=0.15, zorder=1)

        if ax_idx == 0:
            # ── 左图: 港口 + 路网节点 ──
            for n in rn.nodes.values():
                if n.is_port:
                    ax.scatter(n.x, wh - n.y, c='red', s=80, marker='s',
                               edgecolors='white', lw=1, zorder=4)
                    ax.annotate(n.port_name, (n.x, wh - n.y), xytext=(4, 4),
                                textcoords='offset points', fontsize=7,
                                color='white', fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.1', facecolor='red', alpha=0.8), zorder=5)
                elif n.is_gas_station:
                    ax.scatter(n.x, wh - n.y, c='lime', s=60, marker='^',
                               edgecolors='white', lw=1, zorder=4)
            # 船初始位置
            for s in ships:
                node = rn.nodes.get(s.current_node)
                if node:
                    ax.scatter(node.x, wh - node.y, c=ship_colors[s.id], s=150,
                               marker='*', edgecolors='white', lw=1.5, zorder=6)
                    ax.annotate(s.name, (node.x, wh - node.y + 20),
                                fontsize=8, color=ship_colors[s.id], fontweight='bold',
                                ha='center', zorder=7)
            ax.set_title('路网 + 船舶初始位置', fontsize=13, fontweight='bold')

        else:
            # ── 右图: 任务分配线 ──
            task_map = {t.id: t for t in tasks}
            for s in ships:
                color = ship_colors[s.id]
                light = ship_light[s.id]

                # 画船的任务序列: 船位→T0装货→T0卸货→T1装货→T1卸货→...
                prev_node = s.current_node
                for idx, tid in enumerate(s.execution_order):
                    t = task_map.get(tid)
                    if not t: continue
                    pu_node = rn.nodes.get(t.pickup_node)
                    de_node = rn.nodes.get(t.delivery_node)
                    if not pu_node or not de_node: continue

                    # 线: 前一位置 → 装货港
                    pn = rn.nodes.get(prev_node)
                    if pn:
                        ax.plot([pn.x, pu_node.x], [wh - pn.y, wh - pu_node.y],
                                color=light, lw=1.0, alpha=0.5, linestyle='--', zorder=2)

                    # 线: 装货港 → 卸货港 (粗线)
                    ax.annotate('', xy=(de_node.x, wh - de_node.y),
                                xytext=(pu_node.x, wh - pu_node.y),
                                arrowprops=dict(arrowstyle='->', color=color,
                                                lw=2.5, alpha=0.8),
                                zorder=3)

                    # 任务标签
                    mx, my = (pu_node.x + de_node.x) / 2, wh - (pu_node.y + de_node.y) / 2
                    ax.annotate(f'T{tid}', (mx, my - 15), fontsize=7,
                                color=color, fontweight='bold', ha='center',
                                bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.7), zorder=8)

                    prev_node = t.delivery_node

                # 船位置
                node = rn.nodes.get(s.current_node)
                if node:
                    ax.scatter(node.x, wh - node.y, c=color, s=150,
                               marker='*', edgecolors='white', lw=1.5, zorder=6)

            # 港口标签
            for n in rn.nodes.values():
                if n.is_port:
                    ax.annotate(n.port_name, (n.x, wh - n.y + 15),
                                fontsize=7, color='white', fontweight='bold',
                                ha='center', zorder=5)

            ax.set_title('任务分配 (彩色箭头 = 装货→卸货)', fontsize=13, fontweight='bold')

        ax.set_xlim(c0, c1); ax.set_ylim(r0, r1); ax.set_aspect('equal')

    # ── 底部图例 ──
    legend_parts = []
    for i, s in enumerate(ships):
        total_d = 0
        for tid in s.execution_order:
            t = task_map.get(tid)
            if t:
                total_d += rn.dist_matrix[t.pickup_node, t.delivery_node]
        legend_parts.append(f'{ship_colors[i]} ● {s.name}: {len(s.execution_order)}任务, {total_d/1000:.0f}km')
    fig.text(0.5, 0.02, ' | '.join(legend_parts),
             ha='center', fontsize=9, color='white',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#222222', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight', facecolor='black', edgecolor='none')
    plt.close()
    print(f'可视化已保存: {output_png}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--road-network', default='src/task_planner/output/road_network.json')
    p.add_argument('--map', default='data/maps/binary_map_scaled.png')
    p.add_argument('--output', default='src/task_planner/output/allocation_viz.png')
    p.add_argument('--ports-config', default='src/task_planner/config/ports.yaml')
    p.add_argument('--allocation', default='src/task_planner/output/allocation_demo.json')
    args = p.parse_args()

    visualize(args.road_network, args.map, args.output,
              args.ports_config, args.allocation)

#!/usr/bin/env python3
"""
船舶路网节点转移时序图
- 每艘船从初始节点出发, 沿路网最短路径到达各任务港口
- X轴: 时间, Y轴: 路网节点, 彩色线: 船舶轨迹
"""

import sys, os, json, numpy as np
from collections import defaultdict, deque
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from road_network import load_road_network
from task_assigner import Ship, Task, allocate


def reconstruct_paths(rn):
    """从距离矩阵重建所有节点对的最短路径 (前驱矩阵)"""
    n = len(rn.nodes)
    id2idx = {nid: i for i, nid in enumerate(sorted(rn.nodes.keys()))}
    idx2id = {i: nid for nid, i in id2idx.items()}

    dist = np.full((n, n), np.inf)
    nxt = np.full((n, n), -1, dtype=int)  # 前驱矩阵

    for i in range(n):
        dist[i, i] = 0
        nxt[i, i] = i

    for e in rn.edges:
        u, v = id2idx[e.from_id], id2idx[e.to_id]
        w = e.weight
        dist[u, v] = min(dist[u, v], w)
        dist[v, u] = min(dist[v, u], w)
        nxt[u, v] = v
        nxt[v, u] = u

    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            if dist[i, k] == np.inf: continue
            dik = dist[i, k]
            for j in range(n):
                nd = dik + dist[k, j]
                if nd < dist[i, j]:
                    dist[i, j] = nd
                    nxt[i, j] = nxt[i, k]

    def get_path(u_id, v_id):
        """返回从 u_id 到 v_id 的节点序列 (含两端)"""
        ui, vi = id2idx.get(u_id), id2idx.get(v_id)
        if ui is None or vi is None or nxt[ui, vi] == -1:
            return [u_id, v_id]  # 直连
        path = [u_id]
        while ui != vi:
            ui = nxt[ui, vi]
            path.append(idx2id[ui])
        return path

    return get_path, dist, id2idx, idx2id


def main():
    # ── 加载 ──
    rn = load_road_network(
        'src/task_planner/output/road_network.json',
        'src/task_planner/config/ports.yaml')

    port_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_port])
    node_names = {}
    for nid, n in rn.nodes.items():
        if n.is_port: node_names[nid] = n.port_name
        elif n.is_gas_station: node_names[nid] = f'GS_{n.port_name}'
        else: node_names[nid] = f'N{nid}'

    # ── 重构最短路径 ──
    get_path, dist_mat, id2idx, idx2id = reconstruct_paths(rn)

    # ── 船和任务 ──
    ships = [
        Ship(0, 'Ship_0', 2000, 500, 8.0, rn.nodes[port_ids[0]].x, rn.nodes[port_ids[0]].y, port_ids[0], 450, energy_per_km=2.5),
        Ship(1, 'Ship_1', 1500, 400, 7.5, rn.nodes[port_ids[5]].x, rn.nodes[port_ids[5]].y, port_ids[5], 360, energy_per_km=2.2),
        Ship(2, 'Ship_2', 2500, 600, 7.0, rn.nodes[port_ids[9]].x, rn.nodes[port_ids[9]].y, port_ids[9], 540, energy_per_km=3.0),
        Ship(3, 'Ship_3', 1800, 450, 8.5, rn.nodes[port_ids[3]].x, rn.nodes[port_ids[3]].y, port_ids[3], 405, energy_per_km=2.8),
    ]
    tasks = []
    tid = 0
    for i, pu in enumerate(port_ids):
        for j, de in enumerate(port_ids):
            if i == j: continue
            d = dist_mat[id2idx[pu], id2idx[de]]
            if d <= 0 or d == np.inf: continue
            tasks.append(Task(tid, pu, de, float(np.random.choice([300,500,800,1000,1500])), 0, 3600*8, int(np.random.choice([1,2,3]))))
            tid += 1
            if tid >= 20: break
        if tid >= 20: break

    result = allocate(ships, tasks, rn.dist_matrix, True, True, node_id_to_name=node_names)
    ships = result.ships
    tasks = result.tasks
    task_map = {t.id: t for t in tasks}

    # ── 计算每艘船的完整节点轨迹 ──
    ship_timelines = {}  # ship_id -> [(time_sec, node_id), ...]
    ship_colors = ['#ff3333', '#3388ff', '#33cc33', '#ff9933']
    port_nodes_set = set(port_ids)

    for s in ships:
        timeline = []
        t = 0.0
        current_node = s.current_node
        timeline.append((t, current_node, 'start'))

        for tid in s.execution_order:
            task = task_map.get(tid)
            if not task: continue

            # 当前节点 → 装货港
            path1 = get_path(current_node, task.pickup_node)
            for node in path1[1:]:  # 跳过当前节点
                seg_dist = dist_mat[id2idx[current_node], id2idx[node]]
                dt = seg_dist / s.max_speed
                t += dt
                current_node = node
                timeline.append((t, node, 'to_pickup'))

            # 装货时间
            t += 300  # 5 min
            timeline.append((t, task.pickup_node, 'loading'))

            # 装货港 → 卸货港
            path2 = get_path(task.pickup_node, task.delivery_node)
            for node in path2[1:]:
                seg_dist = dist_mat[id2idx[current_node], id2idx[node]]
                dt = seg_dist / s.max_speed
                t += dt
                current_node = node
                timeline.append((t, node, 'to_delivery'))

            # 卸货时间
            t += 180  # 3 min
            timeline.append((t, task.delivery_node, 'unloading'))

        ship_timelines[s.id] = timeline

    # ── 构建 Y 轴: 路网节点 (按顺序排列) ──
    # 使用拓扑顺序: 从上游到下游排列
    all_nodes_in_timeline = set()
    for tl in ship_timelines.values():
        for _, nid, _ in tl:
            all_nodes_in_timeline.add(nid)
    node_list = sorted(all_nodes_in_timeline)  # 按ID排序作为Y轴

    # 也可按节点Y坐标排序来近似上下游
    node_list.sort(key=lambda nid: rn.nodes[nid].y, reverse=True)
    node_to_y = {nid: i for i, nid in enumerate(node_list)}

    # ── 绘图 ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 14),
                                     gridspec_kw={'height_ratios': [3, 1]})

    # == 上图: 时序图 ==
    for s in ships:
        color = ship_colors[s.id]
        tl = ship_timelines[s.id]
        times = [p[0] / 3600.0 for p in tl]  # 小时
        ys = [node_to_y[p[1]] for p in tl]

        ax1.step(times, ys, where='post', color=color, lw=2.5, alpha=0.85, label=s.name)

        # 标记任务港口 (装货/卸货)
        for i, (t_sec, nid, phase) in enumerate(tl):
            if phase == 'loading':
                ax1.scatter(t_sec / 3600.0, node_to_y[nid],
                           c=color, s=80, marker='^', edgecolors='white', lw=1, zorder=5)
            elif phase == 'unloading':
                ax1.scatter(t_sec / 3600.0, node_to_y[nid],
                           c=color, s=80, marker='v', edgecolors='white', lw=1, zorder=5)

    # Y轴标签
    yticks = list(range(len(node_list)))
    ytick_labels = []
    for nid in node_list:
        name = node_names.get(nid, f'N{nid}')
        if nid in port_nodes_set:
            ytick_labels.append(f'■ {name}')  # 港口标记
        elif rn.nodes[nid].is_gas_station:
            ytick_labels.append(f'▲ {name}')
        else:
            ytick_labels.append(f'  {name}')
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(ytick_labels, fontsize=7)

    ax1.set_xlabel('Time (hours)', fontsize=11)
    ax1.set_ylabel('Road Network Node', fontsize=11)
    ax1.set_title('Ship Timeline: Node Transitions on Road Network', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.2, axis='x')

    # 时间范围
    max_time = max(p[0] for tl in ship_timelines.values() for p in tl) / 3600.0
    ax1.set_xlim(0, max_time * 1.05)

    # == 下图: 甘特图 (每艘船的任务时间条) ==
    for s in ships:
        color = ship_colors[s.id]
        tl = ship_timelines[s.id]
        y = s.id

        # 找到任务阶段
        i = 0
        while i < len(tl):
            if tl[i][2] in ('loading', 'to_delivery', 'unloading'):
                start_t = tl[i][0]
                # 找到这个任务结束
                j = i
                while j < len(tl) and tl[j][2] != 'unloading':
                    j += 1
                if j < len(tl):
                    end_t = tl[j][0]
                    # 找到任务ID
                    for tid in s.execution_order:
                        task = task_map.get(tid)
                        if task and task.pickup_node == tl[i][1]:
                            label = f'T{tid}'
                            break
                        elif task and task.delivery_node == tl[i][1]:
                            label = ''
                            break
                    else:
                        label = ''

                    ax2.barh(y, (end_t - start_t) / 3600.0, left=start_t / 3600.0,
                            height=0.6, color=color, alpha=0.7, edgecolor='white')
                    if label:
                        ax2.text((start_t + end_t) / 7200.0, y, label,
                                ha='center', va='center', fontsize=7, color='white', fontweight='bold')
                    i = j + 1
                else:
                    i += 1
            else:
                i += 1

    ax2.set_yticks([s.id for s in ships])
    ax2.set_yticklabels([s.name for s in ships], fontsize=10)
    ax2.set_xlabel('Time (hours)', fontsize=11)
    ax2.set_title('Task Timeline (Gantt)', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, max_time * 1.05)
    ax2.grid(True, alpha=0.2, axis='x')

    # 图例
    legend_patches = [mpatches.Patch(color=ship_colors[s.id], label=f"{s.name}")
                      for s in ships]
    ax2.legend(handles=legend_patches, loc='upper right', fontsize=9)

    plt.tight_layout()
    out = 'src/task_planner/output/timeline.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    # ── 文本输出: 每艘船的详细节点序列 ──
    print(f"Timeline saved: {out}\n")
    for s in ships:
        print(f"{'='*70}")
        print(f"  {s.name}  航速={s.max_speed}m/s")
        print(f"{'='*70}")
        tl = ship_timelines[s.id]
        for t_sec, nid, phase in tl:
            name = node_names.get(nid, f'N{nid}')
            marker = ''
            if phase == 'loading': marker = ' 📦 装货'
            elif phase == 'unloading': marker = ' 📤 卸货'
            elif phase == 'start': marker = ' ▶ 起始'
            print(f"  {t_sec/60:7.1f}min  {name:12s} {marker}")
        print(f"  总耗时: {tl[-1][0]/3600:.1f}h, 途经 {len(set(p[1] for p in tl))} 个节点")
        print()


if __name__ == '__main__':
    main()

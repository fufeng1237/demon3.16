#!/usr/bin/env python3
"""
Route Evolution 可视化 — 三面板联动
  ① 路网地图 (船位置 + 航线)
  ② Route 列表 (当前 + 演化对比)
  ③ Gantt 时间轴 (任务执行进度)

输出: 多帧 GIF + 静态 Route Evolution 图
"""

import sys, os, numpy as np
from copy import deepcopy
from collections import defaultdict
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.animation as animation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))

from road_network import load_road_network
from real_time_scheduler import RealTimeScheduler
from scheduler import Scheduler
from alns_scheduler import RouteNode


def main():
    rn = load_road_network(f'{BASE}/../output/road_network.json',
                           f'{BASE}/../config/ports.yaml')
    port_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_port])
    gas_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_gas_station])
    node_names = {nid: (n.port_name if n.port_name else f'N{nid}')
                  for nid, n in rn.nodes.items()}

    sh = RealTimeScheduler(rn, port_ids, gas_ids, node_names)
    specs = [(0,"S0",2000,500,8.0,0),(1,"S1",1500,400,7.5,5),(2,"S2",2500,600,7.0,9),
             (3,"S3",1800,450,8.5,3),(4,"S4",2200,550,7.8,2),(5,"S5",1600,380,8.2,7)]
    for sid, name, cap, en, sp, pi in specs:
        sh.add_ship(sid, name, cap, en, sp, port_ids[pi])

    np.random.seed(42); tid = 0
    for i, pu in enumerate(port_ids):
        for j, de in enumerate(port_ids):
            if i == j: continue
            d = rn.dist_matrix[pu, de]
            if d <= 0 or d == np.inf: continue
            sh.add_task(tid, pu, de, float(np.random.choice([300,500,800,1000,1200])),
                       int(np.random.choice([1,2,3])), float('inf'))
            tid += 1
            if tid >= 20: break
        if tid >= 20: break

    sched = Scheduler(rn, sh.ships, sh.tasks, port_ids, gas_ids, node_names)
    sched.initialize()

    # ── 图像数据 ──
    img = np.array(Image.open('/root/demon3.16/data/maps/binary_map_scaled.png'))
    if img.ndim == 3: img = img[:,:,0]
    bg = np.flipud(img); h_px, w_px = img.shape; ps = 2.0
    world_w, world_h = w_px * ps, h_px * ps
    water = bg > 127; wr, wc = np.where(water); m = 50
    xlim = (max(0,wc.min()-m)*ps, min(w_px,wc.max()+m)*ps)
    ylim = (max(0,wr.min()-m)*ps, min(h_px,wr.max()+m)*ps)
    colors = ['#ff3333','#3388ff','#33cc33','#ff9933','#ff44ff','#44ffff']

    rn_nodes = rn.nodes
    # Dijkstra 路径缓存
    path_cache = {}
    def get_path(fr, to):
        k = (fr, to)
        if k in path_cache: return path_cache[k]
        n = rn.dist_matrix.shape[0]; dist = np.full(n, np.inf)
        prev = np.full(n, -1, dtype=int); vis = np.zeros(n, dtype=bool)
        dist[fr] = 0
        for _ in range(n):
            u = np.argmin(np.where(vis, np.inf, dist))
            if dist[u] == np.inf: break; vis[u] = True
            if u == to: break
            for v in rn.adj.get(u, []):
                w = rn.dist_matrix[u, v]
                if w < np.inf and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w; prev[v] = u
        path = []; cur = to
        while cur >= 0: path.append(cur); cur = prev[cur]
        path.reverse()
        if path[0] != fr: path = [fr, to]
        path_cache[k] = path; return path

    def route_path_pts(ship_node, route):
        pts = []; cur = ship_node
        for rn in route:
            for nid in get_path(cur, rn.node_id)[1:]:
                nd = rn_nodes.get(nid)
                if nd: pts.append([nd.x, world_h - nd.y])
            cur = rn.node_id
        return pts

    # ── Route 快照历史 ──
    route_snapshots = []
    def take_snapshot(label):
        routes_copy = {}
        for sid in sorted(sched.ships.keys()):
            r = sched.routes.get(sid, [])
            routes_copy[sid] = [rn.node_id for rn in r]
        curs = {sid: s.current_node for sid, s in sched.ships.items()}
        route_snapshots.append((label, curs, routes_copy))

    take_snapshot("t=0")

    # ── 模拟执行, 每完成一个节点记录 ──
    for step in range(25):
        sched.step(60)
        # 检查是否有船完成了节点
        take_snapshot(f"t={(step+1)*1:.0f}min")

    # 选取有变化的快照
    key_snapshots = [route_snapshots[0]]
    prev = route_snapshots[0][2]
    for label, curs, routes in route_snapshots[1:]:
        if routes != prev:
            key_snapshots.append((label, curs, routes))
            prev = routes
    key_snapshots = key_snapshots[:8]  # 最多8帧

    # ======== 构建多帧动画 ========
    def make_frame(frame_idx):
        fig = plt.figure(figsize=(26, 14))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3,
                               height_ratios=[2.5, 1])

        label, curs, routes = key_snapshots[frame_idx]
        prev_label, prev_curs, prev_routes = key_snapshots[max(0, frame_idx-1)]

        # ① 路网地图 (左列, 占两行)
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.imshow(bg, extent=[0, world_w, 0, world_h], origin='lower', cmap='gray')
        for e in rn.edges:
            n1, n2 = rn.nodes[e.from_id], rn.nodes[e.to_id]
            ax1.plot([n1.x, n2.x], [world_h-n1.y, world_h-n2.y], 'cyan', lw=0.3, alpha=0.2, zorder=1)
        for n in rn.nodes.values():
            if n.is_port:
                ax1.scatter(n.x, world_h-n.y, c='red', s=40, marker='s', edgecolors='white', lw=0.5, zorder=3)
                ax1.annotate(n.port_name[:4], (n.x, world_h-n.y+15), fontsize=5, color='white', ha='center')

        for i, sid in enumerate(sorted(sched.ships.keys())):
            cur = curs.get(sid, 0); node = rn_nodes.get(cur)
            route_ids = routes.get(sid, [])
            # 绘制当前 Route 沿路网路径
            if route_ids and node:
                fake_route = [RouteNode(nid, "PASS", -1) for nid in route_ids]
                pts = route_path_pts(cur, fake_route)
                if pts:
                    xs, ys = zip(*pts)
                    ax1.plot(xs, ys, color=colors[i], lw=2, alpha=0.7, zorder=5)
            if node:
                ax1.scatter(node.x, world_h-node.y, c=colors[i], s=150, marker='*',
                           edgecolors='white', lw=1.5, zorder=7)
                ax1.annotate(f'S{i}', (node.x+20, world_h-node.y-10),
                            fontsize=7, color=colors[i], fontweight='bold')
        ax1.set_xlim(xlim); ax1.set_ylim(ylim); ax1.set_aspect('equal')
        ax1.set_title(f'① Road Network + Routes ({label})', fontsize=11, fontweight='bold')

        # ② Route 演化 (右上)
        ax2 = fig.add_subplot(gs[0, 1:])
        ax2.axis('off')
        ax2.set_title('② Route Evolution (before → after)', fontsize=11, fontweight='bold')
        for i, sid in enumerate(sorted(sched.ships.keys())):
            y = 0.95 - i * 0.15
            prev_ids = prev_routes.get(sid, [])
            curr_ids = routes.get(sid, [])
            ax2.text(0.02, y, f'{sched.ships[sid].name}:', fontsize=8, fontweight='bold',
                    color=colors[i], transform=ax2.transAxes)
            prev_str = '→'.join(str(n)[-2:] for n in prev_ids[:6]) if prev_ids else '-'
            curr_str = '→'.join(str(n)[-2:] for n in curr_ids[:6]) if curr_ids else '-'
            changed = prev_str != curr_str
            ax2.text(0.15, y, f'前: [{prev_str}]', fontsize=7, fontfamily='monospace',
                    transform=ax2.transAxes, color='#ff9999' if changed else '#888')
            ax2.text(0.15, y-0.04, f'后: [{curr_str}]', fontsize=7, fontfamily='monospace',
                    transform=ax2.transAxes, color='#99ff99' if changed else '#888')
            if changed:
                ax2.text(0.95, y-0.02, '⟳', fontsize=12, color='yellow', transform=ax2.transAxes, ha='right')

        # ③ Gantt 时间轴 (下右)
        ax3 = fig.add_subplot(gs[1, 1:])
        ax3.set_xlim(0, 30); ax3.set_ylim(0, 7)
        for i, sid in enumerate(sorted(sched.ships.keys())):
            ship = sched.ships[sid]
            # 已完成任务画绿色条
            for j, tid in enumerate(ship.completed_tasks):
                ax3.barh(6-i, 0.5, left=j*1.0, height=0.6, color='#44ff44', alpha=0.7)
            # 未完成任务灰色
            route = sched.routes.get(sid, [])
            n_tasks = len([rn for rn in route if rn.action in ('PICKUP','DELIVERY')])
            for j in range(len(ship.completed_tasks), len(ship.completed_tasks) + n_tasks):
                ax3.barh(6-i, 0.3, left=j*1.0, height=0.4, color='#666666', alpha=0.4)
        ax3.set_yticks([6-i for i in range(len(sched.ships))])
        ax3.set_yticklabels([f'S{i}' for i in range(len(sched.ships))], fontsize=8)
        ax3.set_xlabel('Task index'); ax3.set_title('③ Gantt (Task Progress)', fontsize=11, fontweight='bold')

        # 全局信息
        completed = sum(1 for t in sched.tasks.values() if t.status == 'completed')
        fig.suptitle(f'Route Evolution — {label} | Completed: {completed}/{len(sched.tasks)} | '
                     f'Rolling Horizon Active',
                     fontsize=13, fontweight='bold', y=0.98)

        return fig

    # 生成动画帧
    import tempfile, os
    tmpdir = tempfile.mkdtemp()
    frames = []
    for fi in range(len(key_snapshots)):
        fig = make_frame(fi)
        fpath = f'{tmpdir}/frame_{fi:03d}.png'
        fig.savefig(fpath, dpi=100, bbox_inches='tight', facecolor='#111111')
        plt.close(fig)
        frames.append(fpath)

    # 合并为 GIF
    images = [Image.open(f) for f in frames]
    out = f'{BASE}/../output/route_evolution.gif'
    images[0].save(out, save_all=True, append_images=images[1:], duration=1200, loop=0)
    print(f'Route Evolution: {out}')
    # 最后一帧作为静态图
    import shutil; shutil.copy(frames[-1], f'{BASE}/../output/route_evolution_final.png')
    shutil.rmtree(tmpdir)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""动画: 全船队沿路网移动 + Route + 实时状态 (60帧, 每帧50ms, 快速播放)"""

import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from road_network import load_road_network
from real_time_scheduler import RealTimeScheduler
from scheduler import Scheduler
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

BASE = os.path.dirname(os.path.abspath(__file__))


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
            if tid >= 16: break
        if tid >= 16: break

    scheduler = Scheduler(rn, sh.ships, sh.tasks, port_ids, gas_ids, node_names)
    scheduler.initialize()

    img = np.array(Image.open('/root/demon3.16/data/maps/binary_map_scaled.png'))
    if img.ndim == 3: img = img[:,:,0]
    bg = np.flipud(img); h_px, w_px = img.shape; ps = 2.0
    world_w, world_h = w_px * ps, h_px * ps
    water = bg > 127; wr, wc = np.where(water); m = 50
    xlim = (max(0,wc.min()-m)*ps, min(w_px,wc.max()+m)*ps)
    ylim = (max(0,wr.min()-m)*ps, min(h_px,wr.max()+m)*ps)
    colors = ['#ff3333','#3388ff','#33cc33','#ff9933','#ff44ff','#44ffff',
              '#ffff44','#ff8844','#88ff44','#4488ff']

    n_ships = len(scheduler.ships)
    fig, ax = plt.subplots(figsize=(20, 9))
    ax.imshow(bg, extent=[0, world_w, 0, world_h], origin='lower', cmap='gray')
    for e in rn.edges:
        n1, n2 = rn.nodes[e.from_id], rn.nodes[e.to_id]
        ax.plot([n1.x, n2.x], [world_h-n1.y, world_h-n2.y], 'cyan', lw=0.4, alpha=0.25, zorder=1)
    for n in rn.nodes.values():
        if n.is_port:
            ax.scatter(n.x, world_h-n.y, c='red', s=50, marker='s', edgecolors='white', lw=1, zorder=3)
            ax.annotate(n.port_name[:5], (n.x, world_h-n.y+20), fontsize=6, color='white', ha='center')

    scatters, labels, rlines = [], [], []
    for i in range(n_ships):
        sc = ax.scatter([], [], c=colors[i], s=180, marker='*', edgecolors='white', lw=2, zorder=6)
        scatters.append(sc); rlines.append(ax.plot([],[],color=colors[i],lw=0.7,alpha=0.35,linestyle='--',zorder=2)[0])
        labels.append(ax.annotate('',(0,0),fontsize=7,color=colors[i],fontweight='bold',zorder=7))

    info = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=8, color='white',
                   fontfamily='monospace', va='top',
                   bbox=dict(boxstyle='round', facecolor='#111', alpha=0.85))
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
    ax.set_title(f'{n_ships} Ships × {len(scheduler.tasks)} Tasks — Real-time Transport',
                 fontsize=13, fontweight='bold')

    rn_nodes = rn.nodes
    # Dijkstra 路径缓存: (from, to) → [node_ids]
    path_cache = {}
    def get_path(fr, to):
        k = (fr, to)
        if k in path_cache: return path_cache[k]
        n = rn.dist_matrix.shape[0]
        dist = np.full(n, np.inf); prev = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool); dist[fr] = 0
        for _ in range(n):
            u = np.argmin(np.where(visited, np.inf, dist))
            if dist[u] == np.inf: break
            visited[u] = True
            if u == to: break
            for v in rn.adj.get(u, []):
                w = rn.dist_matrix[u, v]
                if w < np.inf and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w; prev[v] = u
        path = []; cur = to
        while cur >= 0: path.append(cur); cur = prev[cur]
        path.reverse()
        if path[0] != fr: path = [fr, to]
        path_cache[k] = path
        return path

    def route_to_points(ship_node, route):
        """将 Route 展开为沿路网的完整路径点序列"""
        pts = []
        cur = ship_node
        node = rn_nodes.get(cur)
        if node: pts.append([node.x, world_h - node.y])
        for rn in route:
            path = get_path(cur, rn.node_id)
            for nid in path[1:]:  # 跳过起点
                nd = rn_nodes.get(nid)
                if nd: pts.append([nd.x, world_h - nd.y])
            cur = rn.node_id
        return pts

    # 保存初始 Route
    initial_routes = {sid: list(r) for sid, r in scheduler.routes.items()}
    initial_ship_nodes = {sid: s.current_node for sid, s in scheduler.ships.items()}
    # 预计算初始路径
    initial_paths = {}
    for sid in sorted(scheduler.ships.keys()):
        route = initial_routes.get(sid, [])
        if route:
            initial_paths[sid] = route_to_points(initial_ship_nodes[sid], route)

    def update(frame):
        scheduler.step(15)  # 每帧15秒, 慢速展示
        t_min = scheduler.current_time / 60
        completed = sum(1 for t in scheduler.tasks.values() if t.status == 'completed')
        lines = [f"t={t_min:.0f}min  done={completed}/{len(scheduler.tasks)}"]
        for i, sid in enumerate(sorted(scheduler.ships.keys())):
            s = scheduler.ships[sid]; node = rn_nodes.get(s.current_node)
            if node:
                scatters[i].set_offsets([[node.x, world_h - node.y]])
                labels[i].set_position((node.x + 25, world_h - node.y - 15))
                labels[i].set_text(f'{s.name}\ne={s.energy:.0f} l={s.load:.0f}')
            route = scheduler.routes.get(sid, [])
            if route:
                pts = route_to_points(s.current_node, route[:6])
                if len(pts) > 1:
                    xs, ys = zip(*pts); rlines[i].set_data(xs, ys)
            lines.append(f"{s.name}: e={s.energy:.0f} l={s.load:.0f}t route={len(route)}")
        info.set_text('\n'.join(lines))
        return scatters + labels + rlines + [info]

    ani = animation.FuncAnimation(fig, update, frames=80, interval=100, blit=False)
    out_anim = f'{BASE}/../output/animation.gif'
    ani.save(out_anim, writer='pillow', fps=4, dpi=90)
    print(f'Animation: {out_anim}')

    # 静态 Route 总览图 (动画前已保存 initial_routes)
    fig2, ax2 = plt.subplots(figsize=(20, 9))
    ax2.imshow(bg, extent=[0, world_w, 0, world_h], origin='lower', cmap='gray')
    for e in rn.edges:
        n1, n2 = rn.nodes[e.from_id], rn.nodes[e.to_id]
        ax2.plot([n1.x, n2.x], [world_h-n1.y, world_h-n2.y], 'cyan', lw=0.4, alpha=0.25, zorder=1)
    for n in rn.nodes.values():
        if n.is_port:
            ax2.scatter(n.x, world_h-n.y, c='red', s=60, marker='s', edgecolors='white', lw=1, zorder=3)
            ax2.annotate(n.port_name[:5], (n.x, world_h-n.y+25), fontsize=7, color='white', ha='center')
    for i, sid in enumerate(sorted(scheduler.ships.keys())):
        pts = initial_paths.get(sid, [])
        if pts:
            xs, ys = zip(*pts)
            ship_name = scheduler.ships[sid].name
            route_len = len(initial_routes.get(sid, []))
            ax2.plot(xs, ys, color=colors[i], lw=3, alpha=0.8, zorder=5,
                    label=f'{ship_name} ({route_len} nodes, {len(pts)} path pts)')
        node = rn_nodes.get(initial_ship_nodes[sid])
        if node:
            ax2.scatter(node.x, world_h-node.y, c=colors[i], s=250, marker='*',
                       edgecolors='white', lw=2, zorder=6)
        if node:
            ax2.scatter(node.x, world_h-node.y, c=colors[i], s=250, marker='*', edgecolors='white', lw=2, zorder=6)
    ax2.set_xlim(xlim); ax2.set_ylim(ylim); ax2.set_aspect('equal')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.set_title(f'{n_ships} Ships Route Overview — Total Fleet Routes', fontsize=14, fontweight='bold')
    route_png = f'{BASE}/../output/route_overview.png'
    plt.tight_layout(); plt.savefig(route_png, dpi=150, bbox_inches='tight', facecolor='black'); plt.close()
    print(f'Route overview: {route_png}')


if __name__ == '__main__':
    main()

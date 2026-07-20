#!/usr/bin/env python3
"""从 road_network.json 生成 Graph-ALNS 分配路线图 (叠加在地图上)"""
import sys, os, heapq, numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from road_network import load_road_network
from static_assignment import build_static_scene, solve_initial_assignment

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get('TASK_PLANNER_OUTPUT', f'{BASE}/../../output')
PORTS_PATH = os.environ.get('TASK_PLANNER_PORTS', f'{BASE}/../../config/ports.yaml')
USVS_PATH = os.environ.get('TASK_PLANNER_USVS', f'{BASE}/../../config/usvs.txt')
TASKS_PATH = os.environ.get('TASK_PLANNER_TASKS', f'{BASE}/../../config/tasks.txt')

# ── 加载路网 ──
rn = load_road_network(f'{OUT_DIR}/road_network.json', PORTS_PATH)
port_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_port])
gas_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_gas_station])
node_names = {nid: (n.port_name if n.port_name else f'N{nid}')
              for nid, n in rn.nodes.items()}

def find_nearest_node(wx, wy):
    best, best_d = None, float('inf')
    for nid, n in rn.nodes.items():
        d = np.sqrt((n.x - wx) ** 2 + (n.y - wy) ** 2)
        if d < best_d:
            best_d = d
            best = nid
    return best

# ── 读取配置 ──
import re
ship_cfg, task_cfg = [], []
with open(USVS_PATH) as f:
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        p = line.replace('USV:', '').strip().split(',')
        if len(p) >= 8:
            ship_cfg.append((int(p[0]), f'S{int(p[0])}',
                             int(p[4]), float(p[6]), float(p[3]),
                             int(p[1]), int(p[2])))
with open(TASKS_PATH) as f:
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        m = re.match(r'Task\s+(\d+):\s*pickup\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'
                     r'\s*delivery\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', line)
        if m:
            task_cfg.append((int(m.group(1)), int(m.group(2)), int(m.group(3)),
                             int(m.group(4)), int(m.group(5))))

n_ships = len(ship_cfg)
n_tasks = len(task_cfg)

# ── 构建静态快照 & 一次性初始分配（不执行、不重分配） ──
sh = build_static_scene(rn, USVS_PATH, TASKS_PATH, seed=42)

print(f'Running static Graph-ALNS: {n_ships} ships × {n_tasks} tasks...')
routes = solve_initial_assignment(rn, sh.ships, sh.tasks, node_names)

# ── 地图 ──
map_path = os.environ.get('TASK_PLANNER_MAP', '/root/demon3.16/data/maps/binary_map_scaled.png')
img = np.array(Image.open(map_path))
if img.ndim == 3:
    img = img[:, :, 0]
h_px, w_px = img.shape
ps = 2.0
wh = h_px * ps
ww = w_px * ps
bg = np.flipud(img)
water = bg > 127
wr, wc = np.where(water)
m = 50

# ── Dijkstra 路径缓存 ──
path_cache = {}

def get_path(fr, to):
    k = (fr, to)
    if k in path_cache:
        return path_cache[k]
    dist = {fr: 0}
    prev = {}
    q = [(0, fr)]
    while q:
        d, u = heapq.heappop(q)
        if d > dist.get(u, float('inf')):
            continue
        if u == to:
            break
        for v in rn.adj.get(u, []):
            w = rn.dist_matrix[u, v]
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(q, (nd, v))
    if to not in prev:
        path = [fr, to]
    else:
        path = []
        cur = to
        while cur != fr:
            path.append(cur)
            cur = prev[cur]
        path.append(fr)
        path.reverse()
    path_cache[k] = path
    return path


# ── 绘图 ──
colors = ['#ff3333', '#3388ff', '#33cc33', '#ff9933',
          '#ff44ff', '#44ffff', '#ffff44', '#ff8844']
fig, ax = plt.subplots(figsize=(26, 13))
ax.imshow(bg, extent=[0, ww, 0, wh], origin='lower', cmap='gray')

# 路网边 (淡色背景)
for e in rn.edges:
    n1, n2 = rn.nodes[e.from_id], rn.nodes[e.to_id]
    ax.plot([n1.x, n2.x], [wh - n1.y, wh - n2.y],
            'cyan', lw=0.3, alpha=0.2, zorder=1)

# 港口节点
for n in rn.nodes.values():
    if n.is_port:
        ax.scatter(n.x, wh - n.y, c='red', s=30, marker='s',
                   edgecolors='white', lw=0.5, zorder=3)

# 每艘船的路线
legend_items = []
detail_lines = ["Route Details (Graph-ALNS):", ""]

for i, sid in enumerate(sorted(sh.ships.keys())):
    route = routes.get(sid, [])
    ship = sh.ships[sid]
    cur = ship.current_node
    pts = []
    node = rn.nodes.get(cur)
    if node:
        pts.append([node.x, wh - node.y])

    for rnode in route:
        path = get_path(cur, rnode.node_id)
        for nid in path[1:]:
            nd = rn.nodes.get(nid)
            if nd:
                pts.append([nd.x, wh - nd.y])
        cur = rnode.node_id

    if len(pts) > 1:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=colors[i], lw=3, alpha=0.8, zorder=5)

        n_t = len([r for r in route if r.action == 'PICKUP'])
        pu_nodes = [(r.node_id, node_names.get(r.node_id, f'N{r.node_id}'))
                    for r in route if r.action == 'PICKUP']
        de_nodes = [(r.node_id, node_names.get(r.node_id, f'N{r.node_id}'))
                    for r in route if r.action == 'DELIVERY']

        detail_lines.append(f'{ship.name}: {n_t} tasks, {len(route)} nodes')
        for j in range(min(n_t, 4)):
            if j < len(pu_nodes) and j < len(de_nodes):
                detail_lines.append(f'  {pu_nodes[j][1]} → {de_nodes[j][1]}')
        if n_t > 4:
            detail_lines.append(f'  ... +{n_t - 4} more')

        legend_items.append(Patch(color=colors[i],
                                  label=f'{ship.name}: {n_t} tasks'))

# 船起始位置 (星号)
for i, sid in enumerate(sorted(sh.ships.keys())):
    node = rn.nodes.get(sh.ships[sid].current_node)
    if node:
        ax.scatter(node.x, wh - node.y, c=colors[i], s=300, marker='*',
                   edgecolors='white', lw=2, zorder=7)

# 详情文本框
detail_text = '\n'.join(detail_lines)
ax.text(0.02, 0.98, detail_text, transform=ax.transAxes, fontsize=8,
        fontfamily='monospace', va='top', color='white',
        bbox=dict(boxstyle='round', facecolor='#111', alpha=0.85))

ax.set_xlim(max(0, wc.min() - m) * ps, min(w_px, wc.max() + m) * ps)
ax.set_ylim(max(0, wr.min() - m) * ps, min(h_px, wr.max() + m) * ps)
ax.set_aspect('equal')
ax.legend(handles=legend_items, fontsize=8, loc='upper right')
ax.set_title(f'Graph-ALNS Route Allocation: {n_ships} Ships × {n_tasks} Tasks',
             fontsize=14, fontweight='bold')

plt.tight_layout()
out_path = f'{OUT_DIR}/allocation_routes.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='black')
plt.close()
print(f'Saved: {out_path}')

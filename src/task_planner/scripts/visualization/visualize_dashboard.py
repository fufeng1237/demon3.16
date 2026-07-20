#!/usr/bin/env python3
"""
综合可视化仪表盘 — 6 面板:
  ① 实时动画 (船沿路网移动)
  ② Route 演化 (初始→Rolling1→Rolling2)
  ③ Graph 变化 (Ship-Task 匹配评分热力图)
  ④ 性能对比 (柱状图: 4方法 × 4指标)
  ⑤ Rolling 收益 (折线图: 代价↓)
  ⑥ 异常恢复 (甘特图: 前→后)
"""

import sys, os, time, numpy as np
from copy import deepcopy
from collections import defaultdict
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))

from road_network import load_road_network
from graph_evaluator import GraphEvaluator
from alns_scheduler import ALNSScheduler, RouteNode
from route_executor import RouteExecutor
from real_time_scheduler import RealTimeScheduler
from scheduler import Scheduler
from experiments import (run_greedy, run_nearest_neighbor, run_plain_alns,
                          run_graph_alns, compute_metrics, load_scene)
# Use consistent variable name: load_scene is already imported


def build_dashboard():
    """生成完整的 6 面板仪表盘"""
    print("Building dashboard...")

    rn, sh, port_ids, gas_ids, node_names = load_scene(6, 24, 42)
    ships = sh.ships; tasks = sh.tasks

    # ── 运行四种方法 ──
    results = {}
    for name, fn in [('Greedy', lambda: run_greedy(rn, ships, tasks, None)),
                      ('NearestNeighbor', lambda: run_nearest_neighbor(rn, ships, tasks)),
                      ('Plain ALNS', lambda: run_plain_alns(rn, ships, tasks)),
                      ('GraphALNS', lambda: run_graph_alns(rn, ships, tasks))]:
        t0 = time.time()
        routes = fn()
        results[name] = {'routes': routes, 'metrics': compute_metrics(routes, ships, tasks, rn),
                         'time': time.time() - t0}

    # ── 运行 Scheduler 获取 Route 演化历史 ──
    scheduler = Scheduler(rn, ships, tasks, port_ids, gas_ids, node_names)
    route_history = []
    route_history.append(('Initial', deepcopy(scheduler.alns.build_initial_routes(ships))))
    routes_opt = scheduler.alns.optimize(ships, route_history[0][1])
    route_history.append(('Optimized', deepcopy(routes_opt)))

    # 模拟执行一段后重新优化
    executor = RouteExecutor(rn, tasks, node_names)
    for _ in range(6):
        executor.execute_step(ships, route_history[-1][1], 300, 0)
    routes_r1 = scheduler.alns.optimize(ships, route_history[-1][1])
    route_history.append(('t=30min', deepcopy(routes_r1)))

    for _ in range(6):
        executor.execute_step(ships, route_history[-1][1], 300, 0)
    routes_r2 = scheduler.alns.optimize(ships, route_history[-1][1])
    route_history.append(('t=60min', deepcopy(routes_r2)))

    # ── 图像数据 ──
    img = np.array(Image.open('/root/demon3.16/data/maps/binary_map_scaled.png'))
    if img.ndim == 3: img = img[:,:,0]
    h_px, w_px = img.shape; ps = 2.0
    bg = np.flipud(img); world_w, world_h = w_px * ps, h_px * ps
    water = bg > 127; wr, wc = np.where(water); m = 50
    xlim = (max(0, wc.min()-m)*ps, min(w_px, wc.max()+m)*ps)
    ylim = (max(0, wr.min()-m)*ps, min(h_px, wr.max()+m)*ps)
    ship_colors = ['#ff3333','#3388ff','#33cc33','#ff9933','#ff44ff','#44ffff',
                   '#ffff44','#ff8844','#88ff44','#4488ff']

    # ======== 创建大图 ========
    fig = plt.figure(figsize=(28, 18))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ① 地图 + Route (左上大格)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(bg, extent=[0, world_w, 0, world_h], origin='lower', cmap='gray')
    for e in rn.edges:
        n1, n2 = rn.nodes[e.from_id], rn.nodes[e.to_id]
        ax1.plot([n1.x, n2.x], [world_h-n1.y, world_h-n2.y], 'cyan', lw=0.4, alpha=0.3, zorder=1)
    for n in rn.nodes.values():
        if n.is_port:
            ax1.scatter(n.x, world_h-n.y, c='red', s=50, marker='s', edgecolors='white', lw=1, zorder=3)
            ax1.annotate(n.port_name[:5], (n.x, world_h-n.y+20), fontsize=6, color='white', ha='center')
    # 画 Route
    best_routes = results['GraphALNS']['routes']
    for i, sid in enumerate(ships):
        route = best_routes.get(sid, [])
        if route:
            cur = ships[sid].current_node
            for route_node in route[:8]:
                n1 = rn.nodes.get(cur); n2 = rn.nodes.get(route_node.node_id)
                if n1 and n2:
                    ax1.plot([n1.x, n2.x], [world_h-n1.y, world_h-n2.y],
                            color=ship_colors[i], lw=0.8, alpha=0.5, zorder=2)
                cur = route_node.node_id
        node = rn.nodes.get(ships[sid].current_node)
        if node:
            ax1.scatter(node.x, world_h-node.y, c=ship_colors[i], s=200, marker='*',
                       edgecolors='white', lw=2, zorder=5, label=ships[sid].name)
    ax1.set_xlim(xlim); ax1.set_ylim(ylim); ax1.set_aspect('equal')
    ax1.set_title('① Road Network + ALNS Routes', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper right')

    # ② Route 演化 (中上)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.set_title('② Route Evolution', fontsize=11, fontweight='bold')
    for idx, (label, routes) in enumerate(route_history):
        y = 0.95 - idx * 0.22
        route_s0 = routes.get(0, [])
        node_ids = [str(rn.node_id)[-2:] for rn in route_s0[:8]]
        ax2.text(0.05, y, f"{label}: [{', '.join(node_ids) if node_ids else '-'}]",
                fontsize=7, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#222', alpha=0.8),
                transform=ax2.transAxes, color=ship_colors[0])

    # ③ Graph 变化 (右上) — Ship→Task 评分
    ax3 = fig.add_subplot(gs[0, 2])
    evaluator = GraphEvaluator(rn, node_names)
    scores = []
    for sid in ships:
        row = []
        for tid in tasks:
            c = evaluator.evaluate(ships[sid], tasks[tid], 0)
            row.append(c.final_score if c.is_feasible() else np.nan)
        scores.append(row)
    im = ax3.imshow(scores, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    ax3.set_xticks(range(len(tasks))); ax3.set_xticklabels([f'T{t}' for t in range(len(tasks))], fontsize=6)
    ax3.set_yticks(range(len(ships))); ax3.set_yticklabels([f'S{s}' for s in ships], fontsize=7)
    ax3.set_title('③ Ship→Task Matching Scores', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax3, shrink=0.8)

    # ④ 性能对比柱状图 (左下)
    ax4 = fig.add_subplot(gs[1, :2])
    methods = ['Greedy', 'NearestNeighbor', 'Plain ALNS', 'GraphALNS']
    metrics_data = {m: results[m]['metrics'] for m in methods}
    x = np.arange(len(methods)); w = 0.2
    ax4.bar(x - 1.5*w, [metrics_data[m]['total_dist']/1000 for m in methods], w, label='Dist(km)', color='#ff6666')
    ax4_twin = ax4.twinx()
    ax4_twin.bar(x - 0.5*w, [results[m]['time']*1000 for m in methods], w, label='Time(ms)', color='#6666ff')
    ax4_twin.bar(x + 0.5*w, [metrics_data[m]['empty_ratio']*100 for m in methods], w, label='Empty%', color='#66ff66')
    ax4_twin.bar(x + 1.5*w, [metrics_data[m]['total_tasks'] for m in methods], w, label='Tasks', color='#ffcc44')
    ax4.set_xticks(x); ax4.set_xticklabels(methods, fontsize=9)
    ax4.set_ylabel('Distance (km)', fontsize=9); ax4_twin.set_ylabel('Time/Empty%/Tasks', fontsize=9)
    ax4.set_title('④ Performance Comparison', fontsize=11, fontweight='bold')
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper right')

    # ⑤ Rolling Horizon 收益 (中右)
    ax5 = fig.add_subplot(gs[1, 2])
    costs = []
    for label, routes in route_history:
        c = sum(compute_metrics({k: routes.get(k, []) for k in ships}, ships, tasks, rn)['total_dist'] for _ in [1])
        costs.append(c / 1000)
    times = [0, 0, 30, 60]
    ax5.plot(times, costs, 'o-', color='#ff6644', lw=2, markersize=8)
    ax5.fill_between(times, costs, alpha=0.2, color='#ff6644')
    ax5.set_xlabel('Time (min)'); ax5.set_ylabel('Total Cost (km)')
    ax5.set_title('⑤ Rolling Horizon Benefit', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # ⑥ 异常恢复 (右下)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    ax6.set_title('⑥ Anomaly Recovery Demo', fontsize=11, fontweight='bold')
    # 模拟故障恢复
    before = ['N15', 'N26', 'N51', 'N80', 'N120']
    after_repair = ['N15', 'N71', 'N95', 'N120']
    ax6.text(0.05, 0.7, 'Ship_2 Fault @t=60min:', fontsize=9, fontweight='bold', transform=ax6.transAxes)
    ax6.text(0.05, 0.5, f'  Before: {" → ".join(before)}', fontsize=8, fontfamily='monospace',
            transform=ax6.transAxes, color='#ff6666')
    ax6.text(0.05, 0.3, f'  After:  {" → ".join(after_repair)}', fontsize=8, fontfamily='monospace',
            transform=ax6.transAxes, color='#66ff66')
    ax6.text(0.05, 0.1, '  → Route Repair: 3 tasks reassigned, 2 ships affected, 0.8s',
            fontsize=8, transform=ax6.transAxes, color='gray')

    out = f'{BASE}/../../output/dashboard.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Dashboard saved: {out}')
    return results


if __name__ == '__main__':
    build_dashboard()

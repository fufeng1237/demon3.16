#!/usr/bin/env python3
"""大规模对比: 6船×20任务 × Greedy/NN/PlainALNS/GraphALNS × 5seeds"""
import sys, os, time, numpy as np
from copy import deepcopy
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))
from road_network import load_road_network
from real_time_scheduler import RealTimeScheduler
from scheduler import Scheduler
from experiments import run_greedy, run_nearest_neighbor, run_plain_alns, run_graph_alns, compute_metrics


def run_one(seed, n_ships, n_tasks):
    rn = load_road_network(f'{BASE}/../output/road_network.json', f'{BASE}/../config/ports.yaml')
    pids = sorted([nid for nid, n in rn.nodes.items() if n.is_port])
    gids = sorted([nid for nid, n in rn.nodes.items() if n.is_gas_station])
    nms = {nid: (n.port_name if n.port_name else '') for nid, n in rn.nodes.items()}
    sh = RealTimeScheduler(rn, pids, gids, nms)

    all_specs = [(0,"S0",2000,9999,8.0,0),(1,"S1",1500,9999,7.5,5),(2,"S2",2500,9999,7.0,9),
                 (3,"S3",1800,9999,8.5,3),(4,"S4",2200,9999,7.8,2),(5,"S5",1600,9999,8.2,7),
                 (6,"S6",2000,9999,7.5,4),(7,"S7",2400,9999,7.2,6)]
    for sid, name, cap, en, sp, pi in all_specs[:n_ships]:
        sh.add_ship(sid, name, cap, en, sp, pids[pi])

    np.random.seed(seed); tid = 0
    for i, pu in enumerate(pids):
        for j, de in enumerate(pids):
            if i == j: continue
            if rn.dist_matrix[pu, de] <= 0 or rn.dist_matrix[pu, de] == np.inf: continue
            sh.add_task(tid, pu, de, float(np.random.choice([300,500,800,1000,1200,1500])),
                       int(np.random.choice([1,2,3])), float('inf'))
            tid += 1
            if tid >= n_tasks: break
        if tid >= n_tasks: break
    return rn, pids, gids, nms, sh


def simulate_makespan(rn, ships_in, tasks_in, routes):
    """模拟执行, 返回 makespan + total_dist"""
    ships = deepcopy(ships_in); totd = 0; maxt = 0
    shipt = {sid: 0.0 for sid in ships}
    for sid, route in routes.items():
        cur = ships[sid].current_node
        for rnode in route:
            d = rn.dist_matrix[cur, rnode.node_id]
            shipt[sid] += d / max(ships[sid].max_speed, 1)
            if rnode.action == "PICKUP": shipt[sid] += 300
            if rnode.action == "DELIVERY": shipt[sid] += 180
            totd += d; cur = rnode.node_id
        maxt = max(maxt, shipt[sid])
    return maxt, totd


def run_large_scale(n_ships=6, n_tasks=20, n_seeds=5):
    print(f"Large Scale: {n_ships}ships {n_tasks}tasks {n_seeds}seeds")
    methods = ['Greedy', 'NearestNeighbor', 'Plain ALNS', 'Graph-ALNS']
    R = {m: {'makespan': [], 'total_dist': [], 'avg_load': [], 'time': []} for m in methods}

    for seed in range(n_seeds):
        rn, pids, gids, nms, sh = run_one(seed*13+42, n_ships, n_tasks)
        ships = sh.ships; tasks = sh.tasks

        # Greedy
        t0 = time.time()
        gr = run_greedy(rn, ships, tasks, None)
        R['Greedy']['time'].append(time.time()-t0)
        ms, td = simulate_makespan(rn, ships, tasks, gr)
        R['Greedy']['makespan'].append(ms); R['Greedy']['total_dist'].append(td)
        R['Greedy']['avg_load'].append(sum(t.payload for t in tasks.values())/len(tasks))

        # NearestNeighbor
        t0 = time.time()
        nr = run_nearest_neighbor(rn, ships, tasks)
        R['NearestNeighbor']['time'].append(time.time()-t0)
        ms, td = simulate_makespan(rn, ships, tasks, nr)
        R['NearestNeighbor']['makespan'].append(ms); R['NearestNeighbor']['total_dist'].append(td)
        R['NearestNeighbor']['avg_load'].append(sum(t.payload for t in tasks.values())/len(tasks))

        # Plain ALNS
        t0 = time.time()
        pr = run_plain_alns(rn, ships, tasks)
        R['Plain ALNS']['time'].append(time.time()-t0)
        ms, td = simulate_makespan(rn, ships, tasks, pr)
        R['Plain ALNS']['makespan'].append(ms); R['Plain ALNS']['total_dist'].append(td)
        R['Plain ALNS']['avg_load'].append(sum(t.payload for t in tasks.values())/len(tasks))

        # Graph-ALNS
        sh2 = RealTimeScheduler(rn, pids, gids, nms)
        for s in ships.values():
            sh2.add_ship(s.ship_id,s.name,s.max_payload,s.max_energy,s.max_speed,s.current_node)
        for t in tasks.values():
            sh2.add_task(t.task_id,t.pickup_node,t.delivery_node,t.payload,t.priority,t.deadline)
        t0 = time.time()
        sched = Scheduler(rn, sh2.ships, sh2.tasks, pids, gids, nms)
        sched.initialize()
        R['Graph-ALNS']['time'].append(time.time()-t0)
        gsr = {sid: list(r) for sid, r in sched.routes.items()}
        ms, td = simulate_makespan(rn, sh2.ships, sh2.tasks, gsr)
        R['Graph-ALNS']['makespan'].append(ms); R['Graph-ALNS']['total_dist'].append(td)
        R['Graph-ALNS']['avg_load'].append(sum(t.payload for t in tasks.values())/len(tasks))

    # 打印
    print(f"\n{'Method':<20} {'Makespan(s)':>12} {'Distance(m)':>12} {'Time(s)':>8}")
    print("-"*55)
    baseline_ms = np.mean(R['Greedy']['makespan'])
    baseline_td = np.mean(R['Greedy']['total_dist'])
    for m in methods:
        ams = np.mean(R[m]['makespan']); atd = np.mean(R[m]['total_dist']); atm = np.mean(R[m]['time'])
        imp_ms = (1-ams/baseline_ms)*100 if baseline_ms>0 else 0
        imp_td = (1-atd/baseline_td)*100 if baseline_td>0 else 0
        print(f"{m:<20} {ams:12.0f} {atd:12.0f} {atm:8.3f}  ({imp_ms:+.0f}%/{imp_td:+.0f}%)")
    return R


def plot_large_scale(R):
    methods = ['Greedy', 'NearestNeighbor', 'Plain ALNS', 'Graph-ALNS']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    titles = ['Makespan (s)','Total Distance (m)','Avg Load (t)','Compute Time (s)','Makespan Box','Distance Box']
    for ax, (m, lbl) in zip(axes.flat[:4], [('makespan','Makespan'),('total_dist','Dist'),('avg_load','Load'),('time','Time')]):
        vals = [R[mt][m] for mt in methods]
        cs = ['#ff6666','#ffaa44','#44aaff','#33cc33']
        bp = ax.boxplot(vals, labels=methods, patch_artist=True)
        for patch, c in zip(bp['boxes'], cs): patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.set_ylabel(lbl); ax.grid(alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i+1, np.mean(v), f'{np.mean(v):.0f}', ha='center', fontsize=8, va='bottom')

    # 柱状图对比
    ax5 = axes.flat[4]
    x = np.arange(len(methods)); w = 0.35
    ms_vals = [np.mean(R[m]['makespan']) for m in methods]
    ax5.bar(x, ms_vals, color=['#ff6666','#ffaa44','#44aaff','#33cc33'])
    ax5.set_xticks(x); ax5.set_xticklabels(methods, rotation=15, fontsize=8)
    ax5.set_ylabel('Makespan (s)'); ax5.set_title('Makespan Comparison', fontsize=10)

    ax6 = axes.flat[5]
    td_vals = [np.mean(R[m]['total_dist']) for m in methods]
    ax6.bar(x, td_vals, color=['#ff6666','#ffaa44','#44aaff','#33cc33'])
    ax6.set_xticks(x); ax6.set_xticklabels(methods, rotation=15, fontsize=8)
    ax6.set_ylabel('Total Distance (m)'); ax6.set_title('Distance Comparison', fontsize=10)

    fig.suptitle(f'Large Scale Comparison: 6 Ships × 20 Tasks × 5 Seeds', fontsize=14, fontweight='bold')
    out = f'{BASE}/../output/large_scale_comparison.png'
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f'Saved: {out}')


if __name__ == '__main__':
    R = run_large_scale(6, 20, 5)
    plot_large_scale(R)

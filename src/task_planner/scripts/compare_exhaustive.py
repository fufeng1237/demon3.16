#!/usr/bin/env python3
"""穷举法对比实验: makespan, 总航程, 平均载荷"""
import sys, os, time, itertools, numpy as np
from copy import deepcopy
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))
from road_network import load_road_network
from real_time_scheduler import RealTimeScheduler
from scheduler import Scheduler


def load_scene(n_ships, n_tasks, seed):
    rn = load_road_network(f'{BASE}/../output/road_network.json', f'{BASE}/../config/ports.yaml')
    pids = sorted([nid for nid, n in rn.nodes.items() if n.is_port])
    gids = sorted([nid for nid, n in rn.nodes.items() if n.is_gas_station])
    nms = {nid: (n.port_name if n.port_name else f'N{nid}') for nid, n in rn.nodes.items()}
    sh = RealTimeScheduler(rn, pids, gids, nms)
    specs = [(0,"S0",2000,9999,8.0,0),(1,"S1",1500,9999,7.5,5),(2,"S2",2500,9999,7.0,9)]
    for sid, name, cap, en, sp, pi in specs[:n_ships]:
        sh.add_ship(sid, name, cap, en, sp, pids[pi])
    np.random.seed(seed); tid = 0
    for i, pu in enumerate(pids):
        for j, de in enumerate(pids):
            if i == j: continue
            if rn.dist_matrix[pu, de] <= 0 or rn.dist_matrix[pu, de] == np.inf: continue
            sh.add_task(tid, pu, de, float(np.random.choice([300,500,800])), int(np.random.choice([1,2])), float('inf'))
            tid += 1
            if tid >= n_tasks: break
        if tid >= n_tasks: break
    return rn, pids, gids, nms, sh


def exhaustive(roadnet, ships_in, tasks_in):
    """穷举所有 ship→task 分配"""
    ships = deepcopy(ships_in); tdict = deepcopy(tasks_in)
    n_t = len(tdict); n_s = len(ships)
    best_cost, best_assign = float('inf'), None
    for assign in itertools.product(range(n_s), repeat=n_t):
        ship_tasks = defaultdict(list); ok = True
        for tid, sid in enumerate(assign):
            if tdict[tid].payload > ships[sid].remaining_capacity: ok = False; break
            ship_tasks[sid].append(tid)
        if not ok: continue
        for sid, tids in ship_tasks.items():
            cur = ships[sid].current_node; ordered = []; rem = list(tids)
            while rem:
                bt = min(rem, key=lambda x: roadnet.dist_matrix[cur, tdict[x].pickup_node] + roadnet.dist_matrix[tdict[x].pickup_node, tdict[x].delivery_node])
                ordered.append(bt); cur = tdict[bt].delivery_node; rem.remove(bt)
            ship_tasks[sid] = ordered
        cost = sum(roadnet.dist_matrix[ships[sid].current_node, tdict[sid].pickup_node] + roadnet.dist_matrix[tdict[sid].pickup_node, tdict[sid].delivery_node] for sid in ship_tasks for sid2 in ship_tasks )
        # Actually compute correctly:
        cost = 0
        for sid, tids in ship_tasks.items():
            cur = ships[sid].current_node
            for tid in tids:
                cost += roadnet.dist_matrix[cur, tdict[tid].pickup_node]
                cost += roadnet.dist_matrix[tdict[tid].pickup_node, tdict[tid].delivery_node]
                cur = tdict[tid].delivery_node
        if cost < best_cost: best_cost, best_assign = cost, dict(ship_tasks)
    return best_assign, best_cost


def simulate(roadnet, ships_in, tasks_in, assign):
    ships = deepcopy(ships_in); totd = 0; maxt = 0; lsum = 0; lcnt = 0
    shipt = {sid: 0.0 for sid in assign}
    for sid, tids in assign.items():
        cur = ships[sid].current_node
        for tid in tids:
            tk = tasks_in[tid]
            d1 = roadnet.dist_matrix[cur, tk.pickup_node]
            d2 = roadnet.dist_matrix[tk.pickup_node, tk.delivery_node]
            shipt[sid] += d1/max(ships[sid].max_speed,1) + 300 + d2/max(ships[sid].max_speed,1) + 180
            totd += d1 + d2; cur = tk.delivery_node; lsum += tk.payload; lcnt += 1
            maxt = max(maxt, shipt[sid])
    return maxt, totd, lsum/max(lcnt,1)


def run_comparison(n_ships=3, n_tasks=6, n_seeds=5):
    print(f"Exhaustive vs ALNS: {n_ships}ships {n_tasks}tasks {n_seeds}seeds")
    R = {'Exhaustive': [], 'ALNS': []}
    for seed in range(n_seeds):
        rn, pids, gids, nms, sh = load_scene(n_ships, n_tasks, seed*10+42)

        t0 = time.time()
        ba, bc = exhaustive(rn, sh.ships, sh.tasks)
        etime = time.time()-t0
        ms, td, al = simulate(rn, sh.ships, sh.tasks, ba)
        R['Exhaustive'].append({'makespan':ms,'total_dist':td,'avg_load':al,'time':etime,'cost':bc})

        sh2 = RealTimeScheduler(rn, pids, gids, nms)
        for s in sh.ships.values():
            sh2.add_ship(s.ship_id,s.name,s.max_payload,s.max_energy,s.max_speed,s.current_node)
        for t in sh.tasks.values():
            sh2.add_task(t.task_id,t.pickup_node,t.delivery_node,t.payload,t.priority,t.deadline)
        sched = Scheduler(rn, sh2.ships, sh2.tasks, pids, gids, nms)
        t0 = time.time(); sched.initialize(); atime = time.time()-t0
        alns_assign = {}
        for sid in sh2.ships:
            tids = []
            for rnode in sched.routes.get(sid,[]):
                if rnode.action=="PICKUP" and rnode.task_id>=0: tids.append(rnode.task_id)
            alns_assign[sid] = tids
        ms2, td2, al2 = simulate(rn, sh2.ships, sh2.tasks, alns_assign)
        R['ALNS'].append({'makespan':ms2,'total_dist':td2,'avg_load':al2,'time':atime,'cost':0})

    print(f"\n{'Metric':>15} {'Exhaustive':>12} {'ALNS':>12} {'Ratio':>10}")
    print("-"*52)
    for m in ['makespan','total_dist','avg_load','time']:
        ev = np.mean([r[m] for r in R['Exhaustive']])
        av = np.mean([r[m] for r in R['ALNS']])
        ratio = av/max(1e-6,ev)
        print(f"{m:>15}: {ev:12.1f} {av:12.1f} {ratio*100:9.1f}%")
    return R


def plot_results(R):
    fig, axes = plt.subplots(1,3,figsize=(16,5))
    for ax,(m,lbl) in zip(axes,[('makespan','Makespan(s)'),('total_dist','Total Distance(m)'),('avg_load','Avg Load(t)')]):
        ev=[r[m] for r in R['Exhaustive']]; av=[r[m] for r in R['ALNS']]
        x=np.arange(len(ev)); w=0.35
        ax.bar(x-w/2,ev,w,label='Exhaustive(Optimal)',color='#ff6666',alpha=0.8)
        ax.bar(x+w/2,av,w,label='Graph-ALNS',color='#3388ff',alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels([f'S{s}' for s in range(len(ev))])
        ax.set_ylabel(lbl); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle('Graph-ALNS vs Exhaustive(Optimal)',fontsize=13,fontweight='bold')
    out=f'{BASE}/../output/exhaustive_comparison.png'
    plt.tight_layout(); plt.savefig(out,dpi=120); plt.close()
    print(f'Saved: {out}')


if __name__=='__main__':
    R = run_comparison(3,6,5)
    plot_results(R)

#!/usr/bin/env python3
"""
实验框架: 对比四种方法的性能

方法:
  1. Greedy — 贪心分配
  2. NearestNeighbor — 最近邻
  3. Plain ALNS — 无 Graph 辅助
  4. Graph-guided ALNS — 本方案

指标:
  - 总航程, 总时间, 空驶率, 任务完成率, 计算时间
  - Rolling Horizon 收益
  - 异常恢复时间
"""

import sys, os, time, numpy as np
from copy import deepcopy
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from road_network import load_road_network
from graph_evaluator import GraphEvaluator
from alns_scheduler import ALNSScheduler, RouteNode
from route_executor import RouteExecutor
from real_time_scheduler import RealTimeScheduler
from scheduler import Scheduler

BASE = os.path.dirname(os.path.abspath(__file__))


def load_scene(n_ships=8, n_tasks=30, seed=42):
    """加载路网 + 船 + 任务 (可扩展规模)"""
    rn = load_road_network(f'{BASE}/../../output/road_network.json',
                           f'{BASE}/../../config/ports.yaml')
    port_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_port])
    gas_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_gas_station])
    node_names = {nid: (n.port_name if n.port_name else f'N{nid}')
                  for nid, n in rn.nodes.items()}

    sh = RealTimeScheduler(rn, port_ids, gas_ids, node_names)

    # 多艘船分布在不同港口
    ship_specs = [
        (0, "S0", 2000, 500, 8.0), (1, "S1", 1500, 400, 7.5),
        (2, "S2", 2500, 600, 7.0), (3, "S3", 1800, 450, 8.5),
        (4, "S4", 2200, 550, 7.8), (5, "S5", 1600, 380, 8.2),
        (6, "S6", 2800, 650, 6.8), (7, "S7", 2000, 480, 7.3),
        (8, "S8", 1700, 420, 7.6), (9, "S9", 2400, 580, 7.2),
    ]
    for sid, name, cap, energy, speed in ship_specs[:n_ships]:
        sh.add_ship(sid, name, cap, energy, speed, port_ids[sid % len(port_ids)])

    np.random.seed(seed); tid = 0
    for i, pu in enumerate(port_ids):
        for j, de in enumerate(port_ids):
            if i == j: continue
            d = rn.dist_matrix[pu, de]
            if d <= 0 or d == np.inf: continue
            dl = 3600 * np.random.uniform(1, 8) if tid < n_tasks//3 else float('inf')
            sh.add_task(tid, pu, de, float(np.random.choice([200,300,500,800,1000,1200,1500])),
                       int(np.random.choice([1,2,3])), dl)
            tid += 1
            if tid >= n_tasks: break
        if tid >= n_tasks: break
    return rn, sh, port_ids, gas_ids, node_names


def run_greedy(rn, ships, tasks, evaluator):
    """贪心: 每次选最近的船"""
    ships_copy = {sid: deepcopy(s) for sid, s in ships.items()}
    tasks_copy = {tid: deepcopy(t) for tid, t in tasks.items()}
    for s in ships_copy.values(): s.task_sequence = []
    for t in tasks_copy.values(): t.assigned_ship = -1

    task_list = sorted(tasks_copy.keys(),
                       key=lambda tid: -tasks_copy[tid].payload * tasks_copy[tid].priority)

    for tid in task_list:
        t = tasks_copy[tid]
        best_sid = None; best_d = float('inf')
        for sid, s in ships_copy.items():
            d = rn.dist_matrix[s.current_node, t.pickup_node] + \
                rn.dist_matrix[t.pickup_node, t.delivery_node]
            if d < best_d and t.payload <= s.remaining_capacity:
                best_d = d; best_sid = sid
        if best_sid is not None:
            ships_copy[best_sid].task_sequence.append(tid)
            tasks_copy[tid].assigned_ship = best_sid

    routes = {}
    for sid, s in ships_copy.items():
        route = []
        for tid in s.task_sequence:
            t = tasks_copy[tid]
            route.append(RouteNode(t.pickup_node, "PICKUP", tid))
            route.append(RouteNode(t.delivery_node, "DELIVERY", tid))
        routes[sid] = route
    return routes


def run_nearest_neighbor(rn, ships, tasks):
    """最近邻: 每船依次选最近的未分配任务"""
    ships_copy = {sid: deepcopy(s) for sid, s in ships.items()}
    tasks_copy = {tid: deepcopy(t) for tid, t in tasks.items()}
    for s in ships_copy.values(): s.task_sequence = []
    unassigned = set(tasks_copy.keys())

    for sid, ship in ships_copy.items():
        cur = ship.current_node
        while unassigned:
            best_tid = None; best_d = float('inf')
            for tid in unassigned:
                t = tasks_copy[tid]
                d = rn.dist_matrix[cur, t.pickup_node] + \
                    rn.dist_matrix[t.pickup_node, t.delivery_node]
                if d < best_d and t.payload <= ship.remaining_capacity:
                    best_d = d; best_tid = tid
            if best_tid is None: break
            ship.task_sequence.append(best_tid)
            tasks_copy[best_tid].assigned_ship = sid
            cur = tasks_copy[best_tid].delivery_node
            unassigned.remove(best_tid)

    routes = {}
    for sid, s in ships_copy.items():
        route = []
        for tid in s.task_sequence:
            t = tasks_copy[tid]
            route.append(RouteNode(t.pickup_node, "PICKUP", tid))
            route.append(RouteNode(t.delivery_node, "DELIVERY", tid))
        routes[sid] = route
    return routes


def run_plain_alns(rn, ships, tasks):
    """普通 ALNS: 不用 Graph Evaluator 的 Top-K, 搜索全部船"""
    evaluator = GraphEvaluator(rn)
    # 提高 K 到船总数 = 搜索全部
    alns = ALNSScheduler(evaluator, tasks, rn)
    alns.K_candidates = len(ships)  # 不用 Graph, 搜索全部
    routes = alns.build_initial_routes(ships)
    routes = alns.optimize(ships, routes)
    return routes


def run_graph_alns(rn, ships, tasks):
    """Graph-guided ALNS: 用 Top-K 候选"""
    evaluator = GraphEvaluator(rn)
    alns = ALNSScheduler(evaluator, tasks, rn)
    alns.K_candidates = 3  # Graph 辅助, 只搜索 Top-3
    routes = alns.build_initial_routes(ships)
    routes = alns.optimize(ships, routes)
    return routes


def compute_metrics(routes, ships, tasks, road_net):
    """计算评估指标"""
    total_dist = 0; total_tasks = 0
    empty_dist = 0  # 空驶 = 非 PICKUP→DELIVERY 的航行
    for sid, route in routes.items():
        ship = ships[sid]
        cur = ship.current_node
        for route_node in route:
            d = road_net.dist_matrix[cur, route_node.node_id]
            total_dist += d
            if route_node.action not in ("PICKUP",):
                empty_dist += d
            cur = route_node.node_id
        total_tasks += len([r for r in route if r.action == "PICKUP"])
    return {
        'total_dist': total_dist,
        'empty_dist': empty_dist,
        'total_tasks': total_tasks,
        'empty_ratio': empty_dist / max(1, total_dist),
        'n_routes': sum(1 for r in routes.values() if r)
    }


def run_experiment(n_ships=6, n_tasks=24, n_runs=3):
    """运行完整对比实验"""
    results = defaultdict(list)
    print(f"\n{'='*60}")
    print(f"  Experiment: {n_ships} ships, {n_tasks} tasks, {n_runs} runs")
    print(f"{'='*60}")

    for run_idx in range(n_runs):
        seed = 42 + run_idx * 10
        rn, sh, port_ids, gas_ids, node_names = load_scene(n_ships, n_tasks, seed)
        ships = sh.ships; tasks = sh.tasks

        methods = {
            'Greedy': lambda: run_greedy(rn, ships, tasks, None),
            'NearestNeighbor': lambda: run_nearest_neighbor(rn, ships, tasks),
            'Plain ALNS': lambda: run_plain_alns(rn, ships, tasks),
            'Graph-guided ALNS': lambda: run_graph_alns(rn, ships, tasks),
        }

        for name, fn in methods.items():
            t0 = time.time()
            routes = fn()
            elapsed = time.time() - t0
            metrics = compute_metrics(routes, ships, tasks, rn)
            metrics['time'] = elapsed
            results[name].append(metrics)

    # 打印汇总
    print(f"\n{'Method':<20} {'Dist(km)':>10} {'Empty%':>8} {'Time(s)':>8} {'Tasks':>8}")
    print("-" * 55)
    for name in ['Greedy', 'NearestNeighbor', 'Plain ALNS', 'Graph-guided ALNS']:
        rs = results[name]
        avg_d = np.mean([r['total_dist'] for r in rs]) / 1000
        avg_e = np.mean([r['empty_ratio'] for r in rs]) * 100
        avg_t = np.mean([r['time'] for r in rs])
        avg_n = np.mean([r['total_tasks'] for r in rs])
        print(f"{name:<20} {avg_d:10.1f} {avg_e:8.1f} {avg_t:8.3f} {avg_n:8.0f}")

    # 对比基线
    baseline = np.mean([r['total_dist'] for r in results['Greedy']])
    ours = np.mean([r['total_dist'] for r in results['Graph-guided ALNS']])
    print(f"\n  Graph-guided ALNS vs Greedy: {(1-ours/baseline)*100:.1f}% improvement")

    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ships', type=int, default=6)
    p.add_argument('--tasks', type=int, default=24)
    p.add_argument('--runs', type=int, default=3)
    args = p.parse_args()
    run_experiment(args.ships, args.tasks, args.runs)

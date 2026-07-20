#!/usr/bin/env python3
"""Generate supervised static Ship--Task ranking data from ALNS teacher solutions."""
import argparse, json, os, random
from copy import deepcopy
from pathlib import Path
import numpy as np
from road_network import load_road_network
from static_assignment import build_static_scene
from hetero_graph import build_hetero_graph
from alns_scheduler import ALNSScheduler
from graph_evaluator import GraphEvaluator


def make_sample(rn, usvs, tasks_file, seed, max_iter=500):
    random.seed(seed); np.random.seed(seed)
    scene = build_static_scene(rn, usvs, tasks_file, seed=seed)
    # Vary the fleet state instead of learning only one fixed deployment.
    rng = np.random.default_rng(seed + 100_003)
    candidate_nodes = [nid for nid, n in rn.nodes.items()
                       if n.degree > 0 and not n.is_task_anchor]
    for ship in scene.ships.values():
        if rng.random() < 0.75:
            ship.current_node = int(rng.choice(candidate_nodes))
        ship.energy *= float(rng.uniform(0.65, 0.95))
    for task in scene.tasks.values():
        task.priority = int(rng.choice([1, 2, 3]))
        if rng.random() < 0.35:
            task.deadline = float(rng.choice([5000, 7000, 9000, 12000]))
    graph = build_hetero_graph(scene.ships, scene.tasks, rn)
    alns = ALNSScheduler(GraphEvaluator(rn, tasks=scene.tasks), scene.tasks, rn,
                         config={'max_iter': max_iter, 'use_graph_candidates': True})
    routes = alns.optimize(scene.ships, alns.build_initial_routes(scene.ships))
    owner = {}
    for sid, route in routes.items():
        for node in route:
            if node.action == 'PICKUP': owner[node.task_id] = sid
    pairs, features, labels, targets, marginal_metrics = [], [], [], [], []
    # Counterfactual target: take a task out of the teacher solution, insert it
    # into every feasible ship route, and measure the actual global increase.
    # This teaches ranking by marginal schedule quality rather than raw distance.
    by_task = {}
    for k in range(graph.st_edges.shape[1]):
        si, tj = map(int, graph.st_edges[:, k])
        by_task.setdefault(graph.task_ids[tj], []).append((k, graph.ship_ids[si], si, tj))
    for tid, candidates in by_task.items():
        context = {sid: [node for node in route if node.task_id != tid]
                   for sid, route in routes.items()}
        context_raw = alns._fleet_cost_raw(context, scene.ships)
        values = []
        for k, sid, si, tj in candidates:
            pu, de, delta = alns._best_insert_pair(scene.ships[sid], context[sid], scene.tasks[tid])
            if delta == float('inf'):
                continue
            trial = {s: list(route) for s, route in context.items()}
            from alns_scheduler import RouteNode
            task = scene.tasks[tid]
            trial[sid].insert(pu, RouteNode(task.pickup_node, 'PICKUP', tid))
            trial[sid].insert(de + 1, RouteNode(task.delivery_node, 'DELIVERY', tid))
            raw = alns._fleet_cost_raw(trial, scene.ships)
            metrics = [max(0.0, raw['M'] - context_raw['M']),
                       max(0.0, raw['D'] - context_raw['D']),
                       max(0.0, raw['E'] - context_raw['E']),
                       max(0.0, raw['B'] - context_raw['B'])]
            values.append((k, sid, si, tj, metrics))
        if not values:
            continue
        arr = np.asarray([x[4] for x in values], dtype=float)
        lo, hi = arr.min(axis=0), arr.max(axis=0)
        normalized = (arr - lo) / np.maximum(hi - lo, 1e-6)
        # Same priority order as the static optimizer: makespan first, then
        # distance, energy and balance.  Higher target means better candidate.
        utilities = 1.0 - normalized.dot(np.array([0.70, 0.15, 0.08, 0.07]))
        for row, utility in zip(values, utilities):
            k, sid, si, tj, metrics = row
            pairs.append([si, tj]); features.append(graph.st_feat[k].tolist())
            labels.append(int(owner.get(tid) == sid)); targets.append(float(utility))
            marginal_metrics.append(metrics)
    return {'seed': seed, 'ship_x': graph.ship_x.tolist(), 'task_x': graph.task_x.tolist(),
            'road_x': graph.road_x.tolist(), 'rr_edges': graph.rr_edges.tolist(), 'rr_feat': graph.rr_feat.tolist(),
            'sr_edges': graph.sr_edges.tolist(), 'sr_feat': graph.sr_feat.tolist(), 'tr_edges': graph.tr_edges.tolist(),
            'tr_feat': graph.tr_feat.tolist(), 'tt_edges': graph.tt_edges.tolist(), 'tt_feat': graph.tt_feat.tolist(),
            'pairs': pairs, 'pair_features': features, 'labels': labels, 'targets': targets,
            'marginal_metrics': marginal_metrics, 'target_definition': 'counterfactual_marginal_v1'}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--samples', type=int, default=200); p.add_argument('--seed-start', type=int, default=0); p.add_argument('--output', default='learning_data.jsonl'); p.add_argument('--tasks', default=os.getenv('TASK_PLANNER_TASKS'))
    p.add_argument('--max-iter', type=int, default=500); a = p.parse_args()
    package = Path(__file__).resolve().parents[2]
    rn = load_road_network(str(package/'output/road_network.json'), str(package/'config/ports.txt'))
    tasks_path = Path(a.tasks) if a.tasks else package/'config/tasks.txt'; output = Path(a.output)
    with output.open('w', encoding='utf-8') as f:
        for seed in range(a.seed_start, a.seed_start + a.samples):
            f.write(json.dumps(make_sample(rn, package/'config/usvs.txt', tasks_path, seed, a.max_iter)) + '\n')
            print(f'{seed-a.seed_start+1}/{a.samples}', flush=True)


if __name__ == '__main__': main()

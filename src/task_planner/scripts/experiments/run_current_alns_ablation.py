#!/usr/bin/env python3
"""Fair ablation for the current Graph/HGT-ALNS implementation."""
import argparse, copy, json, os, random, time
from pathlib import Path
import numpy as np

from road_network import load_road_network
from planning_service import GraphALNSPlanner
from inference import LearnedCandidateScorer
from evaluate_static_gnn import scene, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--tasks', default=os.getenv('TASK_PLANNER_TASKS'))
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--iter', type=int, default=100)
    p.add_argument('--k', type=int, default=4)
    p.add_argument('--output', default='current_alns_ablation.json')
    a = p.parse_args()
    pkg = Path(__file__).resolve().parents[2]
    tasks_path = Path(a.tasks) if a.tasks else pkg / 'config/tasks.txt'
    rn = load_road_network(str(pkg / 'output/road_network.json'), str(pkg / 'config/ports.txt'))
    names = {nid: n.port_name or f'N{nid}' for nid, n in rn.nodes.items()}
    gas = [nid for nid, n in rn.nodes.items() if n.is_gas_station]
    variants = {
        'A0_plain_alns': (None, False, {'use_graph_candidates': False, 'use_adaptive_weights': False,
                                         'destroy_ops': ['random', 'worst', 'shaw', 'energy'],
                                         'post_optimize_moves': 0}),
        'A1_rule_graph': (None, False, {'use_graph_candidates': True, 'post_optimize_moves': 0}),
        'A2_hgt': (a.model, False, {'use_graph_candidates': True, 'post_optimize_moves': 0}),
        'A3_hgt_multistart': (a.model, True, {'use_graph_candidates': True, 'post_optimize_moves': 0}),
        'A4_full_hgt_alns': (a.model, True, {'use_graph_candidates': True, 'post_optimize_moves': 12}),
    }
    raw = {name: [] for name in variants}
    for seed in range(a.seeds):
        base = scene(rn, pkg / 'config/usvs.txt', tasks_path, seed)
        for name, (model_path, multi_start, overrides) in variants.items():
            random.seed(seed); np.random.seed(seed)
            ships, tasks = copy.deepcopy(base)
            planner = GraphALNSPlanner(rn, names, max_iter=a.iter, k_candidates=a.k,
                                       gas_ids=gas, use_multi_start=multi_start,
                                       alns_overrides=overrides)
            if model_path:
                planner.candidate_scorer = LearnedCandidateScorer(model_path, k=a.k)
            start = time.perf_counter(); plans = planner.plan(ships, tasks)
            item = evaluate(rn, ships, tasks, plans); item['runtime_s'] = time.perf_counter() - start
            raw[name].append(item)
            print(f'[{seed * len(variants) + list(variants).index(name) + 1}/{a.seeds * len(variants)}] {name}', flush=True)
    mean = {name: {metric: float(np.mean([r[metric] for r in rows])) for metric in rows[0]}
            for name, rows in raw.items()}
    Path(a.output).write_text(json.dumps({'tasks_path': str(tasks_path), 'raw': raw, 'mean': mean}, indent=2))
    print(json.dumps(mean, indent=2))


if __name__ == '__main__':
    main()

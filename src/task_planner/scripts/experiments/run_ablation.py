#!/usr/bin/env python3
"""B1--B7 静态初始分配消融实验。

不导入 RealTimeScheduler，不调用 step/run，也不做任何重分配。每个 seed 的
七种方法使用完全相同的静态场景快照，统计采用配对方式比较 B3（完整方法）。
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from road_network import load_road_network
from base_scheduler import BaseScheduler
from static_assignment import (build_static_scene, greedy_initial_routes,
                               solve_initial_assignment)


VARIANTS = {
    'B1_Plain_ALNS': {
        'label': 'B1 Plain ALNS',
        'config': {'use_graph_candidates': False, 'use_adaptive_weights': False,
                   'use_sa': True, 'destroy_ops': ['random', 'worst', 'shaw', 'energy'],
                   'repair_ops': ['greedy', 'regret2']},
        'initial': 'alns',
    },
    'B2_Graph_Greedy': {'label': 'B2 Graph + Greedy', 'config': None, 'initial': 'greedy'},
    'B3_Graph_ALNS': {
        'label': 'B3 Graph + ALNS (full)',
        'config': {'use_graph_candidates': True, 'use_adaptive_weights': True,
                   'use_sa': True}, 'initial': 'alns',
    },
    'B4_no_adaptive': {
        'label': 'B4 w/o adaptive weights',
        'config': {'use_graph_candidates': True, 'use_adaptive_weights': False,
                   'use_sa': True}, 'initial': 'alns',
    },
    'B5_no_SA': {
        'label': 'B5 w/o simulated annealing',
        'config': {'use_graph_candidates': True, 'use_adaptive_weights': True,
                   'use_sa': False}, 'initial': 'alns',
    },
    'B6_no_bottleneck': {
        'label': 'B6 w/o bottleneck destroy',
        'config': {'use_graph_candidates': True, 'use_adaptive_weights': True,
                   'use_sa': True,
                   'destroy_ops': ['random', 'worst', 'shaw', 'energy']}, 'initial': 'alns',
    },
    'B7_basic_operators': {
        'label': 'B7 basic operators only',
        'config': {'use_graph_candidates': True, 'use_adaptive_weights': False,
                   'use_sa': True, 'destroy_ops': ['random'],
                   'repair_ops': ['greedy']}, 'initial': 'alns',
    },
}


def sign_test_pvalue(diffs):
    """双侧精确符号检验；无需 scipy。零差异不计入样本。"""
    n_pos = sum(x > 1e-9 for x in diffs)
    n_neg = sum(x < -1e-9 for x in diffs)
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = min(n_pos, n_neg)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def bootstrap_mean_ci(values, rng, n_boot=3000):
    a = np.asarray(values, dtype=float)
    if len(a) < 2:
        return [float(a.mean()), float(a.mean())]
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    means = a[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def evaluate(rn, ships, tasks, routes):
    base = BaseScheduler(rn, ships, tasks)
    return {
        'makespan_s': float(base.makespan(routes)),
        'distance_m': float(base.total_distance(routes)),
        'energy_kwh': float(base.total_energy(routes)),
        'load_std_s': float(base.load_std(routes)),
    }


def main():
    parser = argparse.ArgumentParser(description='Static B1--B7 ablation')
    parser.add_argument('--seeds', type=int, default=int(os.getenv('TASK_PLANNER_ABLATION_SEEDS', '20')))
    parser.add_argument('--seed-start', type=int, default=0,
                        help='first seed index; supports checkpointed batch runs')
    parser.add_argument('--resume', action='store_true',
                        help='merge this batch with the existing raw CSV')
    parser.add_argument('--max-iter', type=int, default=int(os.getenv('TASK_PLANNER_ABLATION_MAX_ITER', '300')))
    parser.add_argument('--output', default=os.getenv('TASK_PLANNER_OUTPUT'))
    args = parser.parse_args()
    if args.seeds < 1 or args.max_iter < 1:
        raise ValueError('--seeds and --max-iter must be positive')

    package = Path(__file__).resolve().parents[2]
    output = Path(args.output or package / 'output')
    output.mkdir(parents=True, exist_ok=True)
    ports = Path(os.getenv('TASK_PLANNER_PORTS', package / 'config/ports.txt'))
    usvs = Path(os.getenv('TASK_PLANNER_USVS', package / 'config/usvs.txt'))
    tasks_path = Path(os.getenv('TASK_PLANNER_TASKS', package / 'config/tasks.txt'))
    # Results may be written to a task-scale-specific directory; the road
    # network remains the package-wide canonical artifact.
    rn = load_road_network(str(package / 'output' / 'road_network.json'), str(ports))
    node_names = {nid: n.port_name or f'N{nid}' for nid, n in rn.nodes.items()}
    records = []
    csv_path = output / 'ablation_b1_b7_raw.csv'
    progress_path = output / 'ablation_progress.json'
    if args.resume and csv_path.exists():
        with csv_path.open(encoding='utf-8') as f:
            records = list(csv.DictReader(f))
        for row in records:
            row['seed'] = int(row['seed'])
            for field in ('makespan_s', 'distance_m', 'energy_kwh', 'load_std_s', 'runtime_s'):
                row[field] = float(row[field])
        # replace rows for seeds explicitly re-run in this batch
        batch_seeds = set(range(args.seed_start, args.seed_start + args.seeds))
        records = [r for r in records if int(r['seed']) not in batch_seeds]

    total_jobs = args.seeds * len(VARIANTS)
    completed_jobs = 0
    progress_path.write_text(json.dumps({'status': 'running', 'completed': 0,
                                         'total': total_jobs, 'percent': 0.0,
                                         'tasks_path': str(tasks_path)}, indent=2), encoding='utf-8')
    print(f'Static ablation: B1--B7, {args.seeds} seeds, ALNS max_iter={args.max_iter}', flush=True)
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        # 同一 seed 的每个变体从独立但相同的静态快照开始。
        for key, spec in VARIANTS.items():
            np.random.seed(seed * 1009 + 17)
            random.seed(seed * 1009 + 17)
            scene = build_static_scene(rn, usvs, tasks_path, seed=seed * 1009 + 17)
            start = time.perf_counter()
            if spec['initial'] == 'greedy':
                routes = greedy_initial_routes(rn, scene.ships, scene.tasks)
            else:
                config = dict(spec['config'])
                config['max_iter'] = args.max_iter
                routes = solve_initial_assignment(rn, scene.ships, scene.tasks,
                                                  node_names, alns_config=config)
            metrics = evaluate(rn, scene.ships, scene.tasks, routes)
            metrics.update({'seed': seed, 'variant': key, 'label': spec['label'],
                            'runtime_s': time.perf_counter() - start})
            records.append(metrics)
            completed_jobs += 1
            progress = {'status': 'running', 'completed': completed_jobs,
                        'total': total_jobs, 'percent': round(100.0 * completed_jobs / total_jobs, 1),
                        'seed': seed, 'variant': key, 'label': spec['label'],
                        'tasks_path': str(tasks_path)}
            progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"  [{completed_jobs:>2}/{total_jobs}] {progress['percent']:>5.1f}% "
                  f"seed={seed} {spec['label']} ({metrics['runtime_s']:.1f}s)", flush=True)
        print(f'  seed {seed - args.seed_start + 1}/{args.seeds} done (global seed {seed})', flush=True)

    fields = ['seed', 'variant', 'label', 'makespan_s', 'distance_m', 'energy_kwh', 'load_std_s', 'runtime_s']
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)

    rng = np.random.default_rng(20260716)
    completed_seeds = sorted({int(r['seed']) for r in records})
    summary = {'experiment': {'static_only': True, 'seeds': len(completed_seeds),
                              'seed_indices': completed_seeds,
                              'alns_max_iter': args.max_iter,
                              'reference': 'B3_Graph_ALNS'}, 'variants': {}}
    by_variant = {v: [r for r in records if r['variant'] == v] for v in VARIANTS}
    ref = by_variant['B3_Graph_ALNS']
    for variant, rows in by_variant.items():
        entry = {'label': VARIANTS[variant]['label'], 'metrics': {}, 'paired_vs_B3': {}}
        for metric in ['makespan_s', 'distance_m', 'energy_kwh', 'load_std_s', 'runtime_s']:
            vals = [r[metric] for r in rows]
            entry['metrics'][metric] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                        'ci95': bootstrap_mean_ci(vals, rng), 'n': len(vals)}
            if variant != 'B3_Graph_ALNS':
                # 正值表示 B3 更优（指标越小）。
                diffs = [other[metric] - base[metric] for other, base in zip(rows, ref)]
                entry['paired_vs_B3'][metric] = {
                    'B3_improvement_mean': float(np.mean(diffs)),
                    'B3_improvement_pct': float(np.mean(diffs) / max(np.mean([r[metric] for r in rows]), 1e-9) * 100),
                    'sign_test_p': sign_test_pvalue(diffs),
                }
        summary['variants'][variant] = entry

    json_path = output / 'ablation_b1_b7_summary.json'
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    md = ['# B1--B7 静态初始分配消融实验', '', f'- 场景：固定配置，{args.seeds} 个配对种子',
          f'- ALNS 迭代上限：{args.max_iter}', '- 不包含执行、事件或重分配。', '',
          '| Variant | Makespan (s) | Distance (m) | Energy (kWh) | Runtime (s) |',
          '|---|---:|---:|---:|---:|']
    for v, item in summary['variants'].items():
        m = item['metrics']
        md.append(f"| {item['label']} | {m['makespan_s']['mean']:.1f} | {m['distance_m']['mean']:.1f} | {m['energy_kwh']['mean']:.2f} | {m['runtime_s']['mean']:.2f} |")
    (output / 'ablation_b1_b7_report.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    progress_path.write_text(json.dumps({'status': 'complete', 'completed': total_jobs,
                                         'total': total_jobs, 'percent': 100.0,
                                         'tasks_path': str(tasks_path)}, indent=2), encoding='utf-8')
    print(f'Wrote: {csv_path}\nWrote: {json_path}\nWrote: {output / "ablation_b1_b7_report.md"}')


if __name__ == '__main__':
    main()

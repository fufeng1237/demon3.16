#!/usr/bin/env python3
"""
统一对比: Greedy vs ALNS vs VNS vs Tabu Search vs GA vs GES vs Memetic
7 个算法, 各 5 种子, 统一评估
"""
import sys, os, re, io, time, contextlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from road_network import load_road_network
from static_assignment import build_static_scene, solve_initial_assignment
from alns_scheduler import RouteNode
from base_scheduler import BaseScheduler
from vns_scheduler import VNSScheduler
from tabu_scheduler import TabuScheduler
from ga_scheduler import GAScheduler
from ges_scheduler import GESScheduler
from memetic_scheduler import MemeticScheduler

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get('TASK_PLANNER_OUTPUT', f'{BASE}/../../output')
PORTS_PATH = os.environ.get('TASK_PLANNER_PORTS', f'{BASE}/../../config/ports.yaml')
USVS_PATH = os.environ.get('TASK_PLANNER_USVS', f'{BASE}/../../config/usvs.txt')
TASKS_PATH = os.environ.get('TASK_PLANNER_TASKS', f'{BASE}/../../config/tasks.txt')
rn = load_road_network(f'{OUTPUT_DIR}/road_network.json', PORTS_PATH)
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
N_SEEDS = int(os.environ.get('TASK_PLANNER_SEEDS', '5'))
if N_SEEDS < 1:
    raise ValueError('TASK_PLANNER_SEEDS must be at least 1')
print(f'对比实验: {n_ships} ships × {n_tasks} tasks, {N_SEEDS} seeds')
print(f'算法: Greedy | ALNS | VNS | Tabu Search | GA | GES | Memetic')
print()


def build_scene(seed):
    """静态快照：不创建 RealTimeScheduler，也不推进或重分配任务。"""
    return build_static_scene(rn, USVS_PATH, TASKS_PATH, seed=seed)


def greedy_route(sh):
    """公平贪心算法"""
    ships = sh.ships
    tasks = sh.tasks
    task_order = sorted(tasks.keys(),
                        key=lambda tid: -tasks[tid].payload * tasks[tid].priority)
    for tid in task_order:
        t = tasks[tid]
        best_sid, best_cost = None, float('inf')
        for sid, s in ships.items():
            if t.payload > s.remaining_capacity: continue
            cur = s.current_node
            if s.task_sequence:
                last_tid = s.task_sequence[-1]
                cur = tasks[last_tid].delivery_node
            d1 = rn.dist_matrix[cur, t.pickup_node]
            d2 = rn.dist_matrix[t.pickup_node, t.delivery_node]
            if d1 == np.inf or d2 == np.inf: continue
            energy_need = (d1 + d2) / 1000.0 * s.energy_per_km
            if energy_need > s.energy * 0.7: continue
            cost = d1 + d2 + len(s.task_sequence) * 300
            if cost < best_cost:
                best_cost = cost
                best_sid = sid
        if best_sid is not None:
            ships[best_sid].task_sequence.append(tid)
    return {sid: list(s.task_sequence) for sid, s in ships.items()}


def assignment_to_routes(sh, assign):
    routes = {}
    for sid in sh.ships:
        route = []
        for tid in assign.get(sid, []):
            t = sh.tasks.get(tid)
            if t is None: continue
            route.append(RouteNode(t.pickup_node, "PICKUP", tid))
            route.append(RouteNode(t.delivery_node, "DELIVERY", tid))
        routes[sid] = route
    return routes


def evaluate(sh, routes):
    """统一评估"""
    base = BaseScheduler(rn, sh.ships, sh.tasks)
    base._norm_base = None  # 每次重新计算归一化基准
    ms = base.makespan(routes)
    td = base.total_distance(routes)
    en = base.total_energy(routes)
    ls = base.load_std(routes)
    return ms, td, en, ls


# ================================================================
ALGOS = ['Greedy', 'ALNS', 'VNS', 'Tabu', 'GA', 'GES', 'Memetic']
results = {a: {'ms': [], 'td': [], 'en': [], 'ls': [], 'tm': []}
           for a in ALGOS}

for seed in range(N_SEEDS):
    np.random.seed(seed * 10 + 42)
    sh = build_scene(seed * 10 + 42)

    # ── 1. Greedy ──
    t0 = time.perf_counter()
    assign_g = greedy_route(sh)
    routes_g = assignment_to_routes(sh, assign_g)
    results['Greedy']['tm'].append(time.perf_counter() - t0)
    ms, td, en, ls = evaluate(sh, routes_g)
    results['Greedy']['ms'].append(ms)
    results['Greedy']['td'].append(td)
    results['Greedy']['en'].append(en)
    results['Greedy']['ls'].append(ls)

    # ── 2. ALNS (以贪心为初始解) ──
    greedy_routes_alns = assignment_to_routes(sh, assign_g)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        routes_alns = solve_initial_assignment(
            rn, sh.ships, sh.tasks, node_names,
            initial_routes=greedy_routes_alns, optimize=True)
    results['ALNS']['tm'].append(time.perf_counter() - t0)
    ms, td, en, ls = evaluate(sh, routes_alns)
    results['ALNS']['ms'].append(ms)
    results['ALNS']['td'].append(td)
    results['ALNS']['en'].append(en)
    results['ALNS']['ls'].append(ls)

    # ── 3. VNS ──
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        vns = VNSScheduler(rn, sh.ships, sh.tasks)
        routes_vns = vns.optimize()
    results['VNS']['tm'].append(time.perf_counter() - t0)
    ms, td, en, ls = evaluate(sh, routes_vns)
    results['VNS']['ms'].append(ms)
    results['VNS']['td'].append(td)
    results['VNS']['en'].append(en)
    results['VNS']['ls'].append(ls)

    # ── 4. Tabu Search ──
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        ts = TabuScheduler(rn, sh.ships, sh.tasks)
        routes_ts = ts.optimize()
    results['Tabu']['tm'].append(time.perf_counter() - t0)
    ms, td, en, ls = evaluate(sh, routes_ts)
    results['Tabu']['ms'].append(ms)
    results['Tabu']['td'].append(td)
    results['Tabu']['en'].append(en)
    results['Tabu']['ls'].append(ls)

    # ── 5. GA ──
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        ga = GAScheduler(rn, sh.ships, sh.tasks)
        routes_ga = ga.optimize()
    results['GA']['tm'].append(time.perf_counter() - t0)
    ms, td, en, ls = evaluate(sh, routes_ga)
    results['GA']['ms'].append(ms)
    results['GA']['td'].append(td)
    results['GA']['en'].append(en)
    results['GA']['ls'].append(ls)

    # ── 6. GES (Guided Ejection Search) ──
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        ges = GESScheduler(rn, sh.ships, sh.tasks)
        routes_ges = ges.optimize()
    results['GES']['tm'].append(time.perf_counter() - t0)
    ms, td, en, ls = evaluate(sh, routes_ges)
    results['GES']['ms'].append(ms)
    results['GES']['td'].append(td)
    results['GES']['en'].append(en)
    results['GES']['ls'].append(ls)

    # ── 7. Memetic Algorithm ──
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        mem = MemeticScheduler(rn, sh.ships, sh.tasks)
        routes_mem = mem.optimize()
    results['Memetic']['tm'].append(time.perf_counter() - t0)
    ms, td, en, ls = evaluate(sh, routes_mem)
    results['Memetic']['ms'].append(ms)
    results['Memetic']['td'].append(td)
    results['Memetic']['en'].append(en)
    results['Memetic']['ls'].append(ls)

    print(f'  seed {seed+1}/{N_SEEDS} done', flush=True)


# ================================================================
# 输出表格
# ================================================================
print(f'\n{"=" * 95}')
print(f'  Results: {n_ships} ships × {n_tasks} tasks, {N_SEEDS} seeds (mean ± std)')
print(f'{"=" * 95}')

for key, label in [('ms', 'Makespan (s)'), ('td', 'Distance (m)'),
                    ('en', 'Energy (kWh)'), ('ls', 'Load StdDev'),
                    ('tm', 'Time (s)')]:
    print(f'\n  {label}:')
    print(f'  {"Algo":<12} {"Mean":>12} {"Std":>10} {"vs Greedy":>10} {"Best":>12}')
    print(f'  {"-" * 56}')
    greedy_mean = np.mean(results['Greedy'][key])
    for algo in ALGOS:
        vals = results[algo][key]
        m = np.mean(vals)
        s = np.std(vals)
        pct = (m / greedy_mean - 1) * 100 if greedy_mean > 0 else 0
        arrow = '↑' if pct > 0 else '↓'
        best_v = min(vals) if key != 'tm' else m
        print(f'  {algo:<12} {m:12.1f} {s:10.1f} {abs(pct):7.1f}% {arrow}  {best_v:12.1f}')

# ── 综合排名 ──
print(f'\n{"=" * 95}')
print(f'  Composite Ranking (makespan × 0.5 + distance_norm × 0.3 + energy_norm × 0.2)')
print(f'{"=" * 95}')

# Per-seed normalization
seeds_composite = {a: [] for a in ALGOS}
for s in range(N_SEEDS):
    ms_max = max(results[a]['ms'][s] for a in ALGOS)
    td_max = max(results[a]['td'][s] for a in ALGOS)
    en_max = max(results[a]['en'][s] for a in ALGOS)
    for a in ALGOS:
        cs = (0.5 * results[a]['ms'][s] / ms_max +
              0.3 * results[a]['td'][s] / td_max +
              0.2 * results[a]['en'][s] / en_max)
        seeds_composite[a].append(cs)

print(f'  {"Algo":<12} {"Composite":>12} {"Rank":>6}')
print(f'  {"-" * 30}')
ranked = sorted(ALGOS, key=lambda a: np.mean(seeds_composite[a]))
for rank, algo in enumerate(ranked, 1):
    print(f'  {algo:<12} {np.mean(seeds_composite[algo]):12.4f} {rank:>6}')

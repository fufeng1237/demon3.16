#!/usr/bin/env python3
"""
公平对比: Greedy vs Graph-ALNS
Greedy 完整适配本问题约束: pickup→delivery顺序, 容量, 能耗, 可达性
对比指标: makespan, distance, energy, load_std, composite_cost
"""
import sys, os, re, io, time, contextlib, numpy as np
from copy import deepcopy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from road_network import load_road_network
from scheduler import Scheduler
from real_time_scheduler import RealTimeScheduler
from alns_scheduler import RouteNode

BASE = os.path.dirname(os.path.abspath(__file__))
rn = load_road_network(f'{BASE}/../../output/road_network.json',
                       f'{BASE}/../../config/ports.yaml')
port_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_port])
gas_ids = sorted([nid for nid, n in rn.nodes.items() if n.is_gas_station])
node_names = {nid: (n.port_name if n.port_name else f'N{nid}')
              for nid, n in rn.nodes.items()}


def find_nearest_node(wx, wy):
    """世界坐标 → 最近路网节点"""
    best, best_d = None, float('inf')
    for nid, n in rn.nodes.items():
        d = np.sqrt((n.x - wx) ** 2 + (n.y - wy) ** 2)
        if d < best_d:
            best_d = d
            best = nid
    return best


# ── 读取配置 (自动检测规模) ──
ship_cfg, task_cfg = [], []
with open(f'{BASE}/../../config/usvs.txt') as f:
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        p = line.replace('USV:', '').strip().split(',')
        if len(p) >= 8:
            ship_cfg.append((int(p[0]), f'S{int(p[0])}',
                             int(p[4]), float(p[6]), float(p[3]),
                             int(p[1]), int(p[2])))
with open(f'{BASE}/../../config/tasks.txt') as f:
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        m = re.match(r'Task\s+(\d+):\s*pickup\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'
                     r'\s*delivery\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', line)
        if m:
            task_cfg.append((int(m.group(1)), int(m.group(2)), int(m.group(3)),
                             int(m.group(4)), int(m.group(5))))

n_ships = len(ship_cfg)
n_tasks = len(task_cfg)
u_ships = ship_cfg[:n_ships]
u_tasks = task_cfg[:n_tasks]
print(f'Greedy vs Graph-ALNS: {n_ships} ships × {n_tasks} tasks')
print(f'Constraints: pickup→delivery, capacity, energy≤70%, reachability, load-balance')
print(f'Composite = 0.4×Dist_norm + 0.3×Energy_norm + 0.3×LoadStd_norm')


# ================================================================
#  Scene builder (接受 seed 参数, 不再内部 reset)
# ================================================================
def build_scene(seed=None):
    """构建场景. seed 只影响 task payload 的随机选择."""
    sh = RealTimeScheduler(rn, port_ids, gas_ids, node_names)
    for sid, name, cap, en, sp, px, py in u_ships:
        sh.add_ship(sid, name, cap, en, sp, find_nearest_node(px * 2, py * 2))
    for tid, ppx, ppy, dpx, dpy in u_tasks:
        sh.add_task(tid,
                    find_nearest_node(ppx * 2, ppy * 2),
                    find_nearest_node(dpx * 2, dpy * 2),
                    float(np.random.choice([300, 500, 800, 1000, 1500])),
                    int(np.random.choice([1, 2, 3])),
                    float('inf'))
    return sh


# ================================================================
#  Fair Greedy — 完整约束适配
# ================================================================
def greedy_route(sh):
    """
    公平贪心算法, 适配本问题全部约束:
      1. 容量: payload ≤ remaining_capacity
      2. 可达性: d1,d2 ≠ inf
      3. 能耗: energy_need ≤ energy × 0.7 (保留30%安全余量)
      4. Pickup→Delivery 顺序: 任务从 delivery_node 出发继续
      5. 负载均衡: cost += len(seq) × 300 惩罚
    按 载重×优先级 降序处理任务 (大任务优先).
    """
    ships = sh.ships
    tasks = sh.tasks
    # 大载重 × 高优先级先分配
    task_order = sorted(tasks.keys(),
                        key=lambda tid: -tasks[tid].payload * tasks[tid].priority)

    for tid in task_order:
        t = tasks[tid]
        best_sid, best_cost = None, float('inf')

        for sid, s in ships.items():
            # 约束1: 容量
            if t.payload > s.remaining_capacity:
                continue

            # 当前船的最后位置 (从 delivery_node 出发, 保证 P→D 顺序)
            cur = s.current_node
            if s.task_sequence:
                last_tid = s.task_sequence[-1]
                cur = tasks[last_tid].delivery_node

            d1 = rn.dist_matrix[cur, t.pickup_node]
            d2 = rn.dist_matrix[t.pickup_node, t.delivery_node]

            # 约束2: 可达性
            if d1 == np.inf or d2 == np.inf:
                continue

            # 约束3: 能耗 (保留30%安全余量)
            energy_need = (d1 + d2) / 1000.0 * s.energy_per_km
            if energy_need > s.energy * 0.7:
                continue

            # 代价 = 距离 + 负载均衡惩罚
            cost = d1 + d2 + len(s.task_sequence) * 300
            if cost < best_cost:
                best_cost = cost
                best_sid = sid

        if best_sid is not None:
            ships[best_sid].task_sequence.append(tid)

    return {sid: list(s.task_sequence) for sid, s in ships.items()}


# ================================================================
#  Unified evaluation — 统一使用 ALNS 的代价函数 (跟踪实际路线)
# ================================================================
def assignment_to_routes(sh, assign):
    """将任务ID分配转为 RouteNode 路线 (顺序: P→D 对)"""
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


def evaluate_routes(sh, routes):
    """统一评估: 使用 ALNS 的代价函数 (跟踪实际 RouteNode 序列, 支持任务交叉)"""
    from alns_scheduler import ALNSScheduler
    # 创建轻量 ALNS 实例 (仅用于代价计算)
    alns = ALNSScheduler.__new__(ALNSScheduler)
    alns.tasks = sh.tasks
    alns.rn = rn
    alns.loading_time = 300
    alns.unloading_time = 180
    alns.load_penalty_factor = 0.5
    alns._prev_routes = None

    ms = alns._calc_makespan(routes, sh.ships)
    td = alns._calc_distance(routes, sh.ships)
    en = alns._calc_energy(routes, sh.ships)
    ls = np.sqrt(alns._calc_balance(routes, sh.ships))  # std = sqrt(var)
    return ms, td, en, ls


# ================================================================
#  5种子对比实验
# ================================================================
R_g = {'ms': [], 'td': [], 'en': [], 'ls': [], 'tm': [], 'cs': []}
R_a = {'ms': [], 'td': [], 'en': [], 'ls': [], 'tm': [], 'cs': []}

for seed in range(5):  # 5种子取平均
    # ── Greedy ──
    np.random.seed(seed * 10 + 42)
    sh_g = build_scene()
    for tid in sh_g.tasks:
        sh_g.tasks[tid].payload = float(np.random.choice([300, 500, 800, 1000, 1500]))

    t0 = time.perf_counter()
    assign_g = greedy_route(sh_g)
    R_g['tm'].append(time.perf_counter() - t0)

    routes_g = assignment_to_routes(sh_g, assign_g)
    ms_g, td_g, en_g, ls_g = evaluate_routes(sh_g, routes_g)
    R_g['ms'].append(ms_g)
    R_g['td'].append(td_g)
    R_g['en'].append(en_g)
    R_g['ls'].append(ls_g)

    # ── Graph-ALNS (以贪心结果为初始解) ──
    np.random.seed(seed * 10 + 42)
    sh_a = build_scene()
    for tid in sh_a.tasks:
        sh_a.tasks[tid].payload = float(np.random.choice([300, 500, 800, 1000, 1500]))

    # 先算贪心，转为 RouteNode 格式
    assign_greedy = greedy_route(sh_a)
    greedy_routes = {}
    for sid in sh_a.ships:
        route = []
        for tid in assign_greedy.get(sid, []):
            t = sh_a.tasks.get(tid)
            if t is None: continue
            route.append(RouteNode(t.pickup_node, "PICKUP", tid))
            route.append(RouteNode(t.delivery_node, "DELIVERY", tid))
        greedy_routes[sid] = route

    t0 = time.perf_counter()
    # 抑制 ALNS 内部 print 刷屏
    with contextlib.redirect_stdout(io.StringIO()):
        sched = Scheduler(rn, sh_a.ships, sh_a.tasks, port_ids, gas_ids, node_names)
        sched.initialize(initial_routes=greedy_routes)
    R_a['tm'].append(time.perf_counter() - t0)

    ms_a, td_a, en_a, ls_a = evaluate_routes(sh_a, sched.routes)
    R_a['ms'].append(ms_a)
    R_a['td'].append(td_a)
    R_a['en'].append(en_a)
    R_a['ls'].append(ls_a)

    # ── 综合代价 (逐种子归一化, 避免跨种子量纲不同) ──
    mx_td = max(td_g, td_a, 1)
    mx_en = max(en_g, en_a, 1)
    mx_ls = max(ls_g, ls_a, 1)
    R_g['cs'].append(0.4 * td_g / mx_td + 0.3 * en_g / mx_en + 0.3 * ls_g / mx_ls)
    R_a['cs'].append(0.4 * td_a / mx_td + 0.3 * en_a / mx_en + 0.3 * ls_a / mx_ls)


# ================================================================
#  结果输出
# ================================================================
def avg(L):
    return np.mean(L)


print(f'\n{"Metric":<25} {"Greedy":>12} {"Graph-ALNS":>12} {"Change":>12}')
print('-' * 62)
for key, lbl in [('ms', 'Makespan (s)'), ('td', 'Distance (m)'),
                 ('en', 'Energy (kWh)'), ('ls', 'Load StdDev'),
                 ('cs', 'Composite Cost'), ('tm', 'Algo Time (s)')]:
    gv = avg(R_g[key])
    av = avg(R_a[key])
    pct = (av / gv - 1) * 100
    arrow = '↑' if pct > 0 else '↓'
    print(f'{lbl:<25} {gv:12.1f} {av:12.1f} {abs(pct):8.1f}% {arrow}')

# ── ALNS 多目标代价分解 (最后一个种子) ──
print(f'\n{"─" * 60}')
print(f'ALNS Multi-Objective Cost Breakdown (seed={seed}):')
print(f'  J = 0.70×M + 0.15×D + 0.08×E + 0.05×B + 0.02×S')
print(f'  {"Component":<20} {"Raw":>12} {"Norm":>8} {"Weighted":>10}')
print(f'  {"-" * 50}')
try:
    br = sched.alns.get_cost_breakdown(sched.routes, sh_a.ships)
    for key, label, w in [('D', 'Distance (m)', 0.15), ('M', 'Makespan (s)', 0.70),
                           ('E', 'Energy (kWh)', 0.08), ('B', 'Balance (s²)', 0.05),
                           ('S', 'Stability', 0.02)]:
        raw_v = br.get(f'{key}_raw', 0)
        norm_v = br.get(f'{key}_norm', 0)
        print(f'  {label:<20} {raw_v:12.1f} {norm_v:8.3f} {w * norm_v:10.4f}')
    print(f'  {"─" * 50}')
    jt = br.get('J_total', 0)
    print(f'  {"J_total":<20} {"":>12} {"":>8} {jt:10.4f}')
except Exception as e:
    print(f'  (cost breakdown unavailable: {e})')

# ── 综合代价公式说明 ──
print(f'\nComparison Composite = 0.4 × Dist_norm + 0.3 × Energy_norm + 0.3 × LoadStd_norm')
print(f'  (per-seed normalized, lower is better)')
print(f'\nALNS optimized: J = 0.70×M + 0.15×D + 0.08×E + 0.05×B + 0.02×S')

# ── 单船明细 (最后一个种子的 Graph-ALNS 结果) ──
print(f'\n{"─" * 60}')
print(f'Graph-ALNS Per-Ship Detail (seed={seed}):')
for sid in sorted(sh_a.ships.keys()):
    s = sh_a.ships[sid]
    route = sched.routes.get(sid, [])
    n_t = len([r for r in route if r.action == 'PICKUP'])
    if n_t == 0:
        print(f'  {s.name}: idle')
        continue
    tot_d = 0.0
    cur = s.current_node
    pus = []
    for rnd in route:
        tot_d += rn.dist_matrix[cur, rnd.node_id]
        cur = rnd.node_id
        if rnd.action == 'PICKUP':
            pus.append(node_names.get(rnd.node_id, f'N{rnd.node_id}'))
    print(f'  {s.name}: {n_t:2d} tasks  {tot_d / 1000:6.1f} km  '
          f'{tot_d / 1000 * s.energy_per_km:5.0f} kWh  '
          f'{" → ".join(pus[:5])}{"..." if len(pus) > 5 else ""}')

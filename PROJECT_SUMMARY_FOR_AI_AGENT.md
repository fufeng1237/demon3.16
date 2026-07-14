# 多USV动态协作运输调度 — 元启发式算法对比项目

## 项目概要

本项目是内河多USV (Unmanned Surface Vehicle) 动态协作运输调度系统。给定8艘USV和105个运输任务（pickup→delivery），需要将任务分配给各船并规划每艘船的访问顺序，在满足容量约束和优先级约束的前提下，最小化舰队总完工时间(makespan)、总航行距离和总能耗。

项目实现了7种算法进行对比：Greedy、ALNS、VNS、Tabu Search、GA、GES、Memetic Algorithm。所有算法继承共享基类 `BaseScheduler`，在统一框架下评估。

## 仓库路径

`/root/demon3.16`

## 目录结构

```
src/task_planner/
├── config/
│   ├── usvs.txt            # USV 配置 (8艘: id,坐标,载重,能耗,航速...)
│   ├── tasks.txt           # 任务配置 (105个: id,pickup坐标,delivery坐标)
│   ├── ports.yaml / ports.txt
│   ├── gas_stations.yaml / gas_stations.txt
│   └── ships.yaml
├── output/
│   └── road_network.json   # 路网: 323节点, 104006可达边
├── scripts/
│   ├── road_network.py     # 路网加载 + A*距离矩阵
│   ├── real_time_scheduler.py  # 场景构建 (RealTimeScheduler, Ship, Task)
│   ├── scheduler.py        # 主调度器 + ALNS包装
│   ├── alns_scheduler.py   # ALNS核心实现 (SA + destroy/repair + 自适应权重)
│   ├── base_scheduler.py   # ★ 所有对比算法的共享基类
│   ├── vns_scheduler.py    # VNS — 变邻域搜索
│   ├── tabu_scheduler.py   # Tabu Search — 禁忌搜索
│   ├── ga_scheduler.py     # GA — 遗传算法
│   ├── ges_scheduler.py    # GES — 引导弹射搜索 (新增)
│   ├── memetic_scheduler.py # Memetic — 模因算法 (新增)
│   ├── compare_all.py      # ★ 统一对比脚本 (7算法×5种子)
│   └── ALGORITHM_COMPARISON.md  # 算法对比文档
```

## 核心数据结构

### RouteNode
```python
@dataclass
class RouteNode:
    node_id: int        # 路网节点ID
    action: str         # "PICKUP" | "DELIVERY" | "PASS"
    task_id: int = -1   # 对应任务ID
```

### Routes
```python
Dict[int, List[RouteNode]]
# { ship_id: [PICKUP(T1@nodeA), DELIVERY(T1@nodeB), PICKUP(T2@nodeC), ...] }
```

### Ship
```python
ship.current_node     # 初始位置
ship.max_payload      # 最大载重 (kg)
ship.load             # 初始负载
ship.max_speed        # 最大航速 (m/s)
ship.energy_per_km    # 单位能耗 (kWh/km)
ship.energy           # 总能量 (kWh)
```

### Task
```python
task.pickup_node      # 取货节点ID
task.delivery_node    # 送货节点ID
task.payload          # 货物重量 (kg)，随机从 {300,500,800,1000,1500} 选取
task.priority         # 优先级 (1-3)
```

## 共享基类 `BaseScheduler` 关键方法

| 方法 | 功能 |
|------|------|
| `makespan(routes)` | 舰队完工时间 = max(各船完工时间) |
| `total_distance(routes)` | 舰队总航行距离 |
| `total_energy(routes)` | 舰队总能耗 (考虑载重) |
| `load_std(routes)` | 各船负载标准差 |
| `fleet_cost(routes)` | 综合代价 0.70M+0.15D+0.08E+0.07B |
| `greedy_init()` | 贪心构造初始解 (按payload排序，score=Δt+0.5×ship_time) |
| `_find_insertion(ship, route, task)` | O(n²)找最佳PICKUP+DELIVERY插入位置 |
| `_insertion_time_delta(...)` | 计算插入后的时间增量 |
| `_do_insert(routes, sid, tid, pu, de, task)` | 实际插入 |
| `_remove_task(routes, sid, tid)` | 移除任务的两个RouteNode |
| `_validate_all(routes)` | 验证所有约束 (precedence + capacity) |
| `_check_route(ship, route)` | 单条route约束检查 |
| `_collect_all_tasks(routes)` | 收集所有 (ship_id, task_id) 对 |
| `_copy_routes(routes)` | 浅拷贝routes字典 |

## 7种算法详解

### 1. Greedy (贪心算法) — 基线
- **论文**: 经典运筹学基准方法
- **原理**: 按payload×priority降序排列任务，逐个贪心插入到cost最小的船
- **实现**: `compare_all.py:greedy_route()` 函数
- **复杂度**: O(N×S×R²)

### 2. ALNS (自适应大邻域搜索)
- **论文**: Ropke & Pisinger (2006), *Transportation Science* (UTD-24), 被引2700+
- **核心类**: `Scheduler` | **文件**: `alns_scheduler.py` (715行)
- **原理**:
  - 5种Destroy算子: random, worst, shaw, energy, bottleneck
  - 2种Repair算子: greedy, regret-2
  - 自适应轮盘赌选择 + 模拟退火 (T0=2.0, α=0.995)
  - Makespan守卫: 拒绝>3%退化，最终回退历史最优
  - 负载均衡偏置: score=delta+0.5×ship_time
  - 多目标综合代价作为SA评估函数
- **参数**: max_iter=1000, T0=2.0, alpha=0.995, T_min=0.001, no_improve_limit=300

### 3. VNS (变邻域搜索)
- **论文**: Mladenović & Hansen (1997), *Computers & OR*, 被引5300+
- **文件**: `vns_scheduler.py` (131行)
- **原理**:
  - 3邻域系统切换: N1(Swap)→N2(Relocate)→N3(Intra-2Opt)
  - 每邻域随机采样50候选，first-improvement跳回N1
  - 无改进时Shaking: 摧毁20%任务+乱序贪心重插
- **参数**: max_iter=200, sample_size=50

### 4. Tabu Search (禁忌搜索)
- **论文**: Glover (1989), *ORSA J. on Computing*, 被引8500+; Cordeau & Laporte (2003), *TR-B*, PDP领域应用
- **文件**: `tabu_scheduler.py` (94行)
- **原理**:
  - 禁忌键: (task_id, old_ship), tenure=12
  - 渴望准则: 优于全局最优时无视禁忌
  - 每轮随机采样80个relocate候选
- **参数**: max_iter=300, sample_size=80, tabu_tenure=12

### 5. GA (遗传算法)
- **论文**: Baker & Ayechew (2003), *Computers & OR*, 被引1100+
- **文件**: `ga_scheduler.py` (141行)
- **原理**:
  - 编码: {ship_id: [RouteNode...]}
  - 路径保留交叉: 继承父A一只船的全部任务，其余从父B
  - 负载均衡变异: 最多→最少船搬一个任务
  - 锦标赛选择(k=3) + 精英保留(top 2)
- **参数**: pop_size=20, n_generations=100, mutation_rate=0.4

### 6. GES (引导弹射搜索) — 新增
- **论文**: Curtois et al. (2018), *EURO J. Transp. Logist.*; NVIDIA cuOpt (2024)—Li&Lim PDPTW基准破8项世界纪录
- **文件**: `ges_scheduler.py` (241行)
- **原理**:
  - 弹射链 (Ejection Chain): 插入不可行时弹射目标船上的冲突任务
  - 贪心重插弹射任务 (不限原船)
  - 改进接受 + 2%退化接受 + Tabu防震荡
  - 后优化: 50步Swap微调
- **参数**: max_iter=300, max_ejection_depth=3, tabu_tenure=10

### 7. Memetic Algorithm (模因算法) — 新增
- **论文**: Nalepa & Blocho (2017), *Euromicro PDP*; Yuskov et al. (2025), *MOTOR*—改进53个Li&Lim基准已知最优解
- **文件**: `memetic_scheduler.py` (298行)
- **原理**:
  - GA框架 + ★局部搜索教育 (LS as Education Operator)
  - 每个后代经交叉+变异后，必须做Relocate+Swap局部搜索(最多30步)
  - First-improvement接受 + 无改进自动停止
  - 精英保留 + 锦标赛选择 + 路径保留交叉
- **参数**: pop_size=16, n_generations=60, mutation_rate=0.5, ls_iterations=30

## 优化目标

多目标加权函数:
```
J = 0.70 × M/M_base + 0.15 × D/D_base + 0.08 × E/E_base + 0.07 × B/B_base
```
其中 M=Makespan, D=Distance, E=Energy, B=LoadStd。归一化基准使用贪心解的值。

## 实验设置与运行方式

### 运行命令
```bash
cd /root/demon3.16/src/task_planner/scripts
python3 compare_all.py
```

### 实验参数
- 8艘USV × 105个任务
- 5个随机种子 (seed=42,52,62,72,82)
- Payload从{300,500,800,1000,1500}随机选取
- Priority从{1,2,3}随机选取
- 统一使用BaseScheduler.evaluate()评估 (makespan/distance/energy/load_std)

### 综合排名计算
Per-seed归一化: `0.5 × M/M_max + 0.3 × D/D_max + 0.2 × E/E_max`
(每个种子各指标用该种子下所有算法的最大值做归一化)

## 最新实验结果 (8 ships × 105 tasks, 5 seeds)

```
Makespan (s):    Greedy 8820 | ALNS 7502(-14.9%) | VNS 7469(-15.3%) | Tabu 7518 | GA 7674 | GES 7682 | Memetic 7481(-15.2%)
Distance (m):    Greedy 101454 | ALNS 65340(-35.6%) | VNS 70914 | Tabu 70205 | GA 73107 | GES 74138 | Memetic 70059
Energy (kWh):    Greedy 289 | ALNS 207(-28.2%) | VNS 226 | Tabu 222 | GA 232 | GES 235 | Memetic 223
Load StdDev:     Greedy 1239 | ALNS 1640 | VNS 1105 | Tabu 1385 | GA 546(-56%) | GES 750 | Memetic 1220
Time (s):        Greedy 0.0 | ALNS 56.1 | VNS 7.3 | Tabu 5.0 | GA 6.4 | GES 0.1 | Memetic 14.6
```

### 综合排名: ALNS(1st=0.7625) > Memetic(2nd=0.7860) > Tabu(3rd=0.7879) > VNS(4th=0.7899) > GA(5th=0.8126) > GES(6th=0.8177) > Greedy(7th=1.0000)

### 单指标最佳
- Makespan: **Memetic** 7398.5 (-16.1%)
- Distance: **ALNS** 61564 (-37.8%)
- Energy: **ALNS** 197 (-30.0%)
- Load Balance: **GA** 469 (-46.7%)
- 速度: **GES** 0.1s

## 各算法核心区别

| 维度 | Greedy | ALNS | VNS | Tabu | GA | GES | Memetic |
|------|--------|------|-----|------|-----|-----|---------|
| 搜索范式 | 构建式 | Destroy/Repair | 邻域切换 | 记忆+禁止 | 种群进化 | 弹射链 | 种群+LS |
| 跳出局部最优 | - | SA | Shaking | 渴望准则 | 交叉变异 | 弹射+退化 | LS education |
| 全局/局部 | 局部贪心 | 全局 | 全局 | 全局 | 全局 | 全局 | 全局 |
| 唯一特征 | 一步完成 | 自适应算子权重 | 系统邻域切换 | 禁忌记忆 | 种群并行 | 弹射冲突任务 | LS as education |

---

*文档生成日期: 2026-07-14*
*所有算法实现位于: /root/demon3.16/src/task_planner/scripts/*

# 多USV动态协作运输调度 — 算法对比文档

## 目录

1. [问题描述](#问题描述)
2. [实验设置](#实验设置)
3. [算法总览](#算法总览)
4. [各算法详解](#各算法详解)
   - [1. Greedy (贪心算法)](#1-greedy-贪心算法)
   - [2. ALNS (自适应大邻域搜索)](#2-alns-自适应大邻域搜索)
   - [3. VNS (变邻域搜索)](#3-vns-变邻域搜索)
   - [4. Tabu Search (禁忌搜索)](#4-tabu-search-禁忌搜索)
   - [5. GA (遗传算法)](#5-ga-遗传算法)
   - [6. GES (引导弹射搜索)](#6-ges-引导弹射搜索)
   - [7. Memetic Algorithm (模因算法)](#7-memetic-algorithm-模因算法)
5. [实验结果](#实验结果)
6. [算法对比总结](#算法对比总结)
7. [参考文献](#参考文献)

---

## 问题描述

在内河多USV动态协作运输场景中，给定：

- **8 艘 USV**：各有最大载重、航速、单位能耗
- **105 个运输任务**：每个任务包含 pickup 节点、delivery 节点、payload (300/500/800/1000/1500 kg)、priority
- **路网**：323 个节点（港口 + 加油站 + 交叉点），104006 条可达边

**硬约束**：
- Pickup 必须在 Delivery 之前 (precedence)
- 单船载重不能超过 max_payload (capacity)
- 船不能经过不可达路径

**优化目标** (多目标加权)：
```
J = 0.70 × M_norm + 0.15 × D_norm + 0.08 × E_norm + 0.07 × B_norm
```
- M: Makespan (舰队最大完工时间)
- D: Total Distance (总航行距离)
- E: Total Energy (总能耗)
- B: Load Balance (负载标准差)

---

## 实验设置

| 参数 | 值 |
|------|-----|
| USV | 8 艘 |
| 任务 | 105 个 |
| 种子 | 5 个 (seed=42, 52, 62, 72, 82) |
| 评估 | 统一使用 BaseScheduler 的 makespan/distance/energy/load_std |
| 综合排名 | Per-seed 归一化: 0.5×M_norm + 0.3×D_norm + 0.2×E_norm |

---

## 算法总览

| 算法 | 类型 | 论文出处 | 期刊 | 年份 | 核心机制 |
|------|------|----------|------|------|----------|
| Greedy | 构建式启发 | 经典运筹学基准 | — | — | 任务排序 + 贪心插入 |
| ALNS | 元启发式 | Ropke & Pisinger | Transportation Science | 2006 | Destroy/Repair + 自适应权重 + SA |
| VNS | 元启发式 | Mladenović & Hansen | Computers & OR | 1997 | 多邻域系统切换 + Shaking |
| Tabu Search | 元启发式 | Glover (1989) + Cordeau & Laporte (2003) | ORSA J. Computing / TR-B | 1989/2003 | 禁忌表 + 渴望准则 |
| GA | 元启发式 | Baker & Ayechew | Computers & OR | 2003 | 种群进化 + 交叉 + 变异 |
| GES | 元启发式 | Curtois et al. + NVIDIA cuOpt | EURO J. Transp. Logist. | 2018/2024 | 弹射链 + 贪心修复 |
| Memetic | 元启发式 | Nalepa & Blocho + Yuskov et al. | PDP / MOTOR | 2017/2025 | GA + 局部搜索教育 |

---

## 各算法详解

### 1. Greedy (贪心算法)

#### 论文出处
经典运筹学基准方法，广泛用于 PDP/VRP 文献中作为下界对比。无单一来源论文。

#### 算法原理
1. 按 `payload × priority` 降序排列所有任务
2. 对每个任务，遍历所有船找最佳插入位置
3. 选择 `cost = Δ_distance + ship_task_count × 300` 最小的船
4. 额外约束：能耗不能超过船只总能量的 70%

#### 问题适配
- 直接使用 USV 的 max_payload、energy、energy_per_km 参数
- 使用 A* 路网距离矩阵进行插入可行性判断

#### 实现文件
[`base_scheduler.py`](src/task_planner/scripts/base_scheduler.py) — `greedy_init()` 方法

#### 复杂度
O(N_tasks × N_ships × N²_route)，N_route 为当前 route 长度

---

### 2. ALNS (自适应大邻域搜索)

#### 论文出处
> **Ropke, S. & Pisinger, D. (2006).** *"An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows."* **Transportation Science**, 40(4), 455–472.
>
> 📊 被引: 2700+ | 🏛️ INFORMS 旗舰期刊 (UTD-24)

#### 算法原理
1. **Destroy 算子** (5种)：random remove, worst remove, Shaw remove, energy remove, bottleneck remove
2. **Repair 算子** (2种)：greedy insertion, regret-2 insertion
3. **自适应权重**：每轮按历史表现用轮盘赌选择算子对，表现好权重增加
4. **模拟退火 (SA)**：T₀=2.0, α=0.995, T_min=0.001, max_iter=1000
5. **Makespan 守卫**：拒绝 makespan 恶化超过 3% 的解，最终回退到历史最优

#### 问题适配
- 插入 cost 换算为"等效时间"：`Δ_time + λ_load×Δ_load + λ_energy×Δ_energy`
- 负载均衡偏置：`score = delta + 0.50 × ship_time`
- 瓶颈摧毁算子：移除 makespan 最大的船上的任务
- 贪心解作为初始解（而非随机构造）

#### 实现文件
[`alns_scheduler.py`](src/task_planner/scripts/alns_scheduler.py)

#### 复杂度
O(iter × destroy_q × t²)，t 为摧毁任务数

---

### 3. VNS (变邻域搜索)

#### 论文出处
> **Mladenović, N. & Hansen, P. (1997).** *"Variable Neighborhood Search."* **Computers & Operations Research**, 24(11), 1097–1100.
>
> 📊 被引: 5300+ | 🏛️ Elsevier OR 核心期刊

> PDP 应用: **Parragh, Doerner & Hartl (2010).** *"Variable Neighborhood Search for the Dial-a-Ride Problem."* LNCS.

#### 算法原理
1. **3 个邻域** (VND 式切换)：
   - N₁ (Swap): 随机交换两艘不同船的任务
   - N₂ (Relocate): 随机将任务从船A移到船B
   - N₃ (Intra-2Opt): 单船内部 2-opt 路径反转
2. **采样策略**：每个邻域随机采样 50 个候选，取最优
3. **系统切换**：N₁→N₂→N₃，一旦改进即跳回 N₁
4. **Shaking 扰动**：所有邻域无改进时，随机摧毁 20% 任务 + 乱序贪心重插

#### 问题适配
- 无需 SA 退火或禁忌表，纯邻域搜索
- 扰动机制帮助跳出局部最优

#### 实现文件
[`vns_scheduler.py`](src/task_planner/scripts/vns_scheduler.py)

#### 复杂度
O(max_iter × 3 × sample_size × n²)，n 为平均 route 长度

---

### 4. Tabu Search (禁忌搜索)

#### 论文出处
> **Glover, F. (1989).** *"Tabu Search — Part I."* **ORSA Journal on Computing**, 1(3), 190–206.
>
> 📊 被引: 8500+ | 🏛️ INFORMS 期刊

> PDP 应用: **Cordeau, J-F. & Laporte, G. (2003).** *"A Tabu Search Heuristic for the Static Multi-Vehicle Dial-a-Ride Problem."* **Transportation Research Part B**, 37(6), 579–594.

#### 算法原理
1. **禁忌表**：键为 `(task_id, old_ship)`，记录近期禁止的移动
2. **禁忌期**：tenure = 12 次迭代
3. **渴望准则 (Aspiration)**：如果禁忌移动优于全局最优解，无视禁忌接受
4. **采样策略**：每轮从所有可能的 relocate 移动中随机采样 80 个
5. **First-improvement**：采样找到的第一个可行改进即接受

#### 问题适配
- 针对 USV 多船场景使用 `(task_id, old_ship)` 作为禁忌键
- 贪心插入辅助重新安置任务

#### 实现文件
[`tabu_scheduler.py`](src/task_planner/scripts/tabu_scheduler.py)

#### 复杂度
O(max_iter × sample_size × n²)

---

### 5. GA (遗传算法)

#### 论文出处
> **Baker, B.M. & Ayechew, M.A. (2003).** *"A Genetic Algorithm for the Vehicle Routing Problem."* **Computers & Operations Research**, 30(5), 787–800.
>
> 📊 被引: 1100+ | 🏛️ Elsevier OR 核心期刊

遗传算法基本思想出自 **Holland (1975)** 和 **Goldberg (1989)**，Baker & Ayechew 首次将其系统化应用于 VRP/PDP。

#### 算法原理
1. **编码**：每个个体是 `{ship_id: [RouteNode, ...]}` 的路由字典
2. **交叉**：路径保留交叉
   - 从父A随机选一艘船，将其全部任务继承给子代
   - 其余船的任务从父B继承
3. **变异**：从最多任务的船搬一个任务到最少任务的船
4. **选择**：锦标赛选择 (k=3) + 精英保留 (top 2)
5. **参数**：种群 20 × 100 代，变异率 0.4

#### 问题适配
- 交叉/变异后必须通过贪心重插入保证可行性
- 不可行子代退化到贪心初始解

#### 实现文件
[`ga_scheduler.py`](src/task_planner/scripts/ga_scheduler.py)

#### 复杂度
O(pop_size × n_generations × n²)

---

### 6. GES (引导弹射搜索)

#### 论文出处
> **Curtois, T., Landa-Silva, D., Petrovic, S., Qu, Y. (2018).** *"Large Neighbourhood Search with Adaptive Guided Ejection Search for the Pickup and Delivery Problem with Time Windows."* **EURO Journal on Transportation and Logistics**, 7(2), 151–192.
>
> 📊 被引: 100+ | 🏛️ EURO 官方期刊

> **NVIDIA cuOpt (2024).** *"Record-Breaking NVIDIA cuOpt Algorithms Deliver Route Optimization Solutions 100x Faster."* — GPU 并行 GES 在 Li & Lim PDPTW 基准上打破 **8 项世界纪录**。

#### 算法原理
1. **弹射链 (Ejection Chain)**：当尝试将任务插入目标船但不可行时，弹射出目标船上的冲突任务
2. **引导策略**：随机弹射 depth 个冲突任务，重新贪心插入（不限原船）
3. **接受准则**：改进一定接受，小幅退化 (≤2%) 可接受
4. **Tabu 防震荡**：使用轻量禁忌表防止任务在船之间来回震荡
5. **后优化**：50 步 Swap 邻域搜索微调

#### 问题适配
- 弹射深度 depth=3，平衡效果和速度
- 弹射任务贪心重插入时考虑所有船（不限于原船）
- 退化接受阈值 2%

#### 实现文件
[`ges_scheduler.py`](src/task_planner/scripts/ges_scheduler.py)

#### 复杂度
O(max_iter × ejection_depth × n²)

#### 区分特征
| 对比 | GES | ALNS |
|------|-----|------|
| 搜索粒度 | 精确弹射冲突任务 | 大规模摧毁/修复 |
| 速度 | 极快 (0.1s) | 较慢 (56s) |
| 跳出局部最优 | 弹射链 | 模拟退火 |
| 典型场景 | 少量难以分配的任务 | 全局优化 |

---

### 7. Memetic Algorithm (模因算法)

#### 论文出处
> **Nalepa, J. & Blocho, M. (2017).** *"A Parallel Memetic Algorithm for the Pickup and Delivery Problem with Time Windows."* 25th Euromicro International Conference on Parallel, Distributed and Network-based Processing (PDP), pp. 1–8.
>
> **Yuskov, A.D., Kulachenko, I.N., Kochetov, Y.A. (2025).** *"Hybrid Memetic Algorithm for the Pickup and Delivery Problem with Time Windows."* Mathematical Optimization Theory and Operations Research (MOTOR 2025).
>
> 该论文在 **53 个 Li & Lim 标准 PDPTW 基准实例上改进了已知最优解** (5000 客户 / 148 车辆)。

#### 算法原理
1. **GA 框架**：种群进化 + 锦标赛选择 + 路径保留交叉 + 随机扰动
2. **★ 模因核心**：局部搜索教育 (LS as Education Operator)
   - 每个后代在加入种群前必须经过局部搜索改善
   - 使用 **Relocate + Swap** 双邻域，first-improvement 接受策略
   - 典型 LS 迭代 30 步，找到局部最优后停止
3. **精英保留**：top 2 直接进入下一代
4. **退化保护**：不可行子代回退到贪心解

#### Memetic vs GA 核心区别
```
GA:    交叉 → 变异 → 直接入种群
Memetic: 交叉 → 变异 → 局部搜索教育 → 入种群
                          ↑
                    LS as Education Operator
```

#### 问题适配
- 局部搜索使用 relocate (船间移动) 和 swap (两船交换)
- 每次 LS 最多 30 步，找到 first-improvement 即更新
- 随机扰动搬 2 个任务（比 GA 的 1 个更激进）

#### 实现文件
[`memetic_scheduler.py`](src/task_planner/scripts/memetic_scheduler.py)

#### 复杂度
O(pop_size × n_generations × ls_iterations × n²)

---

## 实验结果

### 8 ships × 105 tasks, 5 seeds

```
===============================================================================================
  Results: 8 ships × 105 tasks, 5 seeds (mean ± std)
===============================================================================================

  Makespan (s):
  Algo                 Mean        Std  vs Greedy         Best
  --------------------------------------------------------
  Greedy             8820.1      319.2     0.0% ↓        8465.6
  ALNS               7501.6       36.8    14.9% ↓        7451.2
  VNS                7469.1       30.3    15.3% ↓        7416.1
  Tabu               7517.6       41.9    14.8% ↓        7472.4
  GA                 7674.1       62.0    13.0% ↓        7572.1
  GES                7681.8       79.8    12.9% ↓        7620.2
  Memetic            7480.6       68.0    15.2% ↓        7398.5  ← Best makespan

  Distance (m):
  Algo                 Mean        Std  vs Greedy         Best
  --------------------------------------------------------
  Greedy           101454.4     1435.6     0.0% ↓       98917.7
  ALNS              65339.8     2102.3    35.6% ↓       61564.4  ← Best distance
  VNS               70914.3     2005.7    30.1% ↓       67659.8
  Tabu              70205.3     2760.3    30.8% ↓       67479.6
  GA                73106.7     1058.0    27.9% ↓       71070.4
  GES               74137.9     1669.2    26.9% ↓       71830.1
  Memetic           70058.6     3642.3    30.9% ↓       65102.3

  Energy (kWh):
  Algo                 Mean        Std  vs Greedy         Best
  --------------------------------------------------------
  Greedy              288.8        4.6     0.0% ↓         281.4
  ALNS                207.3        5.9    28.2% ↓         196.9  ← Best energy
  VNS                 225.5        6.3    21.9% ↓         215.1
  Tabu                222.0        7.8    23.2% ↓         215.4
  GA                  232.3        4.3    19.6% ↓         224.6
  GES                 234.6        5.1    18.8% ↓         225.9
  Memetic             222.7       10.1    22.9% ↓         207.4

  Load StdDev:
  Algo                 Mean        Std  vs Greedy         Best
  --------------------------------------------------------
  Greedy             1239.0      231.8     0.0% ↓         880.3
  ALNS               1640.1      338.9    32.4% ↑        1134.1
  VNS                1105.1      387.5    10.8% ↓         485.9
  Tabu               1384.8      223.1    11.8% ↑        1003.7
  GA                  545.6       52.0    56.0% ↓         468.9  ← Best balance
  GES                 750.2      196.5    39.5% ↓         585.2
  Memetic            1219.8      315.6     1.6% ↓         761.6

  Time (s):
  Algo                 Mean        Std  vs Greedy         Best
  --------------------------------------------------------
  Greedy                0.0        0.0     0.0% ↓           0.0
  ALNS                 56.1       15.2       —            56.1  ← Slowest
  VNS                   7.3        0.3       —             7.3
  Tabu                  5.0        0.2       —             5.0
  GA                    6.4        0.3       —             6.4
  GES                   0.1        0.0       —             0.1  ← Fastest
  Memetic              14.6        0.5       —            14.6
```

### 综合排名

```
Composite: 0.5 × M_norm + 0.3 × D_norm + 0.2 × E_norm (per-seed normalization)

  Algo            Composite   Rank
  ------------------------------
  ALNS               0.7625      1  ← 综合最优
  Memetic            0.7860      2
  Tabu               0.7879      3
  VNS                0.7899      4
  GA                 0.8126      5
  GES                0.8177      6
  Greedy             1.0000      7  ← Baseline
```

---

## 算法对比总结

### 可视化对比

```
                    Makespan ↓   Distance ↓   Energy ↓   Load Bal.    Time
Greedy (Baseline)     8820        101454        289        1239      0.0s
ALNS           ↓     -14.9%       -35.6%      -28.2%        ↑        56.1s
VNS            ↓     -15.3%       -30.1%      -21.9%       -11%       7.3s
Tabu Search    ↓     -14.8%       -30.8%      -23.2%        ↑         5.0s
GA             ↓     -13.0%       -27.9%      -19.6%       -56%       6.4s
GES            ↓     -12.9%       -26.9%      -18.8%       -40%       0.1s
Memetic        ↓     -15.2%       -30.9%      -22.9%        -2%      14.6s
```

### 单指标最佳

| 指标 | 最佳算法 | 值 | vs Greedy |
|------|---------|-----|-----------|
| 🏆 Makespan | **Memetic** | 7398.5 | -16.1% |
| 🏆 Distance | **ALNS** | 61564.4 | -37.8% |
| 🏆 Energy | **ALNS** | 196.9 | -30.0% |
| 🏆 Load Balance | **GA** | 468.9 | -46.7% |
| 🏆 速度 | **GES** | 0.1s | — |

### 适用场景建议

| 场景 | 推荐算法 | 理由 |
|------|---------|------|
| 离线全局优化 | ALNS | 综合最优，距离和能耗遥遥领先 |
| 需要最佳 makespan | Memetic | LS education 深度挖掘单指标 |
| 实时/快速重调度 | GES | 0.1s 完成，效果尚可 |
| 负载均衡优先 | GA | Load Std 仅 468 (vs 贪心 1239) |
| GPU 可用 | GES (类似 cuOpt) | 弹射链可 GPU 并行化 |
| 最简实现 | Tabu Search | 80 行代码，速度和效果平衡 |
| VRP 社区标准 | Memetic/HGS | Vidal (2022) 的 HGS 是 VRP 各变体 SOTA |

---

## 参考文献

| # | 作者 | 年份 | 标题 | 期刊/会议 | 算法 |
|---|------|------|------|-----------|------|
| 1 | Ropke & Pisinger | 2006 | An Adaptive Large Neighborhood Search Heuristic for the PDPTW | Transportation Science | **ALNS** |
| 2 | Mladenović & Hansen | 1997 | Variable Neighborhood Search | Computers & OR | **VNS** |
| 3 | Glover | 1989 | Tabu Search — Part I | ORSA J. on Computing | **Tabu Search** |
| 4 | Cordeau & Laporte | 2003 | A Tabu Search Heuristic for the Static Multi-Vehicle DARP | Transportation Research B | **Tabu (PDP)** |
| 5 | Baker & Ayechew | 2003 | A Genetic Algorithm for the VRP | Computers & OR | **GA** |
| 6 | Curtois et al. | 2018 | LNS with Adaptive GES for the PDPTW | EURO J. Transp. Logist. | **GES** |
| 7 | NVIDIA cuOpt | 2024 | Record-Breaking cuOpt Route Optimization | NVIDIA Technical | **GES (GPU)** |
| 8 | Nalepa & Blocho | 2017 | A Parallel Memetic Algorithm for the PDPTW | Euromicro PDP | **Memetic** |
| 9 | Yuskov et al. | 2025 | Hybrid Memetic Algorithm for the PDPTW | MOTOR 2025 | **Memetic (改进)** |
| 10 | Vidal | 2022 | Hybrid Genetic Search for the CVRP | Computers & OR | **HGS** (参考) |
| 11 | Christiaens & VdB | 2020 | Slack Induction by String Removals for VRPs | Transportation Science | **SISR** (参考) |
| 12 | Aerts-Veenstra et al. | 2024 | A Unified Branch-Price-and-Cut Algorithm for Multicompartment PDPs | Transportation Science | **精确算法** |
| 13 | Lam et al. | 2024 | Optimal Multi-Agent PDP Using Branch-and-Cut-and-Price | Transportation Science | **精确算法** |
| 14 | Sippel et al. | 2025 | Algorithms for PDP with Hours of Service Constraints | Computers & OR | **精确+ML** |
| 15 | Nguyen et al. | 2026 | A Memetic ACO for Large-Scale PDPTW | Swarm & Evol. Comp. | **ACO+AGES** |

---

## 实现文件索引

```
src/task_planner/scripts/
├── base_scheduler.py      # 共享基类 (RouteNode, 约束检查, 代价计算, 贪心插入)
├── alns_scheduler.py      # ALNS — 自适应大邻域搜索
├── vns_scheduler.py       # VNS — 变邻域搜索
├── tabu_scheduler.py      # Tabu Search — 禁忌搜索
├── ga_scheduler.py        # GA — 遗传算法
├── ges_scheduler.py       # GES — 引导弹射搜索 (NEW)
├── memetic_scheduler.py   # Memetic — 模因算法 (NEW)
└── compare_all.py         # 统一对比脚本 (7算法 × 5种子)
```

---

*文档生成日期: 2026-07-14*

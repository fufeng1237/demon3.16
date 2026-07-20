# B1--B7 静态初始分配消融实验

- 场景：固定配置，3 个配对种子
- ALNS 迭代上限：100
- 不包含执行、事件或重分配。

| Variant | Makespan (s) | Distance (m) | Energy (kWh) | Runtime (s) |
|---|---:|---:|---:|---:|
| B1 Plain ALNS | 6740.1 | 63873.3 | 200.62 | 68.85 |
| B2 Graph + Greedy | 8622.0 | 103612.2 | 291.87 | 0.00 |
| B3 Graph + ALNS (full) | 6842.1 | 67467.9 | 212.32 | 45.44 |
| B4 w/o adaptive weights | 6857.9 | 67998.2 | 212.49 | 49.93 |
| B5 w/o simulated annealing | 6798.3 | 66223.4 | 209.86 | 46.61 |
| B6 w/o bottleneck destroy | 6814.3 | 65612.3 | 204.28 | 59.60 |
| B7 basic operators only | 6891.8 | 70385.5 | 217.98 | 36.95 |

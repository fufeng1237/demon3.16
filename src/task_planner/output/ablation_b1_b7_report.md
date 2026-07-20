# B1--B7 静态初始分配消融实验

- 场景：固定配置，10 个配对种子
- ALNS 迭代上限：100
- 不包含执行、事件或重分配。

| Variant | Makespan (s) | Distance (m) | Energy (kWh) | Runtime (s) |
|---|---:|---:|---:|---:|
| B1 Plain ALNS | 3496.0 | 32755.7 | 98.01 | 0.67 |
| B2 Graph + Greedy | 4490.9 | 41581.0 | 116.82 | 0.00 |
| B3 Graph + ALNS (full) | 3537.3 | 34514.4 | 101.61 | 0.40 |
| B4 w/o adaptive weights | 3528.0 | 34127.2 | 100.46 | 0.39 |
| B5 w/o simulated annealing | 3535.8 | 33441.9 | 98.98 | 0.41 |
| B6 w/o bottleneck destroy | 3564.1 | 34609.0 | 102.28 | 0.56 |
| B7 basic operators only | 3580.7 | 36157.3 | 106.08 | 0.28 |

#!/usr/bin/env python3
"""绘制 B1--B7 消融结果：相对 B3 增益和分布图。"""
import csv, os
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

package = Path(__file__).resolve().parents[2]
out = Path(os.getenv('TASK_PLANNER_OUTPUT', package / 'output'))
rows = list(csv.DictReader((out / 'ablation_b1_b7_raw.csv').open(encoding='utf-8')))
groups = defaultdict(list)
labels = {}
for r in rows:
    groups[r['variant']].append(r); labels[r['variant']] = r['label']
keys = list(groups)
short = [labels[k].replace('Graph + ', '').replace('w/o ', '−').replace(' (full)', '') for k in keys]
metrics = [('makespan_s', 'Makespan (s)'), ('distance_m', 'Distance (m)'), ('energy_kwh', 'Energy (kWh)')]
ref = {m: np.mean([float(r[m]) for r in groups['B3_Graph_ALNS']]) for m, _ in metrics}

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={'width_ratios': [1.05, 1.35]})
ax = axes[0]
x = np.arange(len(keys)); width = .24
for j, (m, title) in enumerate(metrics):
    improvement = [(ref[m] - np.mean([float(r[m]) for r in groups[k]])) / ref[m] * 100 for k in keys]
    ax.bar(x + (j-1)*width, improvement, width, label=title)
ax.axhline(0, color='#555', lw=.8); ax.set_xticks(x, short, rotation=28, ha='right')
ax.set_ylabel('Improvement vs B3 (%)'); ax.set_title('Component contribution'); ax.legend(fontsize=8)

ax = axes[1]
data = [[float(r['makespan_s']) for r in groups[k]] for k in keys]
bp = ax.boxplot(data, labels=short, patch_artist=True, showmeans=True)
for i, box in enumerate(bp['boxes']): box.set_facecolor('#2f80ed' if keys[i] == 'B3_Graph_ALNS' else '#b7c9df')
ax.set_ylabel('Makespan (s)'); ax.set_title('Paired-seed makespan distribution'); ax.tick_params(axis='x', rotation=28)
fig.suptitle('Static initial-assignment ablation (B1–B7)', fontweight='bold')
fig.tight_layout()
path = out / 'ablation_b1_b7.png'; fig.savefig(path, dpi=180, bbox_inches='tight'); print(f'Saved: {path}')

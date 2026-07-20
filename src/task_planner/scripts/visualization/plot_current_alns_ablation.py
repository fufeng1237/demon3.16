#!/usr/bin/env python3
"""Compact component-ablation figure for the current Graph/HGT-ALNS."""
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(); p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); data = json.loads(Path(a.input).read_text())['mean']
    labels = ['A0\nPlain', 'A1\nRule graph', 'A2\nHGT', 'A3\nHGT +\nmultistart', 'A4\nFull']
    keys = ['A0_plain_alns', 'A1_rule_graph', 'A2_hgt', 'A3_hgt_multistart', 'A4_full_hgt_alns']
    metrics = [('makespan_s', 'Makespan (s)'), ('distance_m', 'Distance (m)'),
               ('energy_kwh', 'Energy (kWh)'), ('runtime_s', 'Runtime (s)')]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3))
    for ax, (metric, title) in zip(axes, metrics):
        vals = [data[k][metric] for k in keys]
        bars = ax.bar(labels, vals, color=['#7f8c8d'] * 4 + ['#e67e22'])
        ax.set_title(title); ax.grid(axis='y', alpha=.25); ax.tick_params(axis='x', labelsize=8)
        for b, value in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f'{value:.1f}', ha='center', va='bottom', fontsize=7)
    fig.suptitle('80-task current Graph/HGT-ALNS ablation (3 paired seeds)', y=1.03)
    fig.tight_layout(); Path(a.output).parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.output, dpi=180, bbox_inches='tight')
    print(f'Saved: {a.output}')

if __name__ == '__main__': main()

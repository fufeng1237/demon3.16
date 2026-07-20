#!/usr/bin/env python3
"""Create a compact chart from compare_all.py's latest text report."""
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(os.environ.get("TASK_PLANNER_OUTPUT", PACKAGE_DIR / "output"))
REPORT = OUTPUT_DIR / "algorithm_comparison_latest.txt"
OUTPUT = OUTPUT_DIR / "algorithm_comparison_summary.png"
ALGOS = ["Greedy", "ALNS", "VNS", "Tabu", "GA", "GES", "Memetic"]


def parse_report(text: str):
    metrics = {"Makespan (s)": {}, "Distance (m)": {}, "Energy (kWh)": {}, "Time (s)": {}}
    for title in metrics:
        section = re.search(rf"{re.escape(title)}:(.*?)(?=\n\s*[A-Z][A-Za-z ]+\s*\(|\n=|\Z)", text, re.S)
        if not section:
            raise ValueError(f"Missing metric section: {title}")
        for algo, mean in re.findall(r"^\s*(Greedy|ALNS|VNS|Tabu|GA|GES|Memetic)\s+([0-9.]+)", section.group(1), re.M):
            metrics[title][algo] = float(mean)

    composite = {}
    ranking = re.search(r"Composite Ranking.*?\n.*?\n.*?\n(.*)", text, re.S)
    if ranking:
        for algo, score in re.findall(r"^\s*(Greedy|ALNS|VNS|Tabu|GA|GES|Memetic)\s+([0-9.]+)", ranking.group(1), re.M):
            composite[algo] = float(score)
    if set(composite) != set(ALGOS):
        raise ValueError("Missing composite ranking")
    return metrics, composite


def main():
    metrics, composite = parse_report(REPORT.read_text(encoding="utf-8"))
    colors = ["#7f8c8d", "#e74c3c", "#3498db", "#9b59b6", "#2ecc71", "#f39c12", "#16a085"]
    x = np.arange(len(ALGOS))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor("#ffffff")

    baseline = "Greedy"
    width = 0.23
    for offset, (title, color) in zip([-width, 0, width], [("Makespan (s)", "#e74c3c"), ("Distance (m)", "#3498db"), ("Energy (kWh)", "#2ecc71")]):
        values = [100 * (1 - metrics[title][a] / metrics[title][baseline]) for a in ALGOS]
        ax1.bar(x + offset, values, width, label=title.replace(" (s)", "").replace(" (m)", "").replace(" (kWh)", ""), color=color)
    ax1.axhline(0, color="#555", lw=0.8)
    ax1.set_title("Improvement vs Greedy", fontweight="bold")
    ax1.set_ylabel("Improvement (%)")
    ax1.set_xticks(x, ALGOS, rotation=25, ha="right")
    ax1.legend(frameon=False, ncol=3, loc="upper right")
    ax1.grid(axis="y", alpha=0.2)

    ranked = sorted(ALGOS, key=lambda a: composite[a])
    y = np.arange(len(ranked))
    scores = [composite[a] for a in ranked]
    bars = ax2.barh(y, scores, color=[colors[ALGOS.index(a)] for a in ranked])
    ax2.set_yticks(y, ranked)
    ax2.invert_yaxis()
    ax2.set_xlabel("Composite score (lower is better)")
    ax2.set_title("Overall ranking", fontweight="bold")
    ax2.grid(axis="x", alpha=0.2)
    ax2.set_xlim(0, max(scores) * 1.12)
    for bar, score, algo in zip(bars, scores, ranked):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center", fontsize=9)

    fig.suptitle("USV Scheduling Algorithms — Current Task Configuration", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()

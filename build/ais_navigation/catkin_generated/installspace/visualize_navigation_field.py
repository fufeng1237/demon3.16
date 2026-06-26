#!/usr/bin/env python3
"""
visualize_navigation_field.py
Visualize the navigation potential field Phi-bar as a heatmap.

Uses blue-white-red diverging colormap:
  - Red   (Phi -> +1): downstream channel
  - White (Phi ~ 0):   mixed water / free water / intersection
  - Blue  (Phi -> -1): upstream channel

Overlays optionally the binary map for geographic reference.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import os
import sys

# map bounds for title annotation
GEO_BOUNDS = {
    "lon_min": 120.414716,
    "lon_max": 120.644716,
    "lat_min": 31.980283,
    "lat_max": 32.081950,
}
RESOLUTION_M = 0.5


def load_csv(path):
    """Load a CSV file as a numpy array."""
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    print(f"  Loaded {path}: shape={data.shape}, range=[{data.min():.4f}, {data.max():.4f}]")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Visualize navigation potential field"
    )
    parser.add_argument(
        "--csv-path",
        default="/root/demon3.16/src/ais_navigation/map/navigation_potential_field.csv",
        help="Path to the phibar CSV file",
    )
    parser.add_argument(
        "--output",
        default="/root/demon3.16/src/ais_navigation/map/navigation_field.png",
        help="Output image path",
    )
    parser.add_argument(
        "--binary-map",
        default=None,
        help="Optional binary map PNG to overlay",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"[ERROR] CSV not found: {args.csv_path}", file=sys.stderr)
        print("Run generate_navigation_field.py first.", file=sys.stderr)
        sys.exit(1)

    # Load data
    print("=" * 60)
    print("Visualizing navigation potential field")
    print("=" * 60)
    phibar = load_csv(args.csv_path)

    rows, cols = phibar.shape
    width_m = cols * RESOLUTION_M
    height_m = rows * RESOLUTION_M

    # Create figure with two subplots: Phi-bar + histogram
    fig, (ax_map, ax_hist) = plt.subplots(
        1, 2,
        figsize=(18, 8),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    # ---- Panel 1: Phi-bar heatmap ----
    # Custom diverging colormap: blue -> white -> red
    cmap = plt.cm.RdBu_r  # Red-Blue reversed: red positive, blue negative

    # Set neutral (zero) to white
    norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    im = ax_map.imshow(
        phibar,
        cmap=cmap,
        norm=norm,
        origin="upper",
        extent=[0, width_m, 0, height_m],
        aspect="equal",
        interpolation="bilinear",
    )

    ax_map.set_xlabel("X [meters]")
    ax_map.set_ylabel("Y [meters]")
    ax_map.set_title(
        f"Navigation Potential Field $\\bar{{\\Phi}}$\n"
        f"({GEO_BOUNDS['lon_min']:.4f}E - {GEO_BOUNDS['lon_max']:.4f}E, "
        f"{GEO_BOUNDS['lat_min']:.4f}N - {GEO_BOUNDS['lat_max']:.4f}N)"
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax_map, shrink=0.8)
    cbar.set_label("$\\bar{\\Phi}$  (Downstream +1  /  Upstream -1)")

    # Add annotation
    ax_map.text(
        0.02, 0.98,
        "Red = Downstream\nBlue = Upstream\nWhite = Mixed/Free",
        transform=ax_map.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # ---- Panel 2: Value distribution histogram ----
    ax_hist.hist(phibar.ravel(), bins=100, color="gray", edgecolor="black", alpha=0.7)
    ax_hist.axvline(x=0, color="black", linestyle="--", linewidth=1.5)
    ax_hist.set_xlabel("$\\bar{\\Phi}$ value")
    ax_hist.set_ylabel("Cell count")
    ax_hist.set_title("Value Distribution")

    # Count statistics
    n_pos = np.sum(phibar > 0.1)
    n_neg = np.sum(phibar < -0.1)
    n_neutral = np.sum((phibar >= -0.1) & (phibar <= 0.1))
    stats_text = f"Positive (>0.1): {n_pos}\nNegative (<-0.1): {n_neg}\nNeutral (|Φ|≤0.1): {n_neutral}"
    ax_hist.text(
        0.98, 0.95, stats_text,
        transform=ax_hist.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()

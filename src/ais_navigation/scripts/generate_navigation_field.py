#!/usr/bin/env python3
"""
generate_navigation_field.py
Build the navigation potential field Phi-bar from AIS trajectory data.

Method (v2 - local direction-aware):
  1. Read AIS trajectory points from XLS files
  2. Build local dominant direction field theta_local(p) from AIS COG data
  3. Direction-weighted KDE: each point weighted by cos(COG - theta_local)
  4. Phi-bar = weighted_KDE_sum / weighted_KDE_abs_sum  in [-1, 1]

Output:
  - navigation_potential_field.csv
  - phibar_grad_x.csv
  - phibar_grad_y.csv
  - local_direction.csv (for debugging)
"""

import numpy as np
import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

# Geo-referencing parameters (from map3.m MATLAB script and map YAML)
# The PGM occupancy grid image is 1803 x 899 pixels at 0.5 m/pixel
#   World X: pixel_col * 0.5  →  [0, 901.5] meters
#   World Y: pixel_row * 0.5  →  [0, 449.5] meters
# Geographic → pixel transform:
#   pixel_col = 1 + (lon - 120.414716) / 0.23 * 1802
#   pixel_row = 1 + (lat - 32.081950) / (-0.101667) * 898
GEO_REF = {
    "lon_ref": 120.414716,   # longitude at pixel_col=1
    "lat_ref": 32.081950,    # latitude at pixel_row=1
    "d_lon": 120.644716 - 120.414716,   # 0.23 degrees
    "d_lat": 31.980283 - 32.081950,     # -0.101667 degrees
    "img_cols": 152,          # ACTUAL PGM width
    "img_rows": 78,           # ACTUAL PGM height
    "pgm_res": 0.5,           # PGM resolution (m/pixel)
}

# PGM world coordinate extent
WORLD_X_MIN = 0.0
WORLD_X_MAX = GEO_REF["img_cols"] * GEO_REF["pgm_res"]   # 76.0 m
WORLD_Y_MIN = 0.0
WORLD_Y_MAX = GEO_REF["img_rows"] * GEO_REF["pgm_res"]   # 39.0 m

# Grid resolution: must match PGM (0.5 m/cell)
RESOLUTION_M = 0.5

# Grid dimensions: match PGM exactly
GRID_COLS = GEO_REF["img_cols"]  # 152
GRID_ROWS = GEO_REF["img_rows"]  # 78

# KDE kernel bandwidth in meters
# Smaller bandwidth = tighter signals, better direction separation
BANDWIDTH_M = 2.0

# Local direction field radius in meters
LOCAL_DIR_RADIUS_M = 5.0

# Small constant to avoid division by zero
EPSILON = 1e-6

# AIS data directories relative to project root
AIS_DIRS = [
    "/root/demon3.16/数据/向右/",
    "/root/demon3.16/数据/航道数据收集/0-80/",
    "/root/demon3.16/数据/航道数据收集/3.16上通道向西/",
    "/root/demon3.16/数据/航道数据收集/3.16上通道向西 - 副本/",
    "/root/demon3.16/数据/航道数据收集/3.16下通道向西/",
    "/root/demon3.16/数据/航道数据收集/80-160/",
]


def read_ais_data(directories):
    """
    Read all AIS XLS files and return per-point arrays + trajectory direction.

    Each XLS file = one ship voyage. Trajectory direction is determined by
    net X displacement from first to last valid point:
      net_dx > 0 → eastbound  → weight +1
      net_dx < 0 → westbound  → weight -1
      ambiguous  → neutral    → weight  0

    All points in a file share the same direction weight.

    Returns:
      lon, lat, cog: per-point arrays
      traj_weights: per-point trajectory direction (+1, -1, or 0)
    """
    all_lon = []
    all_lat = []
    all_cog = []
    all_traj_weight = []

    n_east = 0
    n_west = 0
    n_neutral = 0

    for d in directories:
        if not os.path.isdir(d):
            print(f"  [WARN] Directory not found: {d}", file=sys.stderr)
            continue

        xls_files = [f for f in os.listdir(d) if f.endswith(".xls") or f.endswith(".xlsx")]
        print(f"  Reading {len(xls_files)} files from {os.path.basename(d)}...")

        for fn in xls_files:
            fp = os.path.join(d, fn)
            file_lons = []
            file_lats = []
            file_cogs = []
            try:
                if fn.endswith(".xlsx"):
                    import openpyxl
                    wb = openpyxl.load_workbook(fp, read_only=True)
                    sh = wb.active
                    for row_idx, row in enumerate(sh.iter_rows(values_only=True)):
                        if row_idx < 2: continue
                        if row[1] is None or row[2] is None: continue
                        try:
                            file_lons.append(float(row[1]))
                            file_lats.append(float(row[2]))
                            file_cogs.append(float(row[6]) if row[6] is not None else np.nan)
                        except (ValueError, IndexError, TypeError): continue
                    wb.close()
                else:
                    import xlrd
                    wb = xlrd.open_workbook(fp)
                    sh = wb.sheet_by_index(0)
                    for r in range(2, sh.nrows):
                        try:
                            file_lons.append(float(sh.cell_value(r, 1)))
                            file_lats.append(float(sh.cell_value(r, 2)))
                            file_cogs.append(float(sh.cell_value(r, 6)))
                        except (ValueError, IndexError): continue
            except Exception as e:
                print(f"    Skipping {fn}: {e}", file=sys.stderr)
                continue

            if len(file_lons) < 3:
                continue  # too few points to determine direction

            # Determine trajectory direction from net X displacement
            # Use robust median of first 1/3 vs last 1/3 to avoid outliers
            n = len(file_lons)
            first_lon = np.median(file_lons[:max(1, n // 3)])
            last_lon = np.median(file_lons[-max(1, n // 3):])
            net_dx = last_lon - first_lon

            if net_dx > 0.001:       # eastbound (left→right)
                weight = 1.0
                n_east += 1
            elif net_dx < -0.001:    # westbound (right→left)
                weight = -1.0
                n_west += 1
            else:
                weight = 0.0
                n_neutral += 1

            all_lon.extend(file_lons)
            all_lat.extend(file_lats)
            all_cog.extend(file_cogs)
            all_traj_weight.extend([weight] * len(file_lons))

    print(f"  Trajectories: eastbound={n_east}, westbound={n_west}, neutral={n_neutral}")
    print(f"  Total points: {len(all_lon)}")

    return (np.array(all_lon, dtype=np.float64),
            np.array(all_lat, dtype=np.float64),
            np.array(all_cog, dtype=np.float64),
            np.array(all_traj_weight, dtype=np.float64))


def latlon_to_world(lon, lat):
    """
    Convert geographic coordinates to world coordinates (meters)
    using the same affine transform as the PGM occupancy grid.
    """
    pixel_col = 1.0 + (lon - GEO_REF["lon_ref"]) / GEO_REF["d_lon"] * (GEO_REF["img_cols"] - 1)
    pixel_row = 1.0 + (lat - GEO_REF["lat_ref"]) / GEO_REF["d_lat"] * (GEO_REF["img_rows"] - 1)
    world_x = pixel_col * GEO_REF["pgm_res"]
    world_y = pixel_row * GEO_REF["pgm_res"]
    return world_x, world_y


def compute_local_direction_field(wx, wy, cog, radius_m):
    """
    Compute the local dominant COG direction at each grid cell center.

    For each cell center in the (GRID_ROWS x GRID_COLS) grid, collect all
    AIS points within radius_m, then histogram-vote to find the dominant
    COG direction.

    Args:
      wx, wy: world coords of AIS points (within PGM bounds, valid COG)
      cog: COG values (0-360 deg)
      radius_m: search radius in meters

    Returns:
      theta_local: (GRID_ROWS, GRID_COLS) array of dominant COG [deg],
                   NaN where no AIS points are available.
    """
    rows, cols = GRID_ROWS, GRID_COLS
    theta_local = np.full((rows, cols), np.nan, dtype=np.float64)

    # Grid cell centers in world coords
    xs = np.linspace(RESOLUTION_M / 2.0, WORLD_X_MAX - RESOLUTION_M / 2.0, cols)
    ys = np.linspace(RESOLUTION_M / 2.0, WORLD_Y_MAX - RESOLUTION_M / 2.0, rows)

    for r in range(rows):
        for c in range(cols):
            cx, cy = xs[c], ys[r]
            # Find points within radius
            dist_sq = (wx - cx) ** 2 + (wy - cy) ** 2
            mask = dist_sq <= radius_m ** 2
            cog_nearby = cog[mask]

            if len(cog_nearby) < 3:
                continue  # leave as NaN (too few points)

            # Histogram vote (36 bins for 10° resolution)
            hist, edges = np.histogram(cog_nearby, bins=36, range=(0, 360))
            dominant = edges[np.argmax(hist)] + 5.0  # bin center
            theta_local[r, c] = dominant

        if (r + 1) % 20 == 0:
            print(f"    Computing local direction: row {r + 1}/{rows}...")

    # Fill NaN cells by nearest-neighbor propagation
    theta_local = _fill_nan_nearest(theta_local)

    n_filled = np.sum(~np.isnan(theta_local))
    print(f"    Local direction field: {n_filled}/{rows * cols} cells filled")
    return theta_local


def compute_vector_field(points_wx, points_wy, points_cog,
                         wx_grid, wy_grid, bw_m=2.0):
    """
    基于KDE加权平均的期望航向矢量场 d(p) ∈ ℝ².

    对每个栅格点p，用高斯核对周围AIS点的COG方向做加权平均:
      d_x(p) = Σ cos(COG_i) × K(p, p_i) / Σ K(p, p_i)
      d_y(p) = Σ sin(COG_i) × K(p, p_i) / Σ K(p, p_i)

    然后归一化为单位矢量。

    返回:
      dir_x, dir_y: (ROWS, COLS) 单位矢量分量
    """
    n_points = len(points_wx)
    valid_cog = (points_cog >= 0) & (points_cog <= 360)
    if np.sum(valid_cog) == 0:
        print("    [WARN] No valid COG, using zero vectors")
        return (np.zeros_like(wx_grid), np.zeros_like(wx_grid))

    cog_rad = np.radians(points_cog)
    # COG convention: 0°=north, 90°=east.
    # Math convention: cos(0)=east, sin(0)=north.
    # So: dir_x (east) = sin(COG), dir_y (north) = cos(COG)
    cos_cog = np.sin(cog_rad)   # east component
    sin_cog = np.cos(cog_rad)   # north component

    rows, cols = wx_grid.shape
    n_cells = rows * cols
    h2 = 2.0 * bw_m * bw_m

    sum_w = np.zeros((rows, cols), dtype=np.float64)
    sum_wx = np.zeros((rows, cols), dtype=np.float64)
    sum_wy = np.zeros((rows, cols), dtype=np.float64)

    grid_x_flat = wx_grid.ravel()
    grid_y_flat = wy_grid.ravel()

    BLOCK_SIZE = max(1, min(20000, n_cells))
    for start in range(0, n_cells, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, n_cells)
        gx = grid_x_flat[start:end]
        gy = grid_y_flat[start:end]
        dx = gx[:, np.newaxis] - points_wx[np.newaxis, :]
        dy = gy[:, np.newaxis] - points_wy[np.newaxis, :]
        kernel = np.exp(-(dx * dx + dy * dy) / h2)  # (BLOCK, N)

        sum_w.ravel()[start:end] = np.sum(kernel, axis=1)
        sum_wx.ravel()[start:end] = np.sum(kernel * cos_cog[np.newaxis, :], axis=1)
        sum_wy.ravel()[start:end] = np.sum(kernel * sin_cog[np.newaxis, :], axis=1)

    # Normalize to unit vectors (handle low-density cells)
    mag = np.sqrt(sum_wx * sum_wx + sum_wy * sum_wy)
    min_mag = 1e-6
    valid = (sum_w.ravel() > 1e-3) & (mag.ravel() > min_mag)

    dir_x = np.where(mag > min_mag, sum_wx / mag, 0.0)
    dir_y = np.where(mag > min_mag, sum_wy / mag, 0.0)

    # Fill invalid cells with nearest neighbor
    if not np.all(valid):
        from scipy.interpolate import griddata
        rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        flat_r, flat_c = rr.ravel(), cc.ravel()
        dir_x_flat = dir_x.ravel()
        dir_y_flat = dir_y.ravel()
        dir_x_flat[~valid] = np.nan
        dir_y_flat[~valid] = np.nan
        nan_mask = np.isnan(dir_x_flat)
        if np.sum(~nan_mask) > 0:
            dir_x_flat[nan_mask] = griddata(
                (flat_r[~nan_mask], flat_c[~nan_mask]),
                dir_x_flat[~nan_mask],
                (flat_r[nan_mask], flat_c[nan_mask]), method="nearest")
            dir_y_flat[nan_mask] = griddata(
                (flat_r[~nan_mask], flat_c[~nan_mask]),
                dir_y_flat[~nan_mask],
                (flat_r[nan_mask], flat_c[nan_mask]), method="nearest")
        dir_x = dir_x_flat.reshape(rows, cols)
        dir_y = dir_y_flat.reshape(rows, cols)
        # Renormalize after fill
        mag = np.sqrt(dir_x * dir_x + dir_y * dir_y)
        dir_x = np.where(mag > min_mag, dir_x / mag, 0.0)
        dir_y = np.where(mag > min_mag, dir_y / mag, 0.0)

    print(f"    Vector field: dx_range=[{dir_x.min():.2f},{dir_x.max():.2f}], "
          f"dy_range=[{dir_y.min():.2f},{dir_y.max():.2f}]")
    return dir_x, dir_y


def _fill_nan_nearest(data):
    """Fill NaN values with the nearest non-NaN neighbor."""
    from scipy.interpolate import griddata

    rows, cols = data.shape
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    valid = ~np.isnan(data)

    if np.sum(valid) == 0:
        return np.zeros_like(data)

    filled = griddata(
        (rr[valid], cc[valid]), data[valid],
        (rr, cc), method="nearest"
    )
    return filled


def trajectory_kde(points_wx, points_wy, traj_weights,
                   wx_grid, wy_grid, bw_m=2.0):
    """
    分向轨迹密度估计 (论文公式1-2).

    东行轨迹 → f_down (顺流密度)
    西行轨迹 → f_up   (逆流密度)

    Phi-bar(p) = (f_down - f_up) / (f_down + f_up + epsilon)

    用较小带宽(2m)减少不同方向信号的空间重叠，
    直接用原始密度差计算势场，不做log/归一化压缩。
    """
    n_points = len(points_wx)
    if n_points == 0:
        return (np.zeros_like(wx_grid), np.zeros_like(wx_grid),
                np.zeros_like(wx_grid))

    down_mask = traj_weights > 0.5
    up_mask = traj_weights < -0.5
    n_down = np.sum(down_mask)
    n_up = np.sum(up_mask)
    print(f"    顺流(东行) {n_down} pts, 逆流(西行) {n_up} pts")
    print(f"    带宽 h={bw_m:.1f}m")

    rows, cols = wx_grid.shape
    n_cells = rows * cols
    h2 = 2.0 * bw_m * bw_m

    f_down = np.zeros((rows, cols), dtype=np.float64)
    f_up = np.zeros((rows, cols), dtype=np.float64)

    grid_x_flat = wx_grid.ravel()
    grid_y_flat = wy_grid.ravel()

    BLOCK_SIZE = max(1, min(30000, n_cells))
    for start in range(0, n_cells, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, n_cells)
        gx = grid_x_flat[start:end]
        gy = grid_y_flat[start:end]
        dx = gx[:, np.newaxis] - points_wx[np.newaxis, :]
        dy = gy[:, np.newaxis] - points_wy[np.newaxis, :]
        dist_sq = dx * dx + dy * dy

        f_down.ravel()[start:end] = np.sum(
            np.exp(-dist_sq[:, down_mask] / h2), axis=1)
        f_up.ravel()[start:end] = np.sum(
            np.exp(-dist_sq[:, up_mask] / h2), axis=1)

    print(f"    f_down: [{f_down.min():.2f}, {f_down.max():.2f}]")
    print(f"    f_up:   [{f_up.min():.2f}, {f_up.max():.2f}]")

    # Phi-bar (论文公式2，直接用原始密度)
    phibar = (f_down - f_up) / (f_down + f_up + EPSILON)
    # 对比度拉伸: 小值等比放大，极值不变
    # sign(x) * |x|^γ, γ<1  → 中间值向±1拉近
    gamma = 0.6
    phibar = np.sign(phibar) * np.power(np.abs(phibar), gamma)
    phibar = np.clip(phibar, -1.0, 1.0)

    return phibar, f_down, f_up


def build_grid():
    """
    Build the 2D evaluation grid in world coordinates (meters).

    Grid covers the full PGM occupancy grid extent:
      X: [0, WORLD_X_MAX]
      Y: [0, WORLD_Y_MAX]

    Returns:
      world_x_grid: (ROWS, COLS) array of world X coordinate at each cell center
      world_y_grid: (ROWS, COLS) array of world Y coordinate at each cell center
    """
    # Cell centers in world coordinates
    xs = np.linspace(RESOLUTION_M / 2.0, WORLD_X_MAX - RESOLUTION_M / 2.0, GRID_COLS)
    ys = np.linspace(RESOLUTION_M / 2.0, WORLD_Y_MAX - RESOLUTION_M / 2.0, GRID_ROWS)
    world_x_grid, world_y_grid = np.meshgrid(xs, ys)  # shape (ROWS, COLS)
    return world_x_grid, world_y_grid


def compute_gradient(phibar, resolution_m):
    """
    Compute spatial gradient of phibar using central differences.

    Args:
      phibar: (ROWS, COLS) array
      resolution_m: cell size in meters

    Returns:
      gx: (ROWS, COLS) gradient in x-direction (longitude/columns)
      gy: (ROWS, COLS) gradient in y-direction (latitude/rows)
    """
    gy, gx = np.gradient(phibar, resolution_m, resolution_m)
    return gx, gy  # gx, gy both shape (ROWS, COLS)


def save_csv(data, path):
    """Save 2D array as CSV.

    The C++ loadCSVToLayer applies:
      mat = raw.T; mat = mat.rowwise().reverse(); mat = mat.colwise().reverse()
    This gives: GridMap(i,j) = csv[77-j][151-i]

    We match the original waterway_map CSV convention:
      CSV row 0 = north (high Y), col 0 = west (low X)  [image convention]

    Our data is in world convention: row 0 = south (low Y), col 0 = west (low X).
    So we need np.flipud to convert world → image convention (flip Y only).
    The X-axis ends up reversed in GridMap, matching the existing system behavior.
    """
    data_flipped = np.flipud(data)
    np.savetxt(path, data_flipped, delimiter=",", fmt="%.6f")
    print(f"  Saved {data.shape[0]}x{data.shape[1]} grid to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build navigation potential field from AIS trajectory data"
    )
    parser.add_argument(
        "--output-dir",
        default="/root/demon3.16/src/ais_navigation/map/",
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--bandwidth", type=float, default=BANDWIDTH_M, help="KDE bandwidth in meters"
    )
    parser.add_argument(
        "--local-radius", type=float, default=LOCAL_DIR_RADIUS_M,
        help="Local direction field search radius in meters"
    )
    parser.add_argument(
        "--no-gradients",
        action="store_true",
        default=False,
        help="Skip generating gradient CSV files",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ============================================================
    # Step 1: Read AIS data
    # ============================================================
    print("=" * 60)
    print("Step 1/6: Reading AIS trajectory data...")
    print("=" * 60)
    lon, lat, cog, traj_weights = read_ais_data(AIS_DIRS)

    # Safety: trim all arrays to same length
    min_len = min(len(lon), len(lat), len(cog), len(traj_weights))
    lon = lon[:min_len]; lat = lat[:min_len]; cog = cog[:min_len]; traj_weights = traj_weights[:min_len]

    # Convert lat/lon to world coordinates (meters)
    print("  Converting to world coordinates...")
    world_x, world_y = latlon_to_world(lon, lat)

    # Filter to PGM bounds (no COG filter needed with trajectory direction)
    mask_pgm = (
        (world_x >= WORLD_X_MIN) & (world_x <= WORLD_X_MAX)
        & (world_y >= WORLD_Y_MIN) & (world_y <= WORLD_Y_MAX)
    )
    # Exclude neutral trajectories (weight=0)
    mask_dir = (traj_weights > 0.5) | (traj_weights < -0.5)
    mask_all = mask_pgm & mask_dir

    wx = world_x[mask_all]
    wy = world_y[mask_all]
    traj_w = traj_weights[mask_all]
    cog_valid = cog[mask_all]

    n_east = np.sum(traj_w > 0.5)
    n_west = np.sum(traj_w < -0.5)
    print(f"  Points within PGM bounds: {len(wx)} (eastbound={n_east}, westbound={n_west})")
    if len(wx) == 0:
        print("[ERROR] No valid AIS points!", file=sys.stderr)
        sys.exit(1)

    # ============================================================
    # Step 2: Build evaluation grid
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 2/6: Building evaluation grid...")
    print("=" * 60)
    wx_grid, wy_grid = build_grid()
    print(f"  Grid: {GRID_ROWS} rows x {GRID_COLS} cols")
    print(f"  Resolution: {RESOLUTION_M} m/cell")
    print(f"  Physical size: {GRID_COLS * RESOLUTION_M:.0f}m x "
          f"{GRID_ROWS * RESOLUTION_M:.0f}m")

    # ============================================================
    # Step 3: Compute local direction field (for C++ cost function)
    # ============================================================
    print("\n" + "=" * 60)
    print(f"Step 3/6: Computing local direction field "
          f"(radius={args.local_radius}m)...")
    print("=" * 60)
    theta_local = compute_local_direction_field(wx, wy, cog_valid,
                                                 args.local_radius)

    # ============================================================
    # Step 4: Trajectory-direction KDE
    # ============================================================
    print("\n" + "=" * 60)
    print(f"Step 4/6: Computing trajectory-direction KDE "
          f"(bandwidth={args.bandwidth}m)...")
    print("=" * 60)

    phibar, f_down, f_up = trajectory_kde(
        wx, wy, traj_w,
        wx_grid, wy_grid, args.bandwidth
    )

    # ============================================================
    # Step 5: Analyze Phi-bar
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 5/6: Analyzing Phi-bar...")
    print("=" * 60)
    print(f"  Phi-bar range: [{phibar.min():.4f}, {phibar.max():.4f}]")

    # Count cells by type
    total_density = f_down + f_up
    n_down = np.sum(phibar > 0.1)
    n_up = np.sum(phibar < -0.1)
    n_mixed = np.sum((np.abs(phibar) <= 0.1) & (total_density > 0.01))
    n_free = np.sum((np.abs(phibar) <= 0.1) & (total_density <= 0.01))
    print(f"  Cell breakdown:")
    print(f"    Positive (Phi > +0.1): {n_down}")
    print(f"    Negative (Phi < -0.1): {n_up}")
    print(f"    Mixed    (|Phi| <= 0.1, traffic): {n_mixed}")
    print(f"    Free water (|Phi| <= 0.1, no traffic): {n_free}")

    # ============================================================
    # Step 6: Save outputs
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 6/6: Saving outputs...")
    print("=" * 60)

    # Save Phi-bar
    phibar_path = os.path.join(args.output_dir, "navigation_potential_field.csv")
    save_csv(phibar, phibar_path)

    # Save gradients
    if not args.no_gradients:
        gx, gy = compute_gradient(phibar, RESOLUTION_M)
        gx_path = os.path.join(args.output_dir, "phibar_grad_x.csv")
        gy_path = os.path.join(args.output_dir, "phibar_grad_y.csv")
        save_csv(gx, gx_path)
        save_csv(gy, gy_path)

    # Save local direction field (for debugging)
    dir_path = os.path.join(args.output_dir, "local_direction.csv")
    save_csv(np.nan_to_num(theta_local, nan=0.0), dir_path)

    # Compute and save KDE-weighted vector field d(p) ∈ ℝ²
    print("\n  Computing KDE-weighted direction vector field...")
    dir_x, dir_y = compute_vector_field(wx, wy, cog_valid,
                                        wx_grid, wy_grid, bw_m=2.0)
    save_csv(dir_x, os.path.join(args.output_dir, "dir_x.csv"))
    save_csv(dir_y, os.path.join(args.output_dir, "dir_y.csv"))

    print("\n" + "=" * 60)
    print("Done! Navigation potential field built successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()

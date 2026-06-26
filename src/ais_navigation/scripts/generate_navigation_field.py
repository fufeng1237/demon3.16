#!/usr/bin/env python3
"""
generate_navigation_field.py
Build the navigation potential field Phi-bar from AIS trajectory data.

Method (ref: 基于AIS导航势场的无人艇路径规划.md):
  1. Read AIS trajectory points from XLS files
  2. Classify points as downstream/upstream by COG (Course Over Ground)
  3. Compute Gaussian KDE for each directional set on the map grid
  4. Phi-bar = (f_down - f_up) / (f_down + f_up + epsilon)  in [-1, 1]

Output:
  - navigation_potential_field.csv  (180 rows x 90 cols transpose, values in [-1, 1])
  - phibar_grad_x.csv               (x-gradient of Phi-bar)
  - phibar_grad_y.csv               (y-gradient of Phi-bar)
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

# Grid dimensions calculated from PGM extent
GRID_COLS = int(np.ceil(WORLD_X_MAX / RESOLUTION_M))  # ~601
GRID_ROWS = int(np.ceil(WORLD_Y_MAX / RESOLUTION_M))  # ~300

# KDE kernel bandwidth in meters
BANDWIDTH_M = 5.0

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
    Read all AIS XLS files and return arrays of (lon, lat, cog).

    XLS format (rows 0 and 1 are headers):
      Col 0: mmsi
      Col 1: 经度 (longitude)
      Col 2: 纬度 (latitude)
      Col 3: 时间 (time)
      Col 4: 速度 (speed, knots)
      Col 5: 船首向 (heading, deg)
      Col 6: 对地航向 (COG, deg 0-360)
      Col 7: 航行状态 (status)
    """
    all_lon = []
    all_lat = []
    all_cog = []

    for d in directories:
        if not os.path.isdir(d):
            print(f"  [WARN] Directory not found: {d}", file=sys.stderr)
            continue

        xls_files = [f for f in os.listdir(d) if f.endswith(".xls") or f.endswith(".xlsx")]
        print(f"  Reading {len(xls_files)} files from {os.path.basename(d)}...")

        for fn in xls_files:
            fp = os.path.join(d, fn)
            try:
                # Use xlrd for .xls, openpyxl for .xlsx
                if fn.endswith(".xlsx"):
                    import openpyxl
                    wb = openpyxl.load_workbook(fp, read_only=True)
                    sh = wb.active
                    for row_idx, row in enumerate(sh.iter_rows(values_only=True)):
                        if row_idx < 2:
                            continue
                        if row[1] is None or row[2] is None:
                            continue
                        # Convert FIRST, append only if all succeed (atomicity)
                        try:
                            lon_val = float(row[1])
                            lat_val = float(row[2])
                            cog_val = float(row[6]) if row[6] is not None else np.nan
                        except (ValueError, IndexError, TypeError):
                            continue
                        all_lon.append(lon_val)
                        all_lat.append(lat_val)
                        all_cog.append(cog_val)
                    wb.close()
                else:
                    import xlrd
                    wb = xlrd.open_workbook(fp)
                    sh = wb.sheet_by_index(0)
                    for r in range(2, sh.nrows):
                        # Convert FIRST, append only if all succeed (atomicity)
                        try:
                            lon_val = float(sh.cell_value(r, 1))
                            lat_val = float(sh.cell_value(r, 2))
                            cog_val = float(sh.cell_value(r, 6))
                        except (ValueError, IndexError):
                            continue
                        all_lon.append(lon_val)
                        all_lat.append(lat_val)
                        all_cog.append(cog_val)
            except Exception as e:
                print(f"    Skipping {fn}: {e}", file=sys.stderr)
                continue

    lon_arr = np.array(all_lon, dtype=np.float64)
    lat_arr = np.array(all_lat, dtype=np.float64)
    cog_arr = np.array(all_cog, dtype=np.float64)

    return lon_arr, lat_arr, cog_arr


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


def filter_and_classify(world_x, world_y, cog):
    """
    Filter points to the PGM world coordinate bounds, then classify by COG.

    Classification approach:
      - Compute the dominant COG direction using a histogram
      - Downstream: COG within +/- 90 degrees of the dominant direction
      - Upstream: COG within +/- 90 degrees of the opposite direction
      - Points with invalid COG (NaN, out of 0-360) are excluded
    """
    # Filter by world coordinate bounds (PGM extent)
    mask_pgm = (
        (world_x >= WORLD_X_MIN)
        & (world_x <= WORLD_X_MAX)
        & (world_y >= WORLD_Y_MIN)
        & (world_y <= WORLD_Y_MAX)
    )
    wx_f = world_x[mask_pgm]
    wy_f = world_y[mask_pgm]
    cog_f = cog[mask_pgm]

    print(f"  Points within PGM world bounds: {len(wx_f)} / {len(world_x)}")

    # Filter valid COG
    valid_cog_mask = (cog_f >= 0) & (cog_f <= 360) & ~np.isnan(cog_f)
    valid_wx = wx_f[valid_cog_mask]
    valid_wy = wy_f[valid_cog_mask]
    valid_cog = cog_f[valid_cog_mask]

    if len(valid_cog) == 0:
        print("  [ERROR] No points with valid COG found!", file=sys.stderr)
        return np.array([]), np.array([]), np.array([]), np.array([]), 0.0

    # Compute dominant COG direction using histogram
    hist, edges = np.histogram(valid_cog, bins=72, range=(0, 360))
    dominant_angle = edges[np.argmax(hist)] + 2.5  # bin center
    print(f"  Dominant COG direction: {dominant_angle:.1f}°")

    # Downstream: COG within +/- 90° of dominant angle
    diff = np.abs((valid_cog - dominant_angle + 180) % 360 - 180)
    downstream_mask = diff < 90

    wx_down = valid_wx[downstream_mask]
    wy_down = valid_wy[downstream_mask]
    wx_up = valid_wx[~downstream_mask]
    wy_up = valid_wy[~downstream_mask]

    print(f"  Downstream (T_down): {len(wx_down)} points (COG ~{dominant_angle:.0f}° ± 90°)")
    print(f"  Upstream   (T_up):   {len(wx_up)} points (COG ~{(dominant_angle+180)%360:.0f}° ± 90°)")

    return wx_down, wy_down, wx_up, wy_up, dominant_angle


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


def gaussian_kernel_sum(points_wx, points_wy, wx_grid, wy_grid, bw_m):
    """
    Compute summed Gaussian kernel density on the grid.

    f(p) = sum_i exp(-||p - p_i||^2 / (2 * h^2))

    Uses a vectorized block-wise computation to handle large point sets.

    Args:
      points_wx, points_wy: (N,) trajectory point world coordinates (meters)
      wx_grid, wy_grid: (ROWS, COLS) meshgrid of world coordinates
      bw_m: kernel bandwidth in meters

    Returns:
      density: (ROWS, COLS) density array
    """
    n_points = len(points_wx)
    if n_points == 0:
        return np.zeros_like(wx_grid)

    rows, cols = wx_grid.shape
    n_cells = rows * cols

    h2 = 2.0 * bw_m * bw_m  # 2 * h^2 (all in meters, no conversion needed!)

    density = np.zeros((rows, cols), dtype=np.float64)
    grid_x_flat = wx_grid.ravel()
    grid_y_flat = wy_grid.ravel()

    BLOCK_SIZE = 50000  # number of grid cells per block
    for start in range(0, n_cells, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, n_cells)
        gx = grid_x_flat[start:end]
        gy = grid_y_flat[start:end]

        # (BLOCK, N) squared distances in meters
        dx = gx[:, np.newaxis] - points_wx[np.newaxis, :]
        dy = gy[:, np.newaxis] - points_wy[np.newaxis, :]
        dist_sq = dx * dx + dy * dy

        # exp(-dist^2 / (2*h^2))
        block_density = np.sum(np.exp(-dist_sq / h2), axis=1)
        density.ravel()[start:end] = block_density

    return density


def compute_phibar(f_down, f_up, epsilon=EPSILON):
    """
    Compute normalized navigation potential field.

    Phi-bar(p) = (f_down - f_up) / (f_down + f_up + epsilon)

    Range: [-1, 1]
    - +1: pure downstream channel
    - -1: pure upstream channel
    -  0: mixed water / no historical traffic
    """
    denominator = f_down + f_up + epsilon
    phibar = (f_down - f_up) / denominator
    return np.clip(phibar, -1.0, 1.0)


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

    IMPORTANT: The C++ loadCSVToLayer applies transpose + rowwise.reverse() +
    colwise.reverse() to the CSV data. This transformation expects the CSV to
    be in "image" convention (row 0 = top/north, col 0 = left/west). We flip
    the Y-axis (rows) before saving so the data maps correctly to ROS world
    coordinates after the C++ transformations.
    """
    # Flip rows: CSV row 0 will be highest Y (north), matching image convention
    data_flipped = np.flipud(data)
    np.savetxt(path, data_flipped, delimiter=",", fmt="%.6f")
    print(f"  Saved {data.shape[0]}x{data.shape[1]} grid to {path} (Y-flipped)")


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
    print("Step 1/5: Reading AIS trajectory data...")
    print("=" * 60)
    lon, lat, cog = read_ais_data(AIS_DIRS)
    print(f"  Total trajectory points read: {len(lon)}")

    # Convert lat/lon to world coordinates (meters)
    print("  Converting to world coordinates...")
    world_x, world_y = latlon_to_world(lon, lat)
    print(f"  World X range: [{world_x.min():.1f}, {world_x.max():.1f}] m")
    print(f"  World Y range: [{world_y.min():.1f}, {world_y.max():.1f}] m")

    # ============================================================
    # Step 2: Filter and classify by COG direction
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 2/5: Classifying trajectory points by COG direction...")
    print("=" * 60)
    result = filter_and_classify(world_x, world_y, cog)
    wx_down, wy_down, wx_up, wy_up, dominant_angle = result

    if len(wx_down) == 0 and len(wx_up) == 0:
        print("[ERROR] No points after classification!", file=sys.stderr)
        sys.exit(1)

    # ============================================================
    # Step 3: Build evaluation grid
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 3/5: Building evaluation grid in world coordinates...")
    print("=" * 60)
    wx_grid, wy_grid = build_grid()
    print(f"  Grid: {GRID_ROWS} rows x {GRID_COLS} cols")
    print(f"  Resolution: {RESOLUTION_M} m/cell")
    print(f"  Physical size: {GRID_COLS * RESOLUTION_M:.0f}m x {GRID_ROWS * RESOLUTION_M:.0f}m")
    print(f"  PGM occupancy grid extent: [{WORLD_X_MIN}, {WORLD_X_MAX}] x [{WORLD_Y_MIN}, {WORLD_Y_MAX}] m")

    # ============================================================
    # Step 4: Compute KDE
    # ============================================================
    print("\n" + "=" * 60)
    print(f"Step 4/5: Computing Gaussian KDE (bandwidth={args.bandwidth}m)...")
    print("=" * 60)

    print(f"  Computing f_down from {len(wx_down)} points...")
    f_down = gaussian_kernel_sum(wx_down, wy_down, wx_grid, wy_grid, args.bandwidth)
    print(f"    f_down range: [{f_down.min():.2f}, {f_down.max():.2f}]")

    print(f"  Computing f_up from {len(wx_up)} points...")
    f_up = gaussian_kernel_sum(wx_up, wy_up, wx_grid, wy_grid, args.bandwidth)
    print(f"    f_up range: [{f_up.min():.2f}, {f_up.max():.2f}]")

    # ============================================================
    # Step 5: Compute Phi-bar and save
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 5/5: Computing Phi-bar and saving...")
    print("=" * 60)

    phibar = compute_phibar(f_down, f_up)
    print(f"  Phi-bar range: [{phibar.min():.4f}, {phibar.max():.4f}]")

    # Count cells by type
    n_down = np.sum(phibar > 0.1)
    n_up = np.sum(phibar < -0.1)
    n_mixed = np.sum((phibar >= -0.1) & (phibar <= 0.1) & ((f_down + f_up) > 0.01))
    n_free = np.sum((phibar >= -0.1) & (phibar <= 0.1) & ((f_down + f_up) <= 0.01))
    print(f"  Cell breakdown:")
    print(f"    Downstream cells (Phi > +0.1): {n_down}")
    print(f"    Upstream cells   (Phi < -0.1): {n_up}")
    print(f"    Mixed cells      (|Phi| <= 0.1, traffic): {n_mixed}")
    print(f"    Free water       (|Phi| <= 0.1, no traffic): {n_free}")

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

    print("\n" + "=" * 60)
    print("Done! Navigation potential field built successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()

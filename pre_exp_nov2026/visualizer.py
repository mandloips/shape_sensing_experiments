#!/usr/bin/env -S python3 -s
"""
visualizer.py

Visualizes the ICP-aligned reconstructed guidewire shape against the CAD ground truth
on the same 3D coordinate axes, following the style of SENSORS_2022 Figure 3.
Highlights all 3 embedded sensors (Sensor 1, Sensor 2, Sensor 3).

Inputs:
  - helical_rig_sensed_shape_icp.csv: 500 transformed sensed shape points
  - helix_ground_truth_100mm_icp.csv: ground truth centerline points

Outputs:
  - helical_rig_shape_visualization.png: High-resolution plot comparing both curves
    (Ground truth in red, EM-based sensed shape in green, with markers for Sensors 1, 2, and 3).
"""

import sys
import os

# Ensure clean environment without user site conflicts
if os.environ.get("PYTHONNOUSERSITE") != "1" and not sys.flags.no_user_site:
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, ["-s"] + sys.argv)

import csv
import argparse
import numpy as np
import matplotlib
# Use Agg backend by default for non-interactive / headless saving
if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_point_csv(filepath):
    """
    Loads 3D points from a CSV file.
    """
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        pts = []
        for r in reader:
            x = float(r.get("x_mm", r.get("Tx", r.get("x", 0.0))))
            y = float(r.get("y_mm", r.get("Ty", r.get("y", 0.0))))
            z = float(r.get("z_mm", r.get("Tz", r.get("z", 0.0))))
            pts.append([x, y, z])
    return np.array(pts)


def get_sensor_indices(sensed_pts, csv_path=None):
    """
    Finds the curve indices corresponding to Sensor 1 (Tip), Sensor 2 (Middle),
    and Sensor 3 (Base). On an equidistant curve, Sensor 2 is located at the
    arc-length distance corresponding to segment 1 spacing (~60.5 mm).
    """
    n_pts = len(sensed_pts)
    if n_pts <= 2:
        return 0, 0, max(0, n_pts - 1)

    diffs = np.linalg.norm(np.diff(sensed_pts, axis=0), axis=1)
    cum_s = np.insert(np.cumsum(diffs), 0, 0.0)
    total_length = cum_s[-1]

    l1 = None
    if csv_path:
        cur = os.path.dirname(os.path.abspath(csv_path))
        while cur and cur != os.path.dirname(cur):
            for fname in ["sensor_spacing.csv", os.path.join("data", "sensor_spacing.csv")]:
                cand = os.path.join(cur, fname)
                if os.path.isfile(cand):
                    try:
                        with open(cand, "r") as f:
                            reader = csv.DictReader(f)
                            row1 = next(reader)
                            l1 = float(row1.get("spacing_mm", row1.get("distance_mm", 60.5)))
                            break
                    except Exception:
                        pass
            if l1 is not None:
                break
            cur = os.path.dirname(cur)

    if l1 is None:
        l1 = total_length * (60.5 / 307.0)

    idx_s1 = 0
    idx_s2 = int(np.argmin(np.abs(cum_s - l1)))
    idx_s3 = n_pts - 1
    return idx_s1, idx_s2, idx_s3


def plot_shapes(sensed_csv, gt_csv, output_img=None, show_plot=False):
    if not os.path.isfile(sensed_csv):
        raise FileNotFoundError(f"Sensed shape file not found: {sensed_csv}")
    if not os.path.isfile(gt_csv):
        raise FileNotFoundError(f"Ground truth file not found: {gt_csv}")

    if output_img is None:
        out_dir = os.path.dirname(os.path.abspath(sensed_csv))
        output_img = os.path.join(out_dir, "helical_rig_shape_visualization.png")

    sensed_pts = load_point_csv(sensed_csv)
    gt_pts = load_point_csv(gt_csv)

    print(f"\nVisualizing Guidewire Shape Comparison:")
    print(f"  Sensed shape (green): {sensed_csv} ({len(sensed_pts)} points)")
    print(f"  Ground truth (red):   {gt_csv} ({len(gt_pts)} points)")

    # Identify indices for Sensor 1, Sensor 2, and Sensor 3 along the equidistant curve
    idx_s1, idx_s2, idx_s3 = get_sensor_indices(sensed_pts, sensed_csv)

    s1_pos = sensed_pts[idx_s1]
    s2_pos = sensed_pts[idx_s2]
    s3_pos = sensed_pts[idx_s3]

    print(f"  Sensor 1 (Tip, idx {idx_s1}):    [{s1_pos[0]:.2f}, {s1_pos[1]:.2f}, {s1_pos[2]:.2f}] mm")
    print(f"  Sensor 2 (Middle, idx {idx_s2}): [{s2_pos[0]:.2f}, {s2_pos[1]:.2f}, {s2_pos[2]:.2f}] mm")
    print(f"  Sensor 3 (Base, idx {idx_s3}):   [{s3_pos[0]:.2f}, {s3_pos[1]:.2f}, {s3_pos[2]:.2f}] mm")

    # Create figure with 2 subplots:
    # 1. Main 3D perspective plot (matching SENSORS_2022 Fig 3)
    # 2. 2D top-down projection (X-Y plane)
    fig = plt.figure(figsize=(15, 6))

    # --- Subplot 1: 3D View ---
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")

    # Plot Ground Truth curve in red
    ax3d.plot(
        gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2],
        color="red",
        linewidth=2.5,
        label="Ground truth shape",
        alpha=0.85
    )

    # Plot EM-based sensed curve in green
    ax3d.plot(
        sensed_pts[:, 0], sensed_pts[:, 1], sensed_pts[:, 2],
        color="green",
        linewidth=2.5,
        linestyle="-",
        label="EM-based shape sensing",
        alpha=0.95
    )

    # Mark embedded sensors along the guidewire
    ax3d.scatter(
        s1_pos[0], s1_pos[1], s1_pos[2],
        color="blue", marker="o", s=70, zorder=6,
        label="Sensor 1 (Tip)"
    )
    ax3d.scatter(
        s2_pos[0], s2_pos[1], s2_pos[2],
        color="darkorange", marker="s", s=70, zorder=6,
        label="Sensor 2 (Middle)"
    )
    ax3d.scatter(
        s3_pos[0], s3_pos[1], s3_pos[2],
        color="black", marker="^", s=70, zorder=6,
        label="Sensor 3 (Base)"
    )

    # Text annotations in 3D
    ax3d.text(s1_pos[0] + 3, s1_pos[1] + 3, s1_pos[2] + 2, "S1", color="blue", fontweight="bold", fontsize=10)
    ax3d.text(s2_pos[0] + 3, s2_pos[1] + 3, s2_pos[2] + 2, "S2", color="darkorange", fontweight="bold", fontsize=10)
    ax3d.text(s3_pos[0] + 3, s3_pos[1] + 3, s3_pos[2] + 2, "S3", color="black", fontweight="bold", fontsize=10)

    ax3d.set_xlabel("X (mm)", fontsize=11, labelpad=8)
    ax3d.set_ylabel("Y (mm)", fontsize=11, labelpad=8)
    ax3d.set_zlabel("Z (mm)", fontsize=11, labelpad=8)
    ax3d.set_title("3D Shape Sensing Reconstruction", fontsize=12, fontweight="bold")
    ax3d.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax3d.grid(True, linestyle="--", alpha=0.5)

    # Equalize 3D aspect ratio
    all_pts = np.vstack([gt_pts, sensed_pts])
    max_range = np.array([
        all_pts[:, 0].max() - all_pts[:, 0].min(),
        all_pts[:, 1].max() - all_pts[:, 1].min(),
        all_pts[:, 2].max() - all_pts[:, 2].min()
    ]).max() / 2.0

    mid_x = (all_pts[:, 0].max() + all_pts[:, 0].min()) * 0.5
    mid_y = (all_pts[:, 1].max() + all_pts[:, 1].min()) * 0.5
    mid_z = (all_pts[:, 2].max() + all_pts[:, 2].min()) * 0.5

    ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3d.set_zlim(mid_z - max_range, mid_z + max_range)
    ax3d.view_init(elev=28, azim=45)

    # --- Subplot 2: Top-down (X-Y) Projection ---
    ax2d = fig.add_subplot(1, 2, 2)
    ax2d.plot(gt_pts[:, 0], gt_pts[:, 1], color="red", linewidth=2.5, label="Ground truth shape", alpha=0.85)
    ax2d.plot(sensed_pts[:, 0], sensed_pts[:, 1], color="green", linewidth=2.5, label="EM-based shape sensing", alpha=0.95)

    ax2d.scatter(s1_pos[0], s1_pos[1], color="blue", marker="o", s=80, zorder=6, label="Sensor 1 (Tip)")
    ax2d.scatter(s2_pos[0], s2_pos[1], color="darkorange", marker="s", s=80, zorder=6, label="Sensor 2 (Middle)")
    ax2d.scatter(s3_pos[0], s3_pos[1], color="black", marker="^", s=80, zorder=6, label="Sensor 3 (Base)")

    ax2d.text(s1_pos[0] + 4, s1_pos[1] + 2, "S1", color="blue", fontweight="bold", fontsize=11)
    ax2d.text(s2_pos[0] + 4, s2_pos[1] + 2, "S2", color="darkorange", fontweight="bold", fontsize=11)
    ax2d.text(s3_pos[0] + 4, s3_pos[1] + 2, "S3", color="black", fontweight="bold", fontsize=11)

    ax2d.set_xlabel("X (mm)", fontsize=11)
    ax2d.set_ylabel("Y (mm)", fontsize=11)
    ax2d.set_title("Top-Down Projection (X-Y Plane)", fontsize=12, fontweight="bold")
    ax2d.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_img)), exist_ok=True)
    plt.savefig(output_img, dpi=300, bbox_inches="tight")
    print(f"Saved visualization: {output_img}\n")

    if show_plot and "DISPLAY" in os.environ:
        plt.show()

    return output_img


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ICP-aligned sensed guidewire shape and ground truth."
    )
    parser.add_argument(
        "--sensed",
        default=None,
        help="Path to helical_rig_sensed_shape_icp.csv."
    )
    parser.add_argument(
        "--gt",
        default=None,
        help="Path to helix_ground_truth_100mm_icp.csv."
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output image path (.png)."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot window interactively if GUI is available."
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")

    # Locate sensed ICP file
    sensed_file = args.sensed
    if sensed_file is None:
        candidates = [
            os.path.join(data_dir, "helical_rig_sensed_shape_icp.csv"),
            os.path.join(script_dir, "helical_rig_sensed_shape_icp.csv"),
            "helical_rig_sensed_shape_icp.csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                sensed_file = c
                break

    # Locate ground truth ICP file
    gt_file = args.gt
    if gt_file is None:
        candidates = [
            os.path.join(data_dir, "helix_ground_truth_100mm_icp.csv"),
            os.path.join(script_dir, "helix_ground_truth_100mm_icp.csv"),
            "helix_ground_truth_100mm_icp.csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                gt_file = c
                break

    if sensed_file is None or not os.path.exists(sensed_file):
        print("Error: helical_rig_sensed_shape_icp.csv not found. Please run icp.py first.")
        sys.exit(1)

    if gt_file is None or not os.path.exists(gt_file):
        print("Error: helix_ground_truth_100mm_icp.csv not found. Please run icp.py first.")
        sys.exit(1)

    plot_shapes(sensed_file, gt_file, args.output, args.show)


if __name__ == "__main__":
    main()

#!/usr/bin/env -S python3 -s
"""
compare_rigid_vs_gelatin.py (exp_nov2026)

Generates comparative visualization plots for each shape:
  1) Shape 1: Rigid vs Gelatin (and CAD Ground Truth)
  2) Shape 2: Rigid vs Gelatin (and CAD Ground Truth)
  3) Shape 3: Rigid vs Gelatin (and CAD Ground Truth)
  4) Shape 4: Rigid vs Gelatin (and CAD Ground Truth)
  5) Shape 5: Rigid vs Gelatin (and CAD Ground Truth)

For each shape, creates a comprehensive 3-panel figure:
  - Panel 1: 3D Comparison View (Ground Truth vs Rigid vs Gelatin)
  - Panel 2: Top-Down Projection (X-Y Plane)
  - Panel 3: Arc-Length Error Profile along Guidewire Length (0 to 302 mm)
"""

import sys
import os

# Ensure clean environment without user site conflicts
if os.environ.get("PYTHONNOUSERSITE") != "1" and not sys.flags.no_user_site:
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, ["-s"] + sys.argv)

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_pts(filepath):
    """Loads 3D coordinates from CSV."""
    pts = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            x = float(r.get("X", r.get("x_mm", r.get("Tx", r.get("x", 0.0)))))
            y = float(r.get("Y", r.get("y_mm", r.get("Ty", r.get("y", 0.0)))))
            z = float(r.get("Z", r.get("z_mm", r.get("Tz", r.get("z", 0.0)))))
            pts.append([x, y, z])
    return np.array(pts)


def load_errors(filepath):
    """Loads point-by-point errors from error CSV."""
    with open(filepath, "r", newline="") as f:
        lines = [l for l in f if not l.startswith("#")]
        reader = csv.DictReader(lines)
        return np.array([float(r["error_mm"]) for r in reader])


def get_sensor_indices(pts, l1=59.856):
    """Returns indices corresponding to S1 (Tip), S2 (Middle), and S3 (Base)."""
    n = len(pts)
    if n <= 2:
        return 0, 0, max(0, n - 1)
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum_s = np.insert(np.cumsum(diffs), 0, 0.0)
    idx_s1 = 0
    idx_s2 = int(np.argmin(np.abs(cum_s - l1)))
    idx_s3 = n - 1
    return idx_s1, idx_s2, idx_s3, cum_s


def plot_shape_comparison(shape_num, data_dir, out_dir=None):
    """
    Plots the comparative figure for Shape {shape_num}:
    Rigid vs Gelatin vs CAD Ground Truth.
    """
    s_name = f"s{shape_num}"
    rigid_dir = os.path.join(data_dir, s_name)
    gel_dir = os.path.join(data_dir, f"{s_name}_gelatin")

    rigid_icp_f = os.path.join(rigid_dir, f"exp_{s_name}_sensed_shape_icp.csv")
    gel_icp_f = os.path.join(gel_dir, f"exp_{s_name}_gelatin_sensed_shape_icp.csv")
    gt_f = os.path.join(rigid_dir, f"{s_name}_ground_truth.csv")
    rigid_err_f = os.path.join(rigid_dir, f"exp_{s_name}_sensed_shape_error.csv")
    gel_err_f = os.path.join(gel_dir, f"exp_{s_name}_gelatin_sensed_shape_error.csv")

    for f in [rigid_icp_f, gel_icp_f, gt_f, rigid_err_f, gel_err_f]:
        if not os.path.isfile(f):
            print(f"Error: missing file {f}")
            return None

    # Load data
    pts_rigid = load_pts(rigid_icp_f)
    pts_gel = load_pts(gel_icp_f)
    pts_gt = load_pts(gt_f)
    err_rigid = load_errors(rigid_err_f)
    err_gel = load_errors(gel_err_f)

    r_s1, r_s2, r_s3, cum_s_r = get_sensor_indices(pts_rigid)
    g_s1, g_s2, g_s3, cum_s_g = get_sensor_indices(pts_gel)

    mean_r, std_r = np.mean(err_rigid), np.std(err_rigid)
    mean_g, std_g = np.mean(err_gel), np.std(err_gel)

    fig = plt.figure(figsize=(20, 6))

    # =========================================================================
    # Subplot 1: 3D Comparison View
    # =========================================================================
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")

    # 1. Ground Truth
    ax3d.plot(
        pts_gt[:, 0], pts_gt[:, 1], pts_gt[:, 2],
        color="#d62728", linewidth=2.5, label="Ground Truth (CAD)", alpha=0.85
    )

    # 2. Rigid Sensed Shape
    ax3d.plot(
        pts_rigid[:, 0], pts_rigid[:, 1], pts_rigid[:, 2],
        color="#2ca02c", linewidth=2.2, linestyle="-",
        label=f"Rigid Rig ({mean_r:.2f} ± {std_r:.2f} mm)", alpha=0.95
    )

    # 3. Gelatin Sensed Shape
    ax3d.plot(
        pts_gel[:, 0], pts_gel[:, 1], pts_gel[:, 2],
        color="#1f77b4", linewidth=2.2, linestyle="--",
        label=f"Gelatin Phantom ({mean_g:.2f} ± {std_g:.2f} mm)", alpha=0.95
    )

    # Mark Rigid Sensors (Green markers)
    ax3d.scatter(pts_rigid[r_s1, 0], pts_rigid[r_s1, 1], pts_rigid[r_s1, 2],
                 color="#2ca02c", edgecolor="black", marker="o", s=55, zorder=6)
    ax3d.scatter(pts_rigid[r_s2, 0], pts_rigid[r_s2, 1], pts_rigid[r_s2, 2],
                 color="#2ca02c", edgecolor="black", marker="s", s=55, zorder=6)
    ax3d.scatter(pts_rigid[r_s3, 0], pts_rigid[r_s3, 1], pts_rigid[r_s3, 2],
                 color="#2ca02c", edgecolor="black", marker="^", s=55, zorder=6)

    # Mark Gelatin Sensors (Blue markers)
    ax3d.scatter(pts_gel[g_s1, 0], pts_gel[g_s1, 1], pts_gel[g_s1, 2],
                 color="#1f77b4", edgecolor="navy", marker="o", s=55, zorder=6)
    ax3d.scatter(pts_gel[g_s2, 0], pts_gel[g_s2, 1], pts_gel[g_s2, 2],
                 color="#1f77b4", edgecolor="navy", marker="s", s=55, zorder=6)
    ax3d.scatter(pts_gel[g_s3, 0], pts_gel[g_s3, 1], pts_gel[g_s3, 2],
                 color="#1f77b4", edgecolor="navy", marker="^", s=55, zorder=6)

    # Annotations
    ax3d.text(pts_rigid[r_s1, 0]+2, pts_rigid[r_s1, 1]+2, pts_rigid[r_s1, 2]+2, "S1", color="black", fontsize=9, fontweight="bold")
    ax3d.text(pts_rigid[r_s2, 0]+2, pts_rigid[r_s2, 1]+2, pts_rigid[r_s2, 2]+2, "S2", color="black", fontsize=9, fontweight="bold")
    ax3d.text(pts_rigid[r_s3, 0]+2, pts_rigid[r_s3, 1]+2, pts_rigid[r_s3, 2]+2, "S3", color="black", fontsize=9, fontweight="bold")

    ax3d.set_xlabel("X (mm)", fontsize=10, labelpad=5)
    ax3d.set_ylabel("Y (mm)", fontsize=10, labelpad=5)
    ax3d.set_zlabel("Z (mm)", fontsize=10, labelpad=5)
    ax3d.set_title(f"3D Comparison: Shape {shape_num}", fontsize=12, fontweight="bold")
    ax3d.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax3d.grid(True, linestyle="--", alpha=0.4)

    # Equalize 3D axes
    all_pts = np.vstack([pts_gt, pts_rigid, pts_gel])
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
    ax3d.view_init(elev=26, azim=45)

    # =========================================================================
    # Subplot 2: Top-Down (X-Y) Projection
    # =========================================================================
    ax2d = fig.add_subplot(1, 3, 2)
    ax2d.plot(pts_gt[:, 0], pts_gt[:, 1], color="#d62728", linewidth=2.5, label="Ground Truth (CAD)", alpha=0.85)
    ax2d.plot(pts_rigid[:, 0], pts_rigid[:, 1], color="#2ca02c", linewidth=2.2, linestyle="-", label="Rigid Rig", alpha=0.95)
    ax2d.plot(pts_gel[:, 0], pts_gel[:, 1], color="#1f77b4", linewidth=2.2, linestyle="--", label="Gelatin Phantom", alpha=0.95)

    ax2d.scatter(pts_rigid[r_s1, 0], pts_rigid[r_s1, 1], color="#2ca02c", edgecolor="black", marker="o", s=50, zorder=6)
    ax2d.scatter(pts_rigid[r_s2, 0], pts_rigid[r_s2, 1], color="#2ca02c", edgecolor="black", marker="s", s=50, zorder=6)
    ax2d.scatter(pts_rigid[r_s3, 0], pts_rigid[r_s3, 1], color="#2ca02c", edgecolor="black", marker="^", s=50, zorder=6)

    ax2d.scatter(pts_gel[g_s1, 0], pts_gel[g_s1, 1], color="#1f77b4", edgecolor="navy", marker="o", s=50, zorder=6)
    ax2d.scatter(pts_gel[g_s2, 0], pts_gel[g_s2, 1], color="#1f77b4", edgecolor="navy", marker="s", s=50, zorder=6)
    ax2d.scatter(pts_gel[g_s3, 0], pts_gel[g_s3, 1], color="#1f77b4", edgecolor="navy", marker="^", s=50, zorder=6)

    ax2d.text(pts_rigid[r_s1, 0]+2, pts_rigid[r_s1, 1]+2, "S1", color="black", fontsize=9, fontweight="bold")
    ax2d.text(pts_rigid[r_s2, 0]+2, pts_rigid[r_s2, 1]+2, "S2", color="black", fontsize=9, fontweight="bold")
    ax2d.text(pts_rigid[r_s3, 0]+2, pts_rigid[r_s3, 1]+2, "S3", color="black", fontsize=9, fontweight="bold")

    ax2d.set_xlabel("X (mm)", fontsize=10)
    ax2d.set_ylabel("Y (mm)", fontsize=10)
    ax2d.set_title(f"Top-Down (X-Y) Projection", fontsize=12, fontweight="bold")
    ax2d.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.grid(True, linestyle="--", alpha=0.4)

    # =========================================================================
    # Subplot 3: Arc-Length Error Profile along Guidewire
    # =========================================================================
    ax_err = fig.add_subplot(1, 3, 3)
    s_eval = np.linspace(0.0, cum_s_r[-1], len(err_rigid))

    ax_err.plot(s_eval, err_rigid, color="#2ca02c", linewidth=2.0, label=f"Rigid Error (Mean: {mean_r:.2f} mm)")
    ax_err.plot(s_eval, err_gel, color="#1f77b4", linewidth=2.0, linestyle="--", label=f"Gelatin Error (Mean: {mean_g:.2f} mm)")

    # Mean lines
    ax_err.axhline(mean_r, color="#2ca02c", linestyle=":", alpha=0.7)
    ax_err.axhline(mean_g, color="#1f77b4", linestyle=":", alpha=0.7)

    # Sensor positions
    ax_err.axvline(0, color="gray", linestyle="-.", alpha=0.6)
    ax_err.axvline(59.86, color="gray", linestyle="-.", alpha=0.6)
    ax_err.axvline(cum_s_r[-1], color="gray", linestyle="-.", alpha=0.6)

    ax_err.text(2, max(err_rigid.max(), err_gel.max()) * 0.9, "S1 (Tip)", fontsize=8.5, color="gray", fontweight="bold")
    ax_err.text(62, max(err_rigid.max(), err_gel.max()) * 0.9, "S2 (60 mm)", fontsize=8.5, color="gray", fontweight="bold")
    ax_err.text(cum_s_r[-1] - 35, max(err_rigid.max(), err_gel.max()) * 0.9, "S3 (Base)", fontsize=8.5, color="gray", fontweight="bold")

    ax_err.set_xlabel("Arc Length along Guidewire s (mm)", fontsize=10)
    ax_err.set_ylabel("Reconstruction Error (mm)", fontsize=10)
    ax_err.set_title(f"Error Distribution along Guidewire", fontsize=12, fontweight="bold")
    ax_err.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax_err.grid(True, linestyle="--", alpha=0.4)
    ax_err.set_ylim(bottom=0.0)

    # Save output
    plt.suptitle(f"Shape {shape_num}: Rigid Rig vs Gelatin Phantom Comparison", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file1 = os.path.join(data_dir, f"s{shape_num}_rigid_vs_gelatin.png")
    plt.savefig(out_file1, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_file1}")

    # Also save inside the shape's folder
    out_file2 = os.path.join(rigid_dir, f"s{shape_num}_rigid_vs_gelatin.png")
    plt.savefig(out_file2, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_file2}")

    plt.close(fig)
    return out_file1


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")

    print("\n" + "=" * 65)
    print("Generating Individual Rigid vs Gelatin Comparison Figures:")
    print("=" * 65)

    generated = []
    for i in range(1, 6):
        out = plot_shape_comparison(i, data_dir)
        if out:
            generated.append(out)

    print("\n" + "=" * 65)
    print(f"Successfully generated all {len(generated)} comparison figures.")
    print("=" * 65)


if __name__ == "__main__":
    main()

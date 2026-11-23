#!/usr/bin/env -S python3 -s
"""
icp.py

Implements the Iterative Closest Point (ICP) algorithm (SENSORS_2022 reference,
Rusinkiewicz & Levoy 2001) to align the reconstructed guidewire shape with the
CAD centerline ground truth of the helical rig.

Inputs:
  - helical_rig_sensed_shape.csv: 500 reconstructed shape points
  - helix_ground_truth_100mm.csv: ground truth centerline points of the helical rig

Outputs:
  - helical_rig_sensed_shape_error.csv: point-by-point Euclidean errors, mean error, and standard deviation
  - helical_rig_sensed_shape_icp.csv: transformed sensed shape points after ICP alignment
  - helix_ground_truth_100mm_icp.csv: ground truth points in the common aligned coordinate frame
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


def find_nearest_neighbors(source_pts, target_pts):
    """
    Finds nearest target neighbor for each source point.
    Uses SciPy's cKDTree if available, otherwise vectorized NumPy.
    """
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(target_pts)
        distances, indices = tree.query(source_pts)
        return distances, indices
    except Exception:
        # Vectorized NumPy fallback
        dists = []
        indices = []
        for chunk in np.array_split(source_pts, 5):
            p2 = np.sum(chunk ** 2, axis=1, keepdims=True)
            q2 = np.sum(target_pts ** 2, axis=1, keepdims=True).T
            d2 = np.maximum(p2 + q2 - 2.0 * (chunk @ target_pts.T), 0.0)
            d = np.sqrt(d2)
            indices.append(np.argmin(d, axis=1))
            dists.append(np.min(d, axis=1))
        return np.concatenate(dists), np.concatenate(indices)


def compute_optimal_rigid_transform(src_pts, dst_pts):
    """
    Computes optimal rotation matrix R and translation vector t
    such that (R @ src.T).T + t minimizes sum of squared Euclidean distances to dst.
    Uses Kabsch / SVD algorithm.
    """
    c_src = np.mean(src_pts, axis=0)
    c_dst = np.mean(dst_pts, axis=0)

    src_centered = src_pts - c_src
    dst_centered = dst_pts - c_dst

    # Covariance matrix H
    H = src_centered.T @ dst_centered

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection check: ensure det(R) == +1
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = c_dst - R @ c_src
    return R, t


def run_icp_iteration(source_pts, target_pts, init_R=None, init_t=None, max_iterations=200, tol=1e-6):
    """
    Runs iterative closest point optimization from a given initial pose.
    """
    if init_R is None:
        init_R = np.eye(3)
    if init_t is None:
        init_t = np.mean(target_pts, axis=0) - init_R @ np.mean(source_pts, axis=0)

    curr_pts = (init_R @ source_pts.T).T + init_t
    total_R = init_R.copy()
    total_t = init_t.copy()

    prev_mean_err = float("inf")

    for it in range(max_iterations):
        dists, indices = find_nearest_neighbors(curr_pts, target_pts)
        closest_pts = target_pts[indices]

        # Step transformation
        R_step, t_step = compute_optimal_rigid_transform(curr_pts, closest_pts)

        curr_pts = (R_step @ curr_pts.T).T + t_step
        total_R = R_step @ total_R
        total_t = R_step @ total_t + t_step

        mean_err = np.mean(dists)
        if abs(prev_mean_err - mean_err) < tol:
            break
        prev_mean_err = mean_err

    final_dists, final_indices = find_nearest_neighbors(curr_pts, target_pts)
    return curr_pts, final_dists, total_R, total_t, it + 1


def run_icp(source_pts, target_pts):
    """
    Robust ICP with PCA-based multi-start initialization to guarantee global convergence.
    """
    c_src = np.mean(source_pts, axis=0)
    c_dst = np.mean(target_pts, axis=0)

    _, _, Vt_src = np.linalg.svd(source_pts - c_src)
    _, _, Vt_dst = np.linalg.svd(target_pts - c_dst)

    best_mean_err = float("inf")
    best_result = None

    # Test all 4 proper principal axis orientation alignments
    for s0 in [1, -1]:
        for s1 in [1, -1]:
            s2 = s0 * s1
            S = np.diag([s0, s1, s2])
            R_init = Vt_dst.T @ S @ Vt_src
            t_init = c_dst - R_init @ c_src

            aligned_pts, dists, R_tot, t_tot, iters = run_icp_iteration(
                source_pts, target_pts, init_R=R_init, init_t=t_init
            )
            mean_err = np.mean(dists)
            if mean_err < best_mean_err:
                best_mean_err = mean_err
                best_result = (aligned_pts, dists, R_tot, t_tot, iters)

    return best_result


def load_point_csv(filepath):
    """
    Loads 3D points from a CSV file. Supports headers like (x_mm, y_mm, z_mm)
    or (Tx, Ty, Tz) or (x, y, z).
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


def main():
    parser = argparse.ArgumentParser(
        description="Iterative Closest Point (ICP) alignment and error calculation (SENSORS_2022)."
    )
    parser.add_argument(
        "--sensed",
        default=None,
        help="Path to helical_rig_sensed_shape.csv."
    )
    parser.add_argument(
        "--gt",
        default=None,
        help="Path to helix_ground_truth_100mm.csv."
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to save output files (defaults to data/)."
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")

    # Locate sensed shape file
    sensed_file = args.sensed
    if sensed_file is None:
        candidates = [
            os.path.join(data_dir, "helical_rig_sensed_shape.csv"),
            os.path.join(script_dir, "helical_rig_sensed_shape.csv"),
            "helical_rig_sensed_shape.csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                sensed_file = c
                break

    # Locate ground truth file
    gt_file = args.gt
    if gt_file is None:
        candidates = [
            os.path.join(data_dir, "helix_ground_truth_100mm.csv"),
            os.path.join(script_dir, "helix_ground_truth_100mm.csv"),
            "helix_ground_truth_100mm.csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                gt_file = c
                break

    if sensed_file is None or not os.path.exists(sensed_file):
        print("Error: helical_rig_sensed_shape.csv not found. Please run shape_sensing.py first.")
        sys.exit(1)

    if gt_file is None or not os.path.exists(gt_file):
        print("Error: helix_ground_truth_100mm.csv not found.")
        sys.exit(1)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(sensed_file))
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nRunning ICP Alignment (SENSORS_2022 Reference):")
    print(f"  Sensed shape:  {sensed_file}")
    print(f"  Ground truth:  {gt_file}")

    sensed_pts = load_point_csv(sensed_file)
    gt_pts = load_point_csv(gt_file)

    print(f"  Loaded {len(sensed_pts)} sensed points and {len(gt_pts)} ground truth points.")

    aligned_sensed_pts, point_errors, R_opt, t_opt, iterations = run_icp(sensed_pts, gt_pts)

    mean_error = float(np.mean(point_errors))
    std_error = float(np.std(point_errors))
    min_error = float(np.min(point_errors))
    max_error = float(np.max(point_errors))

    print("=" * 65)
    print(f"ICP Alignment Results (converged in {iterations} iterations):")
    print(f"  Mean Error:               {mean_error:.4f} mm")
    print(f"  Std Dev:                  {std_error:.4f} mm")
    print(f"  Format (Mean ± Std):      {mean_error:.2f} ± {std_error:.2f} mm")
    print(f"  Min Error:                {min_error:.4f} mm")
    print(f"  Max Error:                {max_error:.4f} mm")
    print("=" * 65)

    # 1. Output error CSV with point error, mean error, and std dev
    error_csv_path = os.path.join(out_dir, "helical_rig_sensed_shape_error.csv")
    with open(error_csv_path, "w", newline="") as f:
        # Write informative header comment with mean error and std dev
        f.write(f"# ICP Error Analysis - SENSORS_2022 Reference\n")
        f.write(f"# Mean Error: {mean_error:.6f} mm, Std Dev: {std_error:.6f} mm, Min: {min_error:.6f} mm, Max: {max_error:.6f} mm\n")
        f.write(f"# Summary: {mean_error:.2f} +/- {std_error:.2f} mm\n")
        writer = csv.writer(f)
        writer.writerow(["point_index", "error_mm", "mean_error_mm", "std_error_mm"])
        for idx, err in enumerate(point_errors):
            writer.writerow([idx, f"{err:.6f}", f"{mean_error:.6f}", f"{std_error:.6f}"])
    print(f"Saved: {error_csv_path}")

    # 2. Output transformed sensed shape coordinates
    sensed_icp_path = os.path.join(out_dir, "helical_rig_sensed_shape_icp.csv")
    with open(sensed_icp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["point_index", "x_mm", "y_mm", "z_mm"])
        for idx, pt in enumerate(aligned_sensed_pts):
            writer.writerow([idx, f"{pt[0]:.6f}", f"{pt[1]:.6f}", f"{pt[2]:.6f}"])
    print(f"Saved: {sensed_icp_path}")

    # 3. Output ground truth coordinates in the same frame
    gt_icp_path = os.path.join(out_dir, "helix_ground_truth_100mm_icp.csv")
    with open(gt_icp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["point_index", "x_mm", "y_mm", "z_mm"])
        for idx, pt in enumerate(gt_pts):
            writer.writerow([idx, f"{pt[0]:.6f}", f"{pt[1]:.6f}", f"{pt[2]:.6f}"])
    print(f"Saved: {gt_icp_path}\n")


if __name__ == "__main__":
    main()

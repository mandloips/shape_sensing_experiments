#!/usr/bin/env -S python3 -s
"""
shape_sensing.py

Reconstructs the 3D guidewire shape using piecewise cubic Bezier curves fitted
between consecutive EMT sensors, taking sensor spacings and sensor poses as inputs.
Reference: SENSORS_2022 paper and curve_fitting implementation.

Inputs:
  - sensor_spacing.csv: physical straight distance between consecutive sensors
  - helical_rig_test_median.csv: median position and Markley average quaternion for each sensor

Output:
  - helical_rig_sensed_shape.csv: 500 points sampled along the reconstructed 3D curve
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


def rotate_vector_by_quaternion(vector, q):
    """
    Rotate a 3D vector by a quaternion q = [Q0, Qx, Qy, Qz] (scalar first).
    Formula: v' = v + 2*w*(q_vec x v) + 2*(q_vec x (q_vec x v))
    """
    qw, qx, qy, qz = [float(v) for v in q]
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-12:
        return np.asarray(vector, dtype=float)
    qw /= norm
    qx /= norm
    qy /= norm
    qz /= norm

    v = np.asarray(vector, dtype=float)
    q_vec = np.array([qx, qy, qz], dtype=float)
    uv = np.cross(q_vec, v)
    uuv = np.cross(q_vec, uv)
    return v + 2.0 * (qw * uv + uuv)


def get_bezier_point(t, p0, p1, p2, p3):
    """
    Computes a point on a cubic Bezier curve at parameter t in [0, 1].
    b(t) = (1-t)^3*p0 + 3*(1-t)^2*t*p1 + 3*(1-t)*t^2*p2 + t^3*p3
    """
    return (1.0 - t) ** 3 * p0 + 3.0 * (1.0 - t) ** 2 * t * p1 + 3.0 * (1.0 - t) * (t ** 2) * p2 + (t ** 3) * p3


def bezier_arc_length(p0, p1, p2, p3, steps=100):
    """
    Estimates the arc length of the cubic Bezier curve by summing Euclidean
    distances over discrete parameter steps.
    """
    t_vals = np.linspace(0.0, 1.0, steps)
    points = np.array([get_bezier_point(t, p0, p1, p2, p3) for t in t_vals])
    dist_segments = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    return np.sum(dist_segments)


def objective(params, p0, p3, t_i, t_next, l_straight):
    """
    Optimization cost function: minimizes squared difference between
    the estimated Bezier curve length and the physical straight wire length.
    Equation (5) in SENSORS_2022.
    """
    l1, l2 = params
    p1 = p0 + l1 * t_i
    p2 = p3 - l2 * t_next
    l_estimate = bezier_arc_length(p0, p1, p2, p3, steps=100)
    return (l_straight - l_estimate) ** 2


def optimize_segment(p0, p3, t_i, t_next, l_straight, x0=None):
    """
    Solves for optimal control point distance parameters (l1, l2).
    Uses SciPy's L-BFGS-B with bounds [0, l_straight], with fallback
    to bounded coordinate search if SciPy is not present.
    """
    if x0 is None:
        x0 = [l_straight / 3.0, l_straight / 3.0]

    try:
        from scipy.optimize import minimize
        res = minimize(
            objective,
            x0=x0,
            args=(p0, p3, t_i, t_next, l_straight),
            bounds=[(0.0, l_straight), (0.0, l_straight)],
            method="L-BFGS-B",
            tol=1e-5,
            options={"maxiter": 100, "disp": False}
        )
        return res.x, res.fun, res.success
    except Exception:
        # SciPy-free coordinate descent fallback
        x = np.clip(np.array(x0, dtype=float), 0.0, l_straight)
        best_val = objective(x, p0, p3, t_i, t_next, l_straight)
        step = max(l_straight / 4.0, 1e-4)
        for _ in range(50):
            improved = False
            for axis in (0, 1):
                for direction in (-1.0, 1.0):
                    cand = x.copy()
                    cand[axis] = np.clip(cand[axis] + direction * step, 0.0, l_straight)
                    val = objective(cand, p0, p3, t_i, t_next, l_straight)
                    if val < best_val:
                        best_val = val
                        x = cand
                        improved = True
            if not improved:
                step *= 0.5
            if step < 1e-5:
                break
        return x, best_val, True


def reconstruct_guidewire_shape(sensor_spacing_path, helical_median_path, total_points=500, output_path=None):
    if not os.path.isfile(sensor_spacing_path):
        raise FileNotFoundError(f"Sensor spacing file not found: {sensor_spacing_path}")
    if not os.path.isfile(helical_median_path):
        raise FileNotFoundError(f"Helical median file not found: {helical_median_path}")

    if output_path is None:
        out_dir = os.path.dirname(os.path.abspath(helical_median_path))
        output_path = os.path.join(out_dir, "helical_rig_sensed_shape.csv")

    # 1. Read sensor spacing
    spacings = []
    with open(sensor_spacing_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            val = float(r.get("spacing_mm", r.get("distance_mm", r.get("spacing", 0.0))))
            spacings.append(val)

    # 2. Read helical rig median sensor poses
    sensors = []
    with open(helical_median_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sid = int(r.get("sensor_id", len(sensors) + 1))
            port = r.get("port", f"Sensor {sid}")
            tx = float(r.get("Tx", r.get("x", 0.0)))
            ty = float(r.get("Ty", r.get("y", 0.0)))
            tz = float(r.get("Tz", r.get("z", 0.0)))
            q0 = float(r.get("Q0", r.get("qw", 1.0)))
            qx = float(r.get("Qx", r.get("qx", 0.0)))
            qy = float(r.get("Qy", r.get("qy", 0.0)))
            qz = float(r.get("Qz", r.get("qz", 0.0)))
            sensors.append({
                "sensor_id": sid,
                "port": port,
                "pos": np.array([tx, ty, tz]),
                "quat": np.array([q0, qx, qy, qz])
            })

    sensors.sort(key=lambda s: s["sensor_id"])
    m = len(sensors)
    if m < 2:
        raise ValueError(f"Need at least 2 sensors for shape reconstruction, found {m}")

    if len(spacings) < m - 1:
        raise ValueError(f"Need at least {m-1} spacing values for {m} sensors, found {len(spacings)}")

    # 3. Compute tangents
    # Sensor long axis is Z [0, 0, 1]
    sensor_long_axis = np.array([0.0, 0.0, 1.0])
    positions = [s["pos"] for s in sensors]
    tangents = [rotate_vector_by_quaternion(sensor_long_axis, s["quat"]) for s in sensors]

    print(f"\nReconstructing shape from {m} sensors:")
    print("=" * 65)
    for i, s in enumerate(sensors):
        p = positions[i]
        t = tangents[i]
        print(f"Sensor {s['sensor_id']} ({s['port']}): Pos=[{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}] | Tangent=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")

    # 4. Reconstruct Bezier segments
    num_segments = m - 1
    dense_segments = []

    for i in range(num_segments):
        p0 = positions[i]
        p3 = positions[i + 1]
        t_i = tangents[i]
        t_next = tangents[i + 1]
        l_straight = spacings[i]

        (l1_opt, l2_opt), cost, success = optimize_segment(p0, p3, t_i, t_next, l_straight)
        p1 = p0 + l1_opt * t_i
        p2 = p3 - l2_opt * t_next

        est_len = bezier_arc_length(p0, p1, p2, p3, steps=200)
        print(f"\nSegment {i+1} (Sensor {sensors[i]['sensor_id']} -> Sensor {sensors[i+1]['sensor_id']}):")
        print(f"  Target length (l_straight): {l_straight:.4f} mm")
        print(f"  Estimated curve length:    {est_len:.4f} mm  (diff = {abs(l_straight - est_len):.4e} mm)")
        print(f"  Optimal parameters:        l1 = {l1_opt:.4f} mm, l2 = {l2_opt:.4f} mm")
        print(f"  Control points:")
        print(f"    p0: [{p0[0]:.2f}, {p0[1]:.2f}, {p0[2]:.2f}]")
        print(f"    p1: [{p1[0]:.2f}, {p1[1]:.2f}, {p1[2]:.2f}]")
        print(f"    p2: [{p2[0]:.2f}, {p2[1]:.2f}, {p2[2]:.2f}]")
        print(f"    p3: [{p3[0]:.2f}, {p3[1]:.2f}, {p3[2]:.2f}]")

        # Dense sampling for high-precision arc-length parameterization
        # Sample dense points proportional to segment length (minimum 500)
        steps_segment = max(500, int(l_straight * 10))
        if i == 0:
            t_vals = np.linspace(0.0, 1.0, steps_segment)
        else:
            t_vals = np.linspace(0.0, 1.0, steps_segment)[1:]

        seg_pts = np.array([get_bezier_point(t, p0, p1, p2, p3) for t in t_vals])
        dense_segments.append(seg_pts)

    # 5. Resample EXACTLY total_points at EQUIDISTANT arc-length intervals
    dense_curve = np.vstack(dense_segments)
    diffs = np.linalg.norm(np.diff(dense_curve, axis=0), axis=1)
    cum_s = np.insert(np.cumsum(diffs), 0, 0.0)
    total_length = cum_s[-1]

    # Equidistant arc-length coordinates from 0 to total_length
    s_equidistant = np.linspace(0.0, total_length, total_points)
    reconstructed_points = np.array([
        np.interp(s_equidistant, cum_s, dense_curve[:, d]) for d in range(3)
    ]).T

    step_sizes = np.linalg.norm(np.diff(reconstructed_points, axis=0), axis=1)
    print("=" * 65)
    print(f"Reconstructed curve generated with {len(reconstructed_points)} EQUIDISTANT points.")
    print(f"Total reconstructed curve length: {total_length:.4f} mm")
    print(f"Equidistant step size between consecutive points: {step_sizes.mean():.4f} +/- {step_sizes.std():.1e} mm")

    # 6. Output CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["point_index", "x_mm", "y_mm", "z_mm"])
        for idx, pt in enumerate(reconstructed_points):
            writer.writerow([idx, f"{pt[0]:.6f}", f"{pt[1]:.6f}", f"{pt[2]:.6f}"])

    print("=" * 65)
    print(f"Reconstructed curve generated with {len(reconstructed_points)} points.")
    total_reconstructed_len = np.sum(np.linalg.norm(np.diff(reconstructed_points, axis=0), axis=1))
    print(f"Total reconstructed curve length: {total_reconstructed_len:.4f} mm")
    print(f"Saved: {output_path}\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct 3D guidewire shape from sensor spacing and helical rig test median poses."
    )
    parser.add_argument(
        "--spacing",
        default=None,
        help="Path to sensor_spacing.csv (defaults to data/sensor_spacing.csv)."
    )
    parser.add_argument(
        "--median",
        default=None,
        help="Path to helical_rig_test_median.csv (defaults to data/helical_rig_test_median.csv)."
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to output helical_rig_sensed_shape.csv."
    )
    parser.add_argument(
        "--points",
        "-n",
        type=int,
        default=500,
        help="Number of points on reconstructed curve (default: 500)."
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")

    # Locate spacing file
    spacing_file = args.spacing
    if spacing_file is None:
        candidates = [
            os.path.join(data_dir, "sensor_spacing.csv"),
            os.path.join(script_dir, "sensor_spacing.csv"),
            "sensor_spacing.csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                spacing_file = c
                break

    # Locate median file
    median_file = args.median
    if median_file is None:
        candidates = [
            os.path.join(data_dir, "helical_rig_test_median.csv"),
            os.path.join(script_dir, "helical_rig_test_median.csv"),
            "helical_rig_test_median.csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                median_file = c
                break

    if spacing_file is None or not os.path.exists(spacing_file):
        print("Error: sensor_spacing.csv not found. Please run sensor_spacing_calculator.py first.")
        sys.exit(1)

    if median_file is None or not os.path.exists(median_file):
        print("Error: helical_rig_test_median.csv not found. Please run median_calculator.py first.")
        sys.exit(1)

    reconstruct_guidewire_shape(
        sensor_spacing_path=spacing_file,
        helical_median_path=median_file,
        total_points=args.points,
        output_path=args.output
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env -S python3 -s
"""
median_calculator.py

Calculates the median position (Tx, Ty, Tz) and Markley average quaternion
(Q0, Qx, Qy, Qz) for each sensor from experimental NDI Aurora tracking data.

Output:
  Generates a CSV file with '_median' appended to the input filename, e.g.:
    helical_rig_test.csv -> helical_rig_test_median.csv
    straight_rig_test.csv -> straight_rig_test_median.csv
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


def markley_average_quaternion(quaternions):
    """
    Computes the Markley average quaternion from an (N, 4) array of quaternions.
    Reference:
      F. Landis Markley, 'Averaging Quaternions',
      Journal of Guidance, Control, and Dynamics, Vol. 30, No. 4, 2007.

    Parameters:
      quaternions: (N, 4) array-like of [Q0, Qx, Qy, Qz] (scalar-first format).

    Returns:
      (4,) numpy array of [Q0, Qx, Qy, Qz] normalized average quaternion.
    """
    quats = np.asarray(quaternions, dtype=float)
    if len(quats) == 0:
        raise ValueError("Cannot compute average of empty quaternion list.")
    if len(quats) == 1:
        q = quats[0] / np.linalg.norm(quats[0])
        return q if q[0] >= 0 else -q

    # Normalize input quaternions
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / norms

    # Form the 4x4 matrix M = (1/N) * sum(q_i * q_i^T)
    M = (quats.T @ quats) / len(quats)

    # Eigen-decomposition (M is real symmetric positive semi-definite)
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    # The optimal quaternion is the eigenvector corresponding to the largest eigenvalue
    avg_quat = eigenvectors[:, np.argmax(eigenvalues)]
    avg_quat = avg_quat / np.linalg.norm(avg_quat)

    # By convention, ensure scalar component Q0 is positive
    if avg_quat[0] < 0:
        avg_quat = -avg_quat

    return avg_quat


def process_file(input_path, output_path=None):
    """
    Reads an experimental CSV file, extracts sensor blocks, computes the median
    position and Markley average quaternion for each sensor, and writes the output.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_median{ext}"

    print(f"\nProcessing: {input_path}")
    with open(input_path, "r", newline="") as f:
        reader = list(csv.reader(f))

    if not reader:
        raise ValueError(f"File {input_path} is empty.")

    header = reader[0]
    data_rows = reader[1:]

    # Locate each sensor block in NDI Aurora format
    # Sensor blocks start with headers containing 'Port'
    port_indices = [i for i, col_name in enumerate(header) if "Port" in col_name]

    if not port_indices:
        raise ValueError(f"No sensor ports found in {input_path}")

    sensors = []
    for c in port_indices:
        port_full = header[c].strip()
        port_name = port_full.split(":")[0].strip()

        # NDI CSV columns relative to port column:
        # c+0: Port name
        # c+4: State (e.g. OK)
        # c+5: Q0
        # c+6: Qx
        # c+7: Qy
        # c+8: Qz
        # c+9: Tx
        # c+10: Ty
        # c+11: Tz
        valid_rows = [
            row for row in data_rows
            if len(row) > c + 11 and (len(row) <= c + 4 or row[c + 4].strip().upper() == "OK")
        ]
        if not valid_rows:
            valid_rows = [row for row in data_rows if len(row) > c + 11]

        quats = []
        positions = []
        for row in valid_rows:
            try:
                q0 = float(row[c + 5])
                qx = float(row[c + 6])
                qy = float(row[c + 7])
                qz = float(row[c + 8])
                tx = float(row[c + 9])
                ty = float(row[c + 10])
                tz = float(row[c + 11])
                quats.append([q0, qx, qy, qz])
                positions.append([tx, ty, tz])
            except (ValueError, IndexError):
                continue

        positions = np.array(positions)
        quats = np.array(quats)

        median_pos = np.median(positions, axis=0)
        markley_quat = markley_average_quaternion(quats)

        sensors.append({
            "port": port_name,
            "port_full": port_full,
            "pos": median_pos,
            "quat": markley_quat,
            "samples": len(positions)
        })

    # Sort sensors in physical sequential order:
    # Port 0x0A (Sensor 1, tip), Port 0x0B (Sensor 2, middle), Port 0x0C (Sensor 3, base)
    sensors.sort(key=lambda s: s["port"])
    for idx, s in enumerate(sensors):
        s["sensor_id"] = idx + 1

    # Write output CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sensor_id", "port", "Tx", "Ty", "Tz", "Q0", "Qx", "Qy", "Qz"])
        for s in sensors:
            writer.writerow([
                s["sensor_id"],
                s["port"],
                f"{s['pos'][0]:.6f}",
                f"{s['pos'][1]:.6f}",
                f"{s['pos'][2]:.6f}",
                f"{s['quat'][0]:.8f}",
                f"{s['quat'][1]:.8f}",
                f"{s['quat'][2]:.8f}",
                f"{s['quat'][3]:.8f}"
            ])

    print(f"Saved: {output_path}")
    print(f"{'Sensor':<8} {'Port':<12} {'Position (Tx, Ty, Tz) [mm]':<32} {'Markley Quat (Q0, Qx, Qy, Qz)':<40}")
    print("-" * 95)
    for s in sensors:
        pos_str = f"[{s['pos'][0]:.2f}, {s['pos'][1]:.2f}, {s['pos'][2]:.2f}]"
        quat_str = f"[{s['quat'][0]:.4f}, {s['quat'][1]:.4f}, {s['quat'][2]:.4f}, {s['quat'][3]:.4f}]"
        print(f"Sensor {s['sensor_id']:<2} {s['port']:<12} {pos_str:<32} {quat_str:<40}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute median position and Markley average quaternion for each sensor."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Input CSV files (e.g. helical_rig_test.csv, straight_rig_test.csv)."
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")

    if args.inputs:
        files = args.inputs
    else:
        # Default files to process if no inputs specified
        default_files = [
            os.path.join(data_dir, "straight_rig_test.csv"),
            os.path.join(data_dir, "helical_rig_test.csv")
        ]
        files = [f for f in default_files if os.path.exists(f)]
        if not files:
            print("No input files provided and default files not found.")
            sys.exit(1)

    for f in files:
        if not os.path.isabs(f) and not os.path.exists(f):
            candidate = os.path.join(data_dir, f)
            if os.path.exists(candidate):
                f = candidate
        process_file(f)


if __name__ == "__main__":
    main()

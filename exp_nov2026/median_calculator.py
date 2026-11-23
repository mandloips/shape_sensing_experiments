#!/usr/bin/env -S python3 -s
"""
median_calculator.py (exp_nov2026)

Calculates the median position (Tx, Ty, Tz) and Markley average quaternion
(Q0, Qx, Qy, Qz) for each sensor from experimental NDI Aurora tracking data.

Automatically identifies the physical order of sensors along the guidewire:
  - Finds the ~60 mm pair (Sensor 1 Tip and Sensor 2 Middle).
  - Identifies the Base sensor (Sensor 3).
  - Distinguishes Tip (S1) from Middle (S2) by distance to Base:
      Sensor 2 (Middle) is closer to Base.
      Sensor 1 (Tip) is further from Base.

Output:
  Generates a CSV file with '_median' appended to the input filename.
"""

import sys
import os

# Ensure clean environment without user site conflicts
if os.environ.get("PYTHONNOUSERSITE") != "1" and not sys.flags.no_user_site:
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, ["-s"] + sys.argv)

import csv
import glob
import argparse
import numpy as np


def markley_average_quaternion(quaternions):
    quats = np.asarray(quaternions, dtype=float)
    if len(quats) == 0:
        raise ValueError("Cannot compute average of empty quaternion list.")
    if len(quats) == 1:
        q = quats[0] / np.linalg.norm(quats[0])
        return q if q[0] >= 0 else -q

    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / norms

    M = (quats.T @ quats) / len(quats)
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    avg_quat = eigenvectors[:, np.argmax(eigenvalues)]
    avg_quat = avg_quat / np.linalg.norm(avg_quat)

    if avg_quat[0] < 0:
        avg_quat = -avg_quat

    return avg_quat


def order_sensors_along_guidewire(sensor_dict):
    """
    Robust geometric identification of sensors from Tip to Base:
    1. The two sensors with the shortest distance (~60 mm) are Tip and Middle.
    2. The third sensor is the Base.
    3. The Middle sensor is closer to the Base (~195-242 mm).
    4. The Tip sensor is further from the Base (~223-302 mm).
    """
    pnames = list(sensor_dict.keys())
    if len(pnames) != 3:
        # Fallback to alphanumeric sort if not exactly 3 sensors
        return [
            {"sensor_id": idx + 1, "port": p, **sensor_dict[p]}
            for idx, p in enumerate(sorted(pnames))
        ]

    # Calculate all 3 pairwise distances
    dists = {}
    for i in range(len(pnames)):
        for j in range(i + 1, len(pnames)):
            pA, pB = pnames[i], pnames[j]
            d = np.linalg.norm(sensor_dict[pA]["pos"] - sensor_dict[pB]["pos"])
            dists[(pA, pB)] = d
            dists[(pB, pA)] = d

    # Find the shortest pair: this is (Tip, Middle)
    pair = min([(p1, p2) for p1, p2 in dists.keys() if p1 < p2], key=lambda k: dists[k])
    p_tip_mid = list(pair)
    base_port = [p for p in pnames if p not in p_tip_mid][0]

    # Sensor 2 (Middle) is closer to Base; Sensor 1 (Tip) is further from Base
    d0_base = dists[(p_tip_mid[0], base_port)]
    d1_base = dists[(p_tip_mid[1], base_port)]

    if d0_base > d1_base:
        tip_port = p_tip_mid[0]
        mid_port = p_tip_mid[1]
    else:
        tip_port = p_tip_mid[1]
        mid_port = p_tip_mid[0]

    return [
        {"sensor_id": 1, "port": tip_port, **sensor_dict[tip_port]},
        {"sensor_id": 2, "port": mid_port, **sensor_dict[mid_port]},
        {"sensor_id": 3, "port": base_port, **sensor_dict[base_port]}
    ]


def process_file(input_path, output_path=None):
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

    port_indices = [i for i, col_name in enumerate(header) if "Port" in col_name]
    if not port_indices:
        raise ValueError(f"No sensor ports found in {input_path}")

    sensor_dict = {}
    for c in port_indices:
        port_full = header[c].strip()
        port_name = port_full.split(":")[0].strip()

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

        sensor_dict[port_name] = {
            "pos": np.median(positions, axis=0),
            "quat": markley_average_quaternion(quats),
            "samples": len(positions)
        }

    sensors = order_sensors_along_guidewire(sensor_dict)

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
        role = "Tip" if s['sensor_id'] == 1 else ("Middle" if s['sensor_id'] == 2 else "Base")
        print(f"Sensor {s['sensor_id']:<2} ({role:<6}) {s['port']:<10} {pos_str:<32} {quat_str:<40}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute median position and Markley average quaternion for experimental data in exp_nov2026."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Input CSV files. If omitted, automatically processes exp_straight.csv and all experiment CSVs in data/."
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")

    if args.inputs:
        files = args.inputs
    else:
        candidates = sorted(glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True))
        files = [
            f for f in candidates
            if not f.endswith("_median.csv")
            and not f.endswith("_icp.csv")
            and not f.endswith("_error.csv")
            and not f.endswith("_sensed_shape.csv")
            and "ground_truth" not in os.path.basename(f)
            and "sensor_spacing" not in os.path.basename(f)
        ]

    if not files:
        print("No raw experimental CSV files found to process.")
        sys.exit(1)

    print(f"Found {len(files)} files to process.")
    for f in files:
        if not os.path.isabs(f) and not os.path.exists(f):
            candidate = os.path.join(data_dir, f)
            if os.path.exists(candidate):
                f = candidate
        process_file(f)


if __name__ == "__main__":
    main()

#!/usr/bin/env -S python3 -s
"""
sensor_spacing_calculator.py (exp_nov2026)

Calculates the physical distance between consecutive EMT sensors
(Sensor 1 -> Sensor 2, and Sensor 2 -> Sensor 3) using exp_straight median data.

Output:
  sensor_spacing.csv containing the segment spacing values in mm.
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


def compute_sensor_spacing(input_csv, output_csv=None):
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    if output_csv is None:
        out_dir = os.path.dirname(os.path.abspath(input_csv))
        output_csv = os.path.join(out_dir, "sensor_spacing.csv")

    with open(input_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) < 2:
        raise ValueError(f"Need at least 2 sensors in {input_csv}, found {len(rows)}")

    sensors = []
    for r in rows:
        sid = int(r.get("sensor_id", len(sensors) + 1))
        port = r.get("port", f"Sensor {sid}")
        tx = float(r.get("Tx", r.get("x", r.get("x_mm", 0.0))))
        ty = float(r.get("Ty", r.get("y", r.get("y_mm", 0.0))))
        tz = float(r.get("Tz", r.get("z", r.get("z_mm", 0.0))))
        sensors.append({
            "sensor_id": sid,
            "port": port,
            "pos": np.array([tx, ty, tz])
        })

    sensors.sort(key=lambda s: s["sensor_id"])

    segments = []
    print(f"\nComputing sensor spacing from: {input_csv}")
    print("=" * 60)
    for i in range(len(sensors) - 1):
        s_from = sensors[i]
        s_to = sensors[i + 1]
        dist = np.linalg.norm(s_to["pos"] - s_from["pos"])
        segments.append({
            "segment": i + 1,
            "sensor_from": s_from["sensor_id"],
            "sensor_to": s_to["sensor_id"],
            "from_port": s_from["port"],
            "to_port": s_to["port"],
            "spacing_mm": dist
        })
        print(f"Segment {i+1}: Sensor {s_from['sensor_id']} ({s_from['port']}) -> "
              f"Sensor {s_to['sensor_id']} ({s_to['port']}) = {dist:.4f} mm")

    total_len = sum(s["spacing_mm"] for s in segments)
    print(f"Total distance along catheter: {total_len:.4f} mm")
    print("=" * 60)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["segment", "sensor_from", "sensor_to", "spacing_mm"])
        for s in segments:
            writer.writerow([s["segment"], s["sensor_from"], s["sensor_to"], f"{s['spacing_mm']:.6f}"])

    print(f"Saved sensor spacing to: {output_csv}\n")
    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description="Calculate physical distance between consecutive sensors from exp_straight test."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to exp_straight_median.csv (defaults to data/exp_straight_median.csv)."
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to output sensor_spacing.csv."
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")

    input_file = args.input
    if input_file is None:
        candidates = [
            os.path.join(data_dir, "exp_straight_median.csv"),
            os.path.join(script_dir, "exp_straight_median.csv"),
            "exp_straight_median.csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                input_file = c
                break

    if input_file is None or not os.path.exists(input_file):
        print("Error: exp_straight_median.csv not found.")
        print("Please run median_calculator.py first.")
        sys.exit(1)

    compute_sensor_spacing(input_file, args.output)


if __name__ == "__main__":
    main()

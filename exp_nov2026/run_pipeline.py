#!/usr/bin/env -S python3 -s
"""
run_pipeline.py (exp_nov2026)

Runs the complete shape sensing analysis pipeline across all experimental datasets
in exp_nov2026/data/ (s1, s1_gelatin, s2, s2_gelatin) and produces a comprehensive
summary table matching Table I of SENSORS_2022.
"""

import sys
import os
import subprocess

if os.environ.get("PYTHONNOUSERSITE") != "1" and not sys.flags.no_user_site:
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, ["-s"] + sys.argv)

script_dir = os.path.dirname(os.path.abspath(__file__))

steps = [
    ("Step 1: Compute Median Poses & Markley Quaternions", "median_calculator.py"),
    ("Step 2: Calculate Sensor Spacing from exp_straight", "sensor_spacing_calculator.py"),
    ("Step 3: Reconstruct 3D Shape using Cubic Bezier Curves", "shape_sensing.py"),
    ("Step 4: Align with Ground Truth using ICP and Compute Error", "icp.py"),
    ("Step 5: Generate Visualizations", "visualizer.py"),
    ("Step 6: Generate Rigid vs Gelatin Comparison Visualizations", "compare_rigid_vs_gelatin.py")
]

for title, script in steps:
    print("\n" + "=" * 70)
    print(f"RUNNING {title}")
    print("=" * 70)
    script_path = os.path.join(script_dir, script)
    res = subprocess.run([sys.executable, "-s", script_path])
    if res.returncode != 0:
        print(f"Error running {script}, exiting.")
        sys.exit(res.returncode)

print("\n" + "=" * 70)
print("ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
print("=" * 70)

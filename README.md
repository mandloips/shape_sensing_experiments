# Electromagnetic (EMT) Catheter 3D Shape Sensing & Experimental Verification

This repository contains the complete data processing, 3D shape reconstruction, and ICP ground-truth verification pipeline for flexible catheter shape sensing using embedded Electromagnetic Tracking (EMT) sensors.

---

## 🔬 Pipeline Overview

The pipeline implements continuous 3D catheter curve reconstruction from discrete 6-DoF EMT sensor readings:

1. **Temporal Filtering (`median_calculator.py`):**
   - Extracts spatial median position $(T_x, T_y, T_z)$ over steady-state acquisition windows.
   - Computes orientation using Markley quaternion averaging (eigenvector corresponding to the maximum eigenvalue of the quaternion scatter matrix).
   - Automatically resolves physical sensor sequence ($S_1 \text{ [Tip]} \rightarrow S_2 \text{ [Middle]} \rightarrow S_3 \text{ [Base]}$) based on relative sensor spacings.

2. **Sensor Calibration (`sensor_spacing_calculator.py`):**
   - Computes physical straight-wire arc lengths between adjacent sensors from calibration measurements.
   - Inter-sensor spacings: $S_1 \rightarrow S_2 \approx 59.86\text{ mm}$, $S_2 \rightarrow S_3 \approx 242.15\text{ mm}$ (Total length $\approx 302.01\text{ mm}$).

3. **Piecewise Cubic Bézier Reconstruction (`shape_sensing.py`):**
   - Uses pure rotated $+Z$ orientations ($\mathbf{t}_i = R(q_i) [0, 0, 1]^T$) as endpoint tangent vectors.
   - Solves for control point parameters $(l_1, l_2)$ via bounded L-BFGS-B optimization minimizing the squared difference between the Bézier segment arc-length and the calibrated wire length.
   - Re-parameterizes the entire curve by cumulative Euclidean arc length, producing **500 strictly equidistant points** ($\Delta s = 0.6053\text{ mm}$).

4. **Rigid Registration & Error Evaluation (`icp.py`):**
   - Aligns the reconstructed 500-point curve to CAD ground truth point clouds via Iterative Closest Point (ICP).
   - Uses multi-start initialization (Centroid alignment + PCA reflections) to guarantee convergence to the global minimum.
   - Computes point-by-point Euclidean distances, reporting Mean ± Std, Min, and Max errors.

5. **Visualization (`visualizer.py` & `compare_rigid_vs_gelatin.py`):**
   - Generates 3D perspective views, 2D top-down projections, and arc-length error distributions comparing Rigid Rig vs. Gelatin Phantom against CAD Ground Truth.

---

## 📊 Experimental Results

All reconstructed shapes are evaluated with 500 equidistant points against CAD ground truth:

| Experiment / Configuration | Mean ± Std (mm) | Median (mm) | Min (mm) | Max (mm) | Relative Error (% of 302 mm) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Shape 1** | **$1.10 \pm 0.48$** | 1.18 | 0.23 | 2.53 | 0.36% |
| **Shape 1 (Gelatin)** | **$0.89 \pm 0.52$** | 0.86 | 0.10 | 4.24 | 0.29% |
| **Shape 2** | **$1.74 \pm 0.71$** | 1.81 | 0.38 | 3.59 | 0.58% |
| **Shape 2 (Gelatin)** | **$1.48 \pm 0.53$** | 1.55 | 0.04 | 2.59 | 0.49% |
| **Shape 3** | **$1.72 \pm 0.71$** | 1.70 | 0.36 | 3.72 | 0.57% |
| **Shape 3 (Gelatin)** | **$1.61 \pm 0.62$** | 1.62 | 0.32 | 4.88 | 0.53% |
| **Shape 4** | **$2.84 \pm 0.95$** | 2.72 | 0.83 | 4.96 | 0.94% |
| **Shape 4 (Gelatin)** | **$3.14 \pm 1.18$** | 3.10 | 0.49 | 6.15 | 1.04% |
| **Shape 5** | **$2.43 \pm 0.95$** | 2.42 | 0.75 | 5.30 | 0.80% |
| **Shape 5 (Gelatin)** | **$2.50 \pm 0.96$** | 2.48 | 0.80 | 5.82 | 0.83% |

---

## 📁 Repository Structure

```text
├── exp_nov2026/
│   ├── data/
│   │   ├── exp_straight.csv           # Straight calibration data
│   │   ├── sensor_spacing.csv         # Calibrated inter-sensor spacings
│   │   ├── s1/, s1_gelatin/           # Shape 1 raw, median, sensed, and GT data
│   │   ├── s2/, s2_gelatin/           # Shape 2 raw, median, sensed, and GT data
│   │   ├── s3/, s3_gelatin/           # Shape 3 raw, median, sensed, and GT data
│   │   ├── s4/, s4_gelatin/           # Shape 4 raw, median, sensed, and GT data
│   │   ├── s5/, s5_gelatin/           # Shape 5 raw, median, sensed, and GT data
│   │   ├── all_shapes_comparison.png  # 10-shape grid visualization
│   │   └── s*_rigid_vs_gelatin.png    # Rigid vs Gelatin 3-panel comparisons
│   ├── median_calculator.py
│   ├── sensor_spacing_calculator.py
│   ├── shape_sensing.py
│   ├── icp.py
│   ├── visualizer.py
│   ├── compare_rigid_vs_gelatin.py
│   └── run_pipeline.py                # End-to-end master runner
├── pre_exp_nov2026/
│   ├── data/                          # 100 mm helical rig dataset
│   ├── median_calculator.py
│   ├── sensor_spacing_calculator.py
│   ├── shape_sensing.py
│   ├── icp.py
│   └── visualizer.py
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

### Requirements
- Python 3.8+
- NumPy, SciPy, Matplotlib

### Run End-to-End Analysis
To reproduce all results, re-calculate median poses, reconstruct shapes, evaluate ICP errors, and generate all visualizations:

```bash
cd exp_nov2026
./run_pipeline.py
```

### Run Preliminary Helical Rig Test
```bash
cd pre_exp_nov2026
python3 median_calculator.py
python3 sensor_spacing_calculator.py
python3 shape_sensing.py
python3 icp.py
python3 visualizer.py
```

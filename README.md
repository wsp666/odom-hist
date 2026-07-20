# odom-hist

`odom-hist` is a pip-installable odometry evaluation toolkit for comparing predicted trajectories with a ground-truth trajectory. It is designed for SLAM and odometry experiments where users need repeatable KITTI-style metrics and publication-ready visualizations without exposing private model or algorithm names.

The project packages the original one-off evaluation script into a reusable, user-friendly Python library and CLI that works on both Windows and Ubuntu.

## What it evaluates

Input trajectories use the common TUM text format:

```text
timestamp tx ty tz qx qy qz qw
```

For each prediction baseline, `odom-hist`:

1. Loads ground-truth and predicted poses.
2. Aligns frames by nearest timestamp with a configurable tolerance.
3. Computes KITTI-style segment metrics:
   - translation error in percent
   - rotation error in degrees per 100 m
4. Computes frame-to-frame relative translation and rotation errors.
5. Optionally saves trajectory plots and scatter/histogram error plots.

## Installation

From the repository root:

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

The package is pure Python and declares cross-platform dependencies available on Windows and Ubuntu: `numpy`, `scipy`, and `matplotlib`.

## CLI usage

```bash
odom-hist eval \
  --gt path/to/ground_truth.txt \
  --pred "Model_A" path/to/tum_traj_model_a.txt \
  --pred "Model_B" path/to/tum_traj_model_b.txt \
  --sequence slam-TopS01 \
  --output results_vis
```

Disable plot generation when running on servers or CI:

```bash
odom-hist eval --gt gt.txt --pred Model_A pred.txt --no-plots
```

## Python API

```python
from pathlib import Path
from odom_hist import EvalConfig, EvalVisualizer, OdomEvaluator, SequenceConfig

sequence = SequenceConfig(
    name="slam-TopS01",
    gt=Path("tum_traj_gt.txt"),
    baselines={
        "Model_A": Path("tum_traj_model_a.txt"),
        "Model_B": Path("tum_traj_model_b.txt"),
    },
    output_dir=Path("results_vis"),
)

evaluator = OdomEvaluator(sequence, EvalConfig(timestamp_tolerance=0.06))
result = evaluator.evaluate()

print(result.translation_percent)
print(result.rotation_deg_per_100m)

visualizer = EvalVisualizer(sequence.output_dir)
visualizer.plot_traj(evaluator, sequence.name)
visualizer.plot_scatter_and_hist(
    result.step_translation_m,
    result.step_rotation_rad,
    list(sequence.baselines),
    sequence.name,
)
```

## Project layout

```text
src/odom_hist/
├── cli.py              # command line entry point
├── config.py           # dataclass configuration objects
├── evaluator.py        # high-level alignment and evaluation workflow
├── io.py               # TUM pose loading
├── metrics.py          # transform, KITTI, and step-error metrics
└── visualization.py    # plotting helpers
```

## Notes

- The default timestamp tolerance is `0.06` seconds.
- The default KITTI segment lengths are 100 m through 800 m.
- Matplotlib uses a non-interactive backend by default, so plots can be generated on headless Ubuntu servers and Windows CI.

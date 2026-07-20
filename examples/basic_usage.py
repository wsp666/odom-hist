"""Minimal Python API example for odom-hist."""

from pathlib import Path

from odom_hist import EvalConfig, EvalVisualizer, OdomEvaluator, SequenceConfig

sequence = SequenceConfig(
    name="example-sequence",
    gt=Path("path/to/gt.txt"),
    baselines={"Model_A": Path("path/to/pred.txt")},
    output_dir=Path("results_vis"),
)

evaluator = OdomEvaluator(sequence, EvalConfig(timestamp_tolerance=0.06))
result = evaluator.evaluate()
print(result.translation_percent)
print(result.rotation_deg_per_100m)

visualizer = EvalVisualizer(sequence.output_dir)
visualizer.plot_traj(evaluator, sequence.name)

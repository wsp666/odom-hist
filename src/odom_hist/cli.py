"""Command line interface for odom-hist."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import EvalConfig, SequenceConfig
from .evaluator import OdomEvaluator
from .visualization import EvalVisualizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odom-hist",
        description="Evaluate odometry trajectories against ground truth TUM pose files.",
    )
    subparsers = parser.add_subparsers(dest="command")

    eval_parser = subparsers.add_parser("eval", help="Evaluate one sequence.")
    eval_parser.add_argument("--gt", required=True, type=Path, help="Ground-truth TUM pose file.")
    eval_parser.add_argument(
        "--pred",
        required=True,
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        help="Baseline name and predicted TUM pose file. Repeat for multiple baselines.",
    )
    eval_parser.add_argument("--sequence", default="sequence", help="Sequence name used in output files.")
    eval_parser.add_argument("--output", default="odom_hist_results", type=Path, help="Output directory.")
    eval_parser.add_argument("--tolerance", default=0.06, type=float, help="Timestamp tolerance in seconds.")
    eval_parser.add_argument("--step-size", default=10, type=int, help="KITTI segment start step size.")
    eval_parser.add_argument("--frequency-hz", default=10.0, type=float, help="Trajectory sampling frequency.")
    eval_parser.add_argument("--no-plots", action="store_true", help="Only print metrics; do not save plots.")
    eval_parser.add_argument("--show", action="store_true", help="Display plots interactively.")
    eval_parser.set_defaults(func=run_eval)

    subparsers.add_parser("version", help="Print package version.").set_defaults(func=run_version)
    return parser


def run_eval(args: argparse.Namespace) -> int:
    baselines = {name: Path(path) for name, path in args.pred}
    sequence = SequenceConfig(args.sequence, args.gt, baselines, args.output)
    options = EvalConfig(
        timestamp_tolerance=args.tolerance,
        step_size=args.step_size,
        frequency_hz=args.frequency_hz,
        save_plots=not args.no_plots,
        show_plots=args.show,
    )
    evaluator = OdomEvaluator(sequence, options)
    result = evaluator.evaluate()

    print(f"Sequence: {result.sequence}")
    print("Matched frames:")
    for name, count in result.matched_frames.items():
        print(f"  - {name}: {count}")
    print("Translational error (%):")
    for name, value in result.translation_percent.items():
        print(f"  - {name}: {value:.6f}")
    print("Rotational error (deg/100m):")
    for name, value in result.rotation_deg_per_100m.items():
        print(f"  - {name}: {value:.6f}")

    if options.save_plots or options.show_plots:
        visualizer = EvalVisualizer(sequence.output_dir)
        if options.make_traj_plot:
            visualizer.plot_traj(evaluator, sequence.name, plot_save=options.save_plots, plot_show=options.show_plots)
        if options.make_scatter_plot:
            visualizer.plot_scatter_and_hist(
                result.step_translation_m,
                result.step_rotation_rad,
                list(baselines.keys()),
                sequence.name,
                plot_save=options.save_plots,
                plot_show=options.show_plots,
                bins_num=options.scatter_bins,
            )
    return 0


def run_version(_args: argparse.Namespace) -> int:
    from importlib.metadata import version

    print(version("odom-hist"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

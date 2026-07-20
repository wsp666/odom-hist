"""Configuration models for odom-hist."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class SequenceConfig:
    """Input files and output directory for one odometry sequence.

    Pose files use TUM format by default: ``timestamp tx ty tz qx qy qz qw``.
    ``gt`` is the ground-truth trajectory and ``baselines`` maps method names to
    predicted trajectory files.
    """

    name: str
    gt: Path
    baselines: Mapping[str, Path]
    output_dir: Path


@dataclass(frozen=True)
class EvalConfig:
    """Runtime options shared by all sequence evaluations."""

    timestamp_tolerance: float = 0.06
    segment_lengths: tuple[float, ...] = (100, 200, 300, 400, 500, 600, 700, 800)
    step_size: int = 10
    frequency_hz: float = 10.0
    save_plots: bool = True
    show_plots: bool = False
    make_traj_plot: bool = True
    make_scatter_plot: bool = True
    make_bar_box_plot: bool = False
    scatter_bins: int = 30
    plot_dpi: int = 300
    font_family: str = "DejaVu Serif"
    extra: dict[str, object] = field(default_factory=dict)

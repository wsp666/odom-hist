"""High-level odometry evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import EvalConfig, SequenceConfig
from .io import load_tum_poses
from .metrics import compute_kitti_metrics, get_inverse_tf, rotation_error, translation_error


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics and aligned poses produced by one evaluation run."""

    sequence: str
    translation_percent: dict[str, float]
    rotation_deg_per_100m: dict[str, float]
    step_translation_m: dict[str, list[float]]
    step_rotation_rad: dict[str, list[float]]
    matched_frames: dict[str, int]


class OdomEvaluator:
    """Evaluate predicted odometry trajectories against one ground-truth trajectory."""

    def __init__(self, sequence: SequenceConfig, options: EvalConfig | None = None) -> None:
        self.sequence = sequence
        self.options = options or EvalConfig()
        self.baselines_name = list(sequence.baselines.keys())
        self.gt_raw = load_tum_poses(sequence.gt)
        self.baselines_raw = {
            name: load_tum_poses(path) for name, path in sequence.baselines.items()
        }
        self.gt_data: list[dict[str, dict[int, np.ndarray]]] = []
        self.aligned_data: list[dict[str, dict[int, np.ndarray]]] = []
        self.matched_frames: dict[str, int] = {}
        self.align_all_baselines()
        self.baselines_data = self.aligned_data

    def align_all_baselines(self) -> None:
        """Align GT and predictions by nearest timestamp within tolerance."""

        self.gt_data = []
        self.aligned_data = []
        gt_ts = np.asarray(self.gt_raw["ts"])
        gt_poses_raw = self.gt_raw["poses"]

        common_gt: dict[int, np.ndarray] | None = None
        for name, pred_data in self.baselines_raw.items():
            pred_ts = np.asarray(pred_data["ts"])
            pred_poses_raw = pred_data["poses"]
            aligned_gt: dict[int, np.ndarray] = {}
            aligned_pred: dict[int, np.ndarray] = {}
            next_idx = 0

            for pred_idx, timestamp in enumerate(pred_ts):
                nearest_idx = int(np.argmin(np.abs(gt_ts - timestamp)))
                if abs(float(gt_ts[nearest_idx] - timestamp)) <= self.options.timestamp_tolerance:
                    aligned_gt[next_idx] = gt_poses_raw[nearest_idx]
                    aligned_pred[next_idx] = pred_poses_raw[pred_idx]
                    next_idx += 1

            if not aligned_pred:
                raise ValueError(
                    f"No frames matched for baseline '{name}'. Increase timestamp_tolerance "
                    "or verify that both trajectories use the same timestamp unit."
                )
            if common_gt is None:
                common_gt = aligned_gt
            self.aligned_data.append({name: aligned_pred})
            self.matched_frames[name] = len(aligned_pred)

        self.gt_data = [{"After GT": common_gt or {}}]

    def calcu_t_ape(self, transform_gt: np.ndarray, transform_pred: np.ndarray) -> float:
        """Compatibility method: translational absolute pose error."""

        return translation_error(get_inverse_tf(transform_gt) @ transform_pred)

    def calcu_r_ape(self, transform_gt: np.ndarray, transform_pred: np.ndarray) -> float:
        """Compatibility method: rotational absolute pose error in radians."""

        return rotation_error(get_inverse_tf(transform_gt) @ transform_pred)

    def calcu_pencent(self) -> tuple[dict[str, float], dict[str, float]]:
        """Compatibility method returning KITTI-style aggregate errors."""

        translation_result: dict[str, float] = {}
        rotation_result: dict[str, float] = {}
        gt_dict = self.gt_data[0]["After GT"]
        poses_gt = [gt_dict[idx] for idx in sorted(gt_dict.keys())]

        for item in self.aligned_data:
            name = next(iter(item.keys()))
            pred_dict = item[name]
            poses_pred = [pred_dict[idx] for idx in sorted(pred_dict.keys())]
            usable = min(len(poses_gt), len(poses_pred))
            t_err, r_err = compute_kitti_metrics(
                poses_gt[:usable],
                poses_pred[:usable],
                {self.sequence.name: usable},
                self.options.segment_lengths,
                self.options.step_size,
                self.options.frequency_hz,
            )
            translation_result[name] = t_err
            rotation_result[name] = r_err
        return translation_result, rotation_result

    def calcu_steperrors(self) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        """Compatibility method returning frame-to-frame relative errors."""

        gt_dict = self.gt_data[0]["After GT"]
        step_t: dict[str, list[float]] = {}
        step_r: dict[str, list[float]] = {}

        for item in self.aligned_data:
            name = next(iter(item.keys()))
            pred_dict = item[name]
            common_indices = sorted(set(gt_dict.keys()) & set(pred_dict.keys()))
            t_errors: list[float] = []
            r_errors: list[float] = []
            for prev_idx, cur_idx in zip(common_indices[:-1], common_indices[1:]):
                gt_move = get_inverse_tf(gt_dict[prev_idx]) @ gt_dict[cur_idx]
                pred_move = get_inverse_tf(pred_dict[prev_idx]) @ pred_dict[cur_idx]
                t_errors.append(self.calcu_t_ape(gt_move, pred_move))
                r_errors.append(self.calcu_r_ape(gt_move, pred_move))
            step_t[name] = t_errors
            step_r[name] = r_errors
        return step_t, step_r

    def evaluate(self) -> EvaluationResult:
        """Run all metrics and return a structured result."""

        t_percent, r_deg = self.calcu_pencent()
        step_t, step_r = self.calcu_steperrors()
        return EvaluationResult(
            sequence=self.sequence.name,
            translation_percent=t_percent,
            rotation_deg_per_100m=r_deg,
            step_translation_m=step_t,
            step_rotation_rad=step_r,
            matched_frames=dict(self.matched_frames),
        )


ModelEval = OdomEvaluator
modelEval = OdomEvaluator

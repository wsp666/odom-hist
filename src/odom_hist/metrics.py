"""Odometry error metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def get_inverse_tf(transform: np.ndarray) -> np.ndarray:
    """Invert a 4x4 homogeneous transform."""

    inverse = np.eye(4, dtype=float)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def rotation_error(pose_error: np.ndarray) -> float:
    """Return SO(3) rotation error in radians."""

    trace_value = float(np.trace(pose_error[:3, :3]))
    cos_theta = 0.5 * (trace_value - 1.0)
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def translation_error(pose_error: np.ndarray) -> float:
    """Return Euclidean translation error in metres."""

    return float(np.linalg.norm(pose_error[:3, 3]))


def trajectory_distances(poses: Sequence[np.ndarray]) -> list[float]:
    """Compute cumulative 3D path length for a pose sequence."""

    distances = [0.0]
    for idx in range(1, len(poses)):
        previous = poses[idx - 1][:3, 3]
        current = poses[idx][:3, 3]
        distances.append(distances[idx - 1] + float(np.linalg.norm(current - previous)))
    return distances


def last_frame_from_segment_length(
    distances: Sequence[float], first_frame: int, length: float
) -> int:
    """Find the first frame whose travelled distance exceeds ``length``."""

    target = distances[first_frame] + length
    for idx in range(first_frame, len(distances)):
        if distances[idx] > target:
            return idx
    return -1


def get_stats(errors: Sequence[Sequence[float]]) -> tuple[float, float]:
    """Average KITTI-style translation and rotation errors."""

    if not errors:
        return 0.0, 0.0
    t_err = float(np.mean([item[2] for item in errors]))
    r_err = float(np.mean([item[1] for item in errors]))
    return t_err, r_err


def calc_sequence_errors(
    poses_gt: Sequence[np.ndarray],
    poses_pred: Sequence[np.ndarray],
    lengths: Sequence[float] = (100, 200, 300, 400, 500, 600, 700, 800),
    step_size: int = 10,
    frequency_hz: float = 10.0,
) -> list[list[float]]:
    """Calculate KITTI-style segment errors for one aligned trajectory pair."""

    if len(poses_gt) != len(poses_pred):
        raise ValueError("Ground truth and prediction pose lists must have the same length.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive.")

    errors: list[list[float]] = []
    distances = trajectory_distances(poses_gt)
    for first_frame in range(0, len(poses_gt), step_size):
        for length in lengths:
            last_frame = last_frame_from_segment_length(distances, first_frame, length)
            if last_frame == -1:
                continue
            pose_delta_gt = get_inverse_tf(poses_gt[first_frame]) @ poses_gt[last_frame]
            pose_delta_pred = get_inverse_tf(poses_pred[first_frame]) @ poses_pred[last_frame]
            pose_error = get_inverse_tf(pose_delta_pred) @ pose_delta_gt
            r_err = rotation_error(pose_error)
            t_err = translation_error(pose_error)
            num_frames = float(last_frame - first_frame)
            speed = float(length) / ((1.0 / frequency_hz) * num_frames)
            errors.append([first_frame, r_err / length, t_err / length, length, speed])
    return errors


def compute_kitti_metrics(
    poses_gt: Sequence[np.ndarray],
    poses_pred: Sequence[np.ndarray],
    seq_lens: dict[str, int] | None = None,
    lengths: Sequence[float] = (100, 200, 300, 400, 500, 600, 700, 800),
    step_size: int = 10,
    frequency_hz: float = 10.0,
) -> tuple[float, float]:
    """Return ``(translation_percent, rotation_deg_per_100m)``."""

    if seq_lens is None:
        seq_lens = {"sequence": min(len(poses_gt), len(poses_pred))}

    sequence_indices: list[list[int]] = []
    offset = 0
    max_len = min(len(poses_gt), len(poses_pred))
    for sequence_len in seq_lens.values():
        sequence_indices.append(list(range(offset, min(offset + sequence_len, max_len))))
        offset += sequence_len

    averaged_errors = []
    for indices in sequence_indices:
        if not indices:
            continue
        gt_part = [poses_gt[idx] for idx in indices]
        pred_part = [poses_pred[idx] for idx in indices]
        errors = calc_sequence_errors(gt_part, pred_part, lengths, step_size, frequency_hz)
        if errors:
            averaged_errors.append(get_stats(errors))

    if not averaged_errors:
        return 100.0, 100.0
    avg_t, avg_r = np.mean(averaged_errors, axis=0)
    return float(avg_t * 100.0), float(avg_r * 180.0 / np.pi * 100.0)

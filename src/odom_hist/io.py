"""Trajectory loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_tum_poses(path: str | Path) -> dict[str, object]:
    """Load TUM poses from ``timestamp tx ty tz qx qy qz qw`` text files."""

    pose_path = Path(path).expanduser()
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose file does not exist: {pose_path}")

    data = np.loadtxt(pose_path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 8:
        raise ValueError(
            f"Expected at least 8 columns in TUM pose file {pose_path}; got {data.shape[1]}."
        )

    timestamps = data[:, 0]
    rotations = Rotation.from_quat(data[:, 4:8]).as_matrix()
    poses: dict[int, np.ndarray] = {}
    for idx, _timestamp in enumerate(timestamps):
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = rotations[idx]
        transform[:3, 3] = data[idx, 1:4]
        poses[idx] = transform
    return {"ts": timestamps, "poses": poses, "path": pose_path}

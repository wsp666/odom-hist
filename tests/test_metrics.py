import numpy as np

from odom_hist.metrics import compute_kitti_metrics, get_inverse_tf, trajectory_distances


def make_pose(x=0.0, y=0.0, z=0.0):
    transform = np.eye(4)
    transform[:3, 3] = [x, y, z]
    return transform


def test_inverse_transform_round_trip():
    transform = make_pose(1.0, 2.0, 3.0)
    np.testing.assert_allclose(get_inverse_tf(transform) @ transform, np.eye(4), atol=1e-8)


def test_trajectory_distances():
    poses = [make_pose(0), make_pose(3), make_pose(6)]
    assert trajectory_distances(poses) == [0.0, 3.0, 6.0]


def test_identical_kitti_metrics_are_zero_for_long_trajectory():
    poses = [make_pose(float(idx)) for idx in range(130)]
    t_err, r_err = compute_kitti_metrics(poses, poses, {"seq": len(poses)}, lengths=(100,), step_size=10)
    assert t_err == 0.0
    assert r_err == 0.0

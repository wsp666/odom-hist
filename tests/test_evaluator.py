import numpy as np
from scipy.spatial.transform import Rotation

from odom_hist import EvalConfig, OdomEvaluator, SequenceConfig


def write_tum(path, xs):
    quat = Rotation.identity().as_quat()
    with path.open("w", encoding="utf-8") as file:
        for idx, x in enumerate(xs):
            file.write(f"{idx * 0.1:.1f} {x:.6f} 0 0 {quat[0]} {quat[1]} {quat[2]} {quat[3]}\n")


def test_evaluator_aligns_and_computes_step_errors(tmp_path):
    gt = tmp_path / "gt.txt"
    pred = tmp_path / "pred.txt"
    write_tum(gt, np.arange(5, dtype=float))
    write_tum(pred, np.arange(5, dtype=float))

    sequence = SequenceConfig("seq", gt, {"Model_A": pred}, tmp_path / "out")
    evaluator = OdomEvaluator(sequence, EvalConfig())
    result = evaluator.evaluate()

    assert result.matched_frames == {"Model_A": 5}
    assert result.step_translation_m["Model_A"] == [0.0, 0.0, 0.0, 0.0]
    assert result.step_rotation_rad["Model_A"] == [0.0, 0.0, 0.0, 0.0]

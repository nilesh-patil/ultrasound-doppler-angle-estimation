"""Tests — metrics + predictions dump."""
import numpy as np
import pandas as pd

from uda.config import ExperimentConfig
from uda.evaluation.evaluate import evaluate, metrics


def test_perfect_prediction_is_zero_error_unit_r2():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    m = metrics(y, y.copy())
    assert m["mae"] == 0.0 and m["rmse"] == 0.0 and m["me"] == 0.0
    assert m["mape"] == 0.0
    assert abs(m["r2"] - 1.0) < 1e-12


def test_hand_checked_vector():
    yt = np.array([100.0, 100.0])
    yp = np.array([110.0, 90.0])  # errors +10, -10
    m = metrics(yt, yp)
    assert abs(m["mae"] - 10.0) < 1e-9
    assert abs(m["rmse"] - 10.0) < 1e-9
    assert abs(m["me"] - 0.0) < 1e-9  # bias cancels
    assert abs(m["mape"] - 10.0) < 1e-9


def test_evaluate_writes_metrics_and_predictions(tmp_path):
    cfg = ExperimentConfig(name="rt", backbone={"name": "vgg19", "weights": None})
    meta = pd.DataFrame(
        {
            "image_id": ["a", "b"],
            "patient_id": [0, 1],
            "rotation_deg": [0, 5],
            "split": ["test", "test"],
        }
    )
    row = evaluate(cfg, np.array([88.0, 90.0]), np.array([87.0, 91.0]), meta, out_dir=tmp_path)
    assert (tmp_path / "metrics.csv").exists()
    preds = pd.read_csv(tmp_path / "predictions" / "rt.csv")
    assert {"theta_true", "theta_pred"} <= set(preds.columns)
    assert row["backbone"] == "vgg19" and row["n_test"] == 2


def test_metrics_csv_dedups_on_rerun(tmp_path):
    cfg = ExperimentConfig(name="rt", backbone={"name": "vgg19", "weights": None})
    meta = pd.DataFrame(
        {"image_id": ["a"], "patient_id": [0], "rotation_deg": [0], "split": ["test"]}
    )
    evaluate(cfg, np.array([88.0]), np.array([88.0]), meta, out_dir=tmp_path)
    evaluate(cfg, np.array([88.0]), np.array([80.0]), meta, out_dir=tmp_path)
    df = pd.read_csv(tmp_path / "metrics.csv")
    assert len(df) == 1  # same key replaced, not appended

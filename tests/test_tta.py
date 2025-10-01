"""Tests — rotation test-time augmentation (``uda.evaluation.tta``).

These tests cover the ``uda.evaluation.tta`` module.
Everything here is post-hoc on *saved* predictions: no model is built and every
test runs in well under a second. The module under test is **Keras-free**
(asserted explicitly in a fresh subprocess, mirroring ``tests/test_conformal.py``).

One public entry point is exercised:

* ``tta_aggregate(pred_csv, *, reduce="circular_mean") -> {y_true, y_pred,
  metrics, n_base}`` — de-rotates each row (``base_est = theta_pred -
  rotation_deg``, ``base_true = theta_true - rotation_deg``), reduces the
  de-rotated estimates **circularly** to one prediction per ``image_id``, and
  reports ``uda.evaluation.evaluate.metrics`` over the ``n_base`` reduced estimates.

Vessel orientation is **180-periodic**, so the reduction is the circular mean in
double-angle space ``0.5*atan2(mean(sin 2θ), mean(cos 2θ))`` (or the circular
median minimizing summed signed-wrap distance), never a linear mean — a linear
mean would corrupt any image whose de-rotated estimates straddle the 0/180 seam.
``metrics`` is delegated to ``uda.evaluation.evaluate.metrics`` (single source of truth).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from uda.evaluation import evaluate
from uda.evaluation.tta import tta_aggregate

# Repo-relative path to the honest, full-coverage OOF predictions
# (84 base images × 25 rotations = 2100 rows).
REAL_OOF_CSV = "results/predictions/tuned_densenet201_oof.csv"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _signed_wrap(delta: np.ndarray) -> np.ndarray:
    """Reference signed-wrap into ``(-90, 90]`` (the project's residual scale)."""
    return ((np.asarray(delta, dtype=float) + 90.0) % 180.0) - 90.0


def _circular_mean(angles_deg: np.ndarray) -> float:
    """Reference double-angle circular mean mapped into ``[0, 180)``."""
    t = np.deg2rad(2.0 * np.asarray(angles_deg, dtype=float))
    mean = 0.5 * np.rad2deg(np.arctan2(np.mean(np.sin(t)), np.mean(np.cos(t))))
    return float(mean % 180.0)


def _write_csv(tmp_path, rows, name="synthetic_oof.csv"):
    """Write a synthetic prediction CSV mirroring the real OOF schema."""
    path = tmp_path / name
    pd.DataFrame(
        rows,
        columns=[
            "image_id",
            "patient_id",
            "rotation_deg",
            "theta_true",
            "theta_pred",
        ],
    ).to_csv(path, index=False)
    return path


def _rotated_image(image_id, patient_id, base, rotations, preds):
    """Build rows for one base image: ``theta_true = base + rotation_deg``.

    ``preds`` are the *model* predictions (already in the rotated frame), so the
    de-rotated estimate of row ``i`` is ``preds[i] - rotations[i]``.
    """
    return [
        {
            "image_id": image_id,
            "patient_id": float(patient_id),
            "rotation_deg": float(rot),
            "theta_true": float(base + rot),
            "theta_pred": float(pred),
        }
        for rot, pred in zip(rotations, preds)
    ]


# --------------------------------------------------------------------------- #
# 1. de-rotation correctness — constant base truth + degenerate single rotation
#    (test 1)
# --------------------------------------------------------------------------- #
def test_single_rotation_reproduces_that_rows_base_estimate(tmp_path):
    """One rotation per image is a degenerate reduction: the reduced prediction is
    that row's de-rotated estimate ``theta_pred - rotation_deg`` and ``y_true`` is
    its constant base truth ``theta_true - rotation_deg``."""
    base, rot = 40.0, 30.0
    theta_true = base + rot  # 70
    theta_pred = 76.0  # so base_est = 76 - 30 = 46
    csv = _write_csv(
        tmp_path,
        _rotated_image("img0", 1, base, [rot], [theta_pred]),
    )

    for reduce in ("circular_mean", "median"):
        out = tta_aggregate(csv, reduce=reduce)
        assert out["n_base"] == 1
        assert len(out["y_true"]) == len(out["y_pred"]) == 1
        # de-rotated base truth is the single constant
        assert out["y_true"][0] == pytest.approx(theta_true - rot)  # 40
        # degenerate reduction reproduces the row's de-rotated estimate
        assert out["y_pred"][0] == pytest.approx(theta_pred - rot)  # 46


def test_base_true_is_constant_per_image_and_equals_y_true(tmp_path):
    """``base_true = theta_true - rotation_deg`` is a single constant per ``image_id``
    (since ``theta_true = base + rot``), and ``y_true[image]`` equals that constant."""
    base_a, base_b = 50.0, 120.0
    rots = [-60.0, -20.0, 0.0, 35.0, 60.0]
    rng = np.random.default_rng(0)
    rows = _rotated_image(
        "imgA", 1, base_a, rots, base_a + rng.normal(0, 3, len(rots))
    ) + _rotated_image(
        "imgB", 2, base_b, rots, base_b + rng.normal(0, 3, len(rots))
    )
    csv = _write_csv(tmp_path, rows)

    out = tta_aggregate(csv, reduce="circular_mean")
    # order is not contractually fixed: match y_true against the known base truths
    truths = sorted(np.asarray(out["y_true"], dtype=float).tolist())
    assert truths == pytest.approx([base_a, base_b])


# --------------------------------------------------------------------------- #
# 2. consistent-label rotations collapse to the base angle  (tests 1 + 5)
# --------------------------------------------------------------------------- #
def test_consistent_rotations_collapse_to_base_angle(tmp_path):
    """If the model is perfect (``theta_pred = base + rot``), every de-rotated
    estimate equals ``base`` and the reduction collapses exactly to the base angle."""
    base = 73.0
    rots = [-60.0, -30.0, -10.0, 0.0, 15.0, 45.0, 60.0]
    preds = [base + r for r in rots]  # perfect model in the rotated frame
    csv = _write_csv(tmp_path, _rotated_image("img0", 1, base, rots, preds))

    for reduce in ("circular_mean", "median"):
        out = tta_aggregate(csv, reduce=reduce)
        assert out["y_pred"][0] == pytest.approx(base, abs=1e-6)
        assert out["y_true"][0] == pytest.approx(base, abs=1e-6)


# --------------------------------------------------------------------------- #
# 3. circular reduction respects the 0/180 seam  (test 3)
# --------------------------------------------------------------------------- #
def test_circular_mean_of_1_and_179_is_boundary_not_90(tmp_path):
    """De-rotated estimates ``{1, 179}`` reduce to the 0/180 boundary, NOT to 90
    (a naive linear mean would give the visibly-wrong 90)."""
    # base truth 0; two rotations whose de-rotated estimates are 1 and 179.
    # row1: rot=0,  pred=1   -> base_est = 1
    # row2: rot=10, pred=189 -> base_est = 179
    rows = [
        {
            "image_id": "seam",
            "patient_id": 1.0,
            "rotation_deg": 0.0,
            "theta_true": 0.0,
            "theta_pred": 1.0,
        },
        {
            "image_id": "seam",
            "patient_id": 1.0,
            "rotation_deg": 10.0,
            "theta_true": 10.0,
            "theta_pred": 189.0,
        },
    ]
    csv = _write_csv(tmp_path, rows)

    out = tta_aggregate(csv, reduce="circular_mean")
    pred = out["y_pred"][0]
    # near the seam: ~0 or ~180, i.e. signed-wrap distance to 0 is tiny
    assert abs(_signed_wrap(np.array([pred - 0.0]))[0]) < 1.0
    assert not pred == pytest.approx(90.0, abs=1.0)
    # matches the reference double-angle circular mean of {1, 179}
    assert pred == pytest.approx(_circular_mean(np.array([1.0, 179.0])), abs=1e-6)


def test_circular_mean_reduction_matches_double_angle_formula(tmp_path):
    """On a multi-rotation image the reduced prediction equals the reference
    ``0.5*atan2(mean(sin 2θ), mean(cos 2θ))`` over the de-rotated estimates."""
    base = 12.0
    rots = [-40.0, -5.0, 0.0, 25.0, 55.0]
    # de-rotated estimates spread around the seam: 178, 2, 5, 175, 9
    base_ests = [178.0, 2.0, 5.0, 175.0, 9.0]
    preds = [be + r for be, r in zip(base_ests, rots)]  # pred = base_est + rot
    csv = _write_csv(tmp_path, _rotated_image("img0", 1, base, rots, preds))

    out = tta_aggregate(csv, reduce="circular_mean")
    expected = _circular_mean(np.array(base_ests))
    assert out["y_pred"][0] == pytest.approx(expected, abs=1e-6)


# --------------------------------------------------------------------------- #
# 4. circular median reduction  (test 4)
# --------------------------------------------------------------------------- #
def test_circular_median_minimizes_summed_wrap_distance(tmp_path):
    """``reduce="median"`` returns the de-rotated estimate minimizing summed
    signed-wrap distance to the others; on a noiseless image it equals the base."""
    base = 88.0
    rots = [-30.0, -10.0, 0.0, 20.0, 50.0]
    preds = [base + r for r in rots]  # noiseless -> every base_est == base
    csv = _write_csv(tmp_path, _rotated_image("img0", 1, base, rots, preds))

    out = tta_aggregate(csv, reduce="median")
    pred = out["y_pred"][0]
    # the reduced prediction is the summed-wrap-distance minimizer over candidates
    base_ests = np.array([p - r for p, r in zip(preds, rots)])
    costs = {
        c: float(np.sum(np.abs(_signed_wrap(base_ests - c)))) for c in base_ests
    }
    best = min(costs, key=costs.get)
    assert pred == pytest.approx(best, abs=1e-6)
    assert pred == pytest.approx(base, abs=1e-6)


# --------------------------------------------------------------------------- #
# 5. n_base + shapes  (test 2)
# --------------------------------------------------------------------------- #
def test_n_base_equals_distinct_image_ids_and_shapes(tmp_path):
    """``n_base == #distinct image_id`` and ``len(y_true) == len(y_pred) == n_base``
    (one prediction per base image, not per row)."""
    rng = np.random.default_rng(1)
    rots = [-50.0, -20.0, 0.0, 30.0, 60.0]
    rows = []
    n_images = 7
    for i in range(n_images):
        base = float(rng.uniform(20.0, 150.0))
        preds = base + rng.normal(0.0, 4.0, len(rots))
        rows += _rotated_image(f"img{i:02d}", i % 3, base, rots, preds)
    csv = _write_csv(tmp_path, rows)

    out = tta_aggregate(csv, reduce="circular_mean")
    assert out["n_base"] == n_images
    assert len(out["y_true"]) == n_images
    assert len(out["y_pred"]) == n_images
    # one row per image would be wrong: we have n_images*len(rots) rows
    assert out["n_base"] != len(rows)


# --------------------------------------------------------------------------- #
# 6. TTA reduces error on a noisy synthetic set  (test 5)
# --------------------------------------------------------------------------- #
def test_tta_mae_le_mean_per_rotation_mae(tmp_path):
    """On images whose per-rotation ``base_est`` is the truth plus zero-mean noise,
    the rotation-averaged (TTA) MAE is ``<=`` the mean per-rotation MAE — the
    variance-reduction that is the whole point of TTA."""
    rng = np.random.default_rng(7)
    rots = np.array([-60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0])
    rows = []
    for i in range(40):
        base = float(rng.uniform(40.0, 140.0))
        # de-rotated estimate = base + zero-mean noise; pred = base_est + rot
        base_ests = base + rng.normal(0.0, 8.0, len(rots))
        preds = base_ests + rots
        rows += _rotated_image(f"img{i:02d}", i, base, rots, preds)
    csv = _write_csv(tmp_path, rows)

    out = tta_aggregate(csv, reduce="circular_mean")
    tta_mae = out["metrics"]["mae"]

    # mean per-rotation MAE: de-rotate every row, score raw (no reduction)
    df = pd.read_csv(csv)
    base_est = df["theta_pred"].to_numpy(float) - df["rotation_deg"].to_numpy(float)
    base_true = df["theta_true"].to_numpy(float) - df["rotation_deg"].to_numpy(float)
    per_row_mae = evaluate.metrics(base_true, base_est)["mae"]

    assert tta_mae <= per_row_mae + 1e-9


# --------------------------------------------------------------------------- #
# 7. metrics reuse — single source of truth  (test 6)
# --------------------------------------------------------------------------- #
def test_metrics_dict_equals_evaluate_metrics(tmp_path):
    """The returned ``metrics`` equals ``uda.evaluation.evaluate.metrics(y_true, y_pred)``
    recomputed in the test (keys ``mae, rmse, me, mape, r2``), not a re-implementation."""
    rng = np.random.default_rng(3)
    rots = np.array([-40.0, -10.0, 0.0, 25.0, 55.0])
    rows = []
    for i in range(6):
        base = float(rng.uniform(30.0, 150.0))
        preds = base + rots + rng.normal(0.0, 5.0, len(rots))
        rows += _rotated_image(f"img{i:02d}", i, base, rots, preds)
    csv = _write_csv(tmp_path, rows)

    out = tta_aggregate(csv, reduce="circular_mean")
    expected = evaluate.metrics(
        np.asarray(out["y_true"], dtype=float),
        np.asarray(out["y_pred"], dtype=float),
    )
    assert set(out["metrics"]) == {"mae", "rmse", "me", "mape", "r2"}
    assert out["metrics"] == expected


# --------------------------------------------------------------------------- #
# 8. real OOF csv smoke  (test 7)
# --------------------------------------------------------------------------- #
def test_tta_on_real_oof_csv():
    """End-to-end on the honest OOF predictions (84 base × 25 rotations): finite
    all-around, ``n_base == 84``, and the TTA MAE is ``<=`` the raw per-row MAE."""
    df = pd.read_csv(REAL_OOF_CSV)
    assert df["image_id"].nunique() == 84  # guard the fixture

    out = tta_aggregate(REAL_OOF_CSV, reduce="circular_mean")
    assert set(out) == {"y_true", "y_pred", "metrics", "n_base"}
    assert out["n_base"] == 84
    assert len(out["y_true"]) == 84
    assert len(out["y_pred"]) == 84
    assert all(np.isfinite(v) for v in out["metrics"].values())

    raw_mae = evaluate.metrics(
        df["theta_true"].to_numpy(float), df["theta_pred"].to_numpy(float)
    )["mae"]
    assert out["metrics"]["mae"] <= raw_mae + 1e-9


def test_tta_real_csv_both_reducers_finite():
    """Both reducers run on the real csv and give finite metrics / ``n_base == 84``."""
    for reduce in ("circular_mean", "median"):
        out = tta_aggregate(REAL_OOF_CSV, reduce=reduce)
        assert out["n_base"] == 84
        assert all(np.isfinite(v) for v in out["metrics"].values())


# --------------------------------------------------------------------------- #
# 9. Keras-free fresh-subprocess guard  (test 8)
# --------------------------------------------------------------------------- #
def test_tta_module_is_keras_free():
    """Importing ``uda.evaluation.tta`` must not pull a heavy backend (keras/jax/tensorflow).
    Checked in a FRESH interpreter so other tests' backend imports don't pollute
    ``sys.modules`` (mirror ``tests/test_conformal.py``)."""
    import subprocess

    code = (
        "import uda.evaluation.tta, sys; "
        "bad = [b for b in ('keras', 'jax', 'tensorflow') if b in sys.modules]; "
        "print(','.join(bad)); "
        "sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, (
        f"uda.evaluation.tta pulled a backend: {r.stdout.strip()}\n{r.stderr[-300:]}"
    )

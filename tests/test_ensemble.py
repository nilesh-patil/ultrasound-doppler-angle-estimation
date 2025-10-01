"""Tests for ``uda.evaluation.ensemble`` (post-hoc prediction combiner, Keras-free).

The contract: ``ensemble_predictions`` reads a list of
per-model prediction CSVs (the schema written by ``uda.evaluation.evaluate.evaluate`` —
columns include ``theta_true`` and ``theta_pred``), verifies they share the same
held-out test set (aligned ``theta_true``), and combines their ``theta_pred`` by
either an unweighted ``mean`` or a leakage-free ``stacked`` Ridge meta-learner
(``cross_val_predict``). It returns
``{"y_true", "y_pred", "metrics", "method", "n_models"}`` where ``metrics`` is
exactly ``uda.evaluation.evaluate.metrics(y_true, y_pred)``.

Fixtures are tiny synthetic CSVs in
``tmp_path`` — no model, no training, fast.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uda.evaluation.ensemble import ensemble_predictions
from uda.evaluation.evaluate import metrics

# A fixed held-out test set shared by aligned member CSVs.
THETA_TRUE = np.array([10.0, 25.0, 40.0, 55.0, 70.0, 88.37])


def _write_pred_csv(
    path: Path,
    theta_true: np.ndarray,
    theta_pred: np.ndarray,
    extra: dict[str, object] | None = None,
) -> Path:
    """Write a minimal ``uda.evaluation.evaluate``-style prediction CSV and return its path.

    Only ``theta_true``/``theta_pred`` are contractually read; ``extra`` columns
    (e.g. ``image_id``, ``rotation_deg``) exercise the "ignore extras" invariant.
    """
    data: dict[str, object] = {
        "theta_true": np.asarray(theta_true, dtype=float),
        "theta_pred": np.asarray(theta_pred, dtype=float),
    }
    if extra:
        data.update(extra)
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# mean — value, return shape, metric provenance  (test 1)
# --------------------------------------------------------------------------- #
def test_mean_of_two_csvs_is_columnwise_average(tmp_path):
    pred_a = THETA_TRUE + 2.0
    pred_b = THETA_TRUE - 6.0
    a = _write_pred_csv(tmp_path / "a.csv", THETA_TRUE, pred_a)
    b = _write_pred_csv(tmp_path / "b.csv", THETA_TRUE, pred_b)

    out = ensemble_predictions([a, b], method="mean")

    expected = (pred_a + pred_b) / 2.0
    assert np.allclose(out["y_pred"], expected, atol=1e-9)
    assert np.allclose(out["y_true"], THETA_TRUE, atol=1e-9)


def test_mean_metrics_match_evaluate_metrics_single_source_of_truth(tmp_path):
    pred_a = THETA_TRUE + 2.0
    pred_b = THETA_TRUE - 6.0
    a = _write_pred_csv(tmp_path / "a.csv", THETA_TRUE, pred_a)
    b = _write_pred_csv(tmp_path / "b.csv", THETA_TRUE, pred_b)

    out = ensemble_predictions([a, b], method="mean")

    # metrics must be EXACTLY uda.evaluation.evaluate.metrics over the ensemble prediction,
    # not a re-implementation.
    expected = metrics(THETA_TRUE, (pred_a + pred_b) / 2.0)
    assert out["metrics"] == expected


def test_mean_return_dict_has_contract_shape(tmp_path):
    a = _write_pred_csv(tmp_path / "a.csv", THETA_TRUE, THETA_TRUE + 1.0)
    b = _write_pred_csv(tmp_path / "b.csv", THETA_TRUE, THETA_TRUE - 1.0)

    out = ensemble_predictions([a, b], method="mean")

    assert set(out) == {"y_true", "y_pred", "metrics", "method", "n_models"}
    assert out["method"] == "mean"
    assert out["n_models"] == 2
    assert isinstance(out["y_true"], np.ndarray)
    assert isinstance(out["y_pred"], np.ndarray)
    assert isinstance(out["metrics"], dict)
    # both prediction vectors keep the test-set length
    assert out["y_true"].shape == THETA_TRUE.shape
    assert out["y_pred"].shape == THETA_TRUE.shape


def test_mean_default_method_is_mean(tmp_path):
    a = _write_pred_csv(tmp_path / "a.csv", THETA_TRUE, THETA_TRUE + 1.0)
    b = _write_pred_csv(tmp_path / "b.csv", THETA_TRUE, THETA_TRUE - 1.0)

    out = ensemble_predictions([a, b])  # method omitted -> default "mean"
    assert out["method"] == "mean"


def test_mean_scales_to_n_models(tmp_path):
    # three members; y_pred is the average of all three, n_models == 3.
    preds = [THETA_TRUE + 3.0, THETA_TRUE, THETA_TRUE - 9.0]
    paths = [
        _write_pred_csv(tmp_path / f"m{i}.csv", THETA_TRUE, p)
        for i, p in enumerate(preds)
    ]
    out = ensemble_predictions(paths, method="mean")
    assert out["n_models"] == 3
    assert np.allclose(out["y_pred"], np.mean(preds, axis=0), atol=1e-9)


# --------------------------------------------------------------------------- #
# the averaging value proposition  (test 2)
# --------------------------------------------------------------------------- #
def test_mean_cancels_symmetric_errors_to_zero_mae(tmp_path):
    """Members bracketing the truth (+d / -d) => ensemble MAE is exactly 0."""
    d = 7.5
    a = _write_pred_csv(tmp_path / "hi.csv", THETA_TRUE, THETA_TRUE + d)
    b = _write_pred_csv(tmp_path / "lo.csv", THETA_TRUE, THETA_TRUE - d)

    out = ensemble_predictions([a, b], method="mean")

    # ensemble is exact; each member is off by |d|.
    assert out["metrics"]["mae"] == pytest.approx(0.0, abs=1e-9)
    assert metrics(THETA_TRUE, THETA_TRUE + d)["mae"] == pytest.approx(d, abs=1e-9)
    assert metrics(THETA_TRUE, THETA_TRUE - d)["mae"] == pytest.approx(d, abs=1e-9)


# --------------------------------------------------------------------------- #
# alignment / comparability guard  (test 3)
# --------------------------------------------------------------------------- #
def test_misaligned_theta_true_values_raise_value_error(tmp_path):
    """One differing theta_true row breaks comparability => ValueError."""
    bad_true = THETA_TRUE.copy()
    bad_true[3] += 5.0  # same length, one row differs
    a = _write_pred_csv(tmp_path / "a.csv", THETA_TRUE, THETA_TRUE + 1.0)
    b = _write_pred_csv(tmp_path / "b.csv", bad_true, bad_true + 1.0)

    with pytest.raises(ValueError, match="theta_true"):
        ensemble_predictions([a, b], method="mean")


def test_different_row_counts_raise_value_error(tmp_path):
    a = _write_pred_csv(tmp_path / "a.csv", THETA_TRUE, THETA_TRUE + 1.0)
    short = THETA_TRUE[:-1]
    b = _write_pred_csv(tmp_path / "b.csv", short, short + 1.0)

    with pytest.raises(ValueError):
        ensemble_predictions([a, b], method="mean")


def test_alignment_tolerates_tiny_float_noise(tmp_path):
    """theta_true equal to within 1e-6 (CSV round-trip noise) must NOT raise."""
    jittered = THETA_TRUE + 1e-8  # well under the 1e-6 alignment atol
    a = _write_pred_csv(tmp_path / "a.csv", THETA_TRUE, THETA_TRUE + 1.0)
    b = _write_pred_csv(tmp_path / "b.csv", jittered, jittered - 1.0)

    out = ensemble_predictions([a, b], method="mean")
    assert out["n_models"] == 2


# --------------------------------------------------------------------------- #
# arity guard  (test 4)
# --------------------------------------------------------------------------- #
def test_single_path_raises_value_error(tmp_path):
    a = _write_pred_csv(tmp_path / "only.csv", THETA_TRUE, THETA_TRUE + 1.0)
    with pytest.raises(ValueError):
        ensemble_predictions([a])


@pytest.mark.parametrize("method", ["mean", "stacked"])
def test_single_path_raises_for_both_methods(tmp_path, method):
    a = _write_pred_csv(tmp_path / "only.csv", THETA_TRUE, THETA_TRUE + 1.0)
    with pytest.raises(ValueError):
        ensemble_predictions([a], method=method)


# --------------------------------------------------------------------------- #
# stacked meta-learner — runs, leakage-free by construction, deterministic
# (test 5)
# --------------------------------------------------------------------------- #
def _correlated_members(tmp_path: Path, n: int = 60) -> list[Path]:
    """Three correlated-but-noisy member CSVs over a shared n-row test set."""
    rng = np.random.default_rng(0)
    y = rng.uniform(0.0, 180.0, size=n)
    paths = []
    for i in range(3):
        noise = rng.normal(0.0, 3.0 + i, size=n)
        paths.append(_write_pred_csv(tmp_path / f"member{i}.csv", y, y + noise))
    return paths


def test_stacked_runs_and_returns_finite_predictions(tmp_path):
    paths = _correlated_members(tmp_path)
    out = ensemble_predictions(paths, method="stacked", seed=42, cv_folds=5)

    assert out["method"] == "stacked"
    assert out["n_models"] == 3
    assert out["y_pred"].shape == out["y_true"].shape
    assert out["y_pred"].shape[0] == 60
    assert np.all(np.isfinite(out["y_pred"]))
    # metrics reuse the same evaluate.metrics path; r2 is finite for this case.
    assert np.isfinite(out["metrics"]["r2"])


def test_stacked_metrics_match_evaluate_metrics(tmp_path):
    paths = _correlated_members(tmp_path)
    out = ensemble_predictions(paths, method="stacked", seed=42, cv_folds=5)
    # metrics must be exactly evaluate.metrics(y_true, y_pred) on the OOF preds.
    assert out["metrics"] == metrics(out["y_true"], out["y_pred"])


def test_stacked_is_deterministic_for_a_fixed_seed(tmp_path):
    paths = _correlated_members(tmp_path)
    a = ensemble_predictions(paths, method="stacked", seed=7, cv_folds=5)
    b = ensemble_predictions(paths, method="stacked", seed=7, cv_folds=5)
    # pure function of (P, y_true, seed, cv_folds): bit-equal predictions.
    assert np.array_equal(a["y_pred"], b["y_pred"])
    assert a["metrics"] == b["metrics"]


def test_stacked_seed_controls_the_fold_shuffle(tmp_path):
    """Different seeds drive a different KFold shuffle => OOF preds may differ.

    (We only guarantee predictions *may* differ; we assert the seed is actually wired
    by requiring the two seeds to produce non-identical out-of-fold predictions
    on this noisy 60-row problem.)
    """
    paths = _correlated_members(tmp_path)
    a = ensemble_predictions(paths, method="stacked", seed=1, cv_folds=5)
    b = ensemble_predictions(paths, method="stacked", seed=2, cv_folds=5)
    assert not np.array_equal(a["y_pred"], b["y_pred"])


# --------------------------------------------------------------------------- #
# schema robustness  (test 6)
# --------------------------------------------------------------------------- #
def test_extra_columns_are_ignored(tmp_path):
    """image_id / rotation_deg / split columns must not affect the result."""
    pred_a = THETA_TRUE + 2.0
    pred_b = THETA_TRUE - 6.0
    extra_a = {
        "image_id": [f"img_{i}" for i in range(THETA_TRUE.size)],
        "rotation_deg": np.arange(THETA_TRUE.size) * 5,
        "split": ["test"] * THETA_TRUE.size,
    }
    extra_b = {
        "patient_id": [f"p{i}" for i in range(THETA_TRUE.size)],
        "era": ["v1.1"] * THETA_TRUE.size,
    }
    a = _write_pred_csv(tmp_path / "a.csv", THETA_TRUE, pred_a, extra=extra_a)
    b = _write_pred_csv(tmp_path / "b.csv", THETA_TRUE, pred_b, extra=extra_b)

    out = ensemble_predictions([a, b], method="mean")

    # identical to the no-extras mean.
    assert np.allclose(out["y_pred"], (pred_a + pred_b) / 2.0, atol=1e-9)
    assert out["n_models"] == 2


def test_column_order_does_not_matter(tmp_path):
    """theta_true/theta_pred are read by name, not position."""
    pred_a = THETA_TRUE + 4.0
    # write with columns deliberately reordered / interleaved with extras.
    df = pd.DataFrame(
        {
            "rotation_deg": np.zeros(THETA_TRUE.size),
            "theta_pred": pred_a,
            "image_id": np.arange(THETA_TRUE.size),
            "theta_true": THETA_TRUE,
        }
    )
    a = tmp_path / "a.csv"
    df.to_csv(a, index=False)
    b = _write_pred_csv(tmp_path / "b.csv", THETA_TRUE, THETA_TRUE - 4.0)

    out = ensemble_predictions([a, b], method="mean")
    assert np.allclose(out["y_pred"], THETA_TRUE, atol=1e-9)


# --------------------------------------------------------------------------- #
# input-type flexibility — the signature accepts str | Path
# --------------------------------------------------------------------------- #
def test_accepts_str_paths(tmp_path):
    a = _write_pred_csv(tmp_path / "a.csv", THETA_TRUE, THETA_TRUE + 1.0)
    b = _write_pred_csv(tmp_path / "b.csv", THETA_TRUE, THETA_TRUE - 1.0)
    # pass plain strings rather than Path objects.
    out = ensemble_predictions([str(a), str(b)], method="mean")
    assert out["n_models"] == 2
    assert np.allclose(out["y_pred"], THETA_TRUE, atol=1e-9)

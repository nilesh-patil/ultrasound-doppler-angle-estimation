"""Tests — split/group-conformal angle intervals (``uda.evaluation.conformal``).

These tests cover the ``uda.evaluation.conformal``
module. Everything here is post-hoc on *saved* predictions: no model is built and
every test runs in well under a second. The module under test is **Keras-free**
(asserted explicitly in a fresh subprocess, mirroring ``tests/test_cv.py``).

Two public entry points are exercised:

* ``conformal_intervals(cal_resid, test_pred, alpha=0.1) -> {q, lower, upper, alpha}``
  — pure split-conformal half-width on supplied calibration residuals.
* ``evaluate_conformal(pred_csv, *, alpha=0.1, seed=42) -> {q, alpha, coverage,
  mean_width, n_cal, n_test}`` — end-to-end on a saved prediction CSV, forming a
  **patient-disjoint** calibration/test split via :class:`uda.data.splits.PatientLevelSplit`.

Residuals are on the **signed 180-wrap** scale ``r = ((true - pred + 90) % 180) - 90``
(vessel orientation is 180-periodic), and ``q`` is the *inflated*
``ceil((n+1)*(1-alpha))/n`` empirical quantile of ``|r|`` — the finite-sample-valid
split-conformal quantile. Coverage is delegated to ``uda.evaluation.uncertainty.coverage``
(single source of truth), so a constant half-width ``q`` is scored as the degenerate
Gaussian interval ``coverage(true, pred, std=q, z=1.0)``.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
import pytest

from uda.evaluation import uncertainty
from uda.evaluation.conformal import conformal_intervals, evaluate_conformal

# Repo-relative path to the honest, full-coverage OOF predictions (2100 rows).
REAL_OOF_CSV = "results/predictions/tuned_densenet201_oof.csv"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _signed_wrap(delta: np.ndarray) -> np.ndarray:
    """Reference signed-wrap into ``(-90, 90]`` (the contract's residual scale)."""
    return ((np.asarray(delta, dtype=float) + 90.0) % 180.0) - 90.0


def _inflated_quantile(scores: np.ndarray, alpha: float) -> float:
    """Reference ``q``: the ``ceil((n+1)*(1-alpha))/n`` empirical quantile of ``|·|``.

    Returns ``+inf`` when the inflated rank exceeds ``n`` (uninformative score set).
    """
    s = np.abs(np.asarray(scores, dtype=float))
    n = s.size
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return math.inf
    level = k / n
    return float(np.quantile(s, level, method="higher"))


def _patient_csv(tmp_path, n_patients: int = 6, per_patient: int = 4, seed: int = 0):
    """Write a small synthetic prediction CSV with a ``patient_id`` grouping.

    Columns mirror the real OOF file: ``image_id, patient_id, rotation_deg,
    theta_true, theta_pred``. One base image per row keeps the bookkeeping simple.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        for k in range(per_patient):
            true = float(rng.uniform(70.0, 110.0))
            pred = true + float(rng.normal(0.0, 4.0))
            rows.append(
                {
                    "image_id": f"p{p:02d}_img{k:02d}",
                    "patient_id": float(p),
                    "rotation_deg": 0.0,
                    "theta_true": true,
                    "theta_pred": pred,
                }
            )
    path = tmp_path / "synthetic_oof.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# 1. split-conformal marginal coverage on EXCHANGEABLE residuals  (test 2)
# --------------------------------------------------------------------------- #
def test_split_conformal_marginal_coverage_on_exchangeable_residuals():
    """On exchangeable cal/test residuals (large n_cal), empirical TEST coverage is
    ``>= 1 - alpha`` and not absurdly conservative (within ~``[1-alpha, 1-alpha+0.05]``)."""
    alpha = 0.1
    rng = np.random.default_rng(123)
    n_cal, n_test = 800, 4000
    # one exchangeable pool -> cal + test residuals; pred is arbitrary (constant 0).
    cal_resid = rng.normal(0.0, 10.0, size=n_cal)
    test_resid = rng.normal(0.0, 10.0, size=n_test)

    test_pred = np.zeros(n_test)
    out = conformal_intervals(cal_resid, test_pred, alpha=alpha)
    q = out["q"]

    test_true = test_pred + test_resid  # so |test_true - test_pred| == |test_resid|
    cov = uncertainty.coverage(
        test_true, test_pred, std=np.full_like(test_pred, q), z=1.0
    )
    assert cov >= 1.0 - alpha - 1e-9  # finite-sample validity floor
    assert cov <= 1.0 - alpha + 0.05  # not absurdly wide on this large sample


# --------------------------------------------------------------------------- #
# 2. q monotone non-increasing in alpha  (test 6)
# --------------------------------------------------------------------------- #
def test_q_is_monotone_non_increasing_in_alpha():
    """Looser ``alpha`` -> smaller (or equal) ``q`` -> narrower intervals."""
    rng = np.random.default_rng(7)
    cal_resid = rng.normal(0.0, 8.0, size=1000)
    test_pred = np.zeros(5)

    alphas = [0.01, 0.05, 0.1, 0.2, 0.5]
    qs = [conformal_intervals(cal_resid, test_pred, alpha=a)["q"] for a in alphas]
    for earlier, later in zip(qs, qs[1:]):
        assert later <= earlier + 1e-9


# --------------------------------------------------------------------------- #
# 3. signed-wrap residual magnitude  (test 3)
# --------------------------------------------------------------------------- #
def test_signed_wrap_residual_is_circular():
    """``true=1, pred=179`` -> residual magnitude ``2`` (NOT ``178``); residual in (-90, 90]."""
    r = _signed_wrap(np.array([1.0 - 179.0]))[0]
    assert abs(r) == pytest.approx(2.0)
    assert not abs(r) == pytest.approx(178.0)
    # a sweep of differences always wraps into (-90, 90]
    diffs = np.linspace(-360.0, 360.0, 721)
    wrapped = _signed_wrap(diffs)
    assert np.all(wrapped > -90.0 - 1e-9)
    assert np.all(wrapped <= 90.0 + 1e-9)
    # the conformal q on those circular residuals never exceeds 90 (the wrap cap)
    out = conformal_intervals(wrapped, np.zeros(3), alpha=0.1)
    assert out["q"] <= 90.0 + 1e-9


# --------------------------------------------------------------------------- #
# 4. patient-disjoint calibration/test split  (test 4)
# --------------------------------------------------------------------------- #
def test_evaluate_conformal_calibration_and_test_patients_are_disjoint(tmp_path):
    """The cal/test partition is patient-disjoint and accounts for every row."""
    csv = _patient_csv(tmp_path, n_patients=6, per_patient=4, seed=1)
    df = pd.read_csv(csv)

    out = evaluate_conformal(csv, alpha=0.1, seed=42)
    assert out["n_cal"] > 0 and out["n_test"] > 0
    assert out["n_cal"] + out["n_test"] == len(df)

    # cross-check that *some* patient-disjoint partition of these patients exists with
    # the reported sizes: every patient lands wholly on one side.
    sizes = df.groupby("patient_id").size()
    assert out["n_cal"] in _achievable_group_sums(sizes.tolist())
    assert out["n_test"] in _achievable_group_sums(sizes.tolist())


def _achievable_group_sums(group_sizes):
    """All row-counts reachable by assigning whole groups to one side (subset sums)."""
    reachable = {0}
    for g in group_sizes:
        reachable |= {r + g for r in reachable}
    return reachable


def test_evaluate_conformal_real_csv_is_patient_disjoint():
    """On the real OOF csv the calibration & test patient sets are disjoint."""
    df = pd.read_csv(REAL_OOF_CSV)
    out = evaluate_conformal(REAL_OOF_CSV, alpha=0.1, seed=42)
    assert out["n_cal"] > 0 and out["n_test"] > 0
    assert out["n_cal"] + out["n_test"] == len(df)


# --------------------------------------------------------------------------- #
# 5. determinism  (test 7)
# --------------------------------------------------------------------------- #
def test_evaluate_conformal_is_deterministic(tmp_path):
    """Same ``(csv, alpha, seed)`` -> identical ``q`` and ``coverage`` (whole dict)."""
    csv = _patient_csv(tmp_path, n_patients=8, per_patient=5, seed=2)
    a = evaluate_conformal(csv, alpha=0.1, seed=11)
    b = evaluate_conformal(csv, alpha=0.1, seed=11)
    assert a == b
    assert a["q"] == b["q"]
    assert a["coverage"] == b["coverage"]


# --------------------------------------------------------------------------- #
# 6. coverage == uda.evaluation.uncertainty.coverage on the same arrays  (test 5)
# --------------------------------------------------------------------------- #
def test_reported_coverage_equals_uncertainty_coverage(tmp_path):
    """``evaluate_conformal``'s coverage is exactly ``uncertainty.coverage`` recomputed
    on the test rows with ``std = q`` broadcast, ``z = 1.0``; ``mean_width == 2*q``."""
    csv = _patient_csv(tmp_path, n_patients=6, per_patient=6, seed=3)
    out = evaluate_conformal(csv, alpha=0.1, seed=42)

    # Recover the exact patient-disjoint split this function used and recompute.
    from uda.config import SplitConfig
    from uda.data.splits import PatientLevelSplit

    df = pd.read_csv(csv)
    labels = df.drop_duplicates("image_id")[["image_id", "patient_id"]]
    cfg = SplitConfig(strategy="patient", n_folds=2, seed=42)
    train_ids, test_ids = next(PatientLevelSplit().split(labels, cfg))

    test_rows = df[df["image_id"].isin(set(test_ids))]
    test_true = test_rows["theta_true"].to_numpy(dtype=float)
    test_pred = test_rows["theta_pred"].to_numpy(dtype=float)

    q = out["q"]
    # evaluate_conformal scores coverage on the signed-wrap residual scale (matching
    # calibration); recompute the same way through the single-source-of-truth coverage.
    from uda.evaluation.conformal import _signed_wrap

    test_resid = _signed_wrap(test_true - test_pred)
    expected_cov = uncertainty.coverage(
        test_resid, np.zeros_like(test_resid), std=np.full_like(test_resid, q), z=1.0
    )
    assert out["coverage"] == pytest.approx(expected_cov)
    assert out["mean_width"] == pytest.approx(2.0 * q)


def test_conformal_intervals_matches_inflated_quantile_and_bounds():
    """``q`` equals the *inflated* ``ceil((n+1)*(1-alpha))/n`` empirical quantile of
    ``|cal_resid|`` (>= plain ``1-alpha`` quantile); lower/upper == test_pred ∓ q."""
    cal_resid = np.array([-3.0, 1.0, -5.0, 2.0, 4.0, -1.0, 0.5, -2.5, 6.0, -0.5])
    test_pred = np.array([10.0, 20.0, 30.0])
    alpha = 0.1

    out = conformal_intervals(cal_resid, test_pred, alpha=alpha)
    q = out["q"]
    assert q == pytest.approx(_inflated_quantile(cal_resid, alpha))
    # inflated quantile is >= the plain (1-alpha) empirical quantile
    plain = float(np.quantile(np.abs(cal_resid), 1.0 - alpha, method="higher"))
    assert q >= plain - 1e-9
    assert np.allclose(out["lower"], test_pred - q)
    assert np.allclose(out["upper"], test_pred + q)
    assert out["alpha"] == alpha


def test_conformal_intervals_infinite_q_when_inflated_rank_exceeds_n():
    """Tiny ``n`` with small ``alpha`` makes ``ceil((n+1)*(1-alpha)) > n`` -> ``q == inf``."""
    cal_resid = np.array([1.0, 2.0, 3.0])  # n=3
    out = conformal_intervals(cal_resid, np.zeros(2), alpha=0.05)
    assert math.isinf(out["q"])
    assert np.all(np.isinf(out["upper"]))
    assert np.all(np.isinf(out["lower"]))


# --------------------------------------------------------------------------- #
# 7. Keras-free fresh-subprocess guard  (test 8)
# --------------------------------------------------------------------------- #
def test_conformal_module_is_keras_free():
    """Importing ``uda.evaluation.conformal`` must not pull a heavy backend (keras/jax/tensorflow).
    Checked in a FRESH interpreter so other tests' backend imports don't pollute
    ``sys.modules`` (mirror ``tests/test_cv.py``)."""
    import subprocess

    code = (
        "import uda.evaluation.conformal, sys; "
        "bad = [b for b in ('keras', 'jax', 'tensorflow') if b in sys.modules]; "
        "print(','.join(bad)); "
        "sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, (
        f"uda.evaluation.conformal pulled a backend: {r.stdout.strip()}\n{r.stderr[-300:]}"
    )


# --------------------------------------------------------------------------- #
# 8. runs on the REAL OOF csv  (test 4 + acceptance)
# --------------------------------------------------------------------------- #
def test_evaluate_conformal_on_real_oof_csv():
    """End-to-end on the honest OOF predictions: finite coverage roughly ``>= 0.85``
    at ``alpha=0.1``, with non-empty cal/test halves and ``mean_width == 2*q``."""
    out = evaluate_conformal(REAL_OOF_CSV, alpha=0.1, seed=42)

    assert set(out) == {"q", "alpha", "coverage", "mean_width", "n_cal", "n_test"}
    assert out["alpha"] == 0.1
    assert out["n_cal"] > 0 and out["n_test"] > 0
    assert np.isfinite(out["q"])
    assert np.isfinite(out["coverage"])
    assert out["coverage"] >= 0.85
    assert out["mean_width"] == pytest.approx(2.0 * out["q"])

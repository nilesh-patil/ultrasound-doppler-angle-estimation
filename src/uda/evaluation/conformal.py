"""Split / group-conformal calibrated angle intervals (Keras-free).

This module turns *saved* predictions into
finite-sample-valid prediction intervals, entirely post-hoc — no model is built
and no deep-learning backend (``keras``/``jax``/``tensorflow``) is imported, only
numpy, pandas, and the leakage-free patient splitter.

Two public entry points:

* :func:`conformal_intervals` — pure split-conformal half-width on supplied
  calibration residuals. Given calibration residuals ``cal_resid`` and a vector
  of test point-predictions ``test_pred``, it returns the constant half-width
  ``q`` and the bounds ``test_pred ∓ q``.
* :func:`evaluate_conformal` — end-to-end on a saved prediction CSV. It forms a
  **patient-disjoint** calibration/test partition via
  :class:`uda.data.splits.PatientLevelSplit` (``GroupKFold`` over ``patient_id``,
  ``n_folds=2``), computes calibration residuals on the **signed 180-wrap** scale,
  derives ``q`` on the calibration half, and scores empirical coverage on the
  held-out test half.

Residual scale (critical): vessel orientation is **180-periodic**, so a residual
is the *signed wrap* ``r = ((true - pred + 90) % 180) - 90`` into ``(-90, 90]``.
This makes ``pred=179`` vs ``true=1`` a residual of magnitude ``2`` (not ``178``).
Conformal half-widths therefore live on this wrapped scale and never exceed ``90``.

Quantile (critical): ``q`` is the *inflated* ``ceil((n+1)*(1-alpha))/n`` empirical
quantile of ``|r|`` — the finite-sample-valid split-conformal quantile (not the
plain ``1-alpha`` quantile). When the inflated rank exceeds ``n`` (too few
calibration points for the requested ``alpha``) the score set is uninformative and
``q`` is ``+inf`` (the only honest, valid answer).

Coverage is delegated to :func:`uda.evaluation.uncertainty.coverage` — the single source of
truth — by scoring the constant half-width ``q`` as the degenerate Gaussian
interval ``coverage(true, pred, std=q, z=1.0)`` i.e. ``pred ∓ q``.

Honesty: the calibration and test halves are **patient-disjoint** (group
conformal), so no patient leaks between calibration and test.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from uda.config import SplitConfig
from uda.data.splits import PatientLevelSplit

__all__ = ["conformal_intervals", "evaluate_conformal"]

_IMAGE_ID = "image_id"
_PATIENT_ID = "patient_id"
_THETA_TRUE = "theta_true"
_THETA_PRED = "theta_pred"

# Orientation is 180-periodic; residuals wrap into the half-open interval (-90, 90].
_PERIOD = 180.0
_HALF_PERIOD = 90.0


def _signed_wrap(delta: np.ndarray) -> np.ndarray:
    """Signed wrap of an angle difference into ``(-90, 90]``.

    Vessel orientation is 180-periodic, so the meaningful residual between a true
    and predicted angle is the smallest signed rotation between them. Computed as
    ``((delta + 90) % 180) - 90``, which maps e.g. ``delta = 1 - 179 = -178`` to
    ``+2`` — magnitude ``2``, not ``178``.

    Parameters
    ----------
    delta : numpy.ndarray
        Raw angle differences (degrees), typically ``theta_true - theta_pred``.

    Returns
    -------
    numpy.ndarray
        The wrapped residuals as ``float``, every element in ``(-90, 90]``.
    """
    return ((np.asarray(delta, dtype=float) + _HALF_PERIOD) % _PERIOD) - _HALF_PERIOD


def conformal_intervals(
    cal_resid: np.ndarray, test_pred: np.ndarray, alpha: float = 0.1
) -> dict:
    """Split-conformal constant half-width and the resulting interval bounds.

    The nonconformity score is the absolute calibration residual ``|cal_resid|``.
    The half-width ``q`` is the **inflated** ``ceil((n+1)*(1-alpha))/n`` empirical
    quantile of those scores — the finite-sample-valid split-conformal quantile,
    which is ``>=`` the plain ``1-alpha`` quantile. Under exchangeability of the
    calibration and test residuals this guarantees marginal test coverage
    ``>= 1 - alpha``.

    When the inflated rank ``k = ceil((n+1)*(1-alpha))`` exceeds ``n`` (too few
    calibration points for the requested ``alpha``) the empirical score set cannot
    supply a valid quantile and ``q`` is ``+inf`` — the only choice that preserves
    the coverage guarantee (the interval becomes the whole line).

    Parameters
    ----------
    cal_resid : numpy.ndarray
        Calibration residuals (any sign). Only their magnitude is used. On the
        angle task these are the signed-wrapped residuals in ``(-90, 90]``, so
        ``q`` never exceeds ``90``.
    test_pred : numpy.ndarray
        Test point predictions to center the intervals on.
    alpha : float, optional
        Miscoverage level in ``(0, 1)`` (default ``0.1`` -> ~90% intervals).
        Larger ``alpha`` -> smaller (or equal) ``q`` -> narrower intervals.

    Returns
    -------
    dict
        ``{"q": float, "lower": numpy.ndarray, "upper": numpy.ndarray,
        "alpha": float}`` where ``lower = test_pred - q`` and
        ``upper = test_pred + q`` (both ``+/-inf`` when ``q`` is ``+inf``).
    """
    scores = np.abs(np.asarray(cal_resid, dtype=float))
    n = scores.size
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        q = math.inf
    else:
        level = k / n
        q = float(np.quantile(scores, level, method="higher"))

    test_pred = np.asarray(test_pred, dtype=float)
    return {
        "q": q,
        "lower": test_pred - q,
        "upper": test_pred + q,
        "alpha": alpha,
    }


def evaluate_conformal(pred_csv, *, alpha: float = 0.1, seed: int = 42) -> dict:
    """End-to-end split-conformal evaluation on a saved prediction CSV.

    The CSV must have the OOF schema ``image_id, patient_id, theta_true,
    theta_pred`` (extra columns are ignored). A **patient-disjoint** calibration/
    test partition is formed by taking the first fold yielded by
    :class:`uda.data.splits.PatientLevelSplit` with ``SplitConfig(strategy=
    "patient", n_folds=2, seed=seed)``: its ``train`` base ``image_id`` values are
    the *calibration* patients and its ``test`` values are the held-out *test*
    patients. Because the splitter groups by ``patient_id``, no patient — and
    therefore no image — spans both halves.

    Calibration residuals are the signed-wrapped ``theta_true - theta_pred`` on the
    calibration rows; ``q`` comes from :func:`conformal_intervals`. Test coverage is
    scored on the **same signed-wrap scale** (the angle is 180-periodic, plan
    pitfall #2) through :func:`uda.evaluation.uncertainty.coverage` (single source of truth):
    a test point is covered iff its wrapped residual lies in ``[-q, q]`` — identical
    to ``test_pred ∓ q`` unless a prediction straddles the 0/180 seam.

    Parameters
    ----------
    pred_csv : str or pathlib.Path
        Path to a saved prediction CSV (e.g. an OOF predictions file).
    alpha : float, keyword-only, optional
        Miscoverage level in ``(0, 1)`` (default ``0.1``).
    seed : int, keyword-only, optional
        Seed for the patient-disjoint split (default ``42``). The partition is a
        pure function of ``(csv contents, seed)``, so the whole result is
        reproducible for fixed ``(csv, alpha, seed)``.

    Returns
    -------
    dict
        ``{"q": float, "alpha": float, "coverage": float, "mean_width": float,
        "n_cal": int, "n_test": int}`` where ``mean_width == 2 * q`` (the constant
        interval width) and ``n_cal + n_test`` equals the number of CSV rows.
    """
    df = pd.read_csv(pred_csv)

    # One row per base image carries the patient grouping fed to the splitter.
    labels = df.drop_duplicates(subset=_IMAGE_ID)[[_IMAGE_ID, _PATIENT_ID]]
    cfg = SplitConfig(strategy="patient", n_folds=2, seed=seed)
    cal_ids, test_ids = next(PatientLevelSplit().split(labels, cfg))

    cal_rows = df[df[_IMAGE_ID].isin(set(cal_ids))]
    test_rows = df[df[_IMAGE_ID].isin(set(test_ids))]

    cal_resid = _signed_wrap(
        cal_rows[_THETA_TRUE].to_numpy(dtype=float)
        - cal_rows[_THETA_PRED].to_numpy(dtype=float)
    )
    test_true = test_rows[_THETA_TRUE].to_numpy(dtype=float)
    test_pred = test_rows[_THETA_PRED].to_numpy(dtype=float)

    out = conformal_intervals(cal_resid, test_pred, alpha=alpha)
    q = out["q"]

    # Single source of truth for coverage. Imported lazily *here* rather than at
    # module top-level because ``uda.evaluation.uncertainty`` imports ``keras`` for its
    # MC-dropout helpers; deferring the import keeps ``import uda.evaluation.conformal``
    # backend-free (the Keras-free guard). Score on the signed-wrap scale (matching
    # calibration): covered iff the wrapped residual is in [-q, q], expressed as the
    # interval ``0 ∓ q`` around the wrapped residual.
    from uda.evaluation import uncertainty

    test_resid = _signed_wrap(test_true - test_pred)
    cov = uncertainty.coverage(
        test_resid, np.zeros_like(test_resid), std=np.full_like(test_resid, q), z=1.0
    )

    return {
        "q": q,
        "alpha": alpha,
        "coverage": cov,
        "mean_width": 2.0 * q,
        "n_cal": int(len(cal_rows)),
        "n_test": int(len(test_rows)),
    }

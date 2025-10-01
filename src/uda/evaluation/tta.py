"""Rotation test-time augmentation on saved predictions (Keras-free).

This module turns *saved* rotation-augmented
predictions into one de-rotated, circularly-reduced estimate per base image,
entirely post-hoc — no model is built and no deep-learning backend
(``keras``/``jax``/``tensorflow``) is imported, only numpy, pandas, and the
single-source-of-truth metrics function :func:`uda.evaluation.evaluate.metrics`.

One public entry point:

* :func:`tta_aggregate` — reads a saved prediction CSV with the OOF schema
  ``image_id, patient_id, rotation_deg, theta_true, theta_pred``, **de-rotates**
  every row (``base_est = theta_pred - rotation_deg``,
  ``base_true = theta_true - rotation_deg``), **reduces** the de-rotated
  estimates of each ``image_id`` *circularly* to a single prediction, and
  reports :func:`uda.evaluation.evaluate.metrics` over the ``n_base`` reduced estimates.

De-rotation (critical): each image is the same vessel rotated by
``rotation_deg``, so the model's prediction in the rotated frame minus the
rotation recovers an estimate of the *base* angle. Because ``theta_true =
base + rotation_deg``, the de-rotated truth ``base_true`` is a single constant
per ``image_id`` — the base orientation.

Reduction scale (critical): vessel orientation is **180-periodic**, so the
de-rotated estimates of one image are averaged with a circular mean in
**double-angle** space ``0.5*atan2(mean(sin 2θ), mean(cos 2θ))`` mapped into
``[0, 180)`` (``reduce="circular_mean"``), or with a **circular median** that
returns the candidate minimizing the summed signed-wrap distance
``Σ |((θ - c + 90) % 180) - 90|`` (``reduce="median"``). A naive linear mean
would corrupt any image whose de-rotated estimates straddle the 0/180 seam
(e.g. ``{1, 179}`` averages to ``90`` linearly but to the ``0/180`` boundary
circularly). This circular reduction is the whole point of rotation TTA: it
averages out per-rotation noise without crossing the seam.

``metrics`` is delegated to :func:`uda.evaluation.evaluate.metrics` (single source of
truth), recomputed on the ``n_base`` reduced ``(base_true, reduced_pred)`` pairs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from uda.evaluation import evaluate

__all__ = ["tta_aggregate"]

_IMAGE_ID = "image_id"
_ROTATION_DEG = "rotation_deg"
_THETA_TRUE = "theta_true"
_THETA_PRED = "theta_pred"

# Orientation is 180-periodic; differences wrap into (-90, 90], averages live in [0, 180).
_PERIOD = 180.0
_HALF_PERIOD = 90.0


def _signed_wrap(delta: np.ndarray) -> np.ndarray:
    """Signed wrap of an angle difference into ``(-90, 90]``.

    Vessel orientation is 180-periodic, so the meaningful difference between two
    angles is the smallest signed rotation between them. Computed as
    ``((delta + 90) % 180) - 90``, mapping e.g. ``1 - 179 = -178`` to ``+2``
    (magnitude ``2``, not ``178``).
    """
    return ((np.asarray(delta, dtype=float) + _HALF_PERIOD) % _PERIOD) - _HALF_PERIOD


def _circular_mean(angles_deg: np.ndarray) -> float:
    """Double-angle circular mean of 180-periodic angles, mapped into ``[0, 180)``.

    Averaging in double-angle space ``2θ`` and halving back respects the 0/180
    seam, so e.g. ``{1, 179}`` reduces to the boundary rather than the linear
    mean ``90``.
    """
    t = np.deg2rad(2.0 * np.asarray(angles_deg, dtype=float))
    mean = 0.5 * np.rad2deg(np.arctan2(np.mean(np.sin(t)), np.mean(np.cos(t))))
    return float(mean % _PERIOD)


def _circular_median(angles_deg: np.ndarray) -> float:
    """Circular median: the candidate angle minimizing summed signed-wrap distance.

    Returns the element of ``angles_deg`` with the smallest
    ``Σ_j |signed_wrap(angles_deg_j - c)|`` — the 180-periodic analogue of the
    L1 median, robust to seam-straddling estimates.
    """
    a = np.asarray(angles_deg, dtype=float)
    costs = [float(np.sum(np.abs(_signed_wrap(a - c)))) for c in a]
    return float(a[int(np.argmin(costs))])


_REDUCERS = {
    "circular_mean": _circular_mean,
    "median": _circular_median,
}


def tta_aggregate(pred_csv, *, reduce: str = "circular_mean") -> dict:
    """De-rotate and circularly reduce rotation-augmented predictions per image.

    The CSV must carry the OOF schema ``image_id, patient_id, rotation_deg,
    theta_true, theta_pred`` (extra columns ignored). Every row is de-rotated to
    the base frame — ``base_est = theta_pred - rotation_deg`` and
    ``base_true = theta_true - rotation_deg`` — and the de-rotated estimates of
    each distinct ``image_id`` are reduced **circularly** to one prediction.

    Because ``theta_true = base + rotation_deg``, ``base_true`` is a single
    constant per image (the base orientation), so each image contributes exactly
    one ``(y_true, y_pred)`` pair regardless of how many rotations it has.

    Parameters
    ----------
    pred_csv : str or pathlib.Path
        Path to a saved rotation-augmented prediction CSV.
    reduce : {"circular_mean", "median"}, keyword-only, optional
        Circular reduction over each image's de-rotated estimates (default
        ``"circular_mean"``). ``"circular_mean"`` is the double-angle circular
        mean ``0.5*atan2(mean(sin 2θ), mean(cos 2θ))``; ``"median"`` is the
        summed-signed-wrap-distance minimizer. Both honour the 180-periodicity
        of vessel orientation and never cross the 0/180 seam.

    Returns
    -------
    dict
        ``{"y_true": numpy.ndarray, "y_pred": numpy.ndarray, "metrics": dict,
        "n_base": int}`` where ``y_true``/``y_pred`` have length ``n_base`` (the
        number of distinct ``image_id`` values), ``metrics`` is
        :func:`uda.evaluation.evaluate.metrics` over those reduced pairs (keys
        ``mae, rmse, me, mape, r2``), and ``n_base == len(y_true) == len(y_pred)``.
    """
    if reduce not in _REDUCERS:
        raise ValueError(
            f"reduce must be one of {sorted(_REDUCERS)}; got {reduce!r}"
        )
    reducer = _REDUCERS[reduce]

    df = pd.read_csv(pred_csv)
    base_est = (
        df[_THETA_PRED].to_numpy(dtype=float)
        - df[_ROTATION_DEG].to_numpy(dtype=float)
    )
    base_true = (
        df[_THETA_TRUE].to_numpy(dtype=float)
        - df[_ROTATION_DEG].to_numpy(dtype=float)
    )

    # One row per distinct image, source order preserved (no sort).
    image_ids = df[_IMAGE_ID].to_numpy()
    _, first_idx = np.unique(image_ids, return_index=True)
    distinct = image_ids[np.sort(first_idx)]

    y_true = np.empty(distinct.size, dtype=float)
    y_pred = np.empty(distinct.size, dtype=float)
    for i, img in enumerate(distinct):
        mask = image_ids == img
        # base_true is constant per image; the mean is exact and seam-free.
        y_true[i] = float(np.mean(base_true[mask]))
        y_pred[i] = reducer(base_est[mask])

    n_base = int(distinct.size)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "metrics": evaluate.metrics(y_true, y_pred),
        "n_base": n_base,
    }

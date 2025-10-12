"""Circular fusion of per-image angle estimates (Keras-free).

This module blends two or more *saved* per-image
angle estimates (e.g. a learned model estimate and a hand-crafted geometric one)
into a single fused estimate, entirely post-hoc — no model is built and no
deep-learning backend (``keras``/``jax``/``tensorflow``) is imported, only numpy
and the single-source-of-truth metrics function :func:`uda.evaluation.evaluate.metrics`.
Because :mod:`uda.interpret.fusion` takes *arrays* of estimates it never imports
:mod:`uda.interpret.geometric` (which does image I/O).

Two public entry points:

* :func:`circular_fuse` — the **double-angle weighted circular mean** of ``K``
  per-image estimates, mapped into ``[0, 180)``.
* :func:`evaluate_fusion` — scores the learned, geometric, and fused estimates
  against a common truth via :func:`uda.evaluation.evaluate.metrics`, always surfacing
  ``metrics_learned`` so a weak prior fused with a strong learned estimate is
  never *assumed* to win.

Blend scale (critical): vessel orientation is **180-periodic**, so estimates are
combined with the weighted circular mean in **double-angle** (``2θ``) space
``0.5*atan2(Σ w sin 2θ, Σ w cos 2θ)`` mapped into ``[0, 180)``. A naive linear
``w·a + (1−w)·b`` would corrupt any pair straddling the 0/180 seam: fusing
``179°`` and ``1°`` must return the 0/180 boundary (the circular bisector), not
the visibly-wrong linear ``90°``.

Honesty contract: :func:`evaluate_fusion` reports all three single-
and multi-source metric dicts side by side. Fusion is never assumed to help — a
near-truth learned estimate fused with noisy geometry yields a fused MAE that
sits *between* the two single-source MAEs, never beating the better one.
"""
from __future__ import annotations

import numpy as np

from uda.evaluation import evaluate

__all__ = ["circular_fuse", "evaluate_fusion"]


def _as_kn_array(estimates) -> np.ndarray:
    """Coerce ``estimates`` to a ``(K, N)`` float array.

    Accepts either a length-``K`` sequence of length-``N`` 1-D arrays or an
    already-stacked ``(K, N)`` array. Rows of mismatched length ``N`` cannot be
    aligned and raise :class:`ValueError`.
    """
    if isinstance(estimates, np.ndarray):
        arr = np.asarray(estimates, dtype=float)
        if arr.ndim != 2:
            raise ValueError(
                f"estimates must be a (K, N) array of per-image angles, got "
                f"shape {arr.shape}"
            )
        return arr

    rows = [np.asarray(row, dtype=float).ravel() for row in estimates]
    if not rows:
        raise ValueError("estimates must contain at least one row")
    lengths = {row.shape[0] for row in rows}
    if len(lengths) != 1:
        raise ValueError(
            f"estimate rows must all share length N, got lengths {sorted(lengths)}"
        )
    return np.stack(rows)


def circular_fuse(estimates, *, weights=None) -> np.ndarray:
    """Double-angle weighted circular mean of ``K`` per-image angle estimates.

    Parameters
    ----------
    estimates
        Either a ``(K, N)`` array or a length-``K`` sequence of length-``N``
        1-D arrays of angles in degrees (each a member estimate of the same
        ``N`` images).
    weights
        Optional length-``K`` non-negative weights. ``None`` (default) is the
        uniform ``1/K`` blend. Weights are normalized to sum to ``1`` (so the
        result is scale-invariant); concentrating all weight on member ``k``
        returns that member.

    Returns
    -------
    np.ndarray
        Length-``N`` fused estimate, every value in ``[0, 180)``.
    """
    est = _as_kn_array(estimates)
    k = est.shape[0]

    if weights is None:
        w = np.ones(k, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape[0] != k:
            raise ValueError(
                f"weights must have length K={k}, got {w.shape[0]}"
            )
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("weights must sum to a positive, finite value")
    w = w / total

    # weighted circular mean in double-angle (2θ) space, mapped into [0, 180)
    t = np.deg2rad(2.0 * est)  # (K, N)
    s = np.tensordot(w, np.sin(t), axes=(0, 0))  # (N,)
    c = np.tensordot(w, np.cos(t), axes=(0, 0))  # (N,)
    out = 0.5 * np.rad2deg(np.arctan2(s, c))
    return out % 180.0


def evaluate_fusion(y_true, learned, geometric, *, weights=None) -> dict:
    """Score learned, geometric, and fused estimates against a common truth.

    The fused estimate is :func:`circular_fuse` of the stacked
    ``[learned, geometric]`` members (uniform weights by default). All three
    estimates are scored by the single-source-of-truth
    :func:`uda.evaluation.evaluate.metrics`.

    Returns
    -------
    dict
        ``{metrics_learned, metrics_geometric, metrics_fused, weights, n}`` where
        each ``metrics_*`` is the :func:`uda.evaluation.evaluate.metrics` dict, ``weights``
        is the normalized length-2 fusion weight vector, and ``n`` is the number
        of images scored. ``metrics_learned`` is always surfaced so fusion is
        never *assumed* to win (honesty contract).
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    learned = np.asarray(learned, dtype=float).ravel()
    geometric = np.asarray(geometric, dtype=float).ravel()

    if weights is None:
        w = np.ones(2, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).ravel()
    w = w / w.sum()

    fused = circular_fuse(np.stack([learned, geometric]), weights=w)

    # Canonical order is metrics(y_true, y_pred): MAE/RMSE are symmetric but ME,
    # MAPE and R2 are NOT — the truth must come first or the signed bias and the
    # R2 denominator are wrong.
    return {
        "metrics_learned": evaluate.metrics(yt, learned),
        "metrics_geometric": evaluate.metrics(yt, geometric),
        "metrics_fused": evaluate.metrics(yt, fused),
        "weights": w,
        "n": int(yt.size),
    }

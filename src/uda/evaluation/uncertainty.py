"""Monte-Carlo dropout predictive uncertainty.

Run ``n`` stochastic forward passes with dropout active (``training=True``) and
summarize their spread, then turn the per-element ``(mean, std)`` into Gaussian
predictive intervals and score their empirical coverage.

The summaries live in the model's **encoded output space** (its ``n_outputs``
head outputs). For ``RawDegrees`` that is degrees directly; for ``SinCos2Theta``
the mean/std are in ``(sin, cos)`` space and decoding to a degree interval is out
of scope here — callers using sin/cos should decode each pass through the target
*before* summarizing. The headline MC-dropout run uses ``target=raw`` so the
intervals are already in degrees.

This module reads no config: ``n`` (the wiring point is
``TrainConfig.mc_samples``) is passed explicitly by an eval driver. Wiring into
``uda.training.train``/``uda.evaluation.evaluate`` is out of scope for this module.
"""
from __future__ import annotations

import keras
import numpy as np

__all__ = ["mc_dropout_predict", "predictive_interval", "coverage"]


def mc_dropout_predict(
    model: keras.Model, x: np.ndarray, n: int, *, batch_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of ``n`` stochastic (dropout-on) forward passes.

    Each pass calls ``model(x_batch, training=True)`` so the Dropout masks are
    resampled every time — unlike ``model.predict``, which forces inference mode
    and disables dropout. The whole input is scored once per pass in mini-batches
    of ``batch_size``; the ``n`` passes are stacked to ``(n, N, n_outputs)`` and
    reduced to per-element ``mean`` and population ``std`` (``ddof=0``).

    Parameters
    ----------
    model : keras.Model
        A model containing Dropout layers (e.g. the regression head). With no
        active Dropout the graph is deterministic and ``std`` collapses to ~0.
    x : np.ndarray
        Input batch the model accepts (features for a head, images for a full
        model), shape ``(N, ...)``.
    n : int
        Number of stochastic passes. ``n >= 1`` is required; ``n == 1`` has no
        spread, so ``std`` is 0 everywhere.
    batch_size : int, optional
        Forward-pass mini-batch size (default 128). Batching does not change the
        result — the whole input is summarized once per pass.

    Returns
    -------
    (mean, std) : tuple[np.ndarray, np.ndarray]
        Host ``numpy`` arrays, each shape ``(N, n_outputs)``. ``mean`` is the
        sample mean over passes, ``std`` the population std (``ddof=0``).

    Raises
    ------
    ValueError
        If ``n < 1``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    x = np.asarray(x)
    num = len(x)
    step = max(1, int(batch_size))

    passes = np.stack(
        [
            np.concatenate(
                [
                    keras.ops.convert_to_numpy(
                        model(x[start : start + step], training=True)
                    )
                    for start in range(0, num, step)
                ],
                axis=0,
            )
            for _ in range(n)
        ],
        axis=0,
    )  # (n, N, n_outputs)

    mean = passes.mean(axis=0)
    std = passes.std(axis=0)  # ddof=0 (population) by numpy default
    return np.asarray(mean), np.asarray(std)


def predictive_interval(
    mean: np.ndarray, std: np.ndarray, z: float = 1.96
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian predictive interval ``(mean - z*std, mean + z*std)``.

    Parameters
    ----------
    mean, std : np.ndarray
        Per-element predictive mean and std (e.g. from :func:`mc_dropout_predict`),
        broadcastable to a common shape.
    z : float, optional
        Two-sided normal quantile; ``1.96`` (default) is ~95%.

    Returns
    -------
    (lower, upper) : tuple[np.ndarray, np.ndarray]
        Interval bounds, same (broadcast) shape as ``mean``. The interval is
        centered on ``mean`` with half-width ``z*std``, so it collapses to the
        point ``mean`` wherever ``std == 0``.
    """
    mean = np.asarray(mean, dtype=float)
    half = z * np.asarray(std, dtype=float)
    return mean - half, mean + half


def coverage(
    y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, z: float = 1.96
) -> float:
    """Fraction of ``y_true`` inside ``[mean - z*std, mean + z*std]``.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values, broadcastable against ``mean``/``std``.
    mean, std : np.ndarray
        Predictive mean and std defining the interval (see
        :func:`predictive_interval`).
    z : float, optional
        Two-sided normal quantile (default ``1.96``). Larger ``z`` widens the
        interval, so coverage is monotone non-decreasing in ``z``.

    Returns
    -------
    float
        Empirical coverage in ``[0, 1]`` over all (broadcast) elements. The
        interval is closed, so where ``std == 0`` an element is covered iff
        ``y_true == mean`` exactly.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower, upper = predictive_interval(mean, std, z=z)
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))

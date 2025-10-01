"""Post-hoc prediction combiner — ``mean`` and ``stacked`` ensembles (Keras-free).

:func:`ensemble_predictions` reads a list of
per-model prediction CSVs (the schema written by :func:`uda.evaluation.evaluate.evaluate` —
columns include ``theta_true`` and ``theta_pred``), verifies the members share the
same held-out test set (aligned ``theta_true``, same row order), and combines
their ``theta_pred`` into one prediction by either an unweighted ``mean`` or a
leakage-free ``stacked`` ``Ridge`` meta-learner. The returned metric dict is
*exactly* :func:`uda.evaluation.evaluate.metrics` over the ensemble prediction — the single
source of truth, never re-implemented here.

This module is deliberately Keras-free: it depends only on numpy, pandas, and
scikit-learn, so an ensemble can be assembled from saved CSVs without a backend.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

from uda.evaluation.evaluate import metrics

__all__ = ["ensemble_predictions"]

#: Absolute tolerance for the ``theta_true`` alignment check. Loose enough to
#: absorb CSV float round-trip noise, tight enough to catch a genuinely different
#: held-out test set. We use ``atol=1e-6``.
_ALIGN_ATOL = 1e-6


def _load_predictions(
    pred_csv_paths: list[str | Path],
) -> tuple[np.ndarray, np.ndarray]:
    """Load and align member CSVs into a shared ``y_true`` and a feature matrix.

    Parameters
    ----------
    pred_csv_paths : list of str or Path
        Paths to per-model prediction CSVs. Each must contain at least
        ``theta_true`` and ``theta_pred`` columns; any other columns are ignored.

    Returns
    -------
    (y_true, preds) : tuple[np.ndarray, np.ndarray]
        ``y_true`` is the shared held-out target taken from the first CSV (shape
        ``(n,)``); ``preds`` is the column-stack of each member's ``theta_pred``
        (shape ``(n, k)``, ``k == len(pred_csv_paths)``).

    Raises
    ------
    ValueError
        If fewer than two paths are given, if any member's row count differs from
        the first, or if any member's ``theta_true`` is not equal to the first's
        within ``_ALIGN_ATOL``.
    """
    if len(pred_csv_paths) < 2:
        raise ValueError(
            f"ensemble needs at least 2 prediction CSVs, got {len(pred_csv_paths)}"
        )

    y_true: np.ndarray | None = None
    columns: list[np.ndarray] = []
    for path in pred_csv_paths:
        df = pd.read_csv(Path(path))
        theta_true = df["theta_true"].to_numpy(dtype=np.float64)
        theta_pred = df["theta_pred"].to_numpy(dtype=np.float64)

        if y_true is None:
            y_true = theta_true
        else:
            if theta_true.shape != y_true.shape:
                raise ValueError(
                    f"row-count mismatch across prediction CSVs: file {path} has "
                    f"{theta_true.shape[0]} rows, expected {y_true.shape[0]}"
                )
            if not np.allclose(theta_true, y_true, atol=_ALIGN_ATOL):
                raise ValueError(
                    f"misaligned theta_true across prediction CSVs: file {path} "
                    "differs"
                )
        columns.append(theta_pred)

    assert y_true is not None  # guaranteed: len >= 2 checked above
    return y_true, np.column_stack(columns)


def ensemble_predictions(
    pred_csv_paths: list[str | Path],
    method: Literal["mean", "stacked"] = "mean",
    *,
    seed: int = 42,
    cv_folds: int = 5,
) -> dict:
    """Combine per-model prediction CSVs into one ensemble prediction.

    Parameters
    ----------
    pred_csv_paths : list of str or Path
        Paths to per-model prediction CSVs, each with at least ``theta_true`` and
        ``theta_pred`` columns (the schema written by :func:`uda.evaluation.evaluate.evaluate`).
        At least two paths are required; a single path raises ``ValueError``. Extra
        columns (``image_id``, ``rotation_deg``, ...) are ignored, and the two
        required columns are read by name, not position.
    method : {"mean", "stacked"}, default "mean"
        ``mean`` averages the per-model ``theta_pred`` columns. ``stacked`` fits a
        scikit-learn ``Ridge`` meta-learner over the per-model predictions and
        reports its out-of-fold predictions via ``cross_val_predict`` (so the
        reported ``y_pred`` carries no leakage — each sample is predicted by a
        ``Ridge`` fit on the *other* folds).
    seed : int, default 42
        Seed for the stacked-ensemble CV (``KFold(shuffle=True,
        random_state=seed)``). Ignored for ``mean``.
    cv_folds : int, default 5
        Number of folds for the stacked meta-learner CV.

    Returns
    -------
    dict
        ``{"y_true": np.ndarray, "y_pred": np.ndarray, "metrics": dict,
        "method": str, "n_models": int}`` where ``y_true`` is the shared held-out
        target (from the first CSV, validated equal across members), ``y_pred`` is
        the combined prediction, ``metrics`` is exactly
        :func:`uda.evaluation.evaluate.metrics(y_true, y_pred)`, and ``n_models`` is the number
        of member CSVs.

    Raises
    ------
    ValueError
        If fewer than two paths are given, the members' row counts differ, the
        members' ``theta_true`` columns disagree beyond ``1e-6``, or ``method`` is
        not one of ``{"mean", "stacked"}``.
    """
    y_true, preds = _load_predictions(pred_csv_paths)

    if method == "mean":
        y_pred = preds.mean(axis=1)
    elif method == "stacked":
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        y_pred = cross_val_predict(Ridge(), preds, y_true, cv=cv)
    else:
        raise ValueError(
            f"unknown ensemble method {method!r}; expected 'mean' or 'stacked'"
        )

    return {
        "y_true": y_true,
        "y_pred": np.asarray(y_pred),
        "metrics": metrics(y_true, y_pred),
        "method": method,
        "n_models": preds.shape[1],
    }

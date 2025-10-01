"""Metrics + predictions dump.

Metrics are computed on the held-out test set in **degrees**
(after decoding via the AngleTarget): MAE, RMSE, ME (bias = mean(pred - true)),
MAPE (%), and R². Each run appends one row to ``results/metrics.csv`` keyed by
``(name, split_strategy, target, seed)`` and dumps per-sample predictions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from uda.config import ExperimentConfig

__all__ = ["metrics", "evaluate", "METRICS_CSV_COLUMNS"]

METRICS_CSV_COLUMNS = [
    "name",
    "backbone",
    "pooling",
    "split_strategy",
    "target",
    "seed",
    "era",
    "n_test",
    "mae",
    "rmse",
    "me",
    "mape",
    "r2",
]


def metrics(y_true_deg: np.ndarray, y_pred_deg: np.ndarray) -> dict:
    """Compute regression metrics in degrees.

    ME is the signed bias ``mean(pred - true)``; MAPE is a percentage.
    """
    yt = np.asarray(y_true_deg, dtype=np.float64).ravel()
    yp = np.asarray(y_pred_deg, dtype=np.float64).ravel()
    err = yp - yt
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "me": float(np.mean(err)),
        "mape": float(np.mean(np.abs(err) / np.abs(yt)) * 100.0),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def evaluate(
    cfg: ExperimentConfig,
    y_true_deg: np.ndarray,
    y_pred_deg: np.ndarray,
    test_meta: pd.DataFrame,
    out_dir: str | Path = "results",
) -> dict:
    """Compute metrics, append to ``metrics.csv``, dump predictions; return the row.

    ``test_meta`` is the test-split slice of ``Corpus.meta`` (aligned with the
    prediction order). The predictions CSV gets ``theta_true``/``theta_pred``
    columns appended.
    """
    row = {
        "name": cfg.name,
        "backbone": cfg.backbone.name,
        "pooling": cfg.backbone.pooling,
        "split_strategy": cfg.split.strategy,
        "target": cfg.target.kind,
        "seed": cfg.seed,
        "era": cfg.era,
        "n_test": int(np.asarray(y_true_deg).size),
        **metrics(y_true_deg, y_pred_deg),
    }

    out = Path(out_dir)
    (out / "predictions").mkdir(parents=True, exist_ok=True)

    preds = test_meta.reset_index(drop=True).copy()
    preds["theta_true"] = np.asarray(y_true_deg, dtype=np.float64).ravel()
    preds["theta_pred"] = np.asarray(y_pred_deg, dtype=np.float64).ravel()
    preds.to_csv(out / "predictions" / f"{cfg.name}.csv", index=False)

    metrics_csv = out / "metrics.csv"
    new_row = pd.DataFrame([row], columns=METRICS_CSV_COLUMNS)
    if metrics_csv.exists():
        prev = pd.read_csv(metrics_csv)
        dup = (
            (prev["name"] == cfg.name)
            & (prev["split_strategy"] == cfg.split.strategy)
            & (prev["target"] == cfg.target.kind)
            & (prev["seed"] == cfg.seed)
        )
        prev = prev[~dup]
        new_row = pd.concat([prev, new_row], ignore_index=True)
    new_row.to_csv(metrics_csv, index=False)
    return row

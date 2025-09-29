"""Patient-level k-fold cross-validation harness (Keras-free).

:func:`patient_kfold` drives the existing
:class:`uda.data.splits.PatientLevelSplit` (``GroupKFold`` over ``patient_id``)
and aggregates the per-fold metric dicts returned by an **injected** per-fold
runner.

The runner (``run_fold``) is a dependency injection point: production passes a
closure that builds a corpus on the fold's base ``image_id`` values, trains, and
returns :func:`uda.evaluation.evaluate.metrics`; unit tests pass a cheap stub so no model is
ever built. Consequently all heavy, Keras-dependent work lives entirely inside
the caller's ``run_fold`` and this module imports **no** deep-learning backend
(``keras``/``jax``/``tensorflow``) — only numpy, pandas, and the leakage-free
splitter.

Determinism: the fold partition is a pure function of ``(labels, k, seed)`` via
``GroupKFold(shuffle=True, random_state=seed)`` over the deterministically ordered
base images. Given a deterministic ``run_fold``, the whole result is reproducible.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from uda.config import SplitConfig
from uda.data.splits import PatientLevelSplit

__all__ = ["patient_kfold", "random_kfold"]

_PATIENT_ID = "patient_id"

RunFold = Callable[[np.ndarray, np.ndarray], dict]


def _n_patients(labels: pd.DataFrame) -> int:
    """Number of distinct ``patient_id`` groups in ``labels``."""
    if _PATIENT_ID not in labels.columns:
        raise KeyError(f"labels must contain a '{_PATIENT_ID}' column")
    return int(labels[_PATIENT_ID].nunique())


def _aggregate(folds: list[dict]) -> dict:
    """Mean/std (population, ``ddof=0``) of every metric common to all folds.

    Parameters
    ----------
    folds : list of dict
        Per-fold metric dictionaries as returned by ``run_fold``.

    Returns
    -------
    dict
        ``{metric: {"mean": float, "std": float}}`` for each key present in
        **every** fold dict. Keys missing from any fold are dropped
        (common-keys-only). Order follows the first fold's keys for determinism.
    """
    if not folds:
        return {}
    common = set(folds[0])
    for fold in folds[1:]:
        common &= set(fold)
    aggregate: dict = {}
    for key in folds[0]:  # first-fold order -> deterministic key ordering
        if key not in common:
            continue
        values = np.array([float(fold[key]) for fold in folds], dtype=np.float64)
        aggregate[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),  # population std, ddof=0
        }
    return aggregate


def patient_kfold(
    labels: pd.DataFrame,
    k: int,
    run_fold: RunFold,
    *,
    seed: int = 42,
) -> dict:
    """Run patient-level k-fold CV and aggregate per-fold metric dicts.

    The fold partition is delegated to :class:`uda.data.splits.PatientLevelSplit`
    (``GroupKFold`` over ``patient_id``), so an entire patient — and therefore
    every base ``image_id`` and any rotation of it — lands on exactly one side of
    each fold. No augmentation rows ever enter the ids handed to ``run_fold``:
    the splitter deduplicates to base images first.

    Parameters
    ----------
    labels : pandas.DataFrame
        Must contain ``image_id`` and ``patient_id`` columns (the schema of
        ``data/labels.csv``).
    k : int
        Number of folds. Must satisfy ``2 <= k <= n_patients``; otherwise a
        :class:`ValueError` is raised before scikit-learn is invoked.
    run_fold : Callable[[numpy.ndarray, numpy.ndarray], dict]
        Injected per-fold runner. Called once per fold as
        ``run_fold(train_image_ids, test_image_ids) -> {metric_name: float}``.
        Both arguments are ``numpy.ndarray`` of base ``image_id`` values.
    seed : int, keyword-only
        Seed for the underlying split, passed through ``SplitConfig.seed`` to
        ``GroupKFold(shuffle=True, random_state=seed)``. Different seeds yield
        different fold partitions; the same seed is fully reproducible.

    Returns
    -------
    dict
        ``{"folds": list[dict], "aggregate": dict}`` where ``folds[i]`` is the
        i-th fold's metric dict (exactly what ``run_fold`` returned) and
        ``aggregate[metric] == {"mean": float, "std": float}`` over the folds
        (population std, ``ddof=0``) for every metric key present in **all** fold
        dicts. Keys present in only some folds are dropped from ``aggregate``.

    Raises
    ------
    ValueError
        If ``k < 2`` or ``k > n_patients`` (``GroupKFold`` cannot partition more
        folds than there are patient groups).
    KeyError
        If ``labels`` lacks ``patient_id`` (raised by the patient count) or
        ``image_id`` (raised by the splitter).
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    n_patients = _n_patients(labels)
    if k > n_patients:
        raise ValueError(
            f"k={k} exceeds the number of patients ({n_patients}); "
            "GroupKFold cannot create more folds than groups"
        )

    cfg = SplitConfig(strategy="patient", n_folds=k, seed=seed)
    splitter = PatientLevelSplit()

    fold_metrics: list[dict] = []
    for train_ids, test_ids in splitter.split(labels, cfg):
        fold_metrics.append(run_fold(train_ids, test_ids))

    return {"folds": fold_metrics, "aggregate": _aggregate(fold_metrics)}


def random_kfold(
    n_samples: int,
    k: int,
    run_fold: RunFold,
    *,
    seed: int = 42,
) -> dict:
    """Run a plain random k-fold CV over ``range(n_samples)`` and aggregate folds.

    This is the **paper's "image" protocol**: a random k-fold directly over the
    *augmented* image rows (the full 2100-row corpus), with **no grouping**. Unlike
    :func:`patient_kfold` — which dedups to base images and splits by patient — every
    row is an independent sample here, so rotated copies of one base image may land in
    both train and test. That is exactly the leaky protocol the original paper used and
    the one this function reproduces; use :func:`patient_kfold` for the leakage-free
    number.

    Partitioning is delegated to :class:`sklearn.model_selection.KFold`
    (``shuffle=True, random_state=seed``) over the integer indices ``0..n_samples-1``.
    The per-fold runner is **injected** (the dependency-injection pattern shared with
    :func:`patient_kfold`), so this module stays Keras-free — all training lives in the
    caller's ``run_fold``.

    Parameters
    ----------
    n_samples : int
        Number of rows to partition (e.g. ``len(meta)`` for the 2100-row corpus).
    k : int
        Number of folds. Must satisfy ``2 <= k <= n_samples``; otherwise a
        :class:`ValueError` is raised before scikit-learn is invoked.
    run_fold : Callable[[numpy.ndarray, numpy.ndarray], dict]
        Injected per-fold runner. Called once per fold as
        ``run_fold(train_idx, test_idx) -> {metric_name: float}`` where both arguments
        are integer ``numpy.ndarray`` index arrays into ``range(n_samples)``.
    seed : int, keyword-only
        Seed for ``KFold(shuffle=True, random_state=seed)``. The same seed is fully
        reproducible; different seeds yield different fold partitions.

    Returns
    -------
    dict
        ``{"folds": list[dict], "aggregate": dict}`` — identical shape and aggregate
        semantics to :func:`patient_kfold` (population std, ``ddof=0``, common-keys
        only; reuses the same :func:`_aggregate` helper).

    Raises
    ------
    ValueError
        If ``k < 2`` or ``k > n_samples`` (``KFold`` cannot create more folds than
        samples).
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if k > n_samples:
        raise ValueError(
            f"k={k} exceeds the number of samples ({n_samples}); "
            "KFold cannot create more folds than samples"
        )

    splitter = KFold(n_splits=k, shuffle=True, random_state=seed)

    fold_metrics: list[dict] = []
    for train_idx, test_idx in splitter.split(np.arange(n_samples)):
        fold_metrics.append(run_fold(train_idx, test_idx))

    return {"folds": fold_metrics, "aggregate": _aggregate(fold_metrics)}

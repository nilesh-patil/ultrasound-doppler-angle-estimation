"""Patient-level **nested** k-fold cross-validation harness (Keras-free).

:func:`patient_nested_cv` drives the same
:class:`uda.data.splits.PatientLevelSplit` (``GroupKFold`` over ``patient_id``)
as :mod:`uda.training.cv` for the **outer** split and aggregates the per-outer-fold metric
dicts returned by an **injected** ``run_outer`` runner with the identical
``ddof=0``, common-keys-only semantics.

The *inner* loop is delegated entirely to the caller: the harness hands
``run_outer`` the outer fold's base ``image_id`` arrays plus ``k_inner`` verbatim
and never inspects or re-splits on it. Production passes a closure that runs an
inner ``k_inner``-fold model-selection loop on the outer-train ids, refits, and
returns :func:`uda.evaluation.evaluate.metrics` on the outer-test ids; unit tests pass a
cheap stub so no model is ever built. Consequently all heavy, Keras-dependent
work lives entirely inside the caller's ``run_outer`` and this module imports
**no** deep-learning backend (``keras``/``jax``/``tensorflow``) — only numpy,
pandas, and the leakage-free splitter.

Determinism: the outer fold partition is a pure function of
``(labels, k_outer, seed)`` via ``GroupKFold(shuffle=True, random_state=seed)``
over the deterministically ordered base images. Given a deterministic
``run_outer``, the whole result is reproducible.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from uda.config import SplitConfig
from uda.data.splits import PatientLevelSplit

__all__ = ["patient_nested_cv"]

_PATIENT_ID = "patient_id"

RunOuter = Callable[[np.ndarray, np.ndarray, int], dict]


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
        Per-outer-fold metric dictionaries as returned by ``run_outer``.

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


def patient_nested_cv(
    labels: pd.DataFrame,
    k_outer: int,
    k_inner: int,
    run_outer: RunOuter,
    *,
    seed: int = 42,
) -> dict:
    """Run patient-level **nested** k-fold CV and aggregate per-outer-fold metrics.

    The outer fold partition is delegated to
    :class:`uda.data.splits.PatientLevelSplit` (``GroupKFold`` over
    ``patient_id``), so an entire patient — and therefore every base ``image_id``
    and any rotation of it — lands on exactly one side of each outer fold. No
    augmentation rows ever enter the ids handed to ``run_outer``: the splitter
    deduplicates to base images first. The inner ``k_inner``-fold loop is the
    caller's responsibility — this harness passes ``k_inner`` through verbatim and
    never inspects or re-splits on it.

    Parameters
    ----------
    labels : pandas.DataFrame
        Must contain ``image_id`` and ``patient_id`` columns (the schema of
        ``data/labels.csv``).
    k_outer : int
        Number of outer folds. Must satisfy ``2 <= k_outer <= n_patients``;
        otherwise a :class:`ValueError` is raised before scikit-learn is invoked.
    k_inner : int
        Number of inner folds. Passed through verbatim to ``run_outer`` and
        echoed in the result; the harness performs no inner splitting itself.
    run_outer : Callable[[numpy.ndarray, numpy.ndarray, int], dict]
        Injected per-outer-fold runner. Called once per outer fold as
        ``run_outer(train_image_ids, test_image_ids, k_inner) -> {metric: float}``.
        The first two arguments are ``numpy.ndarray`` of base ``image_id`` values.
    seed : int, keyword-only
        Seed for the underlying outer split, passed through ``SplitConfig.seed``
        to ``GroupKFold(shuffle=True, random_state=seed)``. Different seeds yield
        different outer partitions; the same seed is fully reproducible.

    Returns
    -------
    dict
        ``{"folds": list[dict], "aggregate": dict, "k_inner": int}`` where
        ``folds[i]`` is the i-th outer fold's metric dict (exactly what
        ``run_outer`` returned), ``aggregate[metric] == {"mean": float, "std":
        float}`` over the outer folds (population std, ``ddof=0``) for every
        metric key present in **all** fold dicts (keys present in only some folds
        are dropped), and ``k_inner`` echoes the inner-fold count passed in.

    Raises
    ------
    ValueError
        If ``k_outer < 2`` or ``k_outer > n_patients`` (``GroupKFold`` cannot
        create more outer folds than there are patient groups).
    KeyError
        If ``labels`` lacks ``patient_id`` (raised by the patient count) or
        ``image_id`` (raised by the splitter).
    """
    if k_outer < 2:
        raise ValueError(f"k_outer must be >= 2, got {k_outer}")
    n_patients = _n_patients(labels)
    if k_outer > n_patients:
        raise ValueError(
            f"k_outer={k_outer} exceeds the number of patients ({n_patients}); "
            "GroupKFold cannot create more folds than groups"
        )

    cfg = SplitConfig(strategy="patient", n_folds=k_outer, seed=seed)
    splitter = PatientLevelSplit()

    fold_metrics: list[dict] = []
    for train_ids, test_ids in splitter.split(labels, cfg):
        fold_metrics.append(run_outer(train_ids, test_ids, k_inner))

    return {
        "folds": fold_metrics,
        "aggregate": _aggregate(fold_metrics),
        "k_inner": k_inner,
    }

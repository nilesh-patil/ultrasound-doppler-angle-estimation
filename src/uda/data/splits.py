"""Leakage-aware train/test splitting over the *base* images.

Two strategies are provided behind a common
:class:`SplitStrategy` protocol:

* :class:`ImageLevelSplit` reproduces the paper protocol — an 80/20 holdout (or
  k-fold) over the 84 base ``image_id`` values.
* :class:`PatientLevelSplit` is leakage-free — it splits by ``patient_id`` using
  scikit-learn's grouped splitters so that **no patient (and therefore no base
  image or any rotation of it) ever spans train and test**.

Both strategies operate on, and return, **base** ``image_id`` values only.
Augmentation (the rotation sweep) is applied *after* splitting by the corpus
builder in :mod:`uda.data.dataset` — never here. Doing the split on base images
keeps every rotated copy of a held-out image out of the training set.

Every split is a pure function of ``cfg.seed`` and is therefore reproducible.
"""
from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
)

from uda.config import SplitConfig

__all__ = [
    "SplitStrategy",
    "ImageLevelSplit",
    "PatientLevelSplit",
    "build_split",
]

_IMAGE_ID = "image_id"
_PATIENT_ID = "patient_id"


@runtime_checkable
class SplitStrategy(Protocol):
    """A train/test partitioner over base ``image_id`` values.

    Implementations yield ``(train_image_ids, test_image_ids)`` pairs: a single
    pair for a holdout split, or ``cfg.n_folds`` pairs for k-fold.
    """

    def split(
        self, labels: pd.DataFrame, cfg: SplitConfig
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        ...


def _unique_base_images(labels: pd.DataFrame) -> pd.DataFrame:
    """Return one row per base ``image_id``, ordered deterministically.

    The input frame may already be one-row-per-base-image, but we never rely on
    that: collapsing on ``image_id`` makes the split robust if augmented rows are
    ever passed in by mistake. Rows are sorted by ``image_id`` so the array fed to
    scikit-learn is in a fixed order, which (with a fixed ``random_state``) makes
    the partition fully deterministic.
    """
    if _IMAGE_ID not in labels.columns:
        raise KeyError(f"labels must contain an '{_IMAGE_ID}' column")
    base = labels.drop_duplicates(subset=_IMAGE_ID).sort_values(_IMAGE_ID)
    return base.reset_index(drop=True)


class ImageLevelSplit:
    """Paper protocol: split over base ``image_id`` without replacement.

    A single ``test_size`` holdout by default, or ``cfg.n_folds``-fold when
    ``cfg.n_folds`` is set. Rotated copies of a base image never leak across the
    boundary because augmentation is applied only after this split.
    """

    def split(
        self, labels: pd.DataFrame, cfg: SplitConfig
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        base = _unique_base_images(labels)
        ids = base[_IMAGE_ID].to_numpy()

        if cfg.n_folds is not None:
            splitter = KFold(
                n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed
            )
            folds = splitter.split(ids)
        else:
            splitter = ShuffleSplit(
                n_splits=1, test_size=cfg.test_size, random_state=cfg.seed
            )
            folds = splitter.split(ids)

        for train_idx, test_idx in folds:
            yield ids[train_idx], ids[test_idx]


class PatientLevelSplit:
    """Leakage-free protocol: split by ``patient_id``.

    Uses scikit-learn's grouped splitters so an entire patient lands in exactly
    one side of the partition. Consequently no base ``image_id`` — and no rotation
    of it — appears in both train and test. A single ``test_size`` holdout by
    default (:class:`~sklearn.model_selection.GroupShuffleSplit`), or
    ``cfg.n_folds``-fold (:class:`~sklearn.model_selection.GroupKFold`).
    """

    def split(
        self, labels: pd.DataFrame, cfg: SplitConfig
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        base = _unique_base_images(labels)
        if _PATIENT_ID not in base.columns:
            raise KeyError(
                f"PatientLevelSplit requires a '{_PATIENT_ID}' column in labels"
            )
        ids = base[_IMAGE_ID].to_numpy()
        groups = base[_PATIENT_ID].to_numpy()

        if cfg.n_folds is not None:
            splitter = GroupKFold(
                n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed
            )
            folds = splitter.split(ids, groups=groups)
        else:
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=cfg.test_size, random_state=cfg.seed
            )
            folds = splitter.split(ids, groups=groups)

        for train_idx, test_idx in folds:
            yield ids[train_idx], ids[test_idx]


def build_split(cfg: SplitConfig) -> SplitStrategy:
    """Construct the splitter named by ``cfg.strategy``.

    Parameters
    ----------
    cfg:
        Split configuration; ``cfg.strategy`` selects the implementation
        (``"image"`` or ``"patient"``).

    Returns
    -------
    SplitStrategy
        An :class:`ImageLevelSplit` or :class:`PatientLevelSplit` instance.

    Raises
    ------
    ValueError
        If ``cfg.strategy`` is not a recognized strategy.
    """
    if cfg.strategy == "image":
        return ImageLevelSplit()
    if cfg.strategy == "patient":
        return PatientLevelSplit()
    raise ValueError(f"unknown split strategy: {cfg.strategy!r}")

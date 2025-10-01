"""Adversarial, real-data leakage guarantee over ALL 84 base images.

``test_dataset.py`` verifies the leakage property on a capped, size-patched corpus.
This module asserts the same guarantee on the *full, real* label table at the split
level — cheaply (pure pandas, no augmentation/resize) and authoritatively. This is
the property the entire patient-level re-evaluation (the paper's leakage finding)
rests on, so it gets its own end-to-end test on the data we actually train on.
"""
from __future__ import annotations

from pathlib import Path

from uda.config import SplitConfig
from uda.data.labels import build_labels_csv
from uda.data.splits import build_split

_REPO = Path(__file__).resolve().parents[1]


def _real_labels(tmp_path):
    """Build labels.csv from the repo's data/Results.txt (self-contained)."""
    return build_labels_csv(_REPO / "data" / "Results.txt", tmp_path / "labels.csv")


def _holdout(labels, strategy):
    cfg = SplitConfig(strategy=strategy, test_size=0.2, seed=42)
    train, test = next(iter(build_split(cfg).split(labels, cfg)))
    return set(train.tolist()), set(test.tolist())


def test_patient_split_is_leakage_free_on_all_84(tmp_path):
    labels = _real_labels(tmp_path)
    assert len(labels) == 84
    train, test = _holdout(labels, "patient")

    # base images: disjoint and fully covering
    assert train.isdisjoint(test)
    assert train | test == set(labels["image_id"])

    # the leakage guarantee: no patient spans the split
    lut = labels.set_index("image_id")["patient_id"].to_dict()
    train_patients = {lut[i] for i in train}
    test_patients = {lut[i] for i in test}
    assert train_patients.isdisjoint(test_patients)

    # a non-trivial holdout actually exists
    assert 0 < len(test) < 84


def test_image_split_covers_and_is_disjoint_on_all_84(tmp_path):
    labels = _real_labels(tmp_path)
    train, test = _holdout(labels, "image")
    assert train.isdisjoint(test)
    assert train | test == set(labels["image_id"])
    assert len(test) in (16, 17)  # ~20% of 84


def test_patient_split_is_deterministic(tmp_path):
    labels = _real_labels(tmp_path)
    a = _holdout(labels, "patient")
    b = _holdout(labels, "patient")
    assert a == b

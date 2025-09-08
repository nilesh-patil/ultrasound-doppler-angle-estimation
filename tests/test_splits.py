"""Tests — leakage-aware splitting over base images.

A synthetic labels frame stands in for ``data/labels.csv``: 40 base images spread
across 10 patients (4 images each), with arbitrary in-range angles. The tests
assert determinism, the train/test partition invariants, and — critically — that
``PatientLevelSplit`` keeps every patient (hence every base image) on one side.
"""
import numpy as np
import pandas as pd
import pytest

from uda.config import SplitConfig
from uda.data.splits import (
    ImageLevelSplit,
    PatientLevelSplit,
    SplitStrategy,
    build_split,
)

N_PATIENTS = 10
PER_PATIENT = 4
N_IMAGES = N_PATIENTS * PER_PATIENT


def _labels() -> pd.DataFrame:
    """Synthetic (image_id, patient_id, theta_deg) frame, one row per base image."""
    rng = np.random.default_rng(0)
    rows = []
    for p in range(N_PATIENTS):
        for k in range(PER_PATIENT):
            rows.append(
                {
                    "image_id": f"p{p:02d}_img{k:02d}",
                    "patient_id": f"patient_{p:02d}",
                    "theta_deg": float(rng.uniform(78.0, 104.0)),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def test_build_split_returns_requested_strategy():
    assert isinstance(build_split(SplitConfig(strategy="image")), ImageLevelSplit)
    assert isinstance(build_split(SplitConfig(strategy="patient")), PatientLevelSplit)


def test_strategies_satisfy_protocol():
    assert isinstance(ImageLevelSplit(), SplitStrategy)
    assert isinstance(PatientLevelSplit(), SplitStrategy)


# --------------------------------------------------------------------------- #
# partition invariants — holdout
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy", ["image", "patient"])
def test_holdout_union_is_all_and_disjoint(strategy):
    labels = _labels()
    cfg = SplitConfig(strategy=strategy, test_size=0.25, seed=42)
    (train, test), = list(build_split(cfg).split(labels, cfg))

    train_s, test_s = set(train), set(test)
    all_ids = set(labels["image_id"])

    assert train_s | test_s == all_ids
    assert train_s.isdisjoint(test_s)
    assert len(train) + len(test) == N_IMAGES
    # ids returned are *base* image ids only (no augmentation), so exactly N_IMAGES
    assert len(train_s) + len(test_s) == N_IMAGES


@pytest.mark.parametrize("strategy", ["image", "patient"])
def test_holdout_returns_base_image_ids_only(strategy):
    labels = _labels()
    cfg = SplitConfig(strategy=strategy, test_size=0.25, seed=1)
    (train, test), = list(build_split(cfg).split(labels, cfg))
    returned = set(train) | set(test)
    # never more than the base images: rotations must not appear here
    assert returned == set(labels["image_id"])
    assert len(returned) == N_IMAGES


@pytest.mark.parametrize("strategy", ["image", "patient"])
def test_holdout_test_fraction_is_reasonable(strategy):
    labels = _labels()
    cfg = SplitConfig(strategy=strategy, test_size=0.25, seed=7)
    (train, test), = list(build_split(cfg).split(labels, cfg))
    # roughly 25% held out (patient grouping rounds to whole patients)
    assert 0 < len(test) < N_IMAGES
    frac = len(test) / N_IMAGES
    assert 0.1 <= frac <= 0.4


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy", ["image", "patient"])
def test_deterministic_given_seed(strategy):
    labels = _labels()
    cfg = SplitConfig(strategy=strategy, test_size=0.25, seed=123)

    a = [(tuple(tr), tuple(te)) for tr, te in build_split(cfg).split(labels, cfg)]
    b = [(tuple(tr), tuple(te)) for tr, te in build_split(cfg).split(labels, cfg)]
    assert a == b


@pytest.mark.parametrize("strategy", ["image", "patient"])
def test_different_seed_changes_split(strategy):
    labels = _labels()
    c1 = SplitConfig(strategy=strategy, test_size=0.25, seed=1)
    c2 = SplitConfig(strategy=strategy, test_size=0.25, seed=2)
    (_, test1), = list(build_split(c1).split(labels, c1))
    (_, test2), = list(build_split(c2).split(labels, c2))
    assert set(test1) != set(test2)


# --------------------------------------------------------------------------- #
# the leakage assertion
# --------------------------------------------------------------------------- #
def test_patient_level_patient_sets_are_disjoint():
    labels = _labels()
    cfg = SplitConfig(strategy="patient", test_size=0.25, seed=42)
    (train, test), = list(build_split(cfg).split(labels, cfg))

    id_to_patient = dict(zip(labels["image_id"], labels["patient_id"]))
    train_patients = {id_to_patient[i] for i in train}
    test_patients = {id_to_patient[i] for i in test}

    assert train_patients.isdisjoint(test_patients)
    # every base image of a test patient is on the test side (no leak)
    assert train_patients | test_patients == set(labels["patient_id"])


def test_patient_level_no_base_image_spans_split():
    labels = _labels()
    cfg = SplitConfig(strategy="patient", test_size=0.25, seed=11)
    (train, test), = list(build_split(cfg).split(labels, cfg))
    assert set(train).isdisjoint(set(test))


def test_image_level_base_ids_disjoint():
    labels = _labels()
    cfg = SplitConfig(strategy="image", test_size=0.25, seed=5)
    (train, test), = list(build_split(cfg).split(labels, cfg))
    assert set(train).isdisjoint(set(test))


def test_image_level_may_split_a_patient():
    # Sanity: the *image* strategy is allowed to (and generally will) place
    # images from one patient on both sides — this is exactly the leakage the
    # patient strategy fixes. We only assert it is at least possible here.
    labels = _labels()
    cfg = SplitConfig(strategy="image", test_size=0.25, seed=5)
    (train, test), = list(build_split(cfg).split(labels, cfg))
    id_to_patient = dict(zip(labels["image_id"], labels["patient_id"]))
    train_patients = {id_to_patient[i] for i in train}
    test_patients = {id_to_patient[i] for i in test}
    # the two patient sets overlap (image-level leaks patients by design)
    assert train_patients & test_patients


# --------------------------------------------------------------------------- #
# k-fold
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy", ["image", "patient"])
def test_kfold_yields_k_partitions_covering_all(strategy):
    labels = _labels()
    cfg = SplitConfig(strategy=strategy, n_folds=5, seed=42)
    folds = list(build_split(cfg).split(labels, cfg))
    assert len(folds) == 5

    all_ids = set(labels["image_id"])
    seen_test: set[str] = set()
    for train, test in folds:
        assert set(train).isdisjoint(set(test))
        assert set(train) | set(test) == all_ids
        # each test fold disjoint from the others -> a partition of all ids
        assert seen_test.isdisjoint(set(test))
        seen_test |= set(test)
    assert seen_test == all_ids


def test_patient_kfold_keeps_patients_whole():
    labels = _labels()
    cfg = SplitConfig(strategy="patient", n_folds=5, seed=42)
    id_to_patient = dict(zip(labels["image_id"], labels["patient_id"]))
    for train, test in build_split(cfg).split(labels, cfg):
        train_patients = {id_to_patient[i] for i in train}
        test_patients = {id_to_patient[i] for i in test}
        assert train_patients.isdisjoint(test_patients)


# --------------------------------------------------------------------------- #
# error handling
# --------------------------------------------------------------------------- #
def test_patient_split_requires_patient_column():
    labels = _labels().drop(columns=["patient_id"])
    cfg = SplitConfig(strategy="patient", test_size=0.25, seed=1)
    with pytest.raises(KeyError):
        list(build_split(cfg).split(labels, cfg))


def test_build_split_rejects_unknown_strategy():
    cfg = SplitConfig(strategy="image")
    object.__setattr__(cfg, "strategy", "bogus")  # bypass pydantic to hit the guard
    with pytest.raises(ValueError):
        build_split(cfg)

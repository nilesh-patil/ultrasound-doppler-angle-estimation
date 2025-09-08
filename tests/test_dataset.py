"""Tests — corpus assembly (the integrating step).

These tests exercise the mandatory flow of :func:`uda.data.dataset.build_corpus`:
load labels (building ``labels.csv`` from the repo-root ``Results.txt`` when it is
missing) -> split base images -> augment within each split -> resize to the
backbone native input -> grayscale to 3 channels.

To stay fast and Keras-free they cap the base images (``max_images <= 4``) and
override the backbone native size to a small square via a tiny local backbone
table patch, so no real 224x224/299x299 resize is ever performed. The
``count formula`` test asserts ``84 * 25 == 2100`` arithmetically without building
the full corpus; one capped run confirms the ``cap * 25`` law on real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uda import config as ucfg
from uda.data import augment as _augment
from uda.data import dataset as ds

# A small resize target keeps the capped end-to-end runs fast (no 224/299 work).
SMALL_SIZE = (8, 8)


@pytest.fixture
def small_backbone(monkeypatch):
    """Force every backbone to a tiny ``SMALL_SIZE`` native input for speed."""
    tiny = {name: SMALL_SIZE for name in ds.NATIVE_INPUT_SIZES}
    monkeypatch.setattr(ds, "NATIVE_INPUT_SIZES", tiny, raising=True)
    return SMALL_SIZE


def _experiment(
    tmp_path,
    *,
    strategy: str = "image",
    target_kind: str = "raw",
    test_size: float = 0.25,
) -> ucfg.ExperimentConfig:
    """Build an ExperimentConfig that writes labels.csv into ``tmp_path``.

    ``images_dir`` is left at its repo-relative default (the test relies on the
    real 84 base images), but ``labels_csv`` points at a throwaway path so the
    on-demand build from ``Results.txt`` never touches the repo's data dir.
    """
    return ucfg.ExperimentConfig(
        name="test",
        backbone=ucfg.BackboneConfig(name="vgg19", weights=None),
        target=ucfg.TargetConfig(kind=target_kind),
        data=ucfg.DataConfig(labels_csv=tmp_path / "labels.csv"),
        split=ucfg.SplitConfig(strategy=strategy, test_size=test_size, seed=42),
    )


# --------------------------------------------------------------------------- #
# the count formula (84 * 25 == 2100)
# --------------------------------------------------------------------------- #
def test_count_formula_constants():
    """84 base images x 25 rotations == 2100, stated explicitly."""
    cfg = ucfg.DataConfig()
    assert len(_augment.rotation_angles(cfg)) == 25
    assert 84 * 25 == 2100


def test_default_rotations_is_25():
    assert ucfg.DataConfig().n_rotations == 25


# --------------------------------------------------------------------------- #
# capped end-to-end run: cap * 25 samples, 3 channels, native size
# --------------------------------------------------------------------------- #
# Caps >= 2 so an 80/20 holdout has a non-empty train *and* test side (a single
# base image cannot be split into two non-empty halves).
@pytest.mark.parametrize("cap", [2, 3, 4])
def test_capped_run_yields_cap_times_25_samples(tmp_path, small_backbone, cap):
    cfg = _experiment(tmp_path)
    corpus = ds.build_corpus(cfg, max_images=cap)

    rotations = len(_augment.rotation_angles(cfg.data))
    assert rotations == 25
    total = corpus.x_train.shape[0] + corpus.x_test.shape[0]
    assert total == cap * 25
    assert len(corpus.meta) == cap * 25


def test_x_has_channel_dim_three(tmp_path, small_backbone):
    cfg = _experiment(tmp_path)
    corpus = ds.build_corpus(cfg, max_images=2)

    assert corpus.x_train.ndim == 4
    assert corpus.x_train.shape[-1] == 3
    assert corpus.x_train.shape[1:3] == small_backbone  # native (small) input
    if corpus.x_test.shape[0]:
        assert corpus.x_test.shape[-1] == 3
        assert corpus.x_test.shape[1:3] == small_backbone


def test_images_are_float32_in_unit_range(tmp_path, small_backbone):
    cfg = _experiment(tmp_path)
    corpus = ds.build_corpus(cfg, max_images=2)
    x = corpus.x_train
    assert x.dtype == np.float32
    assert np.isfinite(x).all()
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_train_and_test_both_nonempty_for_cap4(tmp_path, small_backbone):
    cfg = _experiment(tmp_path, test_size=0.25)
    corpus = ds.build_corpus(cfg, max_images=4)
    # 4 base images, 25% holdout -> at least one base image per side.
    assert corpus.x_train.shape[0] > 0
    assert corpus.x_test.shape[0] > 0


# --------------------------------------------------------------------------- #
# targets: y width tracks AngleTarget.n_outputs
# --------------------------------------------------------------------------- #
def test_raw_target_y_is_single_column(tmp_path, small_backbone):
    cfg = _experiment(tmp_path, target_kind="raw")
    corpus = ds.build_corpus(cfg, max_images=2)
    assert corpus.y_train.shape == (corpus.x_train.shape[0], 1)
    assert corpus.y_train.dtype == np.float32
    # rows align with x
    assert corpus.y_train.shape[0] == corpus.x_train.shape[0]


def test_sincos_target_y_has_two_columns(tmp_path, small_backbone):
    cfg = _experiment(tmp_path, target_kind="sincos2theta")
    corpus = ds.build_corpus(cfg, max_images=2)
    assert corpus.y_train.shape == (corpus.x_train.shape[0], 2)
    # encoded sin/cos live on the unit circle
    norms = np.linalg.norm(corpus.y_train, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# meta integrity + label math
# --------------------------------------------------------------------------- #
def test_meta_has_expected_columns_and_splits(tmp_path, small_backbone):
    cfg = _experiment(tmp_path)
    corpus = ds.build_corpus(cfg, max_images=4)
    assert list(corpus.meta.columns) == ds.META_COLUMNS
    assert set(corpus.meta["split"].unique()) <= {"train", "test"}
    # meta rows align 1:1 with stacked x_train then x_test
    n_train = corpus.x_train.shape[0]
    assert (corpus.meta["split"].to_numpy()[:n_train] == "train").all()
    assert (corpus.meta["split"].to_numpy()[n_train:] == "test").all()


def test_meta_rotation_degs_are_the_sweep(tmp_path, small_backbone):
    cfg = _experiment(tmp_path)
    corpus = ds.build_corpus(cfg, max_images=2)
    sweep = set(_augment.rotation_angles(cfg.data))
    # every base image contributes exactly the full rotation sweep
    for image_id, group in corpus.meta.groupby("image_id"):
        assert set(group["rotation_deg"].tolist()) == sweep


def test_decoded_labels_match_rotation_math(tmp_path, small_backbone):
    """RawDegrees: decoded y == wrap_0_180(base_theta + rotation_deg)."""
    cfg = _experiment(tmp_path, target_kind="raw")
    corpus = ds.build_corpus(cfg, max_images=2)

    labels = pd.read_csv(tmp_path / "labels.csv").set_index("image_id")
    # Concatenate y in meta row order (train rows then test rows).
    decoded = np.concatenate(
        [corpus.y_train[:, 0], corpus.y_test[:, 0]]
    ).astype(np.float64)

    base_theta = corpus.meta["image_id"].map(
        lambda i: float(labels.loc[i, "theta_deg"])
    ).to_numpy()
    rot = corpus.meta["rotation_deg"].to_numpy()
    expected = (base_theta + rot) % 180.0
    assert np.allclose(decoded, expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# labels.csv is built on demand from the repo-root Results.txt
# --------------------------------------------------------------------------- #
def test_labels_csv_built_when_missing(tmp_path, small_backbone):
    cfg = _experiment(tmp_path)
    assert not (tmp_path / "labels.csv").exists()
    ds.build_corpus(cfg, max_images=2)
    built = tmp_path / "labels.csv"
    assert built.exists()
    df = pd.read_csv(built)
    assert list(df.columns) == ["image_id", "patient_id", "theta_deg"]
    assert len(df) == 84  # full label table, even though only 2 images consumed


# --------------------------------------------------------------------------- #
# END-TO-END leakage: patient strategy shares no image_id across train/test
# --------------------------------------------------------------------------- #
# The first 12 sorted base images all belong to patient 0; image 13 starts
# patient 1. A patient holdout therefore needs >= 14 base images to put whole
# patients on each side. 14 x 25 == 350 small frames stays fast.
PATIENT_CAP = 14


def test_patient_strategy_no_image_id_spans_train_and_test(tmp_path, small_backbone):
    cfg = _experiment(tmp_path, strategy="patient", test_size=0.25)
    corpus = ds.build_corpus(cfg, max_images=PATIENT_CAP)

    train_ids = set(corpus.meta.loc[corpus.meta["split"] == "train", "image_id"])
    test_ids = set(corpus.meta.loc[corpus.meta["split"] == "test", "image_id"])
    # The end-to-end leakage assertion on the assembled corpus.
    assert train_ids
    assert test_ids
    assert train_ids.isdisjoint(test_ids)


def test_patient_strategy_no_patient_spans_train_and_test(tmp_path, small_backbone):
    cfg = _experiment(tmp_path, strategy="patient", test_size=0.25)
    corpus = ds.build_corpus(cfg, max_images=PATIENT_CAP)
    train_pat = set(corpus.meta.loc[corpus.meta["split"] == "train", "patient_id"])
    test_pat = set(corpus.meta.loc[corpus.meta["split"] == "test", "patient_id"])
    assert train_pat.isdisjoint(test_pat)


# --------------------------------------------------------------------------- #
# native_input_size helper
# --------------------------------------------------------------------------- #
def test_native_input_size_table():
    assert ds.native_input_size("vgg19") == (224, 224)
    assert ds.native_input_size("xception") == (299, 299)
    with pytest.raises(KeyError):
        ds.native_input_size("nope")


# --------------------------------------------------------------------------- #
# augmented (paper) strategy: leaky by design (reproduces Table I)
# --------------------------------------------------------------------------- #
def test_augmented_strategy_count_and_is_leaky(tmp_path, small_backbone):
    cfg = _experiment(tmp_path, strategy="augmented", test_size=0.25)
    corpus = ds.build_corpus(cfg, max_images=4)
    # same total sample count as any strategy
    assert corpus.x_train.shape[0] + corpus.x_test.shape[0] == 4 * 25
    train_ids = set(corpus.meta.loc[corpus.meta["split"] == "train", "image_id"])
    test_ids = set(corpus.meta.loc[corpus.meta["split"] == "test", "image_id"])
    # the defining property of the paper protocol: rotated copies of the same base
    # image land on BOTH sides (leakage).
    assert train_ids & test_ids


def test_augmented_strategy_is_deterministic(tmp_path, small_backbone):
    cfg = _experiment(tmp_path, strategy="augmented")
    a = ds.build_corpus(cfg, max_images=4).meta["split"].tolist()
    b = ds.build_corpus(cfg, max_images=4).meta["split"].tolist()
    assert a == b

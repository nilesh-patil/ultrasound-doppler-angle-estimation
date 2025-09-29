"""Tests — trainer smoke run + native-size consistency.

The smoke run uses ``weights=None`` and patches the native input to 32x32 (on both
the data and backbone sides) so the whole extract -> fit -> predict path runs in
about a second without downloading ImageNet weights.
"""
from __future__ import annotations

import numpy as np

import uda.models.backbones as bb
from uda.models.backbones import BackboneSpec
from uda.config import ExperimentConfig
from uda.data import dataset as ds
from uda.data.dataset import build_corpus
from uda.training.train import train


def _patch_small(monkeypatch, side=32):
    monkeypatch.setattr(ds, "NATIVE_INPUT_SIZES", {k: (side, side) for k in ds.NATIVE_INPUT_SIZES})
    small = {k: BackboneSpec(v.factory, v.preprocess, (side, side)) for k, v in bb.BACKBONES.items()}
    monkeypatch.setattr(bb, "BACKBONES", small)


def test_train_smoke_runs_and_predicts(tmp_path, monkeypatch):
    _patch_small(monkeypatch)
    cfg = ExperimentConfig(
        name="smoke",
        backbone={"name": "vgg19", "weights": None},
        data={"labels_csv": str(tmp_path / "labels.csv")},
        train={"epochs": 3, "batch_size": 16, "early_stopping_patience": 2},
    )
    corpus = build_corpus(cfg, max_images=4)
    res = train(cfg, corpus, out_dir=tmp_path)

    assert res.y_test_pred_deg.shape == res.y_test_true_deg.shape
    assert res.y_test_pred_deg.size > 0
    assert np.isfinite(res.y_test_pred_deg).all()
    assert (tmp_path / "history" / "smoke.csv").exists()
    assert "loss" in res.history and "val_mae" in res.history


def test_finetune_smoke_runs_end_to_end(tmp_path, monkeypatch):
    """The trainable-backbone path trains the assembled model end-to-end."""
    _patch_small(monkeypatch)
    cfg = ExperimentConfig(
        name="ft_smoke",
        backbone={"name": "vgg19", "weights": None, "trainable": True},
        data={"labels_csv": str(tmp_path / "labels.csv")},
        train={"epochs": 2, "batch_size": 16, "early_stopping_patience": 2},
    )
    corpus = build_corpus(cfg, max_images=4)
    res = train(cfg, corpus, out_dir=tmp_path)
    assert res.y_test_pred_deg.shape == res.y_test_true_deg.shape
    assert res.y_test_pred_deg.size > 0
    assert np.isfinite(res.y_test_pred_deg).all()


def test_native_sizes_agree_across_modules():
    """dataset.py and backbones.py duplicate the native sizes (data layer is
    Keras-free) — assert they never drift apart."""
    for name in ds.NATIVE_INPUT_SIZES:
        assert ds.NATIVE_INPUT_SIZES[name] == bb.native_input_size(name)

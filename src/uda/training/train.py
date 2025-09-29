"""Trainer — frozen-feature extraction + shallow-head fit (the paper's recipe).

Because the backbone is frozen, we extract its features
**once** (``backbone.predict``) and fit the head on the cached feature vectors —
identical to end-to-end training with a frozen backbone, but far faster. Adam +
MSE with early stopping on validation MAE; identical across every experiment so the
rows of ``results/metrics.csv`` are comparable by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from uda.models.backbones import build_backbone, preprocess_for
from uda.config import ExperimentConfig
from uda.models.heads import build_head
from uda.models.model import build_model
from uda.seed import set_seed
from uda.models.targets import build_target

__all__ = ["TrainResult", "train"]


@dataclass
class TrainResult:
    """Outputs of a training run (predictions are in **degrees**)."""

    history: dict
    y_test_true_deg: np.ndarray
    y_test_pred_deg: np.ndarray
    test_meta: pd.DataFrame


def _flat_dim(backbone: keras.Model) -> int:
    return int(np.prod(backbone.output_shape[1:]))


def _extract_features(cfg: ExperimentConfig, backbone: keras.Model, x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 0:
        return np.empty((0, _flat_dim(backbone)), dtype=np.float32)
    pre = preprocess_for(cfg.backbone.name, x)
    feats = np.asarray(backbone.predict(pre, batch_size=32, verbose=0), dtype=np.float32)
    # Flatten spatial feature maps — vessel orientation is encoded spatially, so we
    # must NOT average it away (global pooling collapses the angle signal).
    return feats.reshape(feats.shape[0], -1)


def train(cfg: ExperimentConfig, corpus, out_dir: str | Path = "results") -> TrainResult:
    """Train one configuration and return test predictions in degrees.

    Parameters
    ----------
    cfg : ExperimentConfig
        Full experiment config (backbone, head, target, optimizer).
    corpus : uda.data.dataset.Corpus
        Split, augmented corpus (``x_*`` in ``[0,1]``, ``y_*`` already encoded).
    out_dir : str or Path
        Where ``history/<name>.csv`` is written.
    """
    set_seed(cfg.seed)
    target = build_target(cfg.target)

    if cfg.backbone.trainable:
        history, y_pred_enc = _fit_finetune(cfg, corpus)
    else:
        history, y_pred_enc = _fit_frozen(cfg, corpus)

    y_pred_deg = np.asarray(target.decode(y_pred_enc)).ravel()
    y_true_deg = np.asarray(target.decode(np.asarray(corpus.y_test))).ravel()
    test_meta = corpus.meta[corpus.meta["split"] == "test"].reset_index(drop=True)

    out = Path(out_dir)
    (out / "history").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out / "history" / f"{cfg.name}.csv", index=False)

    return TrainResult(
        history=history,
        y_test_true_deg=y_true_deg,
        y_test_pred_deg=y_pred_deg,
        test_meta=test_meta,
    )


def _callbacks(cfg: ExperimentConfig) -> list:
    return [
        keras.callbacks.EarlyStopping(
            monitor=cfg.train.monitor,
            patience=cfg.train.early_stopping_patience,
            restore_best_weights=True,
            mode="min",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=cfg.train.monitor,
            factor=0.5,
            patience=max(3, cfg.train.early_stopping_patience // 2),
            min_lr=1e-6,
            mode="min",
        ),
    ]


def _fit_frozen(cfg: ExperimentConfig, corpus):
    """Frozen backbone: cache features once, fit the head on them (paper recipe)."""
    backbone = build_backbone(cfg.backbone)
    target = build_target(cfg.target)
    f_train = _extract_features(cfg, backbone, corpus.x_train)
    f_test = _extract_features(cfg, backbone, corpus.x_test)

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(f_train.shape[0])
    f_train = f_train[perm]
    y_train = np.asarray(corpus.y_train)[perm]

    head = build_head(f_train.shape[1], cfg.head, target.n_outputs)
    head.compile(
        optimizer=keras.optimizers.Adam(cfg.train.lr),
        loss=cfg.train.loss,
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    history = head.fit(
        f_train,
        y_train,
        validation_split=0.15,
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        callbacks=_callbacks(cfg),
        verbose=0,
    )
    y_pred_enc = np.asarray(head.predict(f_test, batch_size=128, verbose=0))
    return history.history, y_pred_enc


def _fit_finetune(cfg: ExperimentConfig, corpus):
    """Trainable backbone: train the assembled model end-to-end on the images.

    Needed to reproduce the paper's scores and to expose the leakage gap — a frozen
    backbone produces different features for each rotation, so it cannot exploit the
    augmented-split leakage; an adaptable one can.
    """
    model = build_model(cfg)
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.train.lr),
        loss=cfg.train.loss,
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    name = cfg.backbone.name
    x_train = preprocess_for(name, corpus.x_train)
    x_test = preprocess_for(name, corpus.x_test)

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(x_train.shape[0])
    x_train = x_train[perm]
    y_train = np.asarray(corpus.y_train)[perm]

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.15,
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        callbacks=_callbacks(cfg),
        verbose=0,
    )
    y_pred_enc = np.asarray(model.predict(x_test, batch_size=64, verbose=0))
    return history.history, y_pred_enc

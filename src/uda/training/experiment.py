"""Experiment runner — k-fold CV on cached frozen features, either sampling protocol.

To make the bake-off affordable, we extract each frozen backbone's features **once**
over the full augmented corpus, then re-partition those cached features by fold — so
each fold is a fast head fit, not a full corpus build. The same extraction serves
both sampling protocols (the cache key is protocol-independent):

* ``protocol="patient"`` — leakage-free grouped k-fold by ``patient_id``
  (:func:`uda.training.cv.patient_kfold`); the default.
* ``protocol="image"``   — the paper's protocol: random k-fold over the augmented
  image rows (:func:`uda.training.cv.random_kfold`).

Results (mean±std over folds, tagged with a ``protocol`` column) are appended to
``results/era2019_cv.csv``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import keras
import numpy as np
import pandas as pd

from uda.models.backbones import build_backbone
from uda.config import ExperimentConfig
from uda.training.cv import patient_kfold, random_kfold
from uda.data.dataset import build_corpus
from uda.evaluation.evaluate import metrics
from uda.models.heads import build_head
from uda.seed import set_seed
from uda.models.targets import build_target
from uda.training.train import _callbacks, _extract_features

__all__ = [
    "build_full_features",
    "run_cv",
    "run_patient_cv",
    "cv_mape_objective",
    "ERA2019_CSV",
]

ERA2019_CSV = "results/era2019_cv.csv"
_METRICS = ["mae", "rmse", "me", "mape", "r2"]
_PROTOCOLS = ("patient", "image")


def build_full_features(cfg: ExperimentConfig, max_images: int | None = None):
    """Build the full augmented corpus once and extract frozen features.

    Returns ``(feats, y_deg, meta, target)`` where ``feats`` is ``(N, feat_dim)``,
    ``y_deg`` are angles in degrees aligned to ``feats``, and ``meta`` carries
    ``image_id``/``patient_id`` for fold partitioning. The corpus' internal split
    is irrelevant here — we recombine train+test into one pool and re-split by fold.
    """
    set_seed(cfg.seed)
    corpus = build_corpus(cfg, max_images=max_images)
    x_all = np.concatenate([corpus.x_train, corpus.x_test], axis=0)
    meta = corpus.meta.reset_index(drop=True)
    target = build_target(cfg.target)
    y_deg = np.concatenate(
        [
            np.asarray(target.decode(corpus.y_train), dtype=np.float64).ravel(),
            np.asarray(target.decode(corpus.y_test), dtype=np.float64).ravel(),
        ]
    )
    backbone = build_backbone(cfg.backbone)
    feats = _extract_features(cfg, backbone, x_all)
    return feats, y_deg, meta, target


def _fit_predict_masks(cfg, target, feats, y_deg, tr_mask, te_mask) -> dict:
    """Fit the head on the ``tr_mask`` rows, predict the ``te_mask`` rows, score.

    Both masks are boolean arrays over the same ``feats``/``y_deg`` rows; the caller
    decides how they were derived (patient-grouped base-image ids -> rows, or a plain
    random partition of the augmented rows). This is the single source of truth for the
    per-fold head fit shared by both sampling protocols.
    """
    f_tr, f_te = feats[tr_mask], feats[te_mask]
    y_tr_enc = np.asarray(target.encode(y_deg[tr_mask]), dtype=np.float32)

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(f_tr.shape[0])

    head = build_head(f_tr.shape[1], cfg.head, target.n_outputs)
    head.compile(
        optimizer=keras.optimizers.Adam(cfg.train.lr),
        loss=cfg.train.loss,
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    head.fit(
        f_tr[perm],
        y_tr_enc[perm],
        validation_split=0.15,
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        callbacks=_callbacks(cfg),
        verbose=0,
    )
    y_pred = np.asarray(target.decode(np.asarray(head.predict(f_te, batch_size=256, verbose=0)))).ravel()
    return metrics(y_deg[te_mask], y_pred)


def _fit_predict_fold(cfg, target, feats, y_deg, meta, train_ids, test_ids) -> dict:
    """Patient-protocol fold: build row masks from base ``image_id`` values, then fit."""
    tr = meta["image_id"].isin(set(train_ids.tolist())).to_numpy()
    te = meta["image_id"].isin(set(test_ids.tolist())).to_numpy()
    return _fit_predict_masks(cfg, target, feats, y_deg, tr, te)


def _idx_mask(n_rows: int, idx: np.ndarray) -> np.ndarray:
    """Boolean row mask of length ``n_rows`` with ``True`` at the integer ``idx``."""
    mask = np.zeros(n_rows, dtype=bool)
    mask[idx] = True
    return mask


def _cv_over_feats(cfg, target, feats, y_deg, meta, k, protocol, seed):
    """Run one protocol-matched k-fold over already-cached features; return the
    :func:`uda.training.cv` result dict ``{"folds", "aggregate"}``.

    ``protocol="patient"`` partitions the deduped base-image labels by
    ``patient_id`` (leakage-free); ``protocol="image"`` partitions the augmented rows
    of ``meta`` directly with a plain random k-fold (the paper's leaky protocol). Both
    feed the same :func:`_fit_predict_masks` head fit, so feature extraction is shared.
    """
    if protocol == "patient":
        labels = meta.drop_duplicates("image_id")[["image_id", "patient_id"]].reset_index(drop=True)

        def _patient_fold(train_ids, test_ids):
            return _fit_predict_fold(cfg, target, feats, y_deg, meta, train_ids, test_ids)

        return patient_kfold(labels, k=k, run_fold=_patient_fold, seed=seed)

    if protocol == "image":
        n_rows = len(meta)

        def _image_fold(train_idx, test_idx):
            tr = _idx_mask(n_rows, train_idx)
            te = _idx_mask(n_rows, test_idx)
            return _fit_predict_masks(cfg, target, feats, y_deg, tr, te)

        return random_kfold(n_rows, k=k, run_fold=_image_fold, seed=seed)

    raise ValueError(f"unknown protocol {protocol!r}; expected one of {_PROTOCOLS}")


def run_cv(
    cfg: ExperimentConfig,
    k: int = 5,
    out_csv: str | Path = ERA2019_CSV,
    richer: str = "off",
    max_images: int | None = None,
    feature_cache: dict | None = None,
    *,
    protocol: str = "patient",
) -> dict:
    """k-fold CV for one frozen config under ``protocol``; append a mean±std row.

    ``protocol="patient"`` (default) runs leakage-free patient k-fold; ``"image"`` runs
    the paper's random k-fold over the augmented rows. Both share one feature
    extraction, so a populated ``feature_cache`` lets both protocols reuse the same
    cached ``(feats, y_deg, meta)`` for a backbone.

    ``feature_cache`` (optional) is the same dict :func:`cv_mape_objective` fills:
    when a config with the same feature signature was already extracted (e.g. the
    tuned winner of a head-only search), the extraction is reused instead of rebuilt.
    """
    if protocol not in _PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {_PROTOCOLS}")
    if feature_cache is not None and _feature_sig(cfg) in feature_cache:
        feats, y_deg, meta = feature_cache[_feature_sig(cfg)]
        target = build_target(cfg.target)
    else:
        feats, y_deg, meta, target = build_full_features(cfg, max_images=max_images)
        if feature_cache is not None:
            feature_cache[_feature_sig(cfg)] = (feats, y_deg, meta)

    res = _cv_over_feats(cfg, target, feats, y_deg, meta, k=k, protocol=protocol, seed=cfg.seed)
    agg = res["aggregate"]
    row = {
        "name": cfg.name,
        "backbone": cfg.backbone.name,
        "pooling": cfg.backbone.pooling,
        "target": cfg.target.kind,
        "protocol": protocol,
        "richer_aug": richer,
        "era": cfg.era,
        "k": k,
    }
    for m in _METRICS:
        row[f"{m}_mean"] = round(agg[m]["mean"], 4)
        row[f"{m}_std"] = round(agg[m]["std"], 4)
    _append_row(out_csv, row)
    return row


# Backward-compatible alias: the original name was patient-only. Defaulting
# ``protocol="patient"`` keeps existing callers byte-for-byte identical except for the
# new "protocol" column in the appended row.
run_patient_cv = run_cv


def _feature_sig(cfg: ExperimentConfig) -> tuple:
    """Cache key for cached features: everything that changes the extracted feats.

    Head/optimizer knobs and ``target.kind`` are deliberately excluded — they only
    affect the shallow head fit (and ``y`` re-encoding), not the frozen features.
    """
    b = cfg.backbone
    return (
        b.name,
        b.pooling,
        b.weights,
        b.trainable,
        json.dumps(cfg.data.model_dump(mode="json"), sort_keys=True),
    )


def cv_mape_objective(
    base_cfg: ExperimentConfig,
    k: int = 5,
    feature_cache: dict | None = None,
    max_images: int | None = None,
    *,
    protocol: str = "patient",
) -> Callable[[ExperimentConfig], float]:
    """Injected objective for :func:`uda.training.tuning.run_study`: k-fold MAPE mean.

    Returns ``objective(cfg) -> float``. ``protocol="patient"`` (default) scores on
    leakage-free patient k-fold; ``protocol="image"`` scores on the paper's random
    k-fold over the augmented rows. The feature cache is keyed by :func:`_feature_sig`
    (protocol-independent), so a head/optimizer-only search (``dims="head"``) runs
    ``build_full_features`` exactly once **and the same extraction serves either
    protocol** — extract once per backbone, score under whichever protocol is asked.
    The ``target`` is rebuilt per trial (``target.kind`` only changes the encode width,
    not the cached features). Lower is better.
    """
    if protocol not in _PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {_PROTOCOLS}")
    cache = feature_cache if feature_cache is not None else {}

    def objective(cfg: ExperimentConfig) -> float:
        # The tuner is per-backbone: the search space may vary pooling/target but
        # never the backbone identity, which keeps the feature cache to one (or a
        # few, for pooling search) extraction(s) for this study.
        if cfg.backbone.name != base_cfg.backbone.name:
            raise ValueError(
                f"cv_mape_objective is bound to backbone {base_cfg.backbone.name!r}; "
                f"got a trial config for {cfg.backbone.name!r}"
            )
        sig = _feature_sig(cfg)
        if sig not in cache:
            feats, y_deg, meta, _ = build_full_features(cfg, max_images=max_images)
            cache[sig] = (feats, y_deg, meta)
        feats, y_deg, meta = cache[sig]
        target = build_target(cfg.target)

        res = _cv_over_feats(cfg, target, feats, y_deg, meta, k=k, protocol=protocol, seed=cfg.seed)
        return float(res["aggregate"]["mape"]["mean"])

    return objective


def _append_row(path: str | Path, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if p.exists():
        prev = pd.read_csv(p)
        prev = prev[prev["name"] != row["name"]]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(p, index=False)

"""Honest full-coverage 5-fold OOF ensemble of the tuned heads.

Each backbone's **protocol-matched** tuned config (``configs/tuned_<backbone>_<protocol>.yaml``,
falling back to the legacy ``configs/tuned_<backbone>.yaml`` if absent) is run through
5-fold CV under the requested sampling protocol; every augmented sample receives exactly one
**out-of-fold** prediction from a head trained on the other folds. Per-backbone OOF
predictions are persisted (same schema as the single-split prediction dumps) so the
ensemble is reproducible from CSVs without retraining, then combined:

- **mean** — the conservative headline number;
- **stacked** — Ridge ``cross_val_predict`` over the OOF columns (reported for
  reference; its meta-learner CV mixes rotations of an image, a mild optimism).

Two protocols (``--protocol``):

- ``patient`` (default) — leakage-free grouped 5-fold by ``patient_id``; a head never
  sees the held-out patient. Writes ``results/predictions/tuned_<backbone>_oof.csv``.
- ``image`` — the paper's protocol: random 5-fold over the full 2100 augmented rows
  (rotated copies of a base image may span folds). Writes
  ``results/predictions/tuned_<backbone>_image_oof.csv``.

Run: ``pixi run python scripts/oof_ensemble.py``. ~5 extractions + 25 head fits.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from uda.config import load_config
from uda.training.cv import patient_kfold, random_kfold
from uda.evaluation.ensemble import ensemble_predictions
from uda.evaluation.evaluate import metrics
from uda.training.experiment import build_full_features
from uda.models.heads import build_head
from uda.seed import set_seed
from uda.training.train import _callbacks

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results" / "predictions"
BACKBONES = ["densenet201", "resnet50", "vgg19", "xception", "inceptionv3"]

# protocol -> (corpus split strategy, OOF csv suffix)
_SPLIT_STRATEGY = {"patient": "patient", "image": "augmented"}
_OOF_SUFFIX = {"patient": "oof", "image": "image_oof"}


def _fold_masks(meta: pd.DataFrame, protocol: str, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Collect the per-fold (train_mask, test_mask) row masks for ``protocol``.

    ``patient``: patient-grouped 5-fold over the deduped base images, expanded to row
    masks via ``meta.image_id.isin(...)``. ``image``: a plain random 5-fold directly
    over the 2100 augmented rows, expanded from integer index arrays. Reuses the same
    Keras-free splitters as the CV harness so coverage is exhaustive and disjoint.
    """
    masks: list[tuple[np.ndarray, np.ndarray]] = []
    n_rows = len(meta)
    if protocol == "patient":
        labels = meta.drop_duplicates("image_id")[["image_id", "patient_id"]].reset_index(drop=True)

        def _patient_fold(tr_ids, te_ids):
            tr = meta["image_id"].isin(set(tr_ids.tolist())).to_numpy()
            te = meta["image_id"].isin(set(te_ids.tolist())).to_numpy()
            masks.append((tr, te))
            return {"n": int(te.sum())}

        patient_kfold(labels, 5, _patient_fold, seed=seed)
    elif protocol == "image":

        def _image_fold(tr_idx, te_idx):
            tr = np.zeros(n_rows, dtype=bool)
            te = np.zeros(n_rows, dtype=bool)
            tr[tr_idx] = True
            te[te_idx] = True
            masks.append((tr, te))
            return {"n": int(te.sum())}

        random_kfold(n_rows, 5, _image_fold, seed=seed)
    else:
        raise ValueError(f"unknown protocol {protocol!r}")
    return masks


def oof_predictions(cfg, protocol: str, max_images: int | None = None) -> pd.DataFrame:
    """Full-coverage out-of-fold predictions for one tuned config (in meta order)."""
    set_seed(cfg.seed)
    feats, y, meta, target = build_full_features(cfg, max_images=max_images)
    masks = _fold_masks(meta, protocol, seed=cfg.seed)

    pred = np.full(len(meta), np.nan)
    for tr, te in masks:
        y_tr = np.asarray(target.encode(y[tr]), dtype=np.float32)
        perm = np.random.default_rng(cfg.seed).permutation(int(tr.sum()))
        head = build_head(feats.shape[1], cfg.head, target.n_outputs)
        head.compile(
            optimizer=keras.optimizers.Adam(cfg.train.lr),
            loss=cfg.train.loss,
            metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
        )
        head.fit(
            feats[tr][perm], y_tr[perm], validation_split=0.15, epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size, callbacks=_callbacks(cfg), verbose=0,
        )
        pred[te] = np.asarray(
            target.decode(np.asarray(head.predict(feats[te], batch_size=256, verbose=0)))
        ).ravel()

    out = meta[["image_id", "patient_id", "rotation_deg"]].copy()
    out["theta_true"] = y
    out["theta_pred"] = pred
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["image", "patient"], default="patient")
    ap.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="cap base images for fast smoke runs (forwarded to build_full_features)",
    )
    args = ap.parse_args()
    protocol = args.protocol

    PRED.mkdir(parents=True, exist_ok=True)
    paths = []
    for b in BACKBONES:
        # Prefer the protocol-matched tuned config (best-per-protocol); fall back to
        # the legacy single tuned config so a pre-existing patient run still works.
        proto_cfg = ROOT / "configs" / f"tuned_{b}_{protocol}.yaml"
        cfg_path = proto_cfg if proto_cfg.exists() else ROOT / "configs" / f"tuned_{b}.yaml"
        cfg = load_config(cfg_path).model_copy(deep=True)
        cfg.split.strategy = _SPLIT_STRATEGY[protocol]
        df = oof_predictions(cfg, protocol, max_images=args.max_images)
        p = PRED / f"tuned_{b}_{_OOF_SUFFIX[protocol]}.csv"
        df.to_csv(p, index=False)
        m = metrics(df["theta_true"].to_numpy(), df["theta_pred"].to_numpy())
        print(f"{b} [{protocol}]: OOF 5-fold MAPE={m['mape']:.2f} MAE={m['mae']:.2f}  -> {p.name}", flush=True)
        paths.append(str(p))

    mean = ensemble_predictions(paths, method="mean")["metrics"]
    stk = ensemble_predictions(paths, method="stacked")["metrics"]
    print(
        f"OOF-5FOLD-ENSEMBLE [{protocol}]  mean MAPE={mean['mape']:.2f} MAE={mean['mae']:.2f}  |  "
        f"stacked MAPE={stk['mape']:.2f} MAE={stk['mae']:.2f} R2={stk['r2']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

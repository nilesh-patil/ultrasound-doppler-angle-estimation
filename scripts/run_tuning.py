"""Hyperparameter search — tune all 5 replication backbones, then re-ensemble.

For each backbone and each requested sampling protocol: run an Optuna TPE study
(head + optimizer dims) under the protocol-matched k-fold CV on cached frozen
features, record the honest tuned k-fold row in ``results/era2019_cv.csv``,
regenerate the protocol-matched single-split predictions, and finally recompute the
stacked ensemble from the tuned heads.

Two protocols are supported (``--protocol``):

- ``patient`` — leakage-free grouped k-fold by ``patient_id`` (the honest number);
- ``image``   — the paper's protocol: random k-fold over the augmented image rows.

The per-backbone feature extraction runs once and is shared between the study and
the honest CV row via a ``feature_cache`` dict; because the cache key is
protocol-independent, **one extraction per backbone serves both protocols**.
``epochs`` is capped (early stopping governs) to keep the head fits fast.

Run: ``pixi run python scripts/run_tuning.py --trials 40``
     (or ``pixi run -e mac-gpu python scripts/run_tuning.py ...``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from uda.config import dump_config, load_config
from uda.evaluation.ensemble import ensemble_predictions
from uda.training.experiment import cv_mape_objective, run_cv
from uda.training.tuning import run_study

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = ["densenet201", "resnet50", "vgg19", "xception", "inceptionv3"]
PROTOCOLS = ["patient", "image"]

# How each protocol maps onto the corpus split strategy used to regenerate the
# single-split prediction dumps the stacked ensemble consumes.
_SPLIT_STRATEGY = {"patient": "patient", "image": "augmented"}


def _single_split_predictions(cfg, protocol: str) -> str:
    """Train the tuned config on the protocol's single split; dump aligned predictions.

    Produces ``results/predictions/tuned_<backbone>_<protocol>.csv`` with the same
    held-out test rows across backbones (identical split config) — the alignment the
    stacked ensemble requires. Patient artifacts keep their historical ``_patient``
    suffix; the image protocol writes ``_image`` alongside, so prior data is never
    overwritten. The output name is derived from the canonical ``tuned_<backbone>``
    config name so the protocol suffix is applied exactly once (no double-suffix).
    """
    from uda.data.dataset import build_corpus
    from uda.evaluation.evaluate import evaluate
    from uda.training.train import train

    ps = cfg.model_copy(deep=True)
    ps.split.strategy = _SPLIT_STRATEGY[protocol]
    ps.name = f"tuned_{cfg.backbone.name}_{protocol}"
    corpus = build_corpus(ps)
    res = train(ps, corpus, out_dir=str(ROOT / "results"))
    evaluate(ps, res.y_test_true_deg, res.y_test_pred_deg, res.test_meta, out_dir=str(ROOT / "results"))
    return ps.name


def _tune_one(b: str, protocol: str, base, cache: dict, args: argparse.Namespace) -> tuple[str, float]:
    """Tune one backbone under one protocol, reusing ``cache``; return (pred_csv, best_MAPE).

    ``cache`` is shared across protocols for this backbone: the feature signature is
    protocol-independent, so the first protocol extracts the frozen features and every
    later protocol (and the tuned CV row) reuses them — one extraction per backbone, not
    one per (backbone, protocol).
    """
    pbase = base.model_copy(deep=True)
    pbase.split.strategy = _SPLIT_STRATEGY[protocol]
    objective = cv_mape_objective(
        pbase, k=args.k, feature_cache=cache, max_images=args.max_images, protocol=protocol
    )
    print(f"\n=== tuning {b} [{protocol}] ({args.trials} trials, {args.k}-fold) ===", flush=True)
    res = run_study(
        pbase,
        objective,
        n_trials=args.trials,
        study_name=f"tune_{b}_{protocol}",
        out_dir=str(ROOT / "results" / "tuning"),
        configs_dir=str(ROOT / "configs"),
        dims="head",
    )
    print(f"  best MAPE({args.k}-fold)={res['best_value']:.3f}  params={res['best_params']}", flush=True)

    # run_study writes configs/tuned_<backbone>.yaml; copy it to a protocol-suffixed
    # name so the patient/image winners coexist and prior configs are never lost.
    tuned = load_config(res["best_config_path"]).model_copy(deep=True)
    tuned.name = f"tuned_{b}_{protocol}"
    dump_config(tuned, ROOT / "configs" / f"tuned_{b}_{protocol}.yaml")

    row = run_cv(
        tuned, k=args.k, feature_cache=cache, max_images=args.max_images, protocol=protocol
    )  # reuses cached feats
    print(f"  tuned CV row [{protocol}]: MAPE={row['mape_mean']}±{row['mape_std']}  MAE={row['mae_mean']}", flush=True)

    name = _single_split_predictions(tuned, protocol)
    return str(ROOT / "results" / "predictions" / f"{name}.csv"), res["best_value"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80, help="epoch cap; early stopping governs")
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument(
        "--protocol",
        choices=["image", "patient", "both"],
        default="both",
        help="sampling protocol(s) to tune under",
    )
    ap.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="cap base images for fast smoke runs (forwarded to build_full_features)",
    )
    args = ap.parse_args()

    protocols = PROTOCOLS if args.protocol == "both" else [args.protocol]
    pred_paths: dict[str, list[str]] = {p: [] for p in protocols}
    summary: dict[str, list[tuple[str, float]]] = {p: [] for p in protocols}

    # Backbone-outer / protocol-inner: one shared feature cache per backbone serves
    # every protocol, so the expensive frozen extraction runs once per backbone.
    for b in args.backbones:
        base = load_config(ROOT / "configs" / f"replication_{b}.yaml").model_copy(deep=True)
        base.train.epochs = args.epochs
        cache: dict = {}
        for protocol in protocols:
            path, best = _tune_one(b, protocol, base, cache, args)
            pred_paths[protocol].append(path)
            summary[protocol].append((b, best))

    for protocol in protocols:
        print(f"\n##### protocol = {protocol} #####", flush=True)
        print(f"=== per-backbone tuned {args.k}-fold MAPE [{protocol}] ===", flush=True)
        for b, val in summary[protocol]:
            print(f"  {b:14s} {val:6.3f}", flush=True)
        if len(pred_paths[protocol]) >= 2:
            ens = ensemble_predictions(pred_paths[protocol], method="stacked")
            m = ens["metrics"]
            print(f"=== tuned stacked ensemble [{protocol}] ({ens['n_models']} models) ===", flush=True)
            print(
                f"  MAPE={m['mape']:.3f}  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  R2={m['r2']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()

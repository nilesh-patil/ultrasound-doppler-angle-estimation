"""Optuna TPE hyperparameter search over the head + optimizer (Keras-free).

This module is the orchestration + search-space
half of the tuner; it imports only ``optuna``/``pandas``/``uda.config`` and never a
deep-learning backend. The *trained* objective — a patient k-fold CV MAPE over
cached frozen features — is **injected** (production wires
:func:`uda.training.experiment.cv_mape_objective`; unit tests pass a cheap synthetic one),
exactly the dependency-injection pattern used by :func:`uda.training.cv.patient_kfold`.

Why the search is cheap: the expensive backbone feature extraction happens once
per ``(backbone, pooling, data)`` combo inside the objective's feature cache, so
every trial here only re-fits a shallow head. The default ``dims="head"`` search
space never touches the feature-invariant fields (``backbone.name``, ``data.*``),
guaranteeing one extraction reused across all trials.

Determinism: ``TPESampler(seed=seed)`` fixes the trial sequence, and the best
config is rebuilt by replaying ``study.best_params`` through :func:`suggest_config`
on a ``FixedTrial`` — so the persisted YAML always matches the reported winner.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import optuna
import pandas as pd

from uda.config import ExperimentConfig, dump_config

__all__ = ["suggest_config", "run_study"]

Objective = Callable[[ExperimentConfig], float]

_UNIT_CHOICES = [128, 256, 512]
_BATCH_CHOICES = [16, 32, 64]
_PATIENCE_CHOICES = [10, 15, 20]
_POOL_CHOICES = ["grid2", "grid3", "avgmax"]
_TARGET_CHOICES = ["raw", "sincos2theta"]


def suggest_config(
    trial: optuna.trial.Trial,
    base_cfg: ExperimentConfig,
    dims: str = "head",
) -> ExperimentConfig:
    """Define-by-run sample of one :class:`ExperimentConfig` from ``base_cfg``.

    ``base_cfg`` is deep-copied (never mutated). Head + optimizer knobs are always
    sampled; ``"pool"`` and/or ``"target"`` appearing in ``dims`` additionally
    sample ``backbone.pooling`` / ``target.kind`` (these invalidate the feature
    cache, so they are off by default). For ``dims="head"`` the feature-invariant
    fields (``backbone.name``, ``data.*``) are left exactly as in ``base_cfg``.

    The returned config is in-range by construction; its strict-schema validity is
    re-checked whenever it round-trips through ``dump_config``/``load_config``.
    """
    cfg = base_cfg.model_copy(deep=True)

    # --- head: depth/width, regularization ---
    n_layers = trial.suggest_int("n_layers", 1, 2)
    cfg.head.hidden_units = [
        int(trial.suggest_categorical(f"units_l{i}", _UNIT_CHOICES)) for i in range(n_layers)
    ]
    cfg.head.dropout = trial.suggest_float("dropout", 0.0, 0.6)
    cfg.head.l2 = trial.suggest_float("l2", 1e-6, 1e-2, log=True)
    cfg.head.batchnorm = bool(trial.suggest_categorical("batchnorm", [True, False]))

    # --- optimizer / early-stopping schedule (epochs governed by early stopping) ---
    cfg.train.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    cfg.train.batch_size = int(trial.suggest_categorical("batch_size", _BATCH_CHOICES))
    cfg.train.early_stopping_patience = int(
        trial.suggest_categorical("patience", _PATIENCE_CHOICES)
    )

    # --- optional feature-invalidating dims (each adds one cache miss) ---
    if "pool" in dims:
        cfg.backbone.pooling = trial.suggest_categorical("pooling", _POOL_CHOICES)
    if "target" in dims:
        cfg.target.kind = trial.suggest_categorical("target_kind", _TARGET_CHOICES)

    return cfg


def _trials_dataframe(study: optuna.study.Study) -> pd.DataFrame:
    """One row per trial — ``number, value, state, param_<k>...`` (no timestamps).

    Timestamps are deliberately omitted so the CSV is a deterministic function of
    ``(base_cfg, n_trials, objective, seed)``. Conditional params (e.g. ``units_l1``
    only when ``n_layers==2``) are unioned across trials; absent cells are blank.
    """
    rows = []
    for t in study.trials:
        row = {"number": t.number, "value": t.value, "state": t.state.name}
        for key, val in t.params.items():
            row[f"param_{key}"] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    fixed = ["number", "value", "state"]
    param_cols = sorted(c for c in df.columns if c.startswith("param_"))
    return df[fixed + param_cols]


def run_study(
    base_cfg: ExperimentConfig,
    objective: Objective,
    *,
    n_trials: int,
    study_name: str | None = None,
    out_dir: str | Path = "results/tuning",
    configs_dir: str | Path = "configs",
    seed: int | None = None,
    direction: str = "minimize",
    dims: str = "head",
) -> dict:
    """Run an Optuna TPE study over ``suggest_config`` and persist its artifacts.

    Writes ``<out_dir>/<study_name>.csv`` (per-trial table) and
    ``<configs_dir>/tuned_<backbone>.yaml`` (the best config, ``name=
    'tuned_<backbone>'``, via :func:`uda.config.dump_config`). Returns
    ``{"best_params", "best_value", "best_config_path", "trials_csv", "n_trials"}``.

    ``objective(cfg) -> float`` is injected; ``study_name`` defaults to
    ``base_cfg.name`` and ``seed`` to ``base_cfg.seed``.
    """
    study_name = study_name or base_cfg.name
    seed = base_cfg.seed if seed is None else seed

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler, study_name=study_name)

    def _opt(trial: optuna.trial.Trial) -> float:
        cfg = suggest_config(trial, base_cfg, dims=dims)
        return float(objective(cfg))

    study.optimize(_opt, n_trials=n_trials)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_csv = out_dir / f"{study_name}.csv"
    _trials_dataframe(study).to_csv(trials_csv, index=False)

    # Rebuild the winner deterministically by replaying its params.
    best_cfg = suggest_config(optuna.trial.FixedTrial(study.best_params), base_cfg, dims=dims)
    best_name = f"tuned_{base_cfg.backbone.name}"
    best_cfg.name = best_name
    configs_dir = Path(configs_dir)
    configs_dir.mkdir(parents=True, exist_ok=True)
    best_config_path = configs_dir / f"{best_name}.yaml"
    dump_config(best_cfg, best_config_path)

    return {
        "best_params": dict(study.best_params),
        "best_value": float(study.best_value),
        "best_config_path": str(best_config_path),
        "trials_csv": str(trials_csv),
        "n_trials": len(study.trials),
    }

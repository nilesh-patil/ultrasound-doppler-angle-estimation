"""Tests — Optuna hyperparameter search (``uda.training.tuning``).

The module is **Keras-free** (asserted explicitly): search-space construction and
study orchestration import only ``optuna``/``numpy``/``pandas``/``uda.config``.
The trained objective is *injected*, so these tests never build or train a model —
a cheap synthetic objective with a known optimum stands in for the real patient-CV
MAPE. Every test here runs in well under a second.

Mirrors ``tests/test_cv.py`` (inline configs, ``tmp_path`` artifacts, a fresh-
interpreter Keras-free guard).
"""
from __future__ import annotations

import math
import sys

import optuna
import pandas as pd
import pytest

from uda.config import ExperimentConfig, load_config
from uda.training.tuning import run_study, suggest_config

optuna.logging.set_verbosity(optuna.logging.WARNING)


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _base_cfg() -> ExperimentConfig:
    """A frozen DenseNet201 grid3 base config (the v1.1 anchor)."""
    return ExperimentConfig(
        name="tune_test",
        backbone={"name": "densenet201", "pooling": "grid3"},
        target={"kind": "raw"},
        train={"epochs": 50},
    )


# all params for the n_layers==1 path of ``suggest_config(dims="head")``
_HEAD_PARAMS_1 = {
    "n_layers": 1,
    "units_l0": 256,
    "dropout": 0.3,
    "l2": 1e-4,
    "batchnorm": True,
    "lr": 3e-4,
    "batch_size": 32,
    "patience": 15,
}


def _log10_lr_objective(target_lr: float = 3e-4):
    """Synthetic, Keras-free objective minimized at ``train.lr == target_lr``."""
    t = math.log10(target_lr)

    def objective(cfg: ExperimentConfig) -> float:
        return (math.log10(cfg.train.lr) - t) ** 2

    return objective


# --------------------------------------------------------------------------- #
# 1. suggest_config validity
# --------------------------------------------------------------------------- #
def test_suggest_config_returns_valid_in_range_config():
    base = _base_cfg()
    trial = optuna.trial.FixedTrial(_HEAD_PARAMS_1)
    cfg = suggest_config(trial, base, dims="head")

    assert isinstance(cfg, ExperimentConfig)
    # sampled head/optimizer fields are threaded through verbatim
    assert cfg.head.hidden_units == [256]
    assert cfg.head.dropout == pytest.approx(0.3)
    assert cfg.head.l2 == pytest.approx(1e-4)
    assert cfg.head.batchnorm is True
    assert cfg.train.lr == pytest.approx(3e-4)
    assert cfg.train.batch_size == 32
    assert cfg.train.early_stopping_patience == 15


def test_suggest_config_two_layers_sets_hidden_units_length():
    base = _base_cfg()
    params = {**_HEAD_PARAMS_1, "n_layers": 2, "units_l1": 128}
    cfg = suggest_config(optuna.trial.FixedTrial(params), base, dims="head")
    assert cfg.head.hidden_units == [256, 128]


def test_suggest_config_head_dims_leave_feature_invariant_fields_untouched():
    """For dims='head', the backbone + data (which gate feature extraction) are
    never mutated — that is what lets every trial reuse one cached extraction."""
    base = _base_cfg()
    cfg = suggest_config(optuna.trial.FixedTrial(_HEAD_PARAMS_1), base, dims="head")
    assert cfg.backbone.name == base.backbone.name
    assert cfg.backbone.pooling == base.backbone.pooling
    assert cfg.data == base.data
    # base is not mutated in place
    assert base.train.lr == pytest.approx(1e-4)


# --------------------------------------------------------------------------- #
# 2. dims gating
# --------------------------------------------------------------------------- #
def test_dims_head_does_not_sample_pooling_or_target():
    """A FixedTrial WITHOUT 'pooling'/'target_kind' must succeed for dims='head':
    if suggest_config sampled them, FixedTrial would raise on the missing key."""
    base = _base_cfg()
    cfg = suggest_config(optuna.trial.FixedTrial(_HEAD_PARAMS_1), base, dims="head")
    assert cfg.backbone.pooling == base.backbone.pooling
    assert cfg.target.kind == base.target.kind


def test_dims_head_pool_target_samples_pooling_and_target():
    base = _base_cfg()
    params = {**_HEAD_PARAMS_1, "pooling": "grid2", "target_kind": "sincos2theta"}
    cfg = suggest_config(optuna.trial.FixedTrial(params), base, dims="head+pool+target")
    assert cfg.backbone.pooling == "grid2"
    assert cfg.target.kind == "sincos2theta"


# --------------------------------------------------------------------------- #
# 3. run_study convergence
# --------------------------------------------------------------------------- #
def test_run_study_converges_toward_known_optimum(tmp_path):
    base = _base_cfg()
    res = run_study(
        base,
        _log10_lr_objective(3e-4),
        n_trials=40,
        out_dir=tmp_path / "tuning",
        configs_dir=tmp_path / "configs",
        seed=0,
    )
    assert math.isfinite(res["best_value"])
    # within ~0.5 dex of the target lr (best_value is the squared log10 distance)
    assert res["best_value"] < 0.25
    assert abs(math.log10(res["best_params"]["lr"]) - math.log10(3e-4)) < 0.5


def test_run_study_is_deterministic_for_a_fixed_seed(tmp_path):
    base = _base_cfg()
    obj = _log10_lr_objective(3e-4)
    a = run_study(base, obj, n_trials=15, out_dir=tmp_path / "a", configs_dir=tmp_path / "ca", seed=7)
    b = run_study(base, obj, n_trials=15, out_dir=tmp_path / "b", configs_dir=tmp_path / "cb", seed=7)
    assert a["best_params"] == b["best_params"]
    assert a["best_value"] == pytest.approx(b["best_value"])


# --------------------------------------------------------------------------- #
# 4. artifacts
# --------------------------------------------------------------------------- #
def test_run_study_writes_trials_csv_and_best_config_yaml(tmp_path):
    base = _base_cfg()
    res = run_study(
        base,
        _log10_lr_objective(3e-4),
        n_trials=12,
        out_dir=tmp_path / "tuning",
        configs_dir=tmp_path / "configs",
        seed=0,
    )

    trials_csv = res["trials_csv"]
    df = pd.read_csv(trials_csv)
    assert len(df) == 12
    assert {"number", "value", "state"} <= set(df.columns)
    assert any(c.startswith("param_") for c in df.columns)

    best_cfg = load_config(res["best_config_path"])  # round-trips the strict schema
    assert best_cfg.name == "tuned_densenet201"
    assert best_cfg.train.lr == pytest.approx(res["best_params"]["lr"])
    assert best_cfg.train.batch_size == res["best_params"]["batch_size"]
    # the winning config preserves the feature-invariant backbone identity
    assert best_cfg.backbone.name == "densenet201"


def test_every_sampled_config_is_valid_and_in_range(tmp_path):
    """Across a real (mini) TPE study, every config suggest_config produces is a
    valid ExperimentConfig with in-range fields — proven by collecting them all."""
    base = _base_cfg()
    seen: list[ExperimentConfig] = []

    def objective(cfg: ExperimentConfig) -> float:
        seen.append(cfg)
        return float(cfg.train.lr)

    run_study(base, objective, n_trials=25, out_dir=tmp_path / "t", configs_dir=tmp_path / "c", seed=1)
    assert len(seen) == 25
    for cfg in seen:
        assert 1e-5 <= cfg.train.lr <= 1e-3
        assert 0.0 <= cfg.head.dropout <= 0.6
        assert 1e-6 <= cfg.head.l2 <= 1e-2
        assert cfg.train.batch_size in (16, 32, 64)
        assert cfg.train.early_stopping_patience in (10, 15, 20)
        assert 1 <= len(cfg.head.hidden_units) <= 2
        assert all(u in (128, 256, 512) for u in cfg.head.hidden_units)
        assert cfg.backbone.name == "densenet201"  # feature-invariant under dims='head'


# --------------------------------------------------------------------------- #
# 5. Keras-free
# --------------------------------------------------------------------------- #
def test_tuning_module_is_keras_free():
    """Importing ``uda.training.tuning`` must not pull a heavy backend (keras/jax/tensorflow)
    — the trained objective is injected. Checked in a FRESH interpreter so other
    tests' backend imports don't pollute ``sys.modules``."""
    import subprocess

    code = (
        "import uda.training.tuning, sys; "
        "bad = [b for b in ('keras', 'jax', 'tensorflow') if b in sys.modules]; "
        "print(','.join(bad)); "
        "sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"uda.training.tuning pulled a backend: {r.stdout.strip()}\n{r.stderr[-300:]}"

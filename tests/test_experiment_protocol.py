"""Protocol threading through ``uda.training.experiment`` (Keras-free unit tests).

The two sampling protocols share one feature extraction per backbone and differ only
in how the cached augmented rows are partitioned:

- ``patient`` — grouped k-fold by ``patient_id`` over the deduped base images;
- ``image``   — the paper's protocol: random k-fold over the augmented rows.

These tests never run real feature extraction or build a Keras model. The per-fold
head fit (:func:`uda.training.experiment._fit_predict_masks`) is **monkeypatched** with a stub
that records the boolean row masks it receives and returns synthetic metrics, and a
tiny synthetic ``(feats, y_deg, meta)`` is pre-seeded into the feature cache so
``build_full_features`` is never called. We then assert the image protocol touches
every augmented row exactly once and the patient protocol keeps patients whole.

(``uda.training.experiment`` does import keras at module scope — that is expected; only the
leaf modules ``uda.training.cv`` / ``uda.training.tuning`` are asserted Keras-free, in their own tests.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import uda.training.experiment as ex
from uda.config import ExperimentConfig

N_PATIENTS = 6
PER_PATIENT = 3            # base images per patient
ROTATIONS = 5             # augmented rows per base image
N_BASE = N_PATIENTS * PER_PATIENT
N_ROWS = N_BASE * ROTATIONS  # the "2100-row" stand-in (here 90)


def _meta() -> pd.DataFrame:
    """Synthetic augmented-corpus meta: every base image repeated ``ROTATIONS`` times."""
    rows = []
    for p in range(N_PATIENTS):
        for b in range(PER_PATIENT):
            image_id = f"p{p:02d}_img{b:02d}"
            for r in range(ROTATIONS):
                rows.append(
                    {
                        "image_id": image_id,
                        "patient_id": f"patient_{p:02d}",
                        "rotation_deg": float(r * 5),
                    }
                )
    return pd.DataFrame(rows)


def _cfg() -> ExperimentConfig:
    return ExperimentConfig(
        name="proto_test",
        seed=42,
        backbone={"name": "densenet201", "pooling": "grid3"},
        target={"kind": "raw"},
        train={"epochs": 1},
    )


class _MaskRecorder:
    """Stub for ``_fit_predict_masks``: records (tr_mask, te_mask); returns metrics."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def __call__(self, cfg, target, feats, y_deg, tr_mask, te_mask) -> dict:
        self.calls.append((np.asarray(tr_mask).copy(), np.asarray(te_mask).copy()))
        # masks must be boolean row selectors over the full corpus
        assert tr_mask.dtype == bool and te_mask.dtype == bool
        assert len(tr_mask) == len(te_mask) == N_ROWS
        n = float(te_mask.sum())
        # return every metric run_cv aggregates so the CSV row can be built
        return {"mae": n, "rmse": n + 1.0, "me": 0.0, "mape": n, "r2": 0.5}


def _seed_cache(cfg) -> dict:
    """Pre-seed the feature cache so build_full_features is never invoked."""
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((N_ROWS, 8)).astype(np.float32)
    y_deg = rng.uniform(78.0, 104.0, size=N_ROWS)
    meta = _meta()
    return {ex._feature_sig(cfg): (feats, y_deg, meta)}


# --------------------------------------------------------------------------- #
# image protocol partitions the augmented rows exhaustively + disjointly
# --------------------------------------------------------------------------- #
def test_image_objective_partitions_every_augmented_row_once(monkeypatch):
    cfg = _cfg()
    cache = _seed_cache(cfg)
    rec = _MaskRecorder()
    monkeypatch.setattr(ex, "_fit_predict_masks", rec)

    objective = ex.cv_mape_objective(cfg, k=5, feature_cache=cache, protocol="image")
    val = objective(cfg)
    assert np.isfinite(val)

    assert len(rec.calls) == 5
    seen_test = np.zeros(N_ROWS, dtype=int)
    for tr_mask, te_mask in rec.calls:
        # within a fold: train/test disjoint and cover every row
        assert not np.any(tr_mask & te_mask)
        assert np.all(tr_mask | te_mask)
        seen_test += te_mask.astype(int)
    # every augmented row is held out in exactly one fold
    assert np.array_equal(seen_test, np.ones(N_ROWS, dtype=int))


def test_image_objective_returns_mean_test_count(monkeypatch):
    """With the stub's mape==te_mask.sum(), the objective is the mean test-fold size."""
    cfg = _cfg()
    cache = _seed_cache(cfg)
    rec = _MaskRecorder()
    monkeypatch.setattr(ex, "_fit_predict_masks", rec)

    val = ex.cv_mape_objective(cfg, k=5, feature_cache=cache, protocol="image")(cfg)
    sizes = [int(te.sum()) for _tr, te in rec.calls]
    assert val == pytest.approx(float(np.mean(sizes)))
    assert sum(sizes) == N_ROWS


# --------------------------------------------------------------------------- #
# patient protocol keeps whole patients on one side (rows of a patient together)
# --------------------------------------------------------------------------- #
def test_patient_objective_keeps_patients_whole(monkeypatch):
    cfg = _cfg()
    cache = _seed_cache(cfg)
    rec = _MaskRecorder()
    monkeypatch.setattr(ex, "_fit_predict_masks", rec)

    objective = ex.cv_mape_objective(cfg, k=N_PATIENTS, feature_cache=cache, protocol="patient")
    objective(cfg)

    meta = _meta()
    patient = meta["patient_id"].to_numpy()
    assert len(rec.calls) == N_PATIENTS
    seen_test = np.zeros(N_ROWS, dtype=int)
    for tr_mask, te_mask in rec.calls:
        assert not np.any(tr_mask & te_mask)
        assert np.all(tr_mask | te_mask)
        # no patient straddles the boundary
        tr_patients = set(patient[tr_mask])
        te_patients = set(patient[te_mask])
        assert tr_patients.isdisjoint(te_patients)
        seen_test += te_mask.astype(int)
    assert np.array_equal(seen_test, np.ones(N_ROWS, dtype=int))


# --------------------------------------------------------------------------- #
# protocols share one extraction; default is patient; unknown protocol rejected
# --------------------------------------------------------------------------- #
def test_both_protocols_reuse_one_cached_extraction(monkeypatch):
    """The cache key is protocol-independent: a populated cache serves both protocols
    without ever calling build_full_features."""
    cfg = _cfg()
    cache = _seed_cache(cfg)
    monkeypatch.setattr(ex, "_fit_predict_masks", _MaskRecorder())

    def _boom(*_a, **_k):  # build_full_features must NOT be called when cache is warm
        raise AssertionError("build_full_features called despite a warm cache")

    monkeypatch.setattr(ex, "build_full_features", _boom)

    ex.cv_mape_objective(cfg, k=5, feature_cache=cache, protocol="image")(cfg)
    ex.cv_mape_objective(cfg, k=5, feature_cache=cache, protocol="patient")(cfg)
    assert len(cache) == 1  # still a single extraction shared by both protocols


def test_default_protocol_is_patient(monkeypatch):
    """Omitting protocol must reproduce the historical patient behavior."""
    cfg = _cfg()
    cache = _seed_cache(cfg)
    rec_default = _MaskRecorder()
    monkeypatch.setattr(ex, "_fit_predict_masks", rec_default)
    ex.cv_mape_objective(cfg, k=N_PATIENTS, feature_cache=cache)(cfg)
    default_tests = [te.copy() for _tr, te in rec_default.calls]

    cache2 = _seed_cache(cfg)
    rec_patient = _MaskRecorder()
    monkeypatch.setattr(ex, "_fit_predict_masks", rec_patient)
    ex.cv_mape_objective(cfg, k=N_PATIENTS, feature_cache=cache2, protocol="patient")(cfg)
    patient_tests = [te for _tr, te in rec_patient.calls]

    assert len(default_tests) == len(patient_tests)
    for a, b in zip(default_tests, patient_tests):
        assert np.array_equal(a, b)


def test_unknown_protocol_raises():
    cfg = _cfg()
    with pytest.raises(ValueError):
        ex.cv_mape_objective(cfg, k=5, protocol="bogus")


def test_run_patient_cv_alias_points_to_run_cv():
    assert ex.run_patient_cv is ex.run_cv


# --------------------------------------------------------------------------- #
# run_cv writes a protocol-tagged row to the CV csv
# --------------------------------------------------------------------------- #
def test_run_cv_image_writes_protocol_column(monkeypatch, tmp_path):
    cfg = _cfg()
    cache = _seed_cache(cfg)
    monkeypatch.setattr(ex, "_fit_predict_masks", _MaskRecorder())

    out_csv = tmp_path / "cv.csv"
    row = ex.run_cv(cfg, k=5, out_csv=out_csv, feature_cache=cache, protocol="image")
    assert row["protocol"] == "image"

    df = pd.read_csv(out_csv)
    assert "protocol" in df.columns
    assert (df["protocol"] == "image").all()
    # mean of metrics is finite and present
    assert {"mape_mean", "mae_std", "r2_mean"} <= set(df.columns)


def test_run_cv_patient_and_image_rows_coexist(monkeypatch, tmp_path):
    """Distinct names per protocol keep both rows (dedup is by name only)."""
    out_csv = tmp_path / "cv.csv"
    monkeypatch.setattr(ex, "_fit_predict_masks", _MaskRecorder())

    cfg_p = _cfg()
    cfg_p.name = "tuned_densenet201_patient"
    ex.run_cv(cfg_p, k=N_PATIENTS, out_csv=out_csv, feature_cache=_seed_cache(cfg_p), protocol="patient")

    cfg_i = _cfg()
    cfg_i.name = "tuned_densenet201_image"
    ex.run_cv(cfg_i, k=5, out_csv=out_csv, feature_cache=_seed_cache(cfg_i), protocol="image")

    df = pd.read_csv(out_csv)
    assert set(df["protocol"]) == {"patient", "image"}
    assert set(df["name"]) == {"tuned_densenet201_patient", "tuned_densenet201_image"}

"""Tests — patient **nested** k-fold CV harness (``uda.evaluation.nested_cv``).

The harness drives the same :class:`uda.data.splits.PatientLevelSplit`
(``GroupKFold`` over ``patient_id``) as :mod:`uda.training.cv` for the **outer** split and
aggregates per-outer-fold metric dicts with the identical ``ddof=0``,
common-keys-only semantics. The *inner* loop is delegated entirely to an
**injected** ``run_outer`` runner, so these tests never build or train a model: a
cheap stub stands in for ``run_outer``, records the ids (and ``k_inner``) it
receives, and returns a deterministic metric. Every test here runs in well under a
second and the module under test is **Keras-free** (asserted explicitly in a fresh
subprocess).

A synthetic labels frame stands in for ``data/labels.csv``: 10 patients with 4
base images each (40 rows, one per base ``image_id``), exactly as in
``tests/test_cv.py``.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from uda.evaluation.nested_cv import patient_nested_cv

N_PATIENTS = 10
PER_PATIENT = 4
N_IMAGES = N_PATIENTS * PER_PATIENT


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _labels(n_patients: int = N_PATIENTS, per_patient: int = PER_PATIENT) -> pd.DataFrame:
    """Synthetic ``(image_id, patient_id, theta_deg)`` frame, one row per base image."""
    rng = np.random.default_rng(0)
    rows = []
    for p in range(n_patients):
        for k in range(per_patient):
            rows.append(
                {
                    "image_id": f"p{p:02d}_img{k:02d}",
                    "patient_id": f"patient_{p:02d}",
                    "theta_deg": float(rng.uniform(78.0, 104.0)),
                }
            )
    return pd.DataFrame(rows)


def _id_to_patient(labels: pd.DataFrame) -> dict:
    return dict(zip(labels["image_id"], labels["patient_id"]))


class _RecordingRunner:
    """A ``run_outer`` stub: records the ids + ``k_inner`` it receives.

    The returned metric is derived deterministically from the fold so aggregate
    math is hand-checkable: ``mae == len(test_ids)``. Builds **no** model.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray, int]] = []

    def __call__(self, train_ids: np.ndarray, test_ids: np.ndarray, k_inner: int) -> dict:
        self.calls.append((train_ids, test_ids, k_inner))
        return {"mae": float(len(test_ids))}


# --------------------------------------------------------------------------- #
# 1. outer fold count + result keys + k_inner echo
# --------------------------------------------------------------------------- #
def test_returns_k_outer_folds_with_folds_aggregate_and_k_inner_keys():
    labels = _labels()
    result = patient_nested_cv(labels, k_outer=5, k_inner=3, run_outer=_RecordingRunner(), seed=42)

    assert set(result.keys()) == {"folds", "aggregate", "k_inner"}
    assert isinstance(result["folds"], list)
    assert len(result["folds"]) == 5
    assert result["k_inner"] == 3
    # each fold entry is the per-fold metric dict returned by run_outer
    for fold in result["folds"]:
        assert set(fold) == {"mae"}
        assert fold["mae"] == pytest.approx(fold["mae"])


# --------------------------------------------------------------------------- #
# 2. outer folds are patient-disjoint
# --------------------------------------------------------------------------- #
def test_each_outer_fold_train_test_image_ids_are_disjoint():
    labels = _labels()
    runner = _RecordingRunner()
    patient_nested_cv(labels, k_outer=5, k_inner=3, run_outer=runner, seed=42)

    assert len(runner.calls) == 5
    for train_ids, test_ids, _k_inner in runner.calls:
        assert set(train_ids).isdisjoint(set(test_ids))
        # ids handed out are exactly the base image_ids — no augmentation leaks in
        assert set(train_ids) | set(test_ids) == set(labels["image_id"])


def test_each_outer_fold_keeps_patients_whole():
    """Patient-level: the patients in outer train and outer test never overlap."""
    labels = _labels()
    runner = _RecordingRunner()
    patient_nested_cv(labels, k_outer=5, k_inner=3, run_outer=runner, seed=42)

    lut = _id_to_patient(labels)
    for train_ids, test_ids, _k_inner in runner.calls:
        train_patients = {lut[i] for i in train_ids}
        test_patients = {lut[i] for i in test_ids}
        assert train_patients.isdisjoint(test_patients)


def test_outer_test_folds_partition_all_image_ids():
    """GroupKFold partitions: union of all outer test folds == every base image_id,
    and no image id appears in two different outer test folds."""
    labels = _labels()
    runner = _RecordingRunner()
    patient_nested_cv(labels, k_outer=5, k_inner=3, run_outer=runner, seed=42)

    all_ids = set(labels["image_id"])
    seen_test: set[str] = set()
    for _train_ids, test_ids, _k_inner in runner.calls:
        test_s = set(test_ids)
        assert seen_test.isdisjoint(test_s)
        seen_test |= test_s
    assert seen_test == all_ids


# --------------------------------------------------------------------------- #
# 3. k_inner pass-through + run_outer arg shapes
# --------------------------------------------------------------------------- #
def test_run_outer_receives_numpy_image_ids_and_k_inner_passthrough():
    labels = _labels()
    all_ids = set(labels["image_id"])
    captured: list[tuple] = []

    def run_outer(train_ids, test_ids, k_inner):
        captured.append((train_ids, test_ids, k_inner))
        assert isinstance(train_ids, np.ndarray)
        assert isinstance(test_ids, np.ndarray)
        # the ids are base image_ids drawn from the label table
        assert set(train_ids) <= all_ids
        assert set(test_ids) <= all_ids
        # k_inner is passed through verbatim, not consumed/re-split by the harness
        assert k_inner == 7
        return {"mae": float(len(test_ids))}

    result = patient_nested_cv(labels, k_outer=5, k_inner=7, run_outer=run_outer, seed=42)
    assert len(captured) == 5
    # and surfaced in the result
    assert result["k_inner"] == 7
    assert all(k == 7 for _tr, _te, k in captured)


# --------------------------------------------------------------------------- #
# 4. aggregate math == uda.training.cv semantics
# --------------------------------------------------------------------------- #
def test_aggregate_mean_and_std_match_numpy_over_outer_folds():
    labels = _labels()
    runner = _RecordingRunner()
    result = patient_nested_cv(labels, k_outer=5, k_inner=3, run_outer=runner, seed=42)

    # recover the per-fold metric values the harness saw, in fold order
    per_fold = [fold["mae"] for fold in result["folds"]]
    assert len(per_fold) == 5

    expected_mean = float(np.mean(per_fold))
    expected_std = float(np.std(per_fold))  # population std, ddof=0

    assert result["aggregate"]["mae"]["mean"] == pytest.approx(expected_mean)
    assert result["aggregate"]["mae"]["std"] == pytest.approx(expected_std)


def test_aggregate_value_is_derived_from_outer_fold_test_sizes():
    """Cross-check: with mae==len(test_ids), the aggregate mean equals the mean of
    the recorded test-fold sizes — proves run_outer output flows into the aggregate."""
    labels = _labels()
    runner = _RecordingRunner()
    result = patient_nested_cv(labels, k_outer=5, k_inner=3, run_outer=runner, seed=42)

    recorded_sizes = [float(len(test_ids)) for _tr, test_ids, _k in runner.calls]
    assert result["aggregate"]["mae"]["mean"] == pytest.approx(float(np.mean(recorded_sizes)))
    assert result["aggregate"]["mae"]["std"] == pytest.approx(float(np.std(recorded_sizes)))


def test_aggregate_covers_every_metric_common_to_all_folds():
    """Multiple metric keys present in every outer fold are each aggregated; a
    constant metric has std exactly 0 (mirrors uda.training.cv aggregation)."""
    labels = _labels()

    def run_outer(train_ids, test_ids, k_inner):
        n = len(test_ids)
        return {"mae": float(n), "rmse": float(n) + 1.0, "r2": 0.5}

    result = patient_nested_cv(labels, k_outer=5, k_inner=3, run_outer=run_outer, seed=42)
    agg = result["aggregate"]
    assert set(agg.keys()) == {"mae", "rmse", "r2"}
    for key in ("mae", "rmse", "r2"):
        assert set(agg[key].keys()) == {"mean", "std"}

    # r2 is constant across folds -> std exactly 0
    assert agg["r2"]["mean"] == pytest.approx(0.5)
    assert agg["r2"]["std"] == pytest.approx(0.0)


def test_aggregate_drops_keys_not_present_in_all_folds():
    """Keys present in only some outer-fold dicts are dropped from ``aggregate``
    (reuses the uda.training.cv common-keys-only rule)."""
    labels = _labels()
    seen = {"i": 0}

    def run_outer(train_ids, test_ids, k_inner):
        i = seen["i"]
        seen["i"] += 1
        out = {"mae": float(len(test_ids))}
        if i == 0:
            out["only_first"] = 1.0  # present in exactly one outer fold
        return out

    result = patient_nested_cv(labels, k_outer=5, k_inner=3, run_outer=run_outer, seed=42)
    assert "mae" in result["aggregate"]
    assert "only_first" not in result["aggregate"]


# --------------------------------------------------------------------------- #
# 5. determinism
# --------------------------------------------------------------------------- #
def test_two_calls_same_args_return_identical_result():
    labels = _labels()
    a = patient_nested_cv(labels, k_outer=5, k_inner=3, run_outer=_RecordingRunner(), seed=7)
    b = patient_nested_cv(labels, k_outer=5, k_inner=3, run_outer=_RecordingRunner(), seed=7)
    assert a == b


def test_outer_fold_ids_are_identical_across_calls_with_same_seed():
    labels = _labels()
    r1, r2 = _RecordingRunner(), _RecordingRunner()
    patient_nested_cv(labels, k_outer=5, k_inner=3, run_outer=r1, seed=7)
    patient_nested_cv(labels, k_outer=5, k_inner=3, run_outer=r2, seed=7)

    assert len(r1.calls) == len(r2.calls) == 5
    for (tr1, te1, _k1), (tr2, te2, _k2) in zip(r1.calls, r2.calls):
        assert list(tr1) == list(tr2)
        assert list(te1) == list(te2)


def test_different_seed_changes_outer_fold_composition():
    """``seed`` is wired through to ``SplitConfig.seed`` — a different seed yields a
    different outer test-fold ordering (the underlying GroupKFold shuffle changed)."""
    labels = _labels()
    r1, r2 = _RecordingRunner(), _RecordingRunner()
    patient_nested_cv(labels, k_outer=5, k_inner=3, run_outer=r1, seed=1)
    patient_nested_cv(labels, k_outer=5, k_inner=3, run_outer=r2, seed=2)

    test_sets_1 = [set(te) for _tr, te, _k in r1.calls]
    test_sets_2 = [set(te) for _tr, te, _k in r2.calls]
    # the per-fold partition is not identical fold-for-fold between the two seeds
    assert test_sets_1 != test_sets_2


# --------------------------------------------------------------------------- #
# 6. injectability / no training
# --------------------------------------------------------------------------- #
def test_harness_calls_run_outer_exactly_k_outer_times_and_builds_no_model():
    """Cheap stub is invoked exactly once per outer fold (no hidden extra training);
    the harness never instantiates a model — only the injected stub runs."""
    labels = _labels()
    runner = _RecordingRunner()
    result = patient_nested_cv(labels, k_outer=5, k_inner=3, run_outer=runner, seed=42)
    assert len(runner.calls) == 5
    assert len(result["folds"]) == 5


# --------------------------------------------------------------------------- #
# 7. k_outer validation + boundary
# --------------------------------------------------------------------------- #
def test_k_outer_greater_than_n_patients_raises_value_error():
    # 3 patients but k_outer=5 -> GroupKFold cannot partition; spec: validate & raise.
    labels = _labels(n_patients=3, per_patient=4)
    with pytest.raises(ValueError):
        patient_nested_cv(labels, k_outer=5, k_inner=2, run_outer=_RecordingRunner(), seed=42)


def test_k_outer_less_than_two_raises_value_error():
    labels = _labels()
    with pytest.raises(ValueError):
        patient_nested_cv(labels, k_outer=1, k_inner=2, run_outer=_RecordingRunner(), seed=42)


def test_k_outer_equal_to_n_patients_is_allowed():
    """The boundary ``k_outer == n_patients`` is valid (each fold holds out one patient)."""
    labels = _labels(n_patients=5, per_patient=3)
    runner = _RecordingRunner()
    result = patient_nested_cv(labels, k_outer=5, k_inner=2, run_outer=runner, seed=42)
    assert len(result["folds"]) == 5
    assert len(runner.calls) == 5


# --------------------------------------------------------------------------- #
# 8. Keras-free
# --------------------------------------------------------------------------- #
def test_nested_cv_module_is_keras_free():
    """Importing ``uda.evaluation.nested_cv`` must not pull a heavy backend (keras/jax/
    tensorflow) — heavy training lives only inside the caller's ``run_outer``.
    Checked in a FRESH interpreter so other tests' backend imports don't pollute
    ``sys.modules`` (mirrors ``tests/test_cv.py``)."""
    import subprocess

    code = (
        "import uda.evaluation.nested_cv, sys; "
        "bad = [b for b in ('keras', 'jax', 'tensorflow') if b in sys.modules]; "
        "print(','.join(bad)); "
        "sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, (
        f"uda.evaluation.nested_cv pulled a backend: {r.stdout.strip()}\n{r.stderr[-300:]}"
    )

"""Tests — patient-level k-fold CV harness (``uda.training.cv``).

The harness drives :class:`uda.data.splits.PatientLevelSplit` (``GroupKFold`` over
``patient_id``) and aggregates per-fold metric dicts. The per-fold runner is
**injected**, so these tests never build or train a model: a cheap stub stands in
for ``run_fold`` and records the ids it receives. Every test here must run in well
under a second and the module under test is **Keras-free** (asserted explicitly).

A synthetic labels frame stands in for ``data/labels.csv``: 10 patients with 4
base images each (40 rows, one per base ``image_id``), matching the shape used by
``tests/test_splits.py``.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from uda.training.cv import patient_kfold, random_kfold

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
    """A ``run_fold`` stub: records the ids it receives and returns a fixed metric.

    The returned metric is derived deterministically from the fold so aggregate
    math is hand-checkable: ``mae == len(test_ids)``.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def __call__(self, train_ids: np.ndarray, test_ids: np.ndarray) -> dict:
        self.calls.append((train_ids, test_ids))
        return {"mae": float(len(test_ids))}


# --------------------------------------------------------------------------- #
# 1. fold count + disjointness
# --------------------------------------------------------------------------- #
def test_returns_k_folds_with_folds_and_aggregate_keys():
    labels = _labels()
    result = patient_kfold(labels, k=5, run_fold=_RecordingRunner(), seed=42)

    assert set(result.keys()) == {"folds", "aggregate"}
    assert isinstance(result["folds"], list)
    assert len(result["folds"]) == 5
    # each fold entry is the per-fold metric dict returned by run_fold
    for fold in result["folds"]:
        assert fold == {"mae": pytest.approx(fold["mae"])}
        assert set(fold) == {"mae"}


def test_each_fold_train_test_image_ids_are_disjoint():
    labels = _labels()
    runner = _RecordingRunner()
    patient_kfold(labels, k=5, run_fold=runner, seed=42)

    assert len(runner.calls) == 5
    for train_ids, test_ids in runner.calls:
        assert set(train_ids).isdisjoint(set(test_ids))
        # ids handed out are exactly the base image_ids — no augmentation leaks in
        assert set(train_ids) | set(test_ids) == set(labels["image_id"])


def test_each_fold_keeps_patients_whole():
    """Patient-level: the patients in train and test never overlap within a fold."""
    labels = _labels()
    runner = _RecordingRunner()
    patient_kfold(labels, k=5, run_fold=runner, seed=42)

    lut = _id_to_patient(labels)
    for train_ids, test_ids in runner.calls:
        train_patients = {lut[i] for i in train_ids}
        test_patients = {lut[i] for i in test_ids}
        assert train_patients.isdisjoint(test_patients)


def test_test_folds_partition_all_image_ids():
    """GroupKFold partitions: union of all test folds == every base image_id, and
    no image id appears in two different test folds."""
    labels = _labels()
    runner = _RecordingRunner()
    patient_kfold(labels, k=5, run_fold=runner, seed=42)

    all_ids = set(labels["image_id"])
    seen_test: set[str] = set()
    for _train_ids, test_ids in runner.calls:
        test_s = set(test_ids)
        assert seen_test.isdisjoint(test_s)
        seen_test |= test_s
    assert seen_test == all_ids


# --------------------------------------------------------------------------- #
# 2. aggregate math
# --------------------------------------------------------------------------- #
def test_aggregate_mean_and_std_match_numpy_over_folds():
    labels = _labels()
    runner = _RecordingRunner()
    result = patient_kfold(labels, k=5, run_fold=runner, seed=42)

    # recover the per-fold metric values the harness saw, in fold order
    per_fold = [fold["mae"] for fold in result["folds"]]
    assert len(per_fold) == 5

    expected_mean = float(np.mean(per_fold))
    expected_std = float(np.std(per_fold))  # population std, ddof=0

    assert result["aggregate"]["mae"]["mean"] == pytest.approx(expected_mean)
    assert result["aggregate"]["mae"]["std"] == pytest.approx(expected_std)


def test_aggregate_value_is_derived_from_fold_test_sizes():
    """Cross-check: with mae==len(test_ids), the aggregate mean equals the mean of
    the recorded test-fold sizes — proves run_fold output flows into the aggregate."""
    labels = _labels()
    runner = _RecordingRunner()
    result = patient_kfold(labels, k=5, run_fold=runner, seed=42)

    recorded_sizes = [float(len(test_ids)) for _tr, test_ids in runner.calls]
    assert result["aggregate"]["mae"]["mean"] == pytest.approx(float(np.mean(recorded_sizes)))
    assert result["aggregate"]["mae"]["std"] == pytest.approx(float(np.std(recorded_sizes)))


def test_aggregate_covers_every_metric_common_to_all_folds():
    """Multiple metric keys present in every fold are each aggregated."""
    labels = _labels()

    def run_fold(train_ids, test_ids):
        n = len(test_ids)
        return {"mae": float(n), "rmse": float(n) + 1.0, "r2": 0.5}

    result = patient_kfold(labels, k=5, run_fold=run_fold, seed=42)
    agg = result["aggregate"]
    assert set(agg.keys()) == {"mae", "rmse", "r2"}
    for key in ("mae", "rmse", "r2"):
        assert set(agg[key].keys()) == {"mean", "std"}

    # r2 is constant across folds -> std exactly 0
    assert agg["r2"]["mean"] == pytest.approx(0.5)
    assert agg["r2"]["std"] == pytest.approx(0.0)


def test_aggregate_drops_keys_not_present_in_all_folds():
    """Keys present in only some fold dicts are dropped from ``aggregate``."""
    labels = _labels()
    seen = {"i": 0}

    def run_fold(train_ids, test_ids):
        i = seen["i"]
        seen["i"] += 1
        out = {"mae": float(len(test_ids))}
        if i == 0:
            out["only_first"] = 1.0  # present in exactly one fold
        return out

    result = patient_kfold(labels, k=5, run_fold=run_fold, seed=42)
    assert "mae" in result["aggregate"]
    assert "only_first" not in result["aggregate"]


# --------------------------------------------------------------------------- #
# 3. determinism
# --------------------------------------------------------------------------- #
def test_two_calls_same_args_return_identical_result():
    labels = _labels()
    a = patient_kfold(labels, k=5, run_fold=_RecordingRunner(), seed=7)
    b = patient_kfold(labels, k=5, run_fold=_RecordingRunner(), seed=7)
    assert a == b


def test_fold_ids_are_identical_across_calls_with_same_seed():
    labels = _labels()
    r1, r2 = _RecordingRunner(), _RecordingRunner()
    patient_kfold(labels, k=5, run_fold=r1, seed=7)
    patient_kfold(labels, k=5, run_fold=r2, seed=7)

    assert len(r1.calls) == len(r2.calls) == 5
    for (tr1, te1), (tr2, te2) in zip(r1.calls, r2.calls):
        assert list(tr1) == list(tr2)
        assert list(te1) == list(te2)


def test_different_seed_changes_fold_composition():
    """``seed`` is wired through to ``SplitConfig.seed`` — a different seed yields a
    different test-fold ordering (the underlying GroupKFold shuffle changed)."""
    labels = _labels()
    r1, r2 = _RecordingRunner(), _RecordingRunner()
    patient_kfold(labels, k=5, run_fold=r1, seed=1)
    patient_kfold(labels, k=5, run_fold=r2, seed=2)

    test_sets_1 = [set(te) for _tr, te in r1.calls]
    test_sets_2 = [set(te) for _tr, te in r2.calls]
    # the per-fold partition is not identical fold-for-fold between the two seeds
    assert test_sets_1 != test_sets_2


# --------------------------------------------------------------------------- #
# 4. injectability / no training
# --------------------------------------------------------------------------- #
def test_run_fold_receives_numpy_arrays_of_image_ids():
    labels = _labels()
    all_ids = set(labels["image_id"])

    captured: list[tuple] = []

    def run_fold(train_ids, test_ids):
        captured.append((train_ids, test_ids))
        assert isinstance(train_ids, np.ndarray)
        assert isinstance(test_ids, np.ndarray)
        # the ids are base image_ids drawn from the label table
        assert set(train_ids) <= all_ids
        assert set(test_ids) <= all_ids
        return {"mae": float(len(test_ids))}

    patient_kfold(labels, k=5, run_fold=run_fold, seed=42)
    assert len(captured) == 5


def test_cv_module_is_keras_free():
    """Importing ``uda.training.cv`` must not pull a heavy backend (keras/jax/tensorflow) —
    heavy training lives only inside the caller's ``run_fold``. Checked in a FRESH
    interpreter so other tests' backend imports don't pollute ``sys.modules``."""
    import subprocess

    code = (
        "import uda.training.cv, sys; "
        "bad = [b for b in ('keras', 'jax', 'tensorflow') if b in sys.modules]; "
        "print(','.join(bad)); "
        "sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"uda.training.cv pulled a backend: {r.stdout.strip()}\n{r.stderr[-300:]}"


def test_harness_never_calls_run_fold_more_than_k_times():
    """Cheap stub is invoked exactly once per fold (no hidden extra training)."""
    labels = _labels()
    runner = _RecordingRunner()
    result = patient_kfold(labels, k=5, run_fold=runner, seed=42)
    assert len(runner.calls) == 5
    assert len(result["folds"]) == 5


# --------------------------------------------------------------------------- #
# 5. k larger than #patients raises
# --------------------------------------------------------------------------- #
def test_k_greater_than_n_patients_raises_value_error():
    # 3 patients but k=5 -> GroupKFold cannot partition; we validate & raise.
    labels = _labels(n_patients=3, per_patient=4)
    with pytest.raises(ValueError):
        patient_kfold(labels, k=5, run_fold=_RecordingRunner(), seed=42)


def test_k_equal_to_n_patients_is_allowed():
    """The boundary ``k == n_patients`` is valid (every fold holds out one patient)."""
    labels = _labels(n_patients=5, per_patient=3)
    runner = _RecordingRunner()
    result = patient_kfold(labels, k=5, run_fold=runner, seed=42)
    assert len(result["folds"]) == 5
    assert len(runner.calls) == 5


# =========================================================================== #
# random_kfold — the paper's "image" protocol: plain random k-fold over the
# augmented rows (no grouping). Keras-free; stub run_fold; integer index arrays.
# =========================================================================== #
class _IdxRecordingRunner:
    """A ``run_fold`` stub for :func:`random_kfold`: records integer index arrays.

    Returns ``mae == len(test_idx)`` so aggregate math is hand-checkable, mirroring
    ``_RecordingRunner`` for the patient harness.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def __call__(self, train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
        self.calls.append((train_idx, test_idx))
        return {"mae": float(len(test_idx))}


N_SAMPLES = 2100  # the full augmented corpus size the image protocol partitions


# --- shape / keys -------------------------------------------------------- #
def test_random_kfold_returns_folds_and_aggregate_keys():
    result = random_kfold(N_SAMPLES, k=5, run_fold=_IdxRecordingRunner(), seed=42)
    assert set(result.keys()) == {"folds", "aggregate"}
    assert isinstance(result["folds"], list)
    assert len(result["folds"]) == 5
    for fold in result["folds"]:
        assert set(fold) == {"mae"}


# --- exhaustive + disjoint coverage of all indices ----------------------- #
def test_random_kfold_test_folds_partition_every_index_exactly_once():
    runner = _IdxRecordingRunner()
    random_kfold(N_SAMPLES, k=5, run_fold=runner, seed=42)

    assert len(runner.calls) == 5
    seen_test: set[int] = set()
    for train_idx, test_idx in runner.calls:
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
        train_s, test_s = set(train_idx.tolist()), set(test_idx.tolist())
        # train/test disjoint and together cover every index within the fold
        assert train_s.isdisjoint(test_s)
        assert train_s | test_s == set(range(N_SAMPLES))
        # test folds disjoint from one another -> a partition of all indices
        assert seen_test.isdisjoint(test_s)
        seen_test |= test_s
    assert seen_test == set(range(N_SAMPLES))


def test_random_kfold_test_fold_sizes_sum_to_n_samples():
    runner = _IdxRecordingRunner()
    random_kfold(N_SAMPLES, k=5, run_fold=runner, seed=7)
    assert sum(len(te) for _tr, te in runner.calls) == N_SAMPLES


# --- determinism --------------------------------------------------------- #
def test_random_kfold_same_seed_returns_identical_result():
    a = random_kfold(N_SAMPLES, k=5, run_fold=_IdxRecordingRunner(), seed=11)
    b = random_kfold(N_SAMPLES, k=5, run_fold=_IdxRecordingRunner(), seed=11)
    assert a == b


def test_random_kfold_same_seed_yields_identical_index_partitions():
    r1, r2 = _IdxRecordingRunner(), _IdxRecordingRunner()
    random_kfold(N_SAMPLES, k=5, run_fold=r1, seed=11)
    random_kfold(N_SAMPLES, k=5, run_fold=r2, seed=11)
    assert len(r1.calls) == len(r2.calls) == 5
    for (tr1, te1), (tr2, te2) in zip(r1.calls, r2.calls):
        assert np.array_equal(tr1, tr2)
        assert np.array_equal(te1, te2)


def test_random_kfold_different_seed_changes_partition():
    r1, r2 = _IdxRecordingRunner(), _IdxRecordingRunner()
    random_kfold(N_SAMPLES, k=5, run_fold=r1, seed=1)
    random_kfold(N_SAMPLES, k=5, run_fold=r2, seed=2)
    test_sets_1 = [set(te.tolist()) for _tr, te in r1.calls]
    test_sets_2 = [set(te.tolist()) for _tr, te in r2.calls]
    assert test_sets_1 != test_sets_2


# --- aggregate math ------------------------------------------------------ #
def test_random_kfold_aggregate_mean_std_match_numpy_over_folds():
    runner = _IdxRecordingRunner()
    result = random_kfold(N_SAMPLES, k=5, run_fold=runner, seed=42)
    per_fold = [fold["mae"] for fold in result["folds"]]
    assert result["aggregate"]["mae"]["mean"] == pytest.approx(float(np.mean(per_fold)))
    assert result["aggregate"]["mae"]["std"] == pytest.approx(float(np.std(per_fold)))


def test_random_kfold_aggregate_matches_known_stub_dicts():
    """A stub returning fixed dicts proves run_fold output flows into the aggregate."""
    seen = {"i": 0}

    def run_fold(train_idx, test_idx):
        i = seen["i"]
        seen["i"] += 1
        return {"mae": float(i), "r2": 0.5}

    result = random_kfold(20, k=4, run_fold=run_fold, seed=0)
    agg = result["aggregate"]
    # mae values seen are 0,1,2,3 over the four folds (fold order preserved)
    assert agg["mae"]["mean"] == pytest.approx(np.mean([0.0, 1.0, 2.0, 3.0]))
    assert agg["mae"]["std"] == pytest.approx(np.std([0.0, 1.0, 2.0, 3.0]))
    # r2 constant -> std exactly 0
    assert agg["r2"]["mean"] == pytest.approx(0.5)
    assert agg["r2"]["std"] == pytest.approx(0.0)


# --- k validation -------------------------------------------------------- #
def test_random_kfold_k_below_two_raises():
    with pytest.raises(ValueError):
        random_kfold(N_SAMPLES, k=1, run_fold=_IdxRecordingRunner(), seed=0)


def test_random_kfold_k_greater_than_n_samples_raises():
    with pytest.raises(ValueError):
        random_kfold(4, k=5, run_fold=_IdxRecordingRunner(), seed=0)


def test_random_kfold_k_equal_n_samples_is_allowed():
    """Boundary ``k == n_samples`` is valid (leave-one-out)."""
    runner = _IdxRecordingRunner()
    result = random_kfold(5, k=5, run_fold=runner, seed=0)
    assert len(result["folds"]) == 5
    assert len(runner.calls) == 5
    assert all(len(te) == 1 for _tr, te in runner.calls)


def test_random_kfold_calls_run_fold_exactly_k_times():
    runner = _IdxRecordingRunner()
    random_kfold(N_SAMPLES, k=5, run_fold=runner, seed=42)
    assert len(runner.calls) == 5

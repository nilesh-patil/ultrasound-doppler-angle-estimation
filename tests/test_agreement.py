"""Tests — Bland–Altman agreement (``uda.evaluation.agreement``).

These tests cover the ``uda.evaluation.agreement``
module. Everything here is post-hoc on *saved* predictions: no model is built and
every test runs in well under a second. The module under test is **Keras-free**
(asserted explicitly in a fresh subprocess, mirroring ``tests/test_conformal.py``).

Two public entry points are exercised:

* ``bland_altman(method_a, method_b, *, wrap=True) -> {bias, sd, loa_lower,
  loa_upper, mean_axis, diff, label}`` — paired bias + 95% limits of agreement.
* ``agreement_from_csv(pred_csv, *, agg="sample") -> bland_altman dict + {n, agg}``
  — end-to-end on a saved prediction CSV, with ``A = theta_true`` (the **reference**
  MATLAB-GUI reading) vs ``B = theta_pred`` (the model).

The honesty contract: there is exactly **one** human reading per image,
so this is *method-vs-reference* agreement (never inter-observer) — every returned
dict carries ``label == "reference"``. Differences are on the **signed 180-wrap**
scale ``r = ((a - b + 90) % 180) - 90`` (vessel orientation is 180-periodic), and
``agg="patient"`` averages within ``patient_id`` via a **double-angle circular mean**.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from uda.evaluation.agreement import agreement_from_csv, bland_altman

# Repo-relative path to the honest, full-coverage OOF predictions (2100 rows).
REAL_OOF_CSV = "results/predictions/tuned_densenet201_oof.csv"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _signed_wrap(delta: np.ndarray) -> np.ndarray:
    """Reference signed-wrap into ``(-90, 90]`` (the contract's difference scale)."""
    return ((np.asarray(delta, dtype=float) + 90.0) % 180.0) - 90.0


def _circular_mean_deg(theta: np.ndarray) -> float:
    """Reference double-angle circular mean of angles, mapped into ``[0, 180)``."""
    t = np.asarray(theta, dtype=float)
    m = 0.5 * np.arctan2(np.mean(np.sin(np.radians(2.0 * t))),
                         np.mean(np.cos(np.radians(2.0 * t))))
    return float(np.degrees(m) % 180.0)


def _patient_csv(tmp_path, n_patients: int = 6, per_patient: int = 4, seed: int = 0):
    """Write a small synthetic prediction CSV with a ``patient_id`` grouping.

    Columns mirror the real OOF file: ``image_id, patient_id, rotation_deg,
    theta_true, theta_pred``. One base image per row keeps the bookkeeping simple.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        for k in range(per_patient):
            true = float(rng.uniform(70.0, 110.0))
            pred = true + float(rng.normal(0.0, 4.0))
            rows.append(
                {
                    "image_id": f"p{p:02d}_img{k:02d}",
                    "patient_id": float(p),
                    "rotation_deg": 0.0,
                    "theta_true": true,
                    "theta_pred": pred,
                }
            )
    path = tmp_path / "synthetic_oof.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# 1. bias / sd / LoA formula on a fixed, hand-computed array  (test 1)
# --------------------------------------------------------------------------- #
def test_bland_altman_bias_sd_loa_match_hand_computed():
    """On a fixed array: ``bias == mean(diff)``, ``sd == std(diff, ddof=1)`` (the
    *sample* sd), and ``loa_lower/upper == bias ∓ 1.96*sd`` exactly. Inputs are
    chosen so the signed-wrap difference equals the raw difference (no seam)."""
    a = np.array([100.0, 102.0, 98.0, 105.0, 95.0, 101.0])
    b = np.array([99.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    out = bland_altman(a, b, wrap=True)

    diff = _signed_wrap(a - b)
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))

    assert out["bias"] == pytest.approx(bias)
    assert out["sd"] == pytest.approx(sd)
    assert out["loa_lower"] == pytest.approx(bias - 1.96 * sd)
    assert out["loa_upper"] == pytest.approx(bias + 1.96 * sd)
    # the diff array is returned and matches the (wrapped) per-pair difference
    assert np.allclose(out["diff"], diff)


# --------------------------------------------------------------------------- #
# 2. identical inputs collapse  (bias=0, sd=0, LoA collapse)
# --------------------------------------------------------------------------- #
def test_identical_inputs_collapse_bias_sd_and_loa():
    """Identical readings -> ``bias == 0``, ``sd == 0``, and ``loa_lower ==
    loa_upper == 0`` (the limits of agreement collapse onto zero)."""
    a = np.array([10.0, 45.0, 80.0, 120.0, 175.0])
    out = bland_altman(a, a.copy(), wrap=True)

    assert out["bias"] == pytest.approx(0.0)
    assert out["sd"] == pytest.approx(0.0)
    assert out["loa_lower"] == pytest.approx(0.0)
    assert out["loa_upper"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 3. signed-wrap circular difference (179 vs 1 -> magnitude 2)  (test 2)
# --------------------------------------------------------------------------- #
def test_signed_wrap_circular_difference_magnitude_two():
    """``a=1, b=179`` (wrap) -> ``diff == +2`` (NOT ``-178``); every ``diff`` lies in
    ``(-90, 90]``; ``wrap=False`` on the same pair gives the raw ``-178``."""
    a = np.array([1.0])
    b = np.array([179.0])

    out_wrap = bland_altman(a, b, wrap=True)
    assert out_wrap["diff"][0] == pytest.approx(2.0)
    assert not out_wrap["diff"][0] == pytest.approx(-178.0)
    assert out_wrap["bias"] == pytest.approx(2.0)

    # raw mode reproduces the seam-inflated difference (non-periodic diagnostic only)
    out_raw = bland_altman(a, b, wrap=False)
    assert out_raw["diff"][0] == pytest.approx(-178.0)

    # a sweep of differences always wraps into (-90, 90]
    aa = np.linspace(0.0, 180.0, 91)
    bb = np.full_like(aa, 90.0)
    diffs = bland_altman(aa, bb, wrap=True)["diff"]
    assert np.all(diffs > -90.0 - 1e-9)
    assert np.all(diffs <= 90.0 + 1e-9)


# --------------------------------------------------------------------------- #
# 4. constant offset d -> bias == d, sd == 0
# --------------------------------------------------------------------------- #
def test_constant_offset_gives_bias_d_and_zero_sd():
    """A constant signed offset ``d`` between the two readings -> ``bias == d`` and
    ``sd == 0`` (no spread), with the LoA collapsing onto ``d``."""
    d = 7.0
    a = np.array([30.0, 50.0, 70.0, 110.0, 150.0])
    b = a - d  # so signed-wrap(a - b) == +d everywhere (no seam crossing)
    out = bland_altman(a, b, wrap=True)

    assert out["bias"] == pytest.approx(d)
    assert out["sd"] == pytest.approx(0.0)
    assert out["loa_lower"] == pytest.approx(d)
    assert out["loa_upper"] == pytest.approx(d)


# --------------------------------------------------------------------------- #
# 5. mean_axis is the Bland–Altman x-axis and length-matched  (test 4)
# --------------------------------------------------------------------------- #
def test_mean_axis_is_pairwise_average_and_length_matched():
    """``mean_axis == (method_a + method_b)/2`` (the Bland–Altman x-axis), and
    ``len(mean_axis) == len(diff) == n``."""
    a = np.array([20.0, 40.0, 60.0, 80.0])
    b = np.array([22.0, 38.0, 64.0, 79.0])
    out = bland_altman(a, b, wrap=True)

    assert np.allclose(out["mean_axis"], (a + b) / 2.0)
    assert len(out["mean_axis"]) == len(out["diff"]) == len(a)


# --------------------------------------------------------------------------- #
# 6. honesty / label guard  (test 3)
# --------------------------------------------------------------------------- #
def test_label_is_reference_for_both_entry_points(tmp_path):
    """Both ``bland_altman`` and ``agreement_from_csv`` carry ``label == "reference"``
    (the honesty guard: model-vs-single-human-reference, never inter-observer)."""
    out = bland_altman(np.array([10.0, 20.0]), np.array([11.0, 19.0]), wrap=True)
    assert out["label"] == "reference"

    csv = _patient_csv(tmp_path, n_patients=4, per_patient=3, seed=5)
    out_csv = agreement_from_csv(csv, agg="sample")
    assert out_csv["label"] == "reference"


def test_agreement_from_csv_compares_theta_true_reference_to_theta_pred(tmp_path):
    """``agreement_from_csv`` sets A = ``theta_true`` (reference) and B = ``theta_pred``
    (model): bias matches the signed-wrap mean of ``theta_true - theta_pred`` and ``n``
    equals the row count for ``agg="sample"``."""
    csv = _patient_csv(tmp_path, n_patients=5, per_patient=4, seed=6)
    df = pd.read_csv(csv)

    out = agreement_from_csv(csv, agg="sample")
    expected_diff = _signed_wrap(
        df["theta_true"].to_numpy(float) - df["theta_pred"].to_numpy(float)
    )
    assert out["bias"] == pytest.approx(float(np.mean(expected_diff)))
    assert out["n"] == len(df)
    assert out["agg"] == "sample"


# --------------------------------------------------------------------------- #
# 7. patient aggregation: circular mean + n == n_patients + order invariance  (test 5)
# --------------------------------------------------------------------------- #
def test_patient_aggregation_reduces_n_to_n_patients(tmp_path):
    """``agg="patient"`` yields one pair per ``patient_id`` -> ``n == n_patients``."""
    csv = _patient_csv(tmp_path, n_patients=7, per_patient=5, seed=7)
    df = pd.read_csv(csv)
    out = agreement_from_csv(csv, agg="patient")
    assert out["n"] == df["patient_id"].nunique()
    assert out["agg"] == "patient"


def test_patient_aggregation_is_order_invariant(tmp_path):
    """Shuffling the CSV rows does not change the ``agg="patient"`` bias/sd/LoA
    (within-patient circular mean is a set operation, not order-dependent)."""
    csv = _patient_csv(tmp_path, n_patients=6, per_patient=5, seed=8)
    df = pd.read_csv(csv)
    shuffled = df.sample(frac=1.0, random_state=99).reset_index(drop=True)
    shuf_path = tmp_path / "shuffled_oof.csv"
    shuffled.to_csv(shuf_path, index=False)

    base = agreement_from_csv(csv, agg="patient")
    perm = agreement_from_csv(shuf_path, agg="patient")
    assert perm["bias"] == pytest.approx(base["bias"])
    assert perm["sd"] == pytest.approx(base["sd"])
    assert perm["loa_lower"] == pytest.approx(base["loa_lower"])
    assert perm["loa_upper"] == pytest.approx(base["loa_upper"])


def test_patient_aggregation_uses_double_angle_circular_mean(tmp_path):
    """On a seam-straddling patient (angles near 0/180), the aggregated pair matches a
    hand-computed double-angle circular mean — a naive linear mean would be visibly
    wrong."""
    rows = []
    # one patient whose readings straddle the 0/180 seam
    seam_true = np.array([1.0, 179.0, 3.0, 177.0])
    seam_pred = np.array([2.0, 178.0, 1.0, 175.0])
    for k, (tt, pp) in enumerate(zip(seam_true, seam_pred)):
        rows.append({"image_id": f"seam_img{k:02d}", "patient_id": 0.0,
                     "rotation_deg": 0.0, "theta_true": tt, "theta_pred": pp})
    # a second, well-behaved patient so n_patients == 2
    rows.append({"image_id": "calm_img00", "patient_id": 1.0, "rotation_deg": 0.0,
                 "theta_true": 90.0, "theta_pred": 92.0})
    csv = tmp_path / "seam_oof.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    out = agreement_from_csv(csv, agg="patient")
    assert out["n"] == 2

    # hand-compute the seam patient's circular-mean pair and its signed-wrap diff
    a0 = _circular_mean_deg(seam_true)
    b0 = _circular_mean_deg(seam_pred)
    seam_diff = _signed_wrap(np.array([a0 - b0]))[0]
    # the calm patient contributes 90 vs 92 -> diff -2
    calm_diff = _signed_wrap(np.array([90.0 - 92.0]))[0]
    expected_bias = float(np.mean([seam_diff, calm_diff]))
    assert out["bias"] == pytest.approx(expected_bias)
    # a naive linear mean of the seam_true angles (~90) would NOT match the circular ~0
    assert a0 < 10.0 or a0 > 170.0


# --------------------------------------------------------------------------- #
# 8. sample vs patient consistency with one image per patient  (test 6)
# --------------------------------------------------------------------------- #
def test_sample_and_patient_agree_with_one_image_per_patient(tmp_path):
    """With exactly one image per patient, ``agg="sample"`` and ``agg="patient"``
    return the same bias/sd/LoA (the aggregation is a no-op)."""
    csv = _patient_csv(tmp_path, n_patients=8, per_patient=1, seed=9)
    s = agreement_from_csv(csv, agg="sample")
    p = agreement_from_csv(csv, agg="patient")
    assert p["bias"] == pytest.approx(s["bias"])
    assert p["sd"] == pytest.approx(s["sd"])
    assert p["loa_lower"] == pytest.approx(s["loa_lower"])
    assert p["loa_upper"] == pytest.approx(s["loa_upper"])
    assert p["n"] == s["n"]


# --------------------------------------------------------------------------- #
# 9. real OOF csv smoke -> finite bias / LoA  (test 7)
# --------------------------------------------------------------------------- #
def test_agreement_from_csv_on_real_oof_csv():
    """End-to-end on the honest OOF predictions: finite ``bias``/``sd``, ordered LoA
    (``loa_lower < loa_upper``), ``label == "reference"``, and ``n`` equal to the row
    count (sample) / patient count (patient)."""
    df = pd.read_csv(REAL_OOF_CSV)

    s = agreement_from_csv(REAL_OOF_CSV, agg="sample")
    assert set(s) >= {"bias", "sd", "loa_lower", "loa_upper", "mean_axis",
                      "diff", "label", "n", "agg"}
    assert np.isfinite(s["bias"])
    assert np.isfinite(s["sd"])
    assert s["loa_lower"] < s["loa_upper"]
    assert s["label"] == "reference"
    assert s["n"] == len(df)

    p = agreement_from_csv(REAL_OOF_CSV, agg="patient")
    assert np.isfinite(p["bias"])
    assert p["loa_lower"] < p["loa_upper"]
    assert p["label"] == "reference"
    assert p["n"] == df["patient_id"].nunique()


# --------------------------------------------------------------------------- #
# 10. Keras-free fresh-subprocess guard  (test 8)
# --------------------------------------------------------------------------- #
def test_agreement_module_is_keras_free():
    """Importing ``uda.evaluation.agreement`` must not pull a heavy backend (keras/jax/tensorflow).
    Checked in a FRESH interpreter so other tests' backend imports don't pollute
    ``sys.modules`` (mirror ``tests/test_conformal.py``)."""
    import subprocess

    code = (
        "import uda.evaluation.agreement, sys; "
        "bad = [b for b in ('keras', 'jax', 'tensorflow') if b in sys.modules]; "
        "print(','.join(bad)); "
        "sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, (
        f"uda.evaluation.agreement pulled a backend: {r.stdout.strip()}\n{r.stderr[-300:]}"
    )

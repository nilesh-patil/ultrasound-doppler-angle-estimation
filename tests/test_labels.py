"""Tests — robust label parser and heuristic patient grouping."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uda.config import SplitConfig
from uda.data.labels import (
    AUX_COLUMNS,
    LABEL_COLUMNS,
    assign_patient_ids,
    build_labels_csv,
    parse_results,
)
from uda.data.splits import PatientLevelSplit

# Canonical label source under data/ (tests/ lives directly under the repo root).
RESULTS_TXT = Path(__file__).resolve().parent.parent / "data" / "Results.txt"

N_BASE_IMAGES = 84


@pytest.fixture(scope="module")
def parsed() -> pd.DataFrame:
    return parse_results(RESULTS_TXT)


# --------------------------------------------------------------------------- #
# parse_results
# --------------------------------------------------------------------------- #
def test_results_txt_exists():
    assert RESULTS_TXT.is_file()


def test_parse_yields_84_rows(parsed: pd.DataFrame):
    assert len(parsed) == N_BASE_IMAGES


def test_parse_columns(parsed: pd.DataFrame):
    assert list(parsed.columns) == ["image_id", "theta_deg", *AUX_COLUMNS]


def test_theta_in_open_0_180(parsed: pd.DataFrame):
    theta = parsed["theta_deg"]
    assert ((theta > 0) & (theta < 180)).all()


def test_no_nans(parsed: pd.DataFrame):
    assert not parsed.isna().any().any()


def test_image_ids_have_no_extension(parsed: pd.DataFrame):
    assert not parsed["image_id"].str.endswith(".jpg").any()
    assert parsed["image_id"].is_unique


def test_known_row_theta(parsed: pd.DataFrame):
    row = parsed.loc[parsed["image_id"] == "09-41-06_1"]
    assert len(row) == 1
    assert row["theta_deg"].iloc[0] == pytest.approx(88.3653, abs=1e-3)


def test_theta_is_float(parsed: pd.DataFrame):
    assert parsed["theta_deg"].dtype.kind == "f"


# --------------------------------------------------------------------------- #
# assign_patient_ids
# --------------------------------------------------------------------------- #
def test_patient_group_count_in_band(parsed: pd.DataFrame):
    grouped = assign_patient_ids(parsed)
    n_groups = grouped["patient_id"].nunique()
    assert 8 <= n_groups <= 12


def test_every_image_in_exactly_one_group(parsed: pd.DataFrame):
    grouped = assign_patient_ids(parsed)
    # No NaN -> every base image got a patient_id.
    assert grouped["patient_id"].notna().all()
    assert len(grouped) == N_BASE_IMAGES
    # Group sizes partition the 84 images.
    assert grouped.groupby("patient_id").size().sum() == N_BASE_IMAGES


def test_repeated_capture_shares_patient(parsed: pd.DataFrame):
    grouped = assign_patient_ids(parsed)
    lookup = dict(zip(grouped["image_id"], grouped["patient_id"]))
    # _1 and _2 of the same HH-MM-SS acquisition must co-assign.
    assert lookup["09-41-06_1"] == lookup["09-41-06_2"]


def test_groups_contiguous_in_time(parsed: pd.DataFrame):
    """Sorting by acquisition time must yield non-decreasing, gap-free patient ids.

    Contiguity means each patient occupies one unbroken interval on the time axis:
    once we leave a patient we never return to it.
    """
    grouped = assign_patient_ids(parsed)

    def _seconds(image_id: str) -> int:
        h, m, s = image_id.split("_")[0].split("-")
        return int(h) * 3600 + int(m) * 60 + int(s)

    ordered = grouped.assign(_sec=grouped["image_id"].map(_seconds)).sort_values(
        ["_sec", "image_id"]
    )
    ids = ordered["patient_id"].to_numpy()
    # Non-decreasing in time order, and each new patient id is the next integer.
    assert (np.diff(ids) >= 0).all()
    assert list(pd.unique(ids)) == list(range(grouped["patient_id"].nunique()))


def test_patient_ids_start_at_zero_and_are_dense(parsed: pd.DataFrame):
    grouped = assign_patient_ids(parsed)
    ids = sorted(grouped["patient_id"].unique())
    assert ids == list(range(len(ids)))


@pytest.mark.parametrize("gap_seconds", [179, 180, 200, 225])
def test_robust_band_keeps_8_to_12_groups(parsed: pd.DataFrame, gap_seconds: int):
    n_groups = assign_patient_ids(parsed, gap_seconds=gap_seconds)["patient_id"].nunique()
    assert 8 <= n_groups <= 12


def test_default_gap_over_naive_120(parsed: pd.DataFrame):
    """A naive default of 120s over-splits and would fail the band test."""
    n120 = assign_patient_ids(parsed, gap_seconds=120)["patient_id"].nunique()
    assert n120 > 12  # documents why labels.py overrides the default to 180


# --------------------------------------------------------------------------- #
# build_labels_csv
# --------------------------------------------------------------------------- #
def test_build_labels_csv_writes_and_returns(tmp_path):
    out_csv = tmp_path / "labels.csv"
    df = build_labels_csv(RESULTS_TXT, out_csv)

    assert out_csv.is_file()
    assert list(df.columns) == LABEL_COLUMNS
    assert len(df) == N_BASE_IMAGES

    on_disk = pd.read_csv(out_csv)
    assert list(on_disk.columns) == LABEL_COLUMNS
    assert len(on_disk) == N_BASE_IMAGES
    assert not on_disk.isna().any().any()


def test_build_labels_csv_creates_parent_dir(tmp_path):
    out_csv = tmp_path / "nested" / "dir" / "labels.csv"
    build_labels_csv(RESULTS_TXT, out_csv)
    assert out_csv.is_file()


def test_build_labels_csv_theta_matches_parse(tmp_path):
    out_csv = tmp_path / "labels.csv"
    built = build_labels_csv(RESULTS_TXT, out_csv).set_index("image_id")["theta_deg"]
    parsed = parse_results(RESULTS_TXT).set_index("image_id")["theta_deg"]
    pd.testing.assert_series_equal(built.sort_index(), parsed.sort_index())


# --------------------------------------------------------------------------- #
# leakage assertion via PatientLevelSplit
# --------------------------------------------------------------------------- #
def test_patient_grouping_prevents_leakage(parsed: pd.DataFrame):
    """No base image (hence no rotation of it) spans the patient-level split."""
    labels = assign_patient_ids(parsed)
    cfg = SplitConfig(strategy="patient", test_size=0.2, seed=42)
    train_ids, test_ids = next(PatientLevelSplit().split(labels, cfg))

    assert set(train_ids).isdisjoint(set(test_ids))
    assert set(train_ids) | set(test_ids) == set(labels["image_id"])

    pid = dict(zip(labels["image_id"], labels["patient_id"]))
    train_patients = {pid[i] for i in train_ids}
    test_patients = {pid[i] for i in test_ids}
    assert train_patients.isdisjoint(test_patients)

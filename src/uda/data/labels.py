"""Robust label parser for the SPLab Brno Doppler-angle dataset.

``Results.txt`` is tab-separated with CRLF (``\\r\\n``) line
endings; each row is ``filename  theta  c3  c4  c5  c6`` where **column 2 is the
Doppler angle** :math:`\\theta` in degrees (verified ~78-104). The trailing four
columns (``c3..c6``) are auxiliary vessel-wall geometry and are *not* used as the
regression target; they are carried through ``parse_results`` for provenance only.

Patient grouping is recovered heuristically from the ``HH-MM-SS`` acquisition
timestamp embedded in each filename (see :func:`assign_patient_ids`). It is a
**leakage-prevention proxy**, not a claim of true volunteer identity: its sole
guarantee is that every rotation of a given base image stays on one side of any
train/test split. The threshold and resulting group sizes are documented in
``data/README.md``.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

# Default successive-gap threshold (seconds) for clustering acquisition
# timestamps into patients. At 120s the 68
# distinct acquisitions split into 14 groups, which fails the "<= 12
# groups" acceptance test. 180s yields 12 groups summing to all 84 files and
# lies in the robust band [179, 225] that keeps the count within [8, 12].
DEFAULT_GAP_SECONDS = 180

# Raw column names emitted by :func:`parse_results`. The four auxiliary
# vessel-wall geometry measurements keep stable, non-colliding names.
AUX_COLUMNS = ["aux_c3", "aux_c4", "aux_c5", "aux_c6"]

# Columns written to ``data/labels.csv``.
LABEL_COLUMNS = ["image_id", "patient_id", "theta_deg"]


def _strip_suffix(filename: str) -> str:
    """Return the filename stem without its image extension.

    Parameters
    ----------
    filename : str
        A raw filename such as ``"09-41-06_1.jpg"``.

    Returns
    -------
    str
        The stem, e.g. ``"09-41-06_1"``.
    """
    return Path(filename.strip()).stem


def _timestamp_seconds(image_id: str) -> int:
    """Convert the ``HH-MM-SS`` prefix of an image id to seconds past midnight.

    Parameters
    ----------
    image_id : str
        An image id such as ``"09-41-06_1"`` (an optional ``_<n>`` suffix marks a
        repeated capture of the same acquisition and is ignored here).

    Returns
    -------
    int
        ``HH * 3600 + MM * 60 + SS``.
    """
    acquisition = image_id.split("_")[0]
    hours, minutes, seconds = (int(part) for part in acquisition.split("-"))
    return hours * 3600 + minutes * 60 + seconds


def parse_results(results_txt: str | Path) -> pd.DataFrame:
    """Parse ``Results.txt`` into a typed label table.

    The file is opened with universal-newline handling so the CRLF (``\\r\\n``)
    endings are normalized. Column 2 is the Doppler angle; the remaining four
    columns are auxiliary geometry carried through unchanged.

    Parameters
    ----------
    results_txt : str or pathlib.Path
        Path to the tab-separated ``Results.txt``.

    Returns
    -------
    pandas.DataFrame
        Columns ``image_id`` (filename stem), ``theta_deg`` (float), and the four
        raw auxiliary columns ``aux_c3 .. aux_c6`` (float). One row per base image.
    """
    rows: list[dict[str, object]] = []
    # newline="" lets the csv module handle line terminators while Python's
    # universal-newline translation normalizes the CRLF endings on read.
    with open(results_txt, "r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for record in reader:
            if not record or not record[0].strip():
                continue  # tolerate trailing blank lines
            image_id = _strip_suffix(record[0])
            theta_deg = float(record[1])
            aux = [float(value) for value in record[2:6]]
            row: dict[str, object] = {"image_id": image_id, "theta_deg": theta_deg}
            for name, value in zip(AUX_COLUMNS, aux):
                row[name] = value
            rows.append(row)

    return pd.DataFrame(rows, columns=["image_id", "theta_deg", *AUX_COLUMNS])


def assign_patient_ids(
    df: pd.DataFrame, gap_seconds: int = DEFAULT_GAP_SECONDS
) -> pd.DataFrame:
    """Attach a ``patient_id`` recovered from acquisition timestamps.

    Distinct ``HH-MM-SS`` acquisitions are sorted ascending and walked in time; a
    new ``patient_id`` is cut whenever the gap to the previous acquisition is at
    least ``gap_seconds``. Repeated captures of one acquisition (``_1``/``_2`` of
    the same ``HH-MM-SS``) therefore share a single patient, and the groups are
    contiguous, non-overlapping intervals in time.

    This recovers ~8-12 groups (target ~10 volunteers in the paper). It is a
    leakage-prevention proxy, not true identity recovery -- see the module
    docstring and ``data/README.md``.

    Parameters
    ----------
    df : pandas.DataFrame
        Label table from :func:`parse_results` (must contain ``image_id``).
    gap_seconds : int, optional
        Minimum successive gap that starts a new patient. Defaults to
        :data:`DEFAULT_GAP_SECONDS` (180). The leakage experiment is robust to the
        exact value across roughly ``[179, 225]``.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` (original row order preserved) with an added integer
        ``patient_id`` column. Patient ids are contiguous from 0 in time order.
    """
    seconds = df["image_id"].map(_timestamp_seconds)

    # One representative second per distinct acquisition, sorted ascending.
    distinct = sorted(set(seconds.tolist()))

    patient_of_second: dict[int, int] = {}
    current_patient = 0
    previous: int | None = None
    for second in distinct:
        if previous is not None and (second - previous) >= gap_seconds:
            current_patient += 1
        patient_of_second[second] = current_patient
        previous = second

    out = df.copy()
    out["patient_id"] = seconds.map(patient_of_second).astype(int)
    return out


def build_labels_csv(
    results_txt: str | Path,
    out_csv: str | Path,
    gap_seconds: int = DEFAULT_GAP_SECONDS,
) -> pd.DataFrame:
    """Build ``data/labels.csv`` and return the written table.

    Parses ``results_txt``, assigns patient ids, then writes the three canonical
    columns ``image_id, patient_id, theta_deg``. The parent directory is created
    if needed.

    Parameters
    ----------
    results_txt : str or pathlib.Path
        Path to ``Results.txt``.
    out_csv : str or pathlib.Path
        Destination CSV path.
    gap_seconds : int, optional
        Forwarded to :func:`assign_patient_ids`. Defaults to
        :data:`DEFAULT_GAP_SECONDS` (180).

    Returns
    -------
    pandas.DataFrame
        The table written to ``out_csv`` with columns
        ``image_id, patient_id, theta_deg``.
    """
    df = parse_results(results_txt)
    df = assign_patient_ids(df, gap_seconds=gap_seconds)
    labels = df[LABEL_COLUMNS].copy()

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_path, index=False)
    return labels

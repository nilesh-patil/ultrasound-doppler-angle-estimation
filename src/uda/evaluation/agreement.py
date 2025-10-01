"""Bland–Altman method-vs-reference agreement for angle predictions (Keras-free).

This module turns *saved* predictions into a
Bland–Altman agreement summary, entirely post-hoc — no model is built and no
deep-learning backend (``keras``/``jax``/``tensorflow``) is imported, only numpy
and pandas.

Two public entry points:

* :func:`bland_altman` — pure paired-difference agreement. Given two readings
  ``method_a`` and ``method_b`` it returns the ``bias`` (mean signed difference),
  its sample ``sd`` (``ddof=1``), the 95% limits of agreement
  ``loa_lower/upper = bias ∓ 1.96·sd``, the Bland–Altman ``mean_axis``
  ``(a + b)/2``, the per-pair ``diff`` array, and a ``label``.
* :func:`agreement_from_csv` — end-to-end on a saved prediction CSV. It sets
  ``A = theta_true`` (the **reference** MATLAB-GUI reading) and ``B = theta_pred``
  (the model), then defers to :func:`bland_altman`. ``agg="patient"`` first
  collapses every ``patient_id`` to a single pair via a **double-angle circular
  mean** before computing the agreement.

Difference scale (critical): vessel orientation is **180-periodic**, so the
paired difference is the *signed wrap* ``r = ((a - b + 90) % 180) - 90`` into
``(-90, 90]``. This makes ``a=1`` vs ``b=179`` a difference of ``+2`` (not
``-178``). Pass ``wrap=False`` only for a non-periodic raw diagnostic.

Honesty: there is exactly **one** human reading per image (the
MATLAB-GUI angle), so this is *method-vs-reference* agreement — never
inter-observer. The reference is always ``A = theta_true`` and every returned
dict carries ``label == "reference"`` to make that contract impossible to
misread as an inter-observer Bland–Altman plot.

Reuse: residual/error summaries elsewhere go through :func:`uda.evaluation.evaluate.metrics`;
this module deliberately reports only the agreement statistics (bias + limits of
agreement) that ``metrics`` does not, on the same signed-wrap angle scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["bland_altman", "agreement_from_csv"]

_PATIENT_ID = "patient_id"
_THETA_TRUE = "theta_true"
_THETA_PRED = "theta_pred"

# Orientation is 180-periodic; differences wrap into the half-open interval (-90, 90].
_PERIOD = 180.0
_HALF_PERIOD = 90.0

# 95% limits of agreement multiplier (normal approximation: ±1.96·sd).
_LOA_Z = 1.96

# Honesty contract: single human reading => model-vs-reference, never inter-observer.
_LABEL = "reference"


def _signed_wrap(delta: np.ndarray) -> np.ndarray:
    """Signed wrap of an angle difference into ``(-90, 90]``.

    Vessel orientation is 180-periodic, so the meaningful difference between two
    angle readings is the smallest signed rotation between them. Computed as
    ``((delta + 90) % 180) - 90``, which maps e.g. ``delta = 1 - 179 = -178`` to
    ``+2`` — magnitude ``2``, not ``178``.

    Parameters
    ----------
    delta : numpy.ndarray
        Raw angle differences (degrees), typically ``method_a - method_b``.

    Returns
    -------
    numpy.ndarray
        The wrapped differences as ``float``, every element in ``(-90, 90]``.
    """
    return ((np.asarray(delta, dtype=float) + _HALF_PERIOD) % _PERIOD) - _HALF_PERIOD


def _circular_mean_deg(theta: np.ndarray) -> float:
    """Double-angle circular mean of angles (degrees), mapped into ``[0, 180)``.

    Because orientation is 180-periodic, averaging happens in *double-angle*
    space: ``mean = 0.5 * atan2(mean(sin 2t), mean(cos 2t))`` mapped back into
    ``[0, 180)``. This averages seam-straddling readings (e.g. ``1`` and ``179``)
    to ``~0`` rather than the meaningless linear ``~90``.

    Parameters
    ----------
    theta : numpy.ndarray
        Angles in degrees.

    Returns
    -------
    float
        The circular mean angle in ``[0, 180)``.
    """
    t = np.asarray(theta, dtype=float)
    m = 0.5 * np.arctan2(
        np.mean(np.sin(np.radians(2.0 * t))),
        np.mean(np.cos(np.radians(2.0 * t))),
    )
    return float(np.degrees(m) % _PERIOD)


def bland_altman(method_a: np.ndarray, method_b: np.ndarray, *, wrap: bool = True) -> dict:
    """Bland–Altman paired-difference agreement between two angle readings.

    The per-pair difference is ``method_a - method_b``, on the **signed 180-wrap**
    scale by default (``wrap=True``) so it lives in ``(-90, 90]`` (orientation is
    180-periodic). ``bias`` is the mean difference, ``sd`` its *sample* standard
    deviation (``ddof=1``), and the 95% limits of agreement are
    ``loa_lower/upper = bias ∓ 1.96·sd``. The ``mean_axis`` ``(method_a +
    method_b)/2`` is the Bland–Altman plot x-axis.

    Honesty: ``method_a`` is treated as the **reference** reading (the single
    human/MATLAB-GUI angle) and ``method_b`` as the method under test, so the
    returned ``label`` is always ``"reference"`` — this is *not* an
    inter-observer comparison.

    Parameters
    ----------
    method_a : numpy.ndarray
        Reference reading (degrees).
    method_b : numpy.ndarray
        Method-under-test reading (degrees), same length as ``method_a``.
    wrap : bool, keyword-only, optional
        When ``True`` (default) the difference is signed-wrapped into ``(-90, 90]``.
        ``False`` returns the raw ``method_a - method_b`` (a non-periodic
        diagnostic only — it inflates seam-straddling pairs).

    Returns
    -------
    dict
        ``{"bias": float, "sd": float, "loa_lower": float, "loa_upper": float,
        "mean_axis": numpy.ndarray, "diff": numpy.ndarray, "label": "reference"}``.
        ``sd`` and the limits of agreement are ``0`` for a single pair (no spread).
    """
    a = np.asarray(method_a, dtype=float)
    b = np.asarray(method_b, dtype=float)

    raw = a - b
    diff = _signed_wrap(raw) if wrap else raw
    mean_axis = (a + b) / 2.0

    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    loa_lower = bias - _LOA_Z * sd
    loa_upper = bias + _LOA_Z * sd

    return {
        "bias": bias,
        "sd": sd,
        "loa_lower": loa_lower,
        "loa_upper": loa_upper,
        "mean_axis": mean_axis,
        "diff": diff,
        "label": _LABEL,
    }


def agreement_from_csv(pred_csv, *, agg: str = "sample") -> dict:
    """Bland–Altman agreement end-to-end on a saved prediction CSV.

    The CSV must have the OOF schema ``image_id, patient_id, theta_true,
    theta_pred`` (extra columns are ignored). The **reference** is always
    ``A = theta_true`` (the single MATLAB-GUI human reading) and the method under
    test is ``B = theta_pred`` (the model) — so the result is model-vs-reference
    agreement, never inter-observer, and carries ``label == "reference"``.

    Aggregation:

    * ``agg="sample"`` (default) — every CSV row is one Bland–Altman pair.
    * ``agg="patient"`` — each ``patient_id`` is first collapsed to a single pair
      by taking the **double-angle circular mean** of its ``theta_true`` and of
      its ``theta_pred`` (180-periodic average), then those per-patient pairs feed
      the agreement. This is order-invariant (a within-patient set reduction).

    The paired difference is always on the signed 180-wrap scale (via
    :func:`bland_altman` with ``wrap=True``).

    Parameters
    ----------
    pred_csv : str or pathlib.Path
        Path to a saved prediction CSV (e.g. an OOF predictions file).
    agg : {"sample", "patient"}, keyword-only, optional
        Aggregation level (default ``"sample"``).

    Returns
    -------
    dict
        The :func:`bland_altman` dict plus ``{"n": int, "agg": str}`` where ``n``
        is the number of Bland–Altman pairs (CSV rows for ``"sample"``, distinct
        patients for ``"patient"``).
    """
    df = pd.read_csv(pred_csv)

    if agg == "patient":
        groups = df.groupby(_PATIENT_ID, sort=True)
        a = np.array(
            [_circular_mean_deg(g[_THETA_TRUE].to_numpy(dtype=float)) for _, g in groups]
        )
        b = np.array(
            [_circular_mean_deg(g[_THETA_PRED].to_numpy(dtype=float)) for _, g in groups]
        )
    elif agg == "sample":
        a = df[_THETA_TRUE].to_numpy(dtype=float)
        b = df[_THETA_PRED].to_numpy(dtype=float)
    else:  # pragma: no cover - defensive guard
        raise ValueError(f"agg must be 'sample' or 'patient', got {agg!r}")

    out = bland_altman(a, b, wrap=True)
    out["n"] = int(a.size)
    out["agg"] = agg
    return out

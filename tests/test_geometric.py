"""Tests — classical geometric vessel-angle prior (``uda.interpret.geometric``).

These tests cover the ``uda.interpret.geometric``
module. The correctness anchor is a set of tiny **synthetic** images of *known*
orientation — a bright bar through the center at a set angle-to-vertical on a dark
field — so the geometry is checkable without any model or the real labels. Every
test runs in well under a second and the module under test is **Keras-free**
(asserted explicitly in a fresh subprocess, mirroring ``tests/test_tta.py``).

Two public entry points are exercised:

* ``estimate_angle(image_2d, *, method="structure_tensor") -> float`` — the
  dominant linear orientation of the bright structures, measured **to the image's
  VERTICAL axis**, in ``[0, 180)`` (0° = a perfectly vertical wall), 180-periodic.
  A pure function of the pixels — it never sees the labels.
* ``evaluate_geometric(labels_csv, images_dir) -> {y_true, y_pred, metrics, n}`` —
  runs :func:`estimate_angle` over every base image (loaded via
  ``uda.data.images``), joins to ``labels_csv`` on ``image_id``, and scores the
  ``(theta_true, theta_pred)`` pairs with ``uda.evaluation.evaluate.metrics`` (single source
  of truth). Estimation never reads ``theta_deg``.

ANGLE CONVENTION (must match the labels): ``theta`` is measured to the image's
**vertical** axis in ``[0, 180)``. The synthetic generator below pins that
convention: ``angle_to_vertical=0`` draws a vertical wall, ``=90`` a horizontal
wall, ``=30`` a 30° wall. The tests assert recovery within a small tolerance, which
locks the gradient→wall (``+90``) and horizontal→vertical (``90 − ·``) conversions
to the label convention (NOT off by 90°). Orientation is 180-periodic, so all
comparisons use the signed wrap into ``(-90, 90]``.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from uda.evaluation import evaluate
from uda.interpret.geometric import estimate_angle, evaluate_geometric

# Real data — single source of truth for I/O (84 grayscale base images + labels).
REAL_LABELS_CSV = "data/labels.csv"
REAL_IMAGES_DIR = "data/images"
N_BASE = 84

# Tolerance for recovering a known synthetic angle (degrees). The structure tensor
# on a clean bar recovers the truth to well under a degree; ~5° leaves comfortable
# headroom for discretization and any per-method axis bookkeeping.
TOL_DEG = 5.0


# --------------------------------------------------------------------------- #
# helpers — synthetic known-angle image + reference wrap (NOT the estimator)
# --------------------------------------------------------------------------- #
def _signed_wrap(delta) -> np.ndarray:
    """Reference signed-wrap into ``(-90, 90]`` (orientation is 180-periodic)."""
    return ((np.asarray(delta, dtype=float) + 90.0) % 180.0) - 90.0


def _angle_err(est: float, truth: float) -> float:
    """Absolute orientation error in ``[0, 90]`` via the signed wrap."""
    return float(abs(_signed_wrap(np.array([truth - est]))[0]))


def _bar_image(angle_to_vertical_deg: float, size: int = 129, width: float = 2.5):
    """A bright bar through the image center at a KNOWN angle to the VERTICAL axis.

    Convention (matches the labels): ``angle_to_vertical_deg == 0`` is a perfectly
    vertical wall (runs top-to-bottom along rows); ``== 90`` is horizontal (runs
    left-to-right along columns); the angle increases as the wall rotates toward
    horizontal. The wall is a thin Gaussian ridge so the dominant linear structure
    is unambiguous. This is the ground-truth generator — it does NOT call the
    estimator, so the tests check a real recovery, not a tautology.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    c = (size - 1) / 2.0
    x = xx - c  # +x to the right (columns)
    y = yy - c  # +y downward (rows, image convention)
    a = np.deg2rad(float(angle_to_vertical_deg))
    # Wall direction d in (x, y): 0° -> straight down (0, 1); rotate toward +x.
    dx, dy = np.sin(a), np.cos(a)
    # Perpendicular distance of each pixel to the infinite line through the center.
    dist = np.abs(-dy * x + dx * y)
    bar = np.exp(-(dist**2) / (2.0 * width**2))
    return bar.astype(np.float32)


# --------------------------------------------------------------------------- #
# 1. CORRECTNESS ANCHOR — synthetic line at a KNOWN angle is recovered
#    (axis convention pinned)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("angle", [0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0])
def test_recovers_known_synthetic_angle_to_vertical(angle):
    """A bright bar at a known angle-to-vertical is recovered within ~5°.

    Locks the full conversion chain (gradient→wall ``+90`` then
    horizontal→vertical ``90 − ·``) to the label convention. If the
    implementation were off by 90° this assertion would fail by ~90°.
    """
    est = estimate_angle(_bar_image(angle))
    assert isinstance(est, float)
    assert _angle_err(est, angle) <= TOL_DEG


def test_vertical_wall_is_near_zero_and_horizontal_near_ninety():
    """Explicit anchor for the two cardinal orientations (no off-by-90)."""
    v = estimate_angle(_bar_image(0.0))
    h = estimate_angle(_bar_image(90.0))
    assert _angle_err(v, 0.0) <= TOL_DEG
    assert _angle_err(h, 90.0) <= TOL_DEG
    # the two cardinal walls are clearly distinguished (~90° apart)
    assert _angle_err(v, h) >= 90.0 - 2 * TOL_DEG


# --------------------------------------------------------------------------- #
# 2. ROTATING THE LINE BY d SHIFTS THE ESTIMATE BY d (mod 180)
#    (periodicity / equivariance)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("base", [10.0, 40.0, 100.0])
@pytest.mark.parametrize("delta", [20.0, 55.0, -35.0])
def test_rotating_line_shifts_estimate_by_delta_mod_180(base, delta):
    """Rotating the synthetic wall by ``delta`` shifts the estimate by ``delta``
    on the 180-periodic circle (the estimator is rotation-equivariant)."""
    e0 = estimate_angle(_bar_image(base))
    e1 = estimate_angle(_bar_image(base + delta))
    observed_shift = _signed_wrap(np.array([e1 - e0]))[0]
    expected_shift = _signed_wrap(np.array([delta]))[0]
    assert abs(observed_shift - expected_shift) <= 2 * TOL_DEG


def test_wall_and_its_180_rotation_give_the_same_angle():
    """A wall and its 180°-rotated copy are identical orientations (180-periodic),
    and seam-straddling angles near 0/180 are not split."""
    for angle in (0.0, 3.0, 177.0):
        e0 = estimate_angle(_bar_image(angle))
        e180 = estimate_angle(_bar_image(angle + 180.0))
        assert _angle_err(e0, e180) <= TOL_DEG


# --------------------------------------------------------------------------- #
# 3. OUTPUT IS ALWAYS A float IN [0, 180)  (range)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("angle", [0.0, 17.0, 89.0, 90.0, 134.0, 179.0])
def test_output_is_float_in_unit_range(angle):
    est = estimate_angle(_bar_image(angle))
    assert isinstance(est, float)
    assert 0.0 <= est < 180.0


# --------------------------------------------------------------------------- #
# 4. METHODS AGREE ON CLEAN INPUT + unknown method raises
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("angle", [25.0, 70.0, 115.0])
def test_methods_agree_on_clean_synthetic_bar(angle):
    """On a noiseless bar, all three estimators recover the known angle within
    tolerance — cross-checking each method's own axis bookkeeping."""
    img = _bar_image(angle)
    for method in ("structure_tensor", "gradient_histogram", "hough"):
        est = estimate_angle(img, method=method)
        assert isinstance(est, float)
        assert 0.0 <= est < 180.0
        assert _angle_err(est, angle) <= TOL_DEG, f"method={method} angle={angle}"


def test_unknown_method_raises_value_error():
    with pytest.raises(ValueError):
        estimate_angle(_bar_image(45.0), method="definitely_not_a_method")


# --------------------------------------------------------------------------- #
# 5. SPECKLE ROBUSTNESS — Gaussian smoothing beats a raw per-pixel argmax
# --------------------------------------------------------------------------- #
def _raw_pixel_gradient_argmax_to_vertical(image_2d) -> float:
    """A deliberately naive baseline: per-pixel Sobel gradient-orientation argmax
    (no smoothing), converted to the vertical-axis convention. On speckle this is
    dominated by noise — the contrast the structure-tensor ``sigma`` improves on.
    Computed entirely in the test; it does NOT call the module under test."""
    from skimage.filters import sobel_h, sobel_v

    img = np.asarray(image_2d, dtype=float)
    gy = sobel_h(img)  # gradient along rows (vertical)
    gx = sobel_v(img)  # gradient along cols (horizontal)
    mag = np.hypot(gx, gy)
    j = int(np.argmax(mag))
    phi = np.degrees(np.arctan2(gy.ravel()[j], gx.ravel()[j]))  # gradient dir from +x
    wall = phi + 90.0  # wall is perpendicular to the gradient
    to_vertical = (90.0 - wall) % 180.0  # horizontal -> vertical convention
    return float(to_vertical)


def test_speckle_robustness_beats_raw_pixel_argmax():
    """Moderate multiplicative speckle perturbs the structure-tensor estimate only
    slightly (Gaussian-sigma smoothing doing its job), and far less than a raw
    per-pixel gradient-argmax baseline computed in the test."""
    angle = 35.0
    clean = _bar_image(angle, size=129, width=3.0)
    rng = np.random.default_rng(0)
    # multiplicative speckle (Rayleigh-like): the canonical B-mode noise model
    speckle = clean * (1.0 + 0.5 * rng.standard_normal(clean.shape).astype(np.float32))
    speckle = np.clip(speckle, 0.0, None).astype(np.float32)

    st_err = _angle_err(estimate_angle(speckle, method="structure_tensor"), angle)
    raw_err = _angle_err(_raw_pixel_gradient_argmax_to_vertical(speckle), angle)

    # the smoothed estimator stays close to the truth ...
    assert st_err <= 3 * TOL_DEG
    # ... and is meaningfully better than the unsmoothed per-pixel argmax
    assert st_err < raw_err


# --------------------------------------------------------------------------- #
# 6. 2-D GUARD + degenerate constant image
# --------------------------------------------------------------------------- #
def test_three_dimensional_input_raises():
    """A 3-D ``(H, W, 3)`` input is rejected — the caller must pass grayscale via
    ``uda.data.images`` (single source of truth for loading)."""
    rgb = np.zeros((32, 32, 3), dtype=np.float32)
    with pytest.raises((ValueError, Exception)):
        estimate_angle(rgb)


def test_constant_image_returns_finite_angle_without_raising():
    """A texture-free constant image is degenerate but must return a defined,
    finite angle in range (no crash, no NaN)."""
    flat = np.full((48, 48), 0.5, dtype=np.float32)
    est = estimate_angle(flat)
    assert isinstance(est, float)
    assert np.isfinite(est)
    assert 0.0 <= est < 180.0


# --------------------------------------------------------------------------- #
# 7. RUNS ON REAL BASE IMAGES -> finite angle in range
# --------------------------------------------------------------------------- #
def test_runs_on_a_couple_real_base_images():
    """Smoke: load a couple of real base images via ``uda.data.images`` and confirm
    ``estimate_angle`` returns a finite float in ``[0, 180)`` on real speckle."""
    from uda.data import images as uimg

    paths = uimg.list_base_images(REAL_IMAGES_DIR)[:3]
    assert len(paths) >= 2
    for p in paths:
        gray = uimg.load_image_gray(p)
        assert gray.ndim == 2  # the loader hands us 2-D grayscale
        est = estimate_angle(gray)
        assert isinstance(est, float)
        assert np.isfinite(est)
        assert 0.0 <= est < 180.0


# --------------------------------------------------------------------------- #
# 8. evaluate_geometric — shapes + reuse + honest no-peek
# --------------------------------------------------------------------------- #
def test_evaluate_geometric_shapes_and_metrics_reuse():
    """On the real data: ``n == 84``, one ``(y_true, y_pred)`` per base image, all
    metrics finite, and ``metrics`` equals ``uda.evaluation.evaluate.metrics(y_true, y_pred)``
    recomputed in the test (single source of truth, not re-derived)."""
    out = evaluate_geometric(REAL_LABELS_CSV, REAL_IMAGES_DIR)

    assert set(out) == {"y_true", "y_pred", "metrics", "n"}
    assert out["n"] == N_BASE
    y_true = np.asarray(out["y_true"], dtype=float)
    y_pred = np.asarray(out["y_pred"], dtype=float)
    assert len(y_true) == len(y_pred) == N_BASE
    assert np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred))
    assert np.all((y_pred >= 0.0) & (y_pred < 180.0))

    expected = evaluate.metrics(y_true, y_pred)
    assert set(out["metrics"]) == {"mae", "rmse", "me", "mape", "r2"}
    assert all(np.isfinite(v) for v in out["metrics"].values())
    assert out["metrics"] == expected


def test_evaluate_geometric_is_honest_no_peeking_at_labels():
    """Honesty contract: the MAE is reported as-is and only asserted **finite** —
    the test does NOT require the classical baseline to beat the learned model (a
    poor hand-crafted baseline is on-thesis). And no label is consulted during
    estimation: each ``y_pred`` is re-derived from pixels alone (``estimate_angle``
    on the loaded image) and must match the reported prediction bit-for-bit."""
    from uda.data import images as uimg

    out = evaluate_geometric(REAL_LABELS_CSV, REAL_IMAGES_DIR)
    mae = out["metrics"]["mae"]
    assert np.isfinite(mae)
    assert mae >= 0.0  # honest accuracy, whatever it is

    # Re-derive every prediction from pixels only — no labels in the loop.
    recomputed = {
        uimg.image_id(p): estimate_angle(uimg.load_image_gray(p))
        for p in uimg.list_base_images(REAL_IMAGES_DIR)
    }
    # The reported predictions must be exactly the pixel-only estimates: there is
    # no path by which the labels could have tuned the classical output.
    reported = np.asarray(out["y_pred"], dtype=float)
    truth = np.asarray(out["y_true"], dtype=float)
    assert len(reported) == len(truth) == N_BASE
    # every reported y_pred is reproducible from pixels alone
    reported_sorted = np.sort(reported)
    recomputed_sorted = np.sort(np.array(list(recomputed.values()), dtype=float))
    assert reported_sorted == pytest.approx(recomputed_sorted, abs=1e-9)


# --------------------------------------------------------------------------- #
# 9. Keras-free fresh-subprocess guard
# --------------------------------------------------------------------------- #
def test_geometric_module_is_keras_free():
    """Importing ``uda.interpret.geometric`` must not pull a heavy backend
    (keras/jax/tensorflow). Checked in a FRESH interpreter so other tests' backend
    imports don't pollute ``sys.modules`` (mirror ``tests/test_cv.py``)."""
    import subprocess

    code = (
        "import uda.interpret.geometric, sys; "
        "bad = [b for b in ('keras', 'jax', 'tensorflow') if b in sys.modules]; "
        "print(','.join(bad)); "
        "sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, (
        f"uda.interpret.geometric pulled a backend: {r.stdout.strip()}\n{r.stderr[-300:]}"
    )

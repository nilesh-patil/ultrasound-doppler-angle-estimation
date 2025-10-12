"""Classical hand-crafted vessel-angle prior (Keras-free).

This module estimates the dominant linear
orientation of the bright structures in a B-mode ultrasound image *from the
pixels alone* — a deliberately simple, geometry-only baseline with no learned
parameters and no deep-learning backend (``keras``/``jax``/``tensorflow``).
Only numpy + scikit-image + the single-source-of-truth loaders/metrics in
``uda.data.images`` / :func:`uda.evaluation.evaluate.metrics` are used.

Two public entry points:

* :func:`estimate_angle` — the dominant linear orientation of a 2-D grayscale
  image, measured **to the image's VERTICAL axis**, in ``[0, 180)`` (0° = a
  perfectly vertical wall, 90° = horizontal), 180-periodic. A pure function of
  the pixels; it never sees the labels. Three interchangeable backends are
  offered (``structure_tensor`` recommended, with ``gradient_histogram`` and
  ``hough`` alternatives), all returning the same vertical-axis convention.
* :func:`evaluate_geometric` — runs :func:`estimate_angle` over every base
  image (loaded via ``uda.data.images``), joins to ``labels_csv`` on
  ``image_id``, and scores the ``(theta_true, theta_pred)`` pairs with
  :func:`uda.evaluation.evaluate.metrics`. Estimation never reads ``theta_deg``.

ANGLE CONVENTION (must match the labels). The labels' ``theta`` is the Doppler
angle measured to the image's **vertical** axis in ``[0, 180)``. Internally we
first recover the *gradient* orientation (the direction of steepest intensity
change), which is **perpendicular** to the bright wall: ``wall = gradient + 90``.
We then convert that "angle from the horizontal +x axis" to "angle from the
vertical axis": ``to_vertical = 90 − wall``. The two ``±90`` steps are the whole
correctness story (an off-by-90 implementation fails the synthetic-bar tests by
~90°). Orientation is 180-periodic, so every wrap uses the signed wrap into
``(-90, 90]`` and the final answer is mapped into ``[0, 180)``.

HONESTY. This is a hand-crafted geometric baseline on speckly B-mode ultrasound;
it may well be much less accurate than the learned model, and that is on-thesis.
:func:`evaluate_geometric` reports whatever MAE it gets — it is never tuned
toward the labels (no label is consulted during estimation).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from uda.evaluation import evaluate
from uda.data import images as uimg

__all__ = ["estimate_angle", "evaluate_geometric"]

# Orientation is 180-periodic; differences wrap into (-90, 90], answers live in [0, 180).
_PERIOD = 180.0
_HALF_PERIOD = 90.0

# Gaussian smoothing scale (pixels) for the structure tensor / gradient field. This
# is the speckle-robustness knob: it integrates the noisy per-pixel gradients into a
# stable dominant orientation. Chosen for clean recovery on the synthetic bar and
# resilience to multiplicative B-mode speckle (NOT tuned against the real labels).
_SIGMA = 2.0

_LABEL_IMAGE_ID = "image_id"
_LABEL_THETA = "theta_deg"


def _signed_wrap(delta: np.ndarray) -> np.ndarray:
    """Signed wrap of an angle difference into ``(-90, 90]`` (180-periodic)."""
    return ((np.asarray(delta, dtype=float) + _HALF_PERIOD) % _PERIOD) - _HALF_PERIOD


def _gradient_to_vertical(gradient_deg: float) -> float:
    """Convert a gradient orientation to the label's vertical-axis convention.

    The gradient (steepest-intensity direction, measured CCW from the horizontal
    ``+x`` axis) is perpendicular to the bright wall, so ``wall = gradient + 90``.
    Converting "angle from horizontal" to "angle from vertical" is
    ``to_vertical = 90 − wall = −gradient``. The result is mapped into ``[0, 180)``.
    """
    wall = gradient_deg + 90.0
    to_vertical = 90.0 - wall
    return float(to_vertical % _PERIOD)


def _check_2d(image_2d) -> np.ndarray:
    """Validate and return a 2-D float array (grayscale, single source of truth)."""
    img = np.asarray(image_2d, dtype=float)
    if img.ndim != 2:
        raise ValueError(
            f"estimate_angle expects a 2-D grayscale image; got ndim={img.ndim}. "
            "Load images via uda.data.images.load_image_gray."
        )
    return img


def _structure_tensor_gradient_deg(img: np.ndarray) -> float:
    """Dominant gradient orientation (deg, CCW from +x) via the structure tensor.

    The Gaussian-windowed structure tensor ``[[Axx, Axy], [Axy, Ayy]]`` summed over
    the whole image gives the globally dominant axis of intensity variation. Its
    principal (largest-eigenvalue) eigenvector is the mean gradient direction; we
    return its orientation, ``0.5*atan2(2*Axy, Axx − Ayy)`` (the closed-form double
    -angle solution, which is why it is robust to the gradient's ±sign ambiguity).
    """
    from skimage.feature import structure_tensor

    # skimage returns Arr (rows, y), Arc (row-col cross), Acc (cols, x).
    a_rr, a_rc, a_cc = structure_tensor(img, sigma=_SIGMA, order="rc", mode="reflect")
    s_rr = float(np.sum(a_rr))
    s_rc = float(np.sum(a_rc))
    s_cc = float(np.sum(a_cc))
    # Eigenvector of the largest eigenvalue, expressed as a gradient orientation in
    # (x, y) = (cols, rows). In double-angle space (handles the ±gradient sign):
    #   Axx = s_cc (variation along x/cols), Ayy = s_rr, Axy = s_rc.
    axx, ayy, axy = s_cc, s_rr, s_rc
    gradient_rad = 0.5 * np.arctan2(2.0 * axy, axx - ayy)
    return float(np.degrees(gradient_rad))


def _gradient_histogram_gradient_deg(img: np.ndarray) -> float:
    """Dominant gradient orientation (deg, CCW from +x) via a double-angle vote.

    Each pixel's Sobel gradient ``(gx, gy)`` votes for its orientation weighted by
    magnitude. Votes are accumulated in **double-angle** space so opposite gradient
    signs reinforce rather than cancel, and the resultant's half-angle is the
    dominant gradient orientation. (A magnitude-weighted circular mean — the
    histogram is implicit in the vector sum.)
    """
    from skimage.filters import gaussian, sobel_h, sobel_v

    smooth = gaussian(img, sigma=_SIGMA, mode="reflect")
    gy = sobel_h(smooth)  # gradient along rows (y)
    gx = sobel_v(smooth)  # gradient along cols (x)
    phi2 = 2.0 * np.arctan2(gy, gx)  # gradient orientation, doubled
    w = np.hypot(gx, gy) ** 2  # weight by gradient energy
    c = float(np.sum(w * np.cos(phi2)))
    s = float(np.sum(w * np.sin(phi2)))
    gradient_rad = 0.5 * np.arctan2(s, c)
    return float(np.degrees(gradient_rad))


def _hough_to_vertical(img: np.ndarray) -> float:
    """Dominant *wall* orientation (deg to vertical) via the straight-line Hough.

    The Hough transform votes directly for line directions, so it returns the wall
    orientation (not the gradient). skimage's ``hough_line`` parameterises a line by
    its normal angle ``θ`` measured from the horizontal ``+x`` axis (the line's
    normal direction). A vertical wall has a horizontal normal (``θ = 0``); as the
    wall rotates toward horizontal its normal rotates the opposite way, so the
    wall's angle to vertical is ``−θ`` (mod 180). We pick the strongest peak; on a
    blank image (no peaks) we fall back to the structure tensor.
    """
    from skimage import feature
    from skimage.filters import gaussian
    from skimage.transform import hough_line, hough_line_peaks

    smooth = gaussian(img, sigma=_SIGMA, mode="reflect")
    rng = float(np.ptp(smooth))
    if rng <= 0:  # constant image -> no edges; defer to the structure tensor
        return _gradient_to_vertical(_structure_tensor_gradient_deg(img))
    edges = feature.canny(smooth, sigma=1.0)
    if not edges.any():
        return _gradient_to_vertical(_structure_tensor_gradient_deg(img))

    # Fine angle grid over the full (-90, 90] normal range.
    thetas = np.deg2rad(np.linspace(-90.0, 90.0, 360, endpoint=False))
    h, theta_grid, d = hough_line(edges, theta=thetas)
    accum, angles, _ = hough_line_peaks(h, theta_grid, d, num_peaks=1)
    if len(angles) == 0:
        return _gradient_to_vertical(_structure_tensor_gradient_deg(img))
    normal_from_x = float(np.degrees(angles[0]))  # line-normal angle from +x axis
    wall_to_vertical = -normal_from_x  # vertical wall has horizontal normal (θ=0)
    return float(wall_to_vertical % _PERIOD)


_METHODS = ("structure_tensor", "gradient_histogram", "hough")


def estimate_angle(image_2d, *, method: str = "structure_tensor") -> float:
    """Estimate the dominant linear orientation to the image's vertical axis.

    A pure, label-free function of the pixels: it recovers the dominant orientation
    of the bright wall-like structures and returns it **measured to the image's
    vertical axis** in ``[0, 180)`` (0° = vertical wall, 90° = horizontal),
    180-periodic. The ``structure_tensor`` and ``gradient_histogram`` backends
    recover a gradient orientation and convert via ``wall = gradient + 90`` then
    ``to_vertical = 90 − wall``; the ``hough`` backend votes for the wall direction
    directly. A degenerate (constant) image returns a defined finite angle.

    Parameters
    ----------
    image_2d : array-like
        A 2-D grayscale image of shape ``(H, W)`` (e.g. from
        :func:`uda.data.images.load_image_gray`). 3-D inputs are rejected.
    method : {"structure_tensor", "gradient_histogram", "hough"}, keyword-only
        Estimation backend (default ``"structure_tensor"``, recommended for its
        speckle robustness via Gaussian-windowed gradient integration).

    Returns
    -------
    float
        The estimated angle to the vertical axis in ``[0, 180)``.

    Raises
    ------
    ValueError
        If ``image_2d`` is not 2-D, or ``method`` is unknown.
    """
    if method not in _METHODS:
        raise ValueError(
            f"method must be one of {list(_METHODS)}; got {method!r}"
        )
    img = _check_2d(image_2d)

    if method == "hough":
        angle = _hough_to_vertical(img)
    elif method == "gradient_histogram":
        angle = _gradient_to_vertical(_gradient_histogram_gradient_deg(img))
    else:  # structure_tensor
        angle = _gradient_to_vertical(_structure_tensor_gradient_deg(img))

    # Guard against any NaN from a fully degenerate field; keep the answer in range.
    if not np.isfinite(angle):
        return 0.0
    return float(angle % _PERIOD)


def evaluate_geometric(labels_csv, images_dir) -> dict:
    """Score the classical geometric estimator over every base image.

    Loads each base image via ``uda.data.images`` (single source of truth for
    I/O), runs :func:`estimate_angle` on the pixels alone, joins to ``labels_csv``
    on ``image_id``, and scores the ``(theta_true, theta_pred)`` pairs with
    :func:`uda.evaluation.evaluate.metrics`. The labels are read **only** to look up the truth
    for scoring — never during estimation — so the reported MAE is honest (a poor
    hand-crafted baseline vs. the learned model is itself the finding).

    Parameters
    ----------
    labels_csv : str or pathlib.Path
        CSV with columns ``image_id, patient_id, theta_deg`` (the vertical-axis
        truth, in ``[0, 180)``).
    images_dir : str or pathlib.Path
        Directory of the base images (``data/images``).

    Returns
    -------
    dict
        ``{"y_true": numpy.ndarray, "y_pred": numpy.ndarray, "metrics": dict,
        "n": int}`` where ``y_true``/``y_pred`` have length ``n`` (one per base
        image, aligned on ``image_id``), ``metrics`` is
        :func:`uda.evaluation.evaluate.metrics` over those pairs (keys
        ``mae, rmse, me, mape, r2``), and ``n == len(y_true) == len(y_pred)``.
    """
    labels = pd.read_csv(labels_csv)
    truth_by_id = dict(
        zip(
            labels[_LABEL_IMAGE_ID].astype(str),
            labels[_LABEL_THETA].astype(float),
        )
    )

    y_true: list[float] = []
    y_pred: list[float] = []
    for path in uimg.list_base_images(Path(images_dir)):
        img_id = uimg.image_id(path)
        if img_id not in truth_by_id:
            continue
        # Estimation sees pixels only — the label is fetched afterwards for scoring.
        gray = uimg.load_image_gray(path)
        y_pred.append(estimate_angle(gray))
        y_true.append(truth_by_id[img_id])

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return {
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
        "metrics": evaluate.metrics(y_true_arr, y_pred_arr),
        "n": int(y_true_arr.size),
    }

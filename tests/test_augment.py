"""Tests — rotation sweep + CLAHE + normalize."""
from pathlib import Path

import numpy as np
import pytest
from skimage.metrics import structural_similarity as ssim

from uda.config import DataConfig
from uda.data.augment import (
    CLAHE_SSIM_TOLERANCE,
    augment_image,
    clahe,
    rotation_angles,
)

_FIXTURE = Path(__file__).resolve().parents[1] / "data" / "images" / "09-41-06_1.jpg"


def _fixture_gray_u8() -> np.ndarray:
    """Load the canonical fixture as a 2-D ``uint8`` grayscale image."""
    from PIL import Image

    return np.asarray(Image.open(_FIXTURE).convert("L")).astype(np.uint8)


# --- rotation_angles --------------------------------------------------------


def test_rotation_angles_default_count_is_25():
    assert len(rotation_angles(DataConfig())) == 25


def test_rotation_angles_endpoints_and_step():
    angles = rotation_angles(DataConfig())
    assert angles[0] == -60
    assert angles[-1] == 60
    assert angles[angles.index(0)] == 0  # zero rotation is included
    assert all(b - a == 5 for a, b in zip(angles, angles[1:]))


def test_rotation_angles_respects_custom_span():
    cfg = DataConfig(rotation_min_deg=-10, rotation_max_deg=10, rotation_step_deg=10)
    assert rotation_angles(cfg) == [-10, 0, 10]


# --- label math -------------------------------------------------------------


def test_label_math_plus_30_no_wrap_needed():
    img = _fixture_gray_u8()
    theta = 88.37
    out = {rot: nt for _, nt, rot in augment_image(img, theta, DataConfig())}
    assert out[30] == pytest.approx(theta + 30)
    assert out[0] == pytest.approx(theta)
    assert out[-30] == pytest.approx(theta - 30)


def test_label_math_wraps_into_0_180():
    img = _fixture_gray_u8()
    cfg = DataConfig()  # wrap_0_180 defaults to True
    # theta + rotation exceeds 180 -> wraps; e.g. 170 + 30 = 200 -> 20.
    out = {rot: nt for _, nt, rot in augment_image(img, 170.0, cfg)}
    assert out[30] == pytest.approx(20.0)
    # below zero wraps too: 10 + (-30) = -20 -> 160.
    out2 = {rot: nt for _, nt, rot in augment_image(img, 10.0, cfg)}
    assert out2[-30] == pytest.approx(160.0)


def test_label_math_no_wrap_when_disabled():
    img = _fixture_gray_u8()
    cfg = DataConfig(wrap_0_180=False)
    out = {rot: nt for _, nt, rot in augment_image(img, 170.0, cfg)}
    assert out[30] == pytest.approx(200.0)


# --- output image contract --------------------------------------------------


def test_augment_yields_one_per_rotation():
    img = _fixture_gray_u8()
    cfg = DataConfig()
    results = list(augment_image(img, 90.0, cfg))
    assert len(results) == len(rotation_angles(cfg)) == 25
    assert [r for _, _, r in results] == rotation_angles(cfg)


def test_output_images_float32_unit_range_same_shape():
    img = _fixture_gray_u8()
    h, w = img.shape
    for out, _, _ in augment_image(img, 90.0, DataConfig()):
        assert out.dtype == np.float32
        assert out.shape == (h, w)
        assert np.all(np.isfinite(out))
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_accepts_float_unit_range_input():
    img_f = _fixture_gray_u8().astype(np.float32) / 255.0
    out, _, _ = next(augment_image(img_f, 90.0, DataConfig()))
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


# --- CLAHE backends ---------------------------------------------------------


def test_clahe_output_contract_both_backends():
    img = _fixture_gray_u8()
    h, w = img.shape
    for backend in ("skimage", "opencv"):
        out = clahe(img, DataConfig(clahe_backend=backend))
        assert out.dtype == np.float32
        assert out.shape == (h, w)
        assert np.all(np.isfinite(out))
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_clahe_backends_agree_within_ssim_tolerance():
    img = _fixture_gray_u8()
    sk = clahe(img, DataConfig(clahe_backend="skimage"))
    cv = clahe(img, DataConfig(clahe_backend="opencv"))
    score = ssim(sk, cv, data_range=1.0)
    assert score >= CLAHE_SSIM_TOLERANCE


def test_augment_clahe_disabled_still_normalizes():
    img = _fixture_gray_u8()
    cfg = DataConfig(clahe=False)
    out, _, _ = next(augment_image(img, 90.0, cfg))
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0

"""Tests — canonical image loader + provenance (on real data/images)."""
from pathlib import Path

import numpy as np

from uda.data.images import (
    PROVENANCE,
    image_id,
    list_base_images,
    load_image_gray,
)

# Repo root is two levels up from this test file (tests/ -> repo/).
IMAGES_DIR = Path(__file__).resolve().parents[1] / "data" / "images"


def test_lists_exactly_84_base_images():
    paths = list_base_images(IMAGES_DIR)
    assert len(paths) == 84
    assert all(p.suffix == ".jpg" for p in paths)


def test_listing_is_deterministically_sorted():
    a = list_base_images(IMAGES_DIR)
    b = list_base_images(IMAGES_DIR)
    assert a == b
    assert a == sorted(a, key=lambda p: p.name)


def test_loaded_image_is_2d_float32_finite_in_unit_range():
    path = list_base_images(IMAGES_DIR)[0]
    img = load_image_gray(path)
    assert img.ndim == 2
    assert img.dtype == np.float32
    assert np.isfinite(img).all()
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_all_images_load_within_contract():
    # The invariant must hold for every base image, not just the first.
    for path in list_base_images(IMAGES_DIR):
        img = load_image_gray(path)
        assert img.ndim == 2
        assert img.dtype == np.float32
        assert np.isfinite(img).all()
        assert img.min() >= 0.0 and img.max() <= 1.0


def test_image_id_is_filename_stem():
    assert image_id(Path("data/images/09-41-06_1.jpg")) == "09-41-06_1"
    # Independent of directory and works on a real listed path.
    first = list_base_images(IMAGES_DIR)[0]
    assert image_id(first) == first.stem


def test_provenance_mentions_source_and_citation():
    assert isinstance(PROVENANCE, str) and PROVENANCE
    assert "splab.cz" in PROVENANCE
    assert "EMBC" in PROVENANCE
    assert "10.1109/EMBC.2019.8857587" in PROVENANCE

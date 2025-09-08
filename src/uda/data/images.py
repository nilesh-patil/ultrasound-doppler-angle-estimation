"""Canonical image loader + dataset provenance.

The 84 base images live in ``data/images/`` as ``HH-MM-SS[_n].jpg`` and are 1:1
with the rows of ``Results.txt``. This module is the *single* place that reads a
raw image off disk and turns it into the array the rest of the pipeline consumes:
a 2-D ``float32`` grayscale array in ``[0, 1]``. Augmentation, CLAHE, and resizing
build on top of this; none of that happens here.

The on-disk files are stored as 3-channel uint8 JPEGs (a grayscale B-mode scan
replicated across RGB), so :func:`load_image_gray` collapses them to one channel
with :func:`skimage.color.rgb2gray`, which also rescales to ``[0, 1]``.

This module is Keras-free (numpy + scikit-image only).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import color, img_as_float32, io

#: Dataset provenance, mirrored into ``data/README.md``: the SPLab Brno source DB
#: and the EMBC-2019 paper this work replicates.
PROVENANCE: str = (
    "SPLab Brno ultrasound database "
    "(http://splab.cz/en/download/databaze/ultrasound) — "
    "84 B-mode images of the common carotid artery (longitudinal section), "
    "10 volunteers, Sonix OP scanner, 10/14 MHz linear arrays. "
    'Citation: N. Patil, A. Anand, "Automated Ultrasound Doppler Angle '
    'Estimation Using Deep Learning," Annu Int Conf IEEE Eng Med Biol Soc '
    "(EMBC) 2019; 2019:28-31; doi:10.1109/EMBC.2019.8857587."
)

#: Glob pattern for the canonical base images.
_IMAGE_GLOB = "*.jpg"


def list_base_images(images_dir: Path) -> list[Path]:
    """Return the canonical base images, deterministically sorted.

    Parameters
    ----------
    images_dir : Path
        Directory holding the ``HH-MM-SS[_n].jpg`` base images (typically
        ``data/images``).

    Returns
    -------
    list of Path
        The 84 base-image paths sorted lexicographically by filename. The
        ``HH-MM-SS[_n]`` naming makes the lexical order also acquisition order,
        and the sort makes the result reproducible across filesystems.

    Raises
    ------
    FileNotFoundError
        If ``images_dir`` does not exist or contains no matching images.
    """
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images_dir does not exist: {images_dir}")
    paths = sorted(images_dir.glob(_IMAGE_GLOB), key=lambda p: p.name)
    if not paths:
        raise FileNotFoundError(f"no '{_IMAGE_GLOB}' images under {images_dir}")
    return paths


def load_image_gray(path: Path) -> np.ndarray:
    """Load one base image as a 2-D ``float32`` grayscale array in ``[0, 1]``.

    Parameters
    ----------
    path : Path
        Path to a single base image.

    Returns
    -------
    numpy.ndarray
        Grayscale image of shape ``(H, W)``, dtype ``float32``, with values in
        ``[0, 1]``. Multi-channel inputs are converted with
        :func:`skimage.color.rgb2gray`; an already 2-D input is rescaled to
        ``float32`` in ``[0, 1]``.

    Notes
    -----
    The final :func:`numpy.clip` is a guard against tiny floating-point excursions
    outside ``[0, 1]`` from the channel mixing, so the range invariant
    (``range ⊆ [0, 1]``) holds exactly.
    """
    arr = io.imread(Path(path))
    if arr.ndim == 3:
        # Drop an alpha channel if present, then mix RGB -> luminance in [0, 1].
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        gray = color.rgb2gray(arr)
    elif arr.ndim == 2:
        gray = img_as_float32(arr)
    else:  # pragma: no cover - defensive; base images are 2-D or 3-D
        raise ValueError(f"unexpected image ndim={arr.ndim} for {path}")
    gray = img_as_float32(gray)
    np.clip(gray, 0.0, 1.0, out=gray)
    return gray


def image_id(path: Path) -> str:
    """Return the image identifier (the filename stem).

    Parameters
    ----------
    path : Path
        Path to a base image, e.g. ``.../09-41-06_1.jpg``.

    Returns
    -------
    str
        The filename without directory or extension, e.g. ``"09-41-06_1"``. This
        is the key that joins images to label rows in ``Results.txt``.
    """
    return Path(path).stem

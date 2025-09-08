"""Rotation sweep + CLAHE + normalization — the augmentation pipeline.

The 84 base images are each expanded into a sweep of
rotations (default ``[-60, -55, ..., 60]`` → 25 rotations) and the Doppler angle
label is rotated with the image. Contrast is then equalized with CLAHE and the
result normalized to ``float32`` in ``[0, 1]``.

Optional richer augmentation is applied
**after** rotation+CLAHE+normalize via :func:`apply_richer`, gated by the
:class:`~uda.config.DataConfig` knobs ``flip_h``, ``flip_v``, ``gamma_jitter``,
``translate_frac`` and ``speckle_std``. All default off, so the baseline corpus
is byte-for-byte unchanged and no randomness is consumed unless a knob is set.

CLAHE backend semantics
-----------------------
``cfg.clahe_backend`` selects the implementation; both are calibrated to agree
closely (structural similarity well above :data:`CLAHE_SSIM_TOLERANCE` on the
real images). ``cfg.clahe_clip_limit`` carries the *scikit-image* native meaning
(a normalized clip fraction, default ``0.03``).

* ``"skimage"`` — :func:`skimage.exposure.equalize_adapthist` with
  ``kernel_size=(H // 8, W // 8)``, which reproduces scikit-image's own default
  kernel exactly (verified max-abs-diff ``0.0`` versus ``kernel_size=None``), and
  ``clip_limit=cfg.clahe_clip_limit``. This is the faithful default — the paper
  used scikit-image. Output is ``float64`` in ``[0, 1]`` → cast to ``float32``.
* ``"opencv"`` — :func:`cv2.createCLAHE` with
  ``clipLimit=cfg.clahe_clip_limit * 256.0`` and ``tileGridSize=(8, 8)`` in
  ``(cols, rows)`` order. The ``× 256`` factor maps scikit-image's normalized
  clip *fraction* onto OpenCV's absolute per-tile histogram bin-count clip
  (``n_bins = 256``). Returns ``uint8`` → divide by ``255`` → cast to ``float32``.

Both backends require single-channel input. The image is converted to ``uint8``
internally before equalization (scikit-image's adaptive equalization is
dtype-invariant, ~3e-8 difference between ``uint8`` and ``float[0, 1]`` input).
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from uda.config import DataConfig

#: Documented structural-similarity floor between the two CLAHE backends on the
#: project images (measured ~0.956 on ``09-41-06_1``). Tests assert agreement
#: at or above this value.
CLAHE_SSIM_TOLERANCE: float = 0.90


def rotation_angles(cfg: DataConfig) -> list[int]:
    """Return the integer rotation sweep for one image.

    Parameters
    ----------
    cfg : DataConfig
        Source of ``rotation_min_deg``, ``rotation_max_deg`` and
        ``rotation_step_deg``. The defaults (``-60``, ``60``, ``5``) give 25
        rotations.

    Returns
    -------
    list[int]
        Rotation angles in degrees, ``[min, min + step, ..., max]`` inclusive of
        both endpoints.
    """
    return list(
        range(
            cfg.rotation_min_deg,
            cfg.rotation_max_deg + 1,
            cfg.rotation_step_deg,
        )
    )


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Coerce a grayscale image to single-channel ``uint8``.

    Accepts ``uint8`` (passed through) or floating point in ``[0, 1]`` (scaled to
    ``0..255``). Values are clipped to the valid range before casting.
    """
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError(f"clahe expects a 2-D grayscale image, got shape {arr.shape}")
    if arr.dtype == np.uint8:
        return arr
    scaled = np.rint(np.clip(arr.astype(np.float64), 0.0, 1.0) * 255.0)
    return scaled.astype(np.uint8)


def clahe(img: np.ndarray, cfg: DataConfig) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization.

    Dispatches on ``cfg.clahe_backend``. Input may be ``uint8`` or floating point
    in ``[0, 1]``; output is always ``float32`` in ``[0, 1]``.

    Parameters
    ----------
    img : np.ndarray
        2-D grayscale image.
    cfg : DataConfig
        Provides ``clahe_backend`` and ``clahe_clip_limit``.

    Returns
    -------
    np.ndarray
        Equalized image, ``float32`` in ``[0, 1]``, same ``H × W`` as input.

    See Also
    --------
    Module docstring — backend parameter mapping and fidelity notes.
    """
    img_u8 = _to_uint8(img)
    h, w = img_u8.shape

    backend = cfg.clahe_backend
    if backend == "skimage":
        from skimage import exposure

        out = exposure.equalize_adapthist(
            img_u8,
            kernel_size=(h // 8, w // 8),
            clip_limit=cfg.clahe_clip_limit,
        )
        return np.asarray(out, dtype=np.float32)
    if backend == "opencv":
        import cv2

        tile = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit * 256.0,
            tileGridSize=(round(w / (w // 8)), round(h / (h // 8))),
        )
        out = tile.apply(img_u8).astype(np.float32) / 255.0
        return out
    raise ValueError(f"unknown clahe_backend: {backend!r}")  # pragma: no cover


def _wrap_0_180(theta: float) -> float:
    """Wrap an angle in degrees into ``[0, 180)`` (direction-agnostic period)."""
    return float(theta % 180.0)


def _richer_enabled(cfg: DataConfig) -> bool:
    """Whether any richer-augmentation knob is active.

    Parameters
    ----------
    cfg : DataConfig
        Reads ``flip_h``, ``flip_v``, ``gamma_jitter``, ``translate_frac`` and
        ``speckle_std`` only.

    Returns
    -------
    bool
        ``True`` iff at least one of the flip flags is set or one of the
        magnitude knobs is strictly positive. When ``False`` the caller must skip
        :func:`apply_richer` entirely, which guarantees an exact identity and zero
        RNG consumption (so default-config corpora are bit-for-bit unchanged).
    """
    return bool(
        cfg.flip_h
        or cfg.flip_v
        or cfg.gamma_jitter > 0.0
        or cfg.translate_frac > 0.0
        or cfg.speckle_std > 0.0
    )


def _shift_zero_filled(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Integer pixel shift with a zero fill (never wraps content).

    Positive ``dy`` moves content *down* (toward larger row indices), positive
    ``dx`` moves it *right*; the vacated border is filled with ``0``. Unlike
    :func:`numpy.roll`, no content crosses from the opposite edge — wrapping would
    alias anatomy across the frame, so we disallow it.

    Parameters
    ----------
    img : np.ndarray
        2-D source image.
    dy, dx : int
        Row / column shift in pixels (may be negative or zero).

    Returns
    -------
    np.ndarray
        Same shape and dtype as ``img``, shifted and zero-padded.
    """
    h, w = img.shape
    out = np.zeros_like(img)
    ys_dst = slice(max(dy, 0), h + min(dy, 0))
    xs_dst = slice(max(dx, 0), w + min(dx, 0))
    ys_src = slice(max(-dy, 0), h + min(-dy, 0))
    xs_src = slice(max(-dx, 0), w + min(-dx, 0))
    out[ys_dst, xs_dst] = img[ys_src, xs_src]
    return out


def apply_richer(
    img: np.ndarray, cfg: DataConfig, rng: np.random.Generator
) -> np.ndarray:
    """Apply the optional richer augmentations to a single normalized frame.

    The transforms are applied in a fixed order — ``flip_h`` → ``flip_v`` →
    ``gamma`` → ``translate`` → ``speckle`` — and each *stochastic* step draws
    from ``rng`` only when its knob is active. Toggling one knob therefore never
    perturbs another's RNG stream when that other is off, and a call with no
    active stochastic knob (e.g. flips only, or all knobs off) consumes no
    randomness at all. When *no* knob is active the input is returned cast to
    ``float32`` with no other change and ``rng`` is left untouched.

    Parameters
    ----------
    img : np.ndarray
        2-D grayscale frame, ``float`` in ``[0, 1]`` (the output of
        rotation + CLAHE + normalize). Higher-rank input raises ``ValueError``.
    cfg : DataConfig
        Reads ``flip_h``, ``flip_v``, ``gamma_jitter``, ``translate_frac`` and
        ``speckle_std`` only.
    rng : np.random.Generator
        The *only* source of randomness; the same ``rng`` state yields the same
        output.

    Returns
    -------
    np.ndarray
        ``float32`` 2-D image in ``[0, 1]``, same ``H x W`` as ``img``.

    Notes
    -----
    **Label invariance.** This returns only an image and never touches ``theta``.
    Gamma, translation and speckle are strictly label-preserving. Flips are *not*
    geometry-neutral for the Doppler angle — a horizontal or vertical mirror maps
    ``theta -> -theta ≡ 180 - theta`` under the ``[0, 180)`` wrap — so they are
    offered only as angle-*preserving* content augmentation: callers must treat
    mild mirroring as label-preserving noise (an ablation knob), not as a relabel.
    The headline experiment matrix keeps flips off.
    """
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError(
            f"apply_richer expects a 2-D grayscale image, got shape {arr.shape}"
        )

    # Fast path: no knob active => exact float32 identity, no RNG consumed.
    if not _richer_enabled(cfg):
        return arr.astype(np.float32)

    out = arr.astype(np.float32)

    # 1-2. Deterministic mirrors (no RNG). ``.copy()`` so the reversed view does
    #      not alias the source through later in-place steps.
    if cfg.flip_h:
        out = out[:, ::-1].copy()
    if cfg.flip_v:
        out = out[::-1, :].copy()

    # 3. Gamma jitter: symmetric in log-space => E[gamma] = 1 (no brightness
    #    bias). x**g is strictly monotone on [0, 1], so pixel ordering is kept.
    if cfg.gamma_jitter > 0.0:
        g = float(np.exp(rng.uniform(-cfg.gamma_jitter, cfg.gamma_jitter)))
        out = np.power(out, g, dtype=np.float32)

    # 4. Integer pixel shift, zero-filled (never wrapped). Draw dy then dx.
    if cfg.translate_frac > 0.0:
        f = cfg.translate_frac
        h, w = out.shape
        dy = round(float(rng.uniform(-f, f)) * h)
        dx = round(float(rng.uniform(-f, f)) * w)
        out = _shift_zero_filled(out, dy, dx)

    # 5. Additive zero-mean Gaussian "speckle", then clip back into range.
    if cfg.speckle_std > 0.0:
        out = out + rng.normal(0.0, cfg.speckle_std, size=out.shape)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def augment_image(
    img: np.ndarray,
    theta: float,
    cfg: DataConfig,
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[np.ndarray, float, int]]:
    """Yield the rotation-augmented variants of one labeled image.

    For each angle in :func:`rotation_angles`, the image is rotated
    (:func:`skimage.transform.rotate` with ``mode=cfg.rotation_mode`` and
    ``preserve_range=True``), optionally CLAHE-equalized (when ``cfg.clahe``), and
    normalized to ``float32`` in ``[0, 1]`` (when ``cfg.normalize``). When any
    richer-augmentation knob is set (see :func:`_richer_enabled`) the finalized
    frame is then passed through :func:`apply_richer`. The label is rotated with
    the image: ``new_theta = theta + rotation_deg``, wrapped into ``[0, 180)``
    when ``cfg.wrap_0_180``.

    Parameters
    ----------
    img : np.ndarray
        2-D grayscale base image (``uint8`` or float in ``[0, 1]``).
    theta : float
        Doppler angle of the base image, in degrees.
    cfg : DataConfig
        Augmentation configuration.
    rng : np.random.Generator, optional
        Source of randomness for the richer augmentations. Constructed **once**
        via :func:`numpy.random.default_rng` when ``None``, so every rotation of a
        single call draws from one stream. When no richer knob is active
        :func:`apply_richer` is never invoked and ``rng`` is untouched — the
        default-config output is therefore bit-for-bit identical to the
        baseline rotation+CLAHE+normalize pipeline.

    Yields
    ------
    tuple[np.ndarray, float, int]
        ``(image, new_theta, rotation_deg)`` where ``image`` is ``float32`` in
        ``[0, 1]`` with the same ``H × W`` as ``img``.
    """
    from skimage import transform

    base = np.asarray(img)
    if base.ndim != 2:
        raise ValueError(
            f"augment_image expects a 2-D grayscale image, got shape {base.shape}"
        )

    richer = _richer_enabled(cfg)
    if richer and rng is None:
        rng = np.random.default_rng()

    for rotation_deg in rotation_angles(cfg):
        rotated = transform.rotate(
            base,
            rotation_deg,
            mode=cfg.rotation_mode,
            preserve_range=True,
        )

        if cfg.clahe:
            out = clahe(rotated, cfg)
        elif cfg.normalize:
            out = np.asarray(_to_uint8(rotated), dtype=np.float32) / 255.0
        else:
            out = np.asarray(rotated, dtype=np.float32)

        if richer:
            out = apply_richer(out, cfg, rng)

        new_theta = theta + rotation_deg
        if cfg.wrap_0_180:
            new_theta = _wrap_0_180(new_theta)
        else:
            new_theta = float(new_theta)

        yield out, new_theta, rotation_deg

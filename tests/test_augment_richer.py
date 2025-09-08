"""Tests for richer augmentation (``apply_richer``).

These encode the binding contract for richer augmentation — ``apply_richer``,
``_richer_enabled`` and the optional ``rng`` parameter on ``augment_image``
in ``uda.data.augment``.

The fixed apply order (each step consumes RNG only when its knob is active) is
flip_h -> flip_v -> gamma -> translation -> speckle, with a final cast to
``float32`` and clip to ``[0, 1]``. Everything stochastic is a pure function of
the supplied ``np.random.Generator``.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from uda.config import DataConfig
from uda.data.augment import (
    apply_richer,
    augment_image,
    clahe,
    rotation_angles,
)

# --- synthetic fixtures (small + fast; no real images needed) ----------------


def _ramp(h: int = 16, w: int = 16) -> np.ndarray:
    """A 2-D float ``[0, 1]`` ramp with all-distinct values.

    Distinct values matter for the rank-preservation (gamma) test, and the
    asymmetric H-vs-W gradient lets the flip tests catch an axis swap. Values
    are kept strictly inside ``(0, 1)`` so a gamma power map cannot collapse
    ties at the 0/1 endpoints.
    """
    n = h * w
    flat = np.linspace(0.05, 0.95, n, dtype=np.float64)
    return flat.reshape(h, w).astype(np.float64)


# Every richer knob and the cfg kwargs that switch it on. ``translate_frac`` and
# ``speckle_std`` must stay in ``[0, 1)`` per the DataConfig validators.
_KNOBS: dict[str, dict] = {
    "flip_h": {"flip_h": True},
    "flip_v": {"flip_v": True},
    "gamma": {"gamma_jitter": 0.3},
    "translate": {"translate_frac": 0.2},
    "speckle": {"speckle_std": 0.05},
}

# Knobs that consume the RNG stream when active (flips are deterministic).
_STOCHASTIC = {"gamma", "translate", "speckle"}


def _all_knob_combos() -> list[tuple[str, ...]]:
    """Non-empty subsets of the knob names (power set minus the empty set)."""
    names = list(_KNOBS)
    combos: list[tuple[str, ...]] = []
    for r in range(1, len(names) + 1):
        combos.extend(itertools.combinations(names, r))
    return combos


def _cfg_for(combo: tuple[str, ...], **extra) -> DataConfig:
    kwargs: dict = {}
    for name in combo:
        kwargs.update(_KNOBS[name])
    kwargs.update(extra)
    return DataConfig(**kwargs)


def _state(rng: np.random.Generator):
    """Snapshot a generator's internal state for before/after comparison."""
    return rng.bit_generator.state


# --- 1. all-flags-off == identity (no RNG consumed) -------------------------


def test_all_flags_off_is_exact_identity():
    img = _ramp()
    cfg = DataConfig()  # every richer knob defaults OFF
    rng = np.random.default_rng(0)
    out = apply_richer(img, cfg, rng)
    # Same values, just cast to float32 — no transform applied.
    assert out.dtype == np.float32
    assert out.shape == img.shape
    assert np.array_equal(out, img.astype(np.float32))


def test_all_flags_off_does_not_advance_rng():
    img = _ramp()
    cfg = DataConfig()
    rng = np.random.default_rng(0)
    before = _state(rng)
    apply_richer(img, cfg, rng)
    assert _state(rng) == before  # identity path must not touch the stream


def test_augment_image_default_cfg_matches_recomputed_pipeline():
    """Regression guard: default cfg == plain rotation+CLAHE+normalize.

    With all richer knobs at their defaults ``augment_image`` must reproduce the
    baseline frames exactly. We recompute the documented per-rotation body
    (rotate -> CLAHE -> float32) and require frame-by-frame equality, proving the
    richer-aug insertion is a no-op when disabled.
    """
    from skimage import transform

    img = (_ramp(24, 24) * 255).astype(np.uint8)  # uint8 grayscale, like a frame
    cfg = DataConfig()  # clahe=True, normalize=True, all richer knobs off
    base = np.asarray(img)

    produced = list(augment_image(img, 90.0, cfg))
    assert [r for _, _, r in produced] == rotation_angles(cfg)

    for (out, _, rot) in produced:
        rotated = transform.rotate(
            base, rot, mode=cfg.rotation_mode, preserve_range=True
        )
        expected = clahe(rotated, cfg)  # default cfg has clahe=True
        assert out.dtype == np.float32
        assert np.array_equal(out, expected)


# --- 2. range / shape / dtype for every knob combination --------------------


@pytest.mark.parametrize(
    "combo", _all_knob_combos(), ids=lambda c: "+".join(c)
)
def test_output_contract_holds_for_every_knob_combo(combo):
    img = _ramp()
    cfg = _cfg_for(combo)
    rng = np.random.default_rng(123)
    out = apply_richer(img, cfg, rng)
    assert out.dtype == np.float32
    assert out.ndim == 2
    assert out.shape == img.shape
    assert np.all(np.isfinite(out))
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_higher_rank_input_raises_value_error():
    """The contract is a single 2-D frame; a 3-D input must raise ValueError."""
    cfg = DataConfig(gamma_jitter=0.2)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        apply_richer(np.zeros((4, 4, 3), dtype=np.float32), cfg, rng)


# --- 3. determinism: equal seeds => bit-equal output ------------------------


@pytest.mark.parametrize(
    "combo", _all_knob_combos(), ids=lambda c: "+".join(c)
)
def test_equal_seeded_generators_give_identical_output(combo):
    img = _ramp()
    cfg = _cfg_for(combo)
    out_a = apply_richer(img, cfg, np.random.default_rng(0))
    out_b = apply_richer(img, cfg, np.random.default_rng(0))
    assert np.array_equal(out_a, out_b)


# --- 4. per-knob RNG isolation ---------------------------------------------


def test_flips_only_do_not_advance_rng():
    img = _ramp()
    rng = np.random.default_rng(7)
    before = _state(rng)
    apply_richer(img, DataConfig(flip_h=True, flip_v=True), rng)
    assert _state(rng) == before  # deterministic mirrors consume no randomness


@pytest.mark.parametrize("knob", sorted(_STOCHASTIC))
def test_each_stochastic_knob_advances_rng(knob):
    img = _ramp()
    rng = np.random.default_rng(7)
    before = _state(rng)
    apply_richer(img, _cfg_for((knob,)), rng)
    assert _state(rng) != before  # an active stochastic knob must draw from rng


# --- 5. flip correctness ----------------------------------------------------


def test_flip_h_only_is_left_right_mirror():
    img = _ramp()
    out = apply_richer(img, DataConfig(flip_h=True), np.random.default_rng(0))
    assert np.array_equal(out, img[:, ::-1].astype(np.float32))


def test_flip_v_only_is_up_down_mirror():
    img = _ramp()
    out = apply_richer(img, DataConfig(flip_v=True), np.random.default_rng(0))
    assert np.array_equal(out, img[::-1, :].astype(np.float32))


# --- 6. gamma is a rank-preserving (monotone) power map ---------------------


def test_gamma_only_preserves_pixel_ordering():
    img = _ramp()
    # Large jitter + fixed seed: gamma != 1 almost surely, but a positive power
    # map x**g on [0, 1] is strictly monotone, so the value ordering is kept.
    cfg = DataConfig(gamma_jitter=2.0)
    out = apply_richer(img, cfg, np.random.default_rng(1))
    assert np.array_equal(np.argsort(out.ravel()), np.argsort(img.ravel()))
    # And it is genuinely a non-identity transform for this seed/knob.
    assert not np.array_equal(out, img.astype(np.float32))


# --- 7. translation is zero-filled, not wrapped -----------------------------


def test_translation_is_zero_filled_not_wrapped():
    """A seed-pinned shift leaves a 0 border and preserves interior content.

    We avoid ``np.roll`` because wrap-around aliases anatomy across the frame: the
    vacated rows/cols must be exactly 0, and no content may appear wrapped from
    the opposite edge. We find the realized integer (dy, dx) by locating where
    the original top-left pixel landed, then check the vacated border is zero and
    the shifted body equals the original interior.
    """
    h = w = 16
    # Strictly-positive distinct ramp so a wrapped pixel could never be mistaken
    # for the zero fill, and every value is unique for unambiguous matching.
    img = (np.arange(1, h * w + 1, dtype=np.float64) / (h * w + 1)).reshape(h, w)
    cfg = DataConfig(translate_frac=0.25)
    rng = np.random.default_rng(2)
    out = apply_richer(img, cfg, rng)

    f = cfg.translate_frac
    src = np.random.default_rng(2)  # replay the same draws the impl must make
    dy = round(float(src.uniform(-f, f)) * h)
    dx = round(float(src.uniform(-f, f)) * w)

    # Build the expected zero-filled shift (translate-and-pad-with-0).
    expected = np.zeros_like(img)
    ys_dst = slice(max(dy, 0), h + min(dy, 0))
    xs_dst = slice(max(dx, 0), w + min(dx, 0))
    ys_src = slice(max(-dy, 0), h + min(-dy, 0))
    xs_src = slice(max(-dx, 0), w + min(-dx, 0))
    expected[ys_dst, xs_dst] = img[ys_src, xs_src]

    assert np.allclose(out, expected.astype(np.float32), atol=0.0)

    # Explicit border check: at least one vacated strip exists for this seed and
    # every vacated pixel is exactly 0 (no wrap-around content).
    assert dy != 0 or dx != 0, "seed must force a non-zero shift for this test"
    if dy > 0:
        assert np.all(out[:dy, :] == 0.0)
    elif dy < 0:
        assert np.all(out[dy:, :] == 0.0)
    if dx > 0:
        assert np.all(out[:, :dx] == 0.0)
    elif dx < 0:
        assert np.all(out[:, dx:] == 0.0)


# --- 8. speckle: bounded, changes the image, ~unbiased ----------------------


def test_speckle_changes_image_but_stays_in_unit_range():
    img = _ramp()
    cfg = DataConfig(speckle_std=0.05)
    out = apply_richer(img, cfg, np.random.default_rng(3))
    assert not np.array_equal(out, img.astype(np.float32))
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.all(np.isfinite(out))


def test_speckle_is_approximately_unbiased_over_many_seeds():
    """Additive zero-mean Gaussian => mean(out - img) ~ 0 across many draws.

    Use a mid-gray constant frame so clipping at 0/1 cannot bias the residual
    (a ramp near the endpoints would clip asymmetrically). Average the residual
    over many seeds and the whole frame; it must be close to zero.
    """
    img = np.full((16, 16), 0.5, dtype=np.float64)
    cfg = DataConfig(speckle_std=0.05)
    residuals = []
    for seed in range(200):
        out = apply_richer(img, cfg, np.random.default_rng(seed))
        residuals.append(np.asarray(out, dtype=np.float64) - img)
    mean_residual = np.mean(residuals)
    assert abs(mean_residual) < 5e-3  # unbiased to well within a percent

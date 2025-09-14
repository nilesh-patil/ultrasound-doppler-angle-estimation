"""Tests — angle representations (periodicity fix)."""
import numpy as np
import pytest

from uda.config import TargetConfig
from uda.models.targets import (
    AngleTarget,
    RawDegrees,
    SinCos2Theta,
    build_target,
)

# Dense sweep across the full undirected-angle range [0, 180].
THETAS = np.linspace(0.0, 180.0, 181)


def test_n_outputs():
    assert RawDegrees().n_outputs == 1
    assert SinCos2Theta().n_outputs == 2


def test_raw_round_trip_exact():
    t = RawDegrees()
    decoded = t.decode(t.encode(THETAS))
    assert np.allclose(decoded, THETAS, atol=0.0, rtol=0.0)


def test_raw_encode_shape():
    t = RawDegrees()
    y = t.encode(THETAS)
    assert y.shape == (THETAS.size, 1)
    assert t.decode(y).shape == (THETAS.size,)


def test_sincos_encode_shape_and_unit_norm():
    t = SinCos2Theta()
    y = t.encode(THETAS)
    assert y.shape == (THETAS.size, 2)
    # encode produces points on the unit circle: sin^2 + cos^2 == 1.
    assert np.allclose(np.sum(y**2, axis=-1), 1.0, atol=1e-9)


def test_sincos_round_trip_within_tol_modulo_180():
    t = SinCos2Theta()
    decoded = t.decode(t.encode(THETAS))
    # Compare modulo the 180-degree period (0 and 180 map to the same point).
    diff = (decoded - THETAS + 90.0) % 180.0 - 90.0
    assert np.max(np.abs(diff)) < 1e-4


def test_sincos_decode_in_half_open_interval():
    t = SinCos2Theta()
    decoded = t.decode(t.encode(THETAS))
    assert np.all(decoded >= 0.0)
    assert np.all(decoded < 180.0)


def test_sincos_direction_agnostic():
    """theta and theta+180 are the same undirected orientation -> same encoding."""
    t = SinCos2Theta()
    base = np.array([0.0, 12.5, 47.3, 88.37, 104.0, 179.9])
    assert np.allclose(t.encode(base), t.encode(base + 180.0), atol=1e-9)


def test_sincos_decode_collapses_theta_plus_180():
    t = SinCos2Theta()
    base = np.array([5.0, 33.0, 88.37, 150.0])
    d_base = t.decode(t.encode(base))
    d_shift = t.decode(t.encode(base + 180.0))
    assert np.allclose(d_base, d_shift, atol=1e-4)


def test_sincos_decode_tolerates_non_unit_norm():
    """Only the direction of (cos, sin) matters, not the magnitude."""
    t = SinCos2Theta()
    y = t.encode(THETAS)
    decoded = t.decode(3.7 * y)  # scale away from the unit circle
    diff = (decoded - THETAS + 90.0) % 180.0 - 90.0
    assert np.max(np.abs(diff)) < 1e-4


def test_sincos_scalar_input():
    t = SinCos2Theta()
    enc = t.encode(88.37)
    assert enc.shape == (2,)
    dec = t.decode(enc)
    assert np.isscalar(dec) or dec.shape == ()
    assert abs(float(dec) - 88.37) < 1e-4


def test_build_target_dispatch():
    assert isinstance(build_target(TargetConfig(kind="raw")), RawDegrees)
    assert isinstance(build_target(TargetConfig(kind="sincos2theta")), SinCos2Theta)


def test_build_target_returns_angle_target():
    for kind in ("raw", "sincos2theta"):
        tgt = build_target(TargetConfig(kind=kind))
        assert isinstance(tgt, AngleTarget)


def test_build_target_default_config_is_raw():
    assert isinstance(build_target(TargetConfig()), RawDegrees)

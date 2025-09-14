"""Angle representations — the periodicity fix for Doppler-angle regression.

The Doppler insonation angle is undirected: ``theta`` and
``theta + 180`` describe the same vessel orientation. Regressing raw degrees (the
paper's choice) ignores this periodicity and is discontinuous at the wrap point,
so we also provide a ``(sin 2*theta, cos 2*theta)`` encoding whose decode is
continuous and direction-agnostic.

Every target implements the :class:`AngleTarget` protocol: a fixed ``n_outputs``
plus a mutually-inverse ``encode``/``decode`` pair. ``encode`` maps angles in
degrees to network targets; ``decode`` maps network outputs back to degrees in
``[0, 180)``. Decoding to degrees is deliberately kept out of the loss graph
(performed in evaluate.py), so these helpers operate on plain numpy arrays.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from uda.config import TargetConfig


@runtime_checkable
class AngleTarget(Protocol):
    """Reversible mapping between angles in degrees and network targets.

    Attributes
    ----------
    n_outputs : int
        Width of the encoded target vector (the model's final layer size).
    """

    n_outputs: int

    def encode(self, theta_deg: np.ndarray) -> np.ndarray:
        """Map angles in degrees to network targets of shape ``(..., n_outputs)``."""
        ...

    def decode(self, y: np.ndarray) -> np.ndarray:
        """Map network outputs back to angles in degrees within ``[0, 180)``."""
        ...


class RawDegrees:
    """Identity encoding — regress the angle in degrees directly (paper protocol).

    Encoding and decoding are the identity (a trailing singleton axis of width
    ``n_outputs == 1`` is added on encode and removed on decode), so this target
    carries no periodicity fix and is provided to faithfully replicate the
    original method.
    """

    n_outputs: int = 1

    def encode(self, theta_deg: np.ndarray) -> np.ndarray:
        """Return ``theta_deg`` unchanged, shaped ``(..., 1)``.

        Parameters
        ----------
        theta_deg : np.ndarray
            Angles in degrees, any shape.

        Returns
        -------
        np.ndarray
            ``float64`` array with a trailing axis of size 1.
        """
        theta = np.asarray(theta_deg, dtype=np.float64)
        return theta[..., np.newaxis]

    def decode(self, y: np.ndarray) -> np.ndarray:
        """Return the predicted degrees, dropping the trailing singleton axis.

        Parameters
        ----------
        y : np.ndarray
            Network outputs shaped ``(..., 1)`` (a trailing axis of size 1 is
            squeezed; other shapes pass through unchanged).

        Returns
        -------
        np.ndarray
            Angles in degrees as ``float64``.
        """
        out = np.asarray(y, dtype=np.float64)
        if out.ndim >= 1 and out.shape[-1] == 1:
            out = out[..., 0]
        return out


class SinCos2Theta:
    """``(sin 2*theta, cos 2*theta)`` encoding — continuous, direction-agnostic.

    Doubling the angle folds the ``180`` degree period onto a full ``360`` degree
    circle, so the encoding is identical for ``theta`` and ``theta + 180`` and is
    smooth across the wrap point. Decoding inverts via ``atan2`` and halves the
    recovered angle back into ``[0, 180)``.
    """

    n_outputs: int = 2

    def encode(self, theta_deg: np.ndarray) -> np.ndarray:
        """Encode angles as ``[sin(2*theta), cos(2*theta)]`` along a new last axis.

        Parameters
        ----------
        theta_deg : np.ndarray
            Angles in degrees, any shape ``S``.

        Returns
        -------
        np.ndarray
            ``float64`` array of shape ``S + (2,)``; the last axis holds
            ``(sin 2*theta, cos 2*theta)``.
        """
        theta = np.asarray(theta_deg, dtype=np.float64)
        two_theta = np.deg2rad(2.0 * theta)
        return np.stack([np.sin(two_theta), np.cos(two_theta)], axis=-1)

    def decode(self, y: np.ndarray) -> np.ndarray:
        """Decode ``[sin, cos]`` pairs to angles in degrees within ``[0, 180)``.

        Recovers ``2*theta = atan2(sin, cos)`` then halves it; the result is
        wrapped into ``[0, 180)`` so the inverse is single-valued. The encoded
        vector need not be unit-norm — only the direction of ``(cos, sin)`` is
        used.

        Parameters
        ----------
        y : np.ndarray
            Network outputs of shape ``(..., 2)`` holding ``(sin, cos)`` pairs.

        Returns
        -------
        np.ndarray
            Angles in degrees as ``float64`` in ``[0, 180)``.
        """
        out = np.asarray(y, dtype=np.float64)
        sin = out[..., 0]
        cos = out[..., 1]
        two_theta_deg = np.rad2deg(np.arctan2(sin, cos))
        theta = (two_theta_deg / 2.0) % 180.0
        # ``%`` can return a value rounded up to exactly 180.0 for inputs an
        # epsilon below the wrap point; snap those back to 0.0 so the result is
        # strictly half-open in [0, 180).
        theta = np.where(theta >= 180.0, 0.0, theta)
        return theta[()] if np.ndim(theta) == 0 else theta


def build_target(cfg: TargetConfig) -> AngleTarget:
    """Construct the :class:`AngleTarget` selected by ``cfg.kind``.

    Parameters
    ----------
    cfg : uda.config.TargetConfig
        Target configuration; ``cfg.kind`` is ``"raw"`` or ``"sincos2theta"``.

    Returns
    -------
    AngleTarget
        :class:`RawDegrees` for ``"raw"``, :class:`SinCos2Theta` for
        ``"sincos2theta"``.

    Raises
    ------
    ValueError
        If ``cfg.kind`` is not a recognised target kind.
    """
    if cfg.kind == "raw":
        return RawDegrees()
    if cfg.kind == "sincos2theta":
        return SinCos2Theta()
    raise ValueError(f"unknown target kind: {cfg.kind!r}")

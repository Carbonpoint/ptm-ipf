"""Spherical projections used by the legend and pole figure plots."""

from __future__ import annotations

import numpy as np

__all__ = [
    "equal_area",
    "inverse_stereographic",
    "stereographic",
    "upper_hemisphere",
]


def upper_hemisphere(v: np.ndarray) -> np.ndarray:
    """Flip directions onto the upper hemisphere (poles are centrosymmetric)."""
    v = np.atleast_2d(np.asarray(v, dtype=float))
    return np.where(v[:, 2:3] < 0, -v, v)


def stereographic(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stereographic projection from the south pole of unit vectors *v*."""
    v = np.atleast_2d(np.asarray(v, dtype=float))
    denom = 1.0 + v[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        return v[:, 0] / denom, v[:, 1] / denom


def inverse_stereographic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Inverse of :func:`stereographic`; returns unit vectors of shape (n, 3)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r2 = x**2 + y**2
    denom = 1.0 + r2
    return np.stack([2 * x / denom, 2 * y / denom, (1.0 - r2) / denom], axis=-1)


def equal_area(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lambert equal-area projection, scaled so the equator is the unit circle.

    Equal-area is the standard choice for pole figures because equal areas on
    the plot then correspond to equal solid angles, which is what makes a
    density in multiples of a random distribution meaningful.
    """
    v = np.atleast_2d(np.asarray(v, dtype=float))
    scale = np.sqrt(1.0 / (1.0 + v[:, 2]))
    return v[:, 0] * scale, v[:, 1] * scale

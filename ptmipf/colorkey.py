"""Inverse pole figure colour keys.

The colouring implemented here is the EDAX/TSL key as described by Nolze and
Hielscher, *Orientations - perfectly colored*, J. Appl. Cryst. 49 (2016) 1786.
The implementation follows MTEX' ``polarCoordinates`` routine and its Python
port in :mod:`orix.plot`, rewritten here so that colouring several million
atoms needs only NumPy.  Colours agree with MTEX, orix and EDAX OIM to within
floating point noise, which is what makes the maps produced by this package
directly comparable with EBSD orientation maps.
"""

from __future__ import annotations

import numpy as np

from .symmetry import LaueGroup

__all__ = ["IPFColorKey", "hsv_to_rgb"]


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.atleast_2d(np.asarray(v, dtype=float))
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / norm


def _angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between rows of *a* and *b*, numerically safe near 0 and pi."""
    a = _unit(a)
    b = _unit(b)
    dot = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.arccos(dot)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Vectorised HSV to RGB conversion, identical to matplotlib's."""
    hsv = np.asarray(hsv, dtype=float)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = (i.astype(int) % 6)[..., np.newaxis]
    options = np.stack(
        [
            np.stack([v, t, p], axis=-1),
            np.stack([q, v, p], axis=-1),
            np.stack([p, v, t], axis=-1),
            np.stack([p, q, v], axis=-1),
            np.stack([t, p, v], axis=-1),
            np.stack([v, p, q], axis=-1),
        ],
        axis=-2,
    )
    rgb = np.take_along_axis(options, i[..., np.newaxis], axis=-2)[..., 0, :]
    return np.clip(rgb, 0.0, 1.0)


def _hsl_to_hsv(hue, saturation, lightness):
    """Convert HSL to HSV, following MTEX' ``hsl2hsv``."""
    l2 = 2.0 * lightness
    s2 = saturation * np.where(l2 <= 1.0, l2, 2.0 - l2)
    with np.errstate(invalid="ignore", divide="ignore"):
        saturation2 = (2.0 * s2) / (l2 + s2)
    saturation2 = np.nan_to_num(saturation2, nan=0.0)
    value = (l2 + s2) / 2.0
    return hue, saturation2, value


class IPFColorKey:
    """Map crystal directions to inverse pole figure colours.

    Parameters
    ----------
    laue
        The Laue group whose fundamental sector is coloured.

    Notes
    -----
    The three sector vertices are red, green and blue; the sector barycentre is
    white.  For ``m-3m`` this gives the familiar red ``[001]`` / green
    ``[101]`` / blue ``[111]`` key, and for ``6/mmm`` red ``[0001]`` / green
    ``[2-1-10]`` / blue ``[10-10]``.
    """

    def __init__(self, laue: LaueGroup) -> None:
        self.laue = laue
        self.center = laue.center
        self.normals = laue.sector_normals
        self._build_azimuth_correction()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def direction2color(self, directions: np.ndarray) -> np.ndarray:
        """Colour crystal *directions*, shape (n, 3), returning (n, 3) RGB."""
        reduced = self.laue.reduce(directions)
        return self.sector_direction2color(reduced)

    def sector_direction2color(self, reduced: np.ndarray) -> np.ndarray:
        """Colour directions that already lie in the fundamental sector."""
        azimuth, polar = self.polar_coordinates(reduced)
        lightness = 0.5 + polar / 2.0
        hue = np.mod(azimuth / (2.0 * np.pi), 1.0)
        h, s, v = _hsl_to_hsv(hue, 1.0, lightness)
        return hsv_to_rgb(np.stack([h, s, v], axis=-1))

    def orientation2color(
        self, rotations: np.ndarray, sample_direction: np.ndarray
    ) -> np.ndarray:
        """Colour crystal-to-sample rotations by one sample direction.

        Parameters
        ----------
        rotations
            Crystal-to-sample rotation matrices, shape (n, 3, 3).
        sample_direction
            The reference direction in sample coordinates, shape (3,).

        Returns
        -------
        numpy.ndarray
            RGB colours, shape (n, 3).
        """
        d = _unit(np.asarray(sample_direction, dtype=float).reshape(1, 3))[0]
        # R maps crystal -> sample, so R^T maps the sample direction into the
        # crystal frame, which is the direction an inverse pole figure shows.
        crystal_directions = np.einsum("nji,j->ni", rotations, d)
        return self.direction2color(crystal_directions)

    # ------------------------------------------------------------------
    # polar coordinates inside the sector (MTEX ``polarCoordinates``)
    # ------------------------------------------------------------------
    def polar_coordinates(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Azimuthal and radial coordinates of *v* relative to the barycentre.

        The azimuth runs around the barycentre and selects the hue; the radial
        coordinate is 1 at the barycentre and 0 on the sector boundary, and
        sets the lightness.
        """
        v = _unit(v)
        azimuth = self._azimuth(v)
        azimuth = np.interp(azimuth, self._azimuth_grid, self._azimuth_table)
        polar = self._polar(v)
        return azimuth, polar

    def _azimuth(self, v: np.ndarray) -> np.ndarray:
        """MTEX' ``calcAngle``: angle around the barycentre, from the north pole."""
        center = self.center
        rx = np.array([0.0, 0.0, 1.0]) - center
        rx = _unit(rx - np.dot(rx, center) * center)[0]
        ry = _unit(np.cross(center, rx))[0]
        d = _unit(v - center)
        azimuth = np.arctan2(d @ ry, d @ rx)
        azimuth = np.mod(azimuth, 2.0 * np.pi)
        return np.nan_to_num(azimuth, nan=0.0)

    def _polar(self, v: np.ndarray) -> np.ndarray:
        center = self.center
        normal_to_plane = _unit(np.cross(v, center))
        polar = np.full(len(v), np.inf)
        for normal in self.normals:
            boundary = _unit(np.cross(normal_to_plane, normal))
            with np.errstate(invalid="ignore", divide="ignore"):
                distances = _angle_between(-v, boundary) / _angle_between(
                    -center[np.newaxis], boundary
                )
            distances = np.nan_to_num(distances, nan=1.0, posinf=1.0)
            polar = np.minimum(polar, distances)
        return np.clip(polar, 0.0, 1.0)

    # ------------------------------------------------------------------
    # azimuth correction table (MTEX' ``correctAngle``)
    # ------------------------------------------------------------------
    def _build_azimuth_correction(self, m: int = 1000) -> None:
        """Precompute the hue equalisation for a sector with unequal edges.

        Without this correction the fully blue point of the ``m-3m`` sector
        would sit due north of the barycentre instead of at ``[111]``.
        """
        center = self.center
        rx = np.array([0.0, 0.0, 1.0]) - center
        rx = _unit(rx - np.dot(rx, center) * center)[0]

        azimuth_grid = np.linspace(0.0, 2.0 * np.pi, m)
        base_normal = _unit(np.cross(rx, center))[0]
        # Rotate the reference normal about the barycentre.
        cos_a = np.cos(azimuth_grid[:-1])[:, np.newaxis]
        sin_a = np.sin(azimuth_grid[:-1])[:, np.newaxis]
        c = center[np.newaxis]
        normals = (
            base_normal * cos_a
            + np.cross(c, base_normal) * sin_a
            + c * (c @ base_normal) * (1.0 - cos_a)
        )

        # Angular distance from the barycentre to the sector boundary, as a
        # function of azimuth.
        distances = np.full((len(self.normals), m - 1), np.inf)
        for i, normal in enumerate(self.normals):
            distances[i] = _angle_between(np.cross(normal, normals), c)
        polar = np.min(distances, axis=0)

        vertices = _unit(self.laue.sector_vertices)
        if len(vertices) == 3:
            # Give each of the three sector edges an equal share of the hue
            # circle, so the vertices land on pure red, green and blue.
            angle = self._azimuth_raw(vertices, center, rx)
            splits = np.round(m * np.sort(angle) / (2.0 * np.pi)).astype(int)
            if splits[0] < 10:
                splits = splits[1:]
            splits = np.clip(splits, 0, polar.size)
            bounds = sorted({0, *splits.tolist(), polar.size})
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                idx = np.arange(lo, hi)
                if idx.size:
                    polar[idx] /= polar[idx].sum() / 3.0

        table = 2.0 * np.pi * np.cumsum(np.concatenate([[0.0], polar / polar.sum()]))
        self._azimuth_grid = azimuth_grid
        self._azimuth_table = table

    @staticmethod
    def _azimuth_raw(v: np.ndarray, center: np.ndarray, rx: np.ndarray) -> np.ndarray:
        ry = _unit(np.cross(center, rx))[0]
        d = _unit(v - center)
        return np.mod(np.arctan2(d @ ry, d @ rx), 2.0 * np.pi)

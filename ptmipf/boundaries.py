"""Colour the grain boundaries of a flat orientation map.

Two ways of colouring, both used in EBSD practice:

* by misorientation angle, on a continuous colour scale over a chosen range,
  so that a map of boundaries reads as a map of how much the lattice turns
  across each one; and
* by threshold, so that boundaries above an angle are drawn in one colour and
  the rest in another (or left alone), which is the usual high angle versus
  low angle boundary distinction.

The angle across a boundary is the disorientation between the two grains'
mean orientations, reduced by the crystal symmetry.  An optional crystal axis
turns this into the angle between that axis in the two grains instead, which
for hexagonal metals is the c axis tilt, the quantity that separates a twin
from a slightly rotated neighbour.
"""

from __future__ import annotations

import numpy as np

from .polefigure import IDEAL_C_OVER_A, miller_to_cartesian, symmetry_equivalents
from .symmetry import get_laue_group

__all__ = ["boundary_axis_angles", "color_boundaries_by_angle", "color_boundaries_by_threshold"]


def boundary_axis_angles(flat, axis, c_over_a: float = IDEAL_C_OVER_A, plane: bool = False):
    """Angle between one crystal axis in the two grains across each boundary.

    Parameters
    ----------
    flat
        A segmented :class:`~ptmipf.flatmap.FlatMap`.
    axis
        Miller or Miller-Bravais indices of the crystal direction, ``"0001"``
        for the hcp c axis, or a vector in the crystal frame.
    plane
        Treat the indices as a plane normal rather than a direction.

    Returns
    -------
    numpy.ndarray
        The angle in degrees per pixel, NaN off the boundary; the same shape
        and meaning as ``flat.boundary_angle_map`` but for the chosen axis.
    """
    if flat.labels is None or flat.rotations is None:
        raise ValueError("boundary angles need a segmented map")
    from .flatmap import _grain_mean_rotations

    laue = get_laue_group(flat.laue)
    if isinstance(axis, str):
        crystal = miller_to_cartesian(axis, laue, c_over_a, plane=plane)
    else:
        crystal = np.asarray(axis, dtype=float)
        crystal /= np.linalg.norm(crystal)
    equivalents = symmetry_equivalents(crystal, laue)

    means = _grain_mean_rotations(flat.labels, flat.orientation_index, flat.rotations, laue)
    # Sample-frame poles of the axis for every grain, all symmetry equivalents.
    poles = np.einsum("gij,mj->gmi", means, equivalents)

    labels = flat.labels
    out = np.full(labels.shape, np.nan)
    for shift, ax in ((-1, 0), (-1, 1)):
        neighbour = np.roll(labels, shift, axis=ax)
        differs = (labels >= 0) & (neighbour >= 0) & (labels != neighbour)
        differs[-1, :] = False if ax == 0 else differs[-1, :]
        if ax == 1:
            differs[:, -1] = False
        if not differs.any():
            continue
        a, b = labels[differs], neighbour[differs]
        # Smallest angle between any equivalent pole of grain a and of grain b.
        cosines = np.abs(np.einsum("nmi,nki->nmk", poles[a], poles[b]))
        angle = np.degrees(np.arccos(np.clip(cosines.max(axis=(1, 2)), 0.0, 1.0)))
        current = out[differs]
        out[differs] = np.where(np.isnan(current), angle, np.fmax(current, angle))
    return out


def color_boundaries_by_angle(
    flat,
    vmin: float = 0.0,
    vmax: float = 90.0,
    cmap: str = "viridis",
    angles: np.ndarray | None = None,
    below_color=None,
    width: int = 1,
) -> np.ndarray:
    """Paint the boundaries of *flat* by misorientation on a colour scale.

    Parameters
    ----------
    vmin, vmax
        Range of the colour scale in degrees; angles outside are clipped.
    cmap
        Any matplotlib colormap name.
    angles
        Per-pixel angles to use instead of the map's own disorientation, for
        example from :func:`boundary_axis_angles`.
    below_color
        If given, boundaries below *vmin* get this colour rather than the
        bottom of the scale, which separates "low angle" from "not measured".
    width
        Thickness of the drawn boundary in pixels.

    Returns
    -------
    numpy.ndarray
        A copy of ``flat.rgb`` with the boundaries recoloured.
    """
    import matplotlib

    angle_map = _angles_or_raise(flat, angles)
    rgb = flat.rgb.copy()
    on = flat.boundary & ~np.isnan(angle_map)
    if width > 1:
        on = _thicken(on, width)
        angle_map = _spread(angle_map, on)
    colormap = matplotlib.colormaps[cmap]
    scaled = np.clip((angle_map[on] - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    rgb[on] = colormap(scaled)[:, :3]
    if below_color is not None:
        low = on & (angle_map < vmin)
        rgb[low] = np.asarray(matplotlib.colors.to_rgb(below_color))
    return rgb


def color_boundaries_by_threshold(
    flat,
    threshold: float = 15.0,
    high_color="black",
    low_color=None,
    angles: np.ndarray | None = None,
    width: int = 1,
) -> np.ndarray:
    """Draw boundaries above *threshold* in one colour, the rest in another.

    With ``low_color=None`` the low angle boundaries are left as the map drew
    them, which by default means they vanish into the grain colour.  Pass a
    colour to draw them too, faintly for example, which is the conventional
    high angle black, low angle grey rendering.
    """
    from matplotlib.colors import to_rgb

    angle_map = _angles_or_raise(flat, angles)
    rgb = flat.rgb.copy()
    on = flat.boundary & ~np.isnan(angle_map)
    if width > 1:
        on = _thicken(on, width)
        angle_map = _spread(angle_map, on)
    high = on & (angle_map >= threshold)
    low = on & (angle_map < threshold)
    if low_color is not None:
        rgb[low] = np.asarray(to_rgb(low_color))
    else:
        # Restore the grain colour underneath, so the low angle line disappears.
        rgb[low] = _underneath(flat, low)
    rgb[high] = np.asarray(to_rgb(high_color))
    return rgb


def _angles_or_raise(flat, angles):
    """The per-pixel boundary angles, refusing a map that has none."""
    angle_map = flat.boundary_angle_map if angles is None else angles
    if flat.boundary is None or angle_map is None or not flat.boundary.any():
        raise ValueError(
            "this map has no segmented boundaries; build it with boundary_angle > 0"
        )
    return angle_map


def _thicken(mask: np.ndarray, width: int) -> np.ndarray:
    from scipy.ndimage import binary_dilation

    return binary_dilation(mask, iterations=max(0, width - 1))


def _spread(angle_map: np.ndarray, on: np.ndarray) -> np.ndarray:
    """Give thickened boundary pixels the angle of the nearest original one."""
    from scipy.ndimage import distance_transform_edt

    known = ~np.isnan(angle_map)
    if known.all() or not known.any():
        return angle_map
    _, nearest = distance_transform_edt(~known, return_indices=True)
    spread = angle_map[nearest[0], nearest[1]]
    return np.where(on, spread, angle_map)


def _underneath(flat, pixels: np.ndarray) -> np.ndarray:
    """The grain colour a boundary pixel would have had, from its label."""
    rgb = flat.rgb
    labels = flat.labels
    out = np.empty((int(pixels.sum()), 3))
    rows, cols = np.nonzero(pixels)
    for i, (r, c) in enumerate(zip(rows, cols)):
        label = labels[r, c]
        same = labels == label
        same &= ~flat.boundary
        out[i] = rgb[same].mean(axis=0) if same.any() else rgb[r, c]
    return out

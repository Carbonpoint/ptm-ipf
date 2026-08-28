"""Pole figures and inverse pole figure density plots.

Pole figures are drawn in Lambert equal-area projection on the upper
hemisphere, with intensities in multiples of a random distribution (MRD), which
is the convention used by EBSD texture software.
"""

from __future__ import annotations

import re

import numpy as np

from .legend import _format_indices, _sector_edge_points
from .projections import equal_area, stereographic, upper_hemisphere
from .symmetry import LaueGroup, get_laue_group

__all__ = [
    "ipf_density",
    "miller_to_cartesian",
    "parse_miller",
    "pole_directions",
    "pole_figure",
    "symmetry_equivalents",
]

#: Ideal axial ratio of a close-packed hexagonal lattice.
IDEAL_C_OVER_A = np.sqrt(8.0 / 3.0)


def parse_miller(indices: str | tuple) -> tuple[int, ...]:
    """Parse ``0001``, ``10-10``, ``(1 1 1)`` or ``[2,-1,-1,0]`` into indices.

    Three- and four-index (Miller-Bravais) notation are both accepted; the
    redundant third Bravais index is checked and dropped.
    """
    if not isinstance(indices, str):
        values = [int(i) for i in indices]
    else:
        text = indices.strip().strip("()[]{}<>")
        if re.search(r"[,\s]", text):
            values = [int(t) for t in re.split(r"[,\s]+", text) if t]
        else:
            # Compact form such as "10-10": a minus sign binds to the next digit.
            values = []
            sign = 1
            for char in text:
                if char == "-":
                    sign = -1
                    continue
                values.append(sign * int(char))
                sign = 1
    if len(values) == 4:
        h, k, i, l = values
        if h + k + i != 0:
            raise ValueError(
                f"invalid Miller-Bravais indices {indices!r}: h + k + i must be 0"
            )
        values = [h, k, l]
    if len(values) != 3:
        raise ValueError(f"expected 3 or 4 indices, got {values} from {indices!r}")
    return tuple(values)


def _lattice_matrix(laue: LaueGroup, c_over_a: float) -> np.ndarray:
    """Direct lattice vectors (as rows) in the template Cartesian frame."""
    if laue.hexagonal_indices:
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [-0.5, np.sqrt(3) / 2, 0.0],
                [0.0, 0.0, c_over_a],
            ]
        )
    return np.eye(3)


def miller_to_cartesian(
    indices,
    laue: LaueGroup,
    c_over_a: float = IDEAL_C_OVER_A,
    plane: bool = True,
) -> np.ndarray:
    """Convert Miller (-Bravais) indices to a Cartesian crystal direction.

    Parameters
    ----------
    indices
        Three- or four-index notation, e.g. ``0001`` or ``10-11``.
    laue
        Laue group, which fixes the lattice setting.
    c_over_a
        Axial ratio, used for hexagonal lattices.  Only non-basal, non-prismatic
        poles such as ``{10-11}`` depend on it.
    plane
        True for a plane normal ``(hkl)`` (the usual meaning in a pole figure),
        False for a lattice direction ``[uvw]``.

    Returns
    -------
    numpy.ndarray
        Unit vector of shape (3,) in the crystal Cartesian frame.
    """
    h, k, l = parse_miller(indices)
    lattice = _lattice_matrix(laue, c_over_a)
    if plane:
        reciprocal = np.linalg.inv(lattice).T  # rows are b1, b2, b3
        v = np.array([h, k, l], dtype=float) @ reciprocal
    else:
        v = np.array([h, k, l], dtype=float) @ lattice
    return v / np.linalg.norm(v)


def symmetry_equivalents(v: np.ndarray, laue: LaueGroup) -> np.ndarray:
    """All symmetrically equivalent directions of *v*, shape (m, 3).

    Both ``+v`` and ``-v`` are included, as a pole figure is centrosymmetric.
    """
    v = np.asarray(v, dtype=float).reshape(3)
    equivalents = np.concatenate([laue.operators @ v, -(laue.operators @ v)])
    # Remove duplicates, e.g. for [0001] where many operators coincide.
    rounded = np.round(equivalents, 8) + 0.0
    _, index = np.unique(rounded, axis=0, return_index=True)
    return equivalents[np.sort(index)]


def pole_directions(
    rotations: np.ndarray,
    indices,
    laue: LaueGroup,
    c_over_a: float = IDEAL_C_OVER_A,
    plane: bool = True,
    sample_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Sample-frame directions of one pole family for many orientations.

    Parameters
    ----------
    rotations
        Crystal-to-sample rotation matrices, shape (n, 3, 3).
    indices
        Pole family, e.g. ``0001``.
    sample_matrix
        Rows are the sample frame axes in cell coordinates; if given, the poles
        are expressed in that frame instead of in cell coordinates.

    Returns
    -------
    numpy.ndarray
        Unit vectors of shape (n * m, 3) for the *m* equivalent poles.
    """
    crystal = miller_to_cartesian(indices, laue, c_over_a, plane=plane)
    equivalents = symmetry_equivalents(crystal, laue)
    # (n, m, 3): rotate every equivalent pole of every orientation into the cell.
    poles = np.einsum("nij,mj->nmi", rotations, equivalents).reshape(-1, 3)
    if sample_matrix is not None:
        poles = poles @ np.asarray(sample_matrix, dtype=float).T
    return poles


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur, so SciPy is not needed at run time."""
    if sigma <= 0:
        return image
    radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(image, radius, mode="constant")
    blurred = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 0, padded)
    blurred = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, blurred)
    return blurred[radius:-radius, radius:-radius]


def _density_grid(x, y, resolution: int, sigma_bins: float):
    """Kernel density of projected poles on the unit disc, in MRD."""
    edges = np.linspace(-1.0, 1.0, resolution + 1)
    counts, _, _ = np.histogram2d(x, y, bins=(edges, edges))
    density = _gaussian_blur(counts, sigma_bins)

    centers = 0.5 * (edges[:-1] + edges[1:])
    gx, gy = np.meshgrid(centers, centers, indexing="ij")
    inside = gx**2 + gy**2 <= 1.0
    # Equal-area projection: uniform on the hemisphere is uniform on the disc,
    # so MRD is simply the density divided by its mean over the disc.
    mean = density[inside].mean()
    mrd = np.where(inside, density / mean if mean > 0 else 0.0, np.nan)
    return centers, mrd


def _draw_frame(ax, up_label: str, right_label: str) -> None:
    theta = np.linspace(0, 2 * np.pi, 361)
    ax.plot(np.cos(theta), np.sin(theta), color="black", lw=1.0, zorder=5)
    # The centre of the projection is the out-of-page axis, drawn with the usual
    # circle-and-dot symbol rather than named: it is implied by the two in-plane
    # axes and needs no label.
    ax.plot(
        [0], [0], marker="o", markerfacecolor="none", markeredgecolor="black",
        ms=7, markeredgewidth=0.9, linestyle="none", zorder=5,
    )
    ax.plot([0], [0], marker=".", color="black", ms=2.5, linestyle="none", zorder=5)
    pad = 1.06
    ax.text(0, pad, up_label, ha="center", va="bottom", fontsize=10)
    ax.text(pad, 0, right_label, ha="left", va="center", fontsize=10)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")


def pole_figure(
    rotations: np.ndarray,
    poles,
    laue,
    sample_frame=None,
    up: str = "rd",
    right: str = "td",
    center: str = "nd",
    c_over_a: float = IDEAL_C_OVER_A,
    plane: bool = True,
    mode: str = "density",
    resolution: int = 300,
    smoothing: float = 4.0,
    max_orientations: int = 200_000,
    contours: int = 12,
    cmap: str = "viridis",
    filename=None,
    dpi: int = 200,
    seed: int = 0,
):
    """Plot one or more pole figures from crystal-to-sample rotations.

    Parameters
    ----------
    rotations
        Rotation matrices, shape (n, 3, 3), as returned by
        :meth:`~ptmipf.analysis.IPFResult.rotations`.
    poles
        A pole family such as ``0001``, or a sequence of them.
    laue
        Laue group or its name.
    sample_frame
        :class:`~ptmipf.frames.SampleFrame` defining the plot axes.
    up, right, center
        Sample axes placed at the top, at the right, and at the centre.
    mode
        ``"density"`` for a contoured MRD map, ``"scatter"`` for individual poles.
    max_orientations
        Orientations are randomly subsampled to this many before plotting.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    from .frames import SampleFrame

    if not isinstance(laue, LaueGroup):
        laue = get_laue_group(laue)
    if isinstance(poles, (str, tuple)) and not isinstance(poles, list):
        poles = [poles]
    frame = sample_frame or SampleFrame()

    rotations = np.asarray(rotations, dtype=float)
    if len(rotations) > max_orientations:
        rng = np.random.default_rng(seed)
        rotations = rotations[rng.choice(len(rotations), max_orientations, replace=False)]

    # Rows: the plot's right, up and out-of-page axes in cell coordinates.
    basis = np.stack([frame.direction(right), frame.direction(up), frame.direction(center)])
    if np.linalg.det(basis) < 0:
        basis[0] *= -1.0

    fig, axes = plt.subplots(1, len(poles), figsize=(3.3 * len(poles), 3.6), squeeze=False)
    for ax, pole in zip(axes[0], poles):
        directions = pole_directions(
            rotations, pole, laue, c_over_a=c_over_a, plane=plane, sample_matrix=basis
        )
        x, y = equal_area(upper_hemisphere(directions))

        if mode == "scatter":
            ax.scatter(x, y, s=1.0, c="tab:blue", alpha=0.25, linewidths=0, zorder=3)
        else:
            centers, mrd = _density_grid(x, y, resolution, smoothing)
            gx, gy = np.meshgrid(centers, centers, indexing="ij")
            contour = ax.contourf(gx, gy, mrd, levels=contours, cmap=cmap, zorder=3)
            bar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.08)
            bar.set_label("MRD", fontsize=9)

        _draw_frame(ax, frame.label(up), frame.label(right))
        ax.set_title(_format_indices(str(pole), "{}" if plane else "<>"), fontsize=11)

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=dpi)
    return fig


def ipf_density(
    rotations: np.ndarray,
    direction,
    laue,
    sample_frame=None,
    resolution: int = 400,
    smoothing: float = 4.0,
    max_orientations: int = 200_000,
    contours: int = 12,
    cmap: str = "magma",
    filename=None,
    dpi: int = 200,
    seed: int = 0,
):
    """Plot the density of crystal directions inside the IPF fundamental sector.

    This is the inverse pole figure that accompanies an IPF-coloured map: it
    shows how much of the sample actually sits at each colour.
    """
    import matplotlib.pyplot as plt

    from .frames import SampleFrame

    if not isinstance(laue, LaueGroup):
        laue = get_laue_group(laue)
    frame = sample_frame or SampleFrame()
    d = frame.direction(direction)

    rotations = np.asarray(rotations, dtype=float)
    if len(rotations) > max_orientations:
        rng = np.random.default_rng(seed)
        rotations = rotations[rng.choice(len(rotations), max_orientations, replace=False)]

    crystal = np.einsum("nji,j->ni", rotations, d)
    reduced = laue.reduce(crystal)
    x, y = stereographic(reduced)

    edge = _sector_edge_points(laue)
    ex, ey = stereographic(edge)
    pad = 0.03
    xr = (ex.min() - pad, ex.max() + pad)
    yr = (ey.min() - pad, ey.max() + pad)

    bins = (
        np.linspace(*xr, resolution + 1),
        np.linspace(*yr, resolution + 1),
    )
    counts, _, _ = np.histogram2d(x, y, bins=bins)
    density = _gaussian_blur(counts, smoothing)

    cx = 0.5 * (bins[0][:-1] + bins[0][1:])
    cy = 0.5 * (bins[1][:-1] + bins[1][1:])
    gx, gy = np.meshgrid(cx, cy, indexing="ij")
    inside = laue.in_sector(inverse_stereographic_grid(gx, gy)).reshape(gx.shape)
    density = np.where(inside, density, np.nan)
    mean = np.nanmean(density)
    mrd = density / mean if mean > 0 else density

    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    contour = ax.contourf(gx, gy, mrd, levels=contours, cmap=cmap, zorder=2)
    bar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.08)
    bar.set_label("MRD", fontsize=9)
    ax.plot(ex, ey, color="black", lw=1.0, zorder=3)

    vx, vy = stereographic(laue.sector_vertices)
    centroid = np.array([vx.mean(), vy.mean()])
    for px, py, label in zip(vx, vy, laue.vertex_labels):
        offset = np.array([px, py]) - centroid
        offset = offset / np.linalg.norm(offset) * 0.06
        ax.text(
            px + offset[0],
            py + offset[1],
            _format_indices(label),
            ha="center",
            va="center",
            fontsize=10,
        )

    ax.set_title(f"IPF {frame.label(direction)}  ({laue.name})", fontsize=11, pad=18)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.12)
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=dpi)
    return fig


def inverse_stereographic_grid(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    from .projections import inverse_stereographic

    return inverse_stereographic(gx.ravel(), gy.ravel())

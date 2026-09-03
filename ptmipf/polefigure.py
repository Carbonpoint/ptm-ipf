"""Pole figures and inverse pole figure density plots.

Pole figures are drawn in Lambert equal-area projection on the upper
hemisphere, with intensities in multiples of a random distribution (MRD), which
is the convention used by EBSD texture software.
"""

from __future__ import annotations

import re

import numpy as np

from .legend import _format_indices, _sector_edge_points, place_vertex_labels
from .projections import equal_area, stereographic, upper_hemisphere
from .symmetry import LaueGroup, get_laue_group

__all__ = [
    "equal_area_sigma_bins",
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


def _gaussian_blur(image: np.ndarray, sigma) -> np.ndarray:
    """Separable Gaussian blur, so SciPy is not needed at run time.

    *sigma* is one value or one per axis; the axes need separate values
    whenever the two grid spacings differ, which they do for the inverse pole
    figure sector.
    """
    sigmas = (float(sigma), float(sigma)) if np.isscalar(sigma) else tuple(float(v) for v in sigma)
    if max(sigmas) <= 0:
        return image
    radius = max(1, int(np.ceil(3 * max(sigmas))))
    padded = np.pad(image, radius, mode="constant")
    for axis, value in enumerate(sigmas):
        if value <= 0:
            continue
        x = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (x / value) ** 2)
        kernel /= kernel.sum()
        padded = np.apply_along_axis(
            lambda row, k=kernel: np.convolve(row, k, mode="same"), axis, padded
        )
    return padded[radius:-radius, radius:-radius]


def _combine(bandwidth: float, extra: float) -> float:
    """Two Gaussians in series are one Gaussian, widths adding in quadrature."""
    return float(np.hypot(bandwidth, extra))


def equal_area_sigma_bins(degrees: float, resolution: int) -> float:
    """An angular smoothing width in equal-area disc bins.

    The Lambert projection puts a direction at polar angle ``t`` from the
    centre at radius ``sqrt(2) sin(t/2)``, so near the centre one radian of arc
    is ``1/sqrt(2)`` of the disc radius, and that is the conversion used.

    Away from the centre the kernel becomes elliptical: radially it is
    compressed and tangentially stretched, by up to ``sqrt(2)`` each way at the
    rim.  The two factors are exact reciprocals, which is the projection being
    equal-area, so the solid angle the kernel spreads over is right everywhere
    and only its shape drifts.  For the few degrees of smoothing this is meant
    for, that is not visible; it is written down because at 40 degrees it would
    be.
    """
    if degrees <= 0:
        return 0.0
    return float(np.radians(degrees) / np.sqrt(2.0) * (resolution / 2.0))


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


def _levels(values, contours: int, vmin: float | None, vmax: float | None):
    """Contour levels for a density plot, over a fixed range when one is given.

    Without a range every figure is scaled to its own peak, which is the right
    default but makes two figures impossible to compare: the same colour means
    a different multiple of random in each.  Fixing the range is what turns a
    series of pole figures into a sequence that can be read.
    """
    finite = np.asarray(values)[np.isfinite(values)]
    low = float(vmin) if vmin is not None else (float(finite.min()) if finite.size else 0.0)
    high = float(vmax) if vmax is not None else (float(finite.max()) if finite.size else 1.0)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        high = low + 1.0
    return np.linspace(low, high, max(2, int(contours) + 1))


def _extend(values, vmin: float | None, vmax: float | None) -> str:
    """Which ends of the colour bar to mark as running past the range."""
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        return "neither"
    below = vmin is not None and float(finite.min()) < float(vmin)
    above = vmax is not None and float(finite.max()) > float(vmax)
    return {(True, True): "both", (True, False): "min", (False, True): "max"}.get(
        (below, above), "neither"
    )


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
    bandwidth: float = 4.0,
    smoothing: float = 0.0,
    max_orientations: int = 200_000,
    contours: int = 12,
    cmap="viridis",
    filename=None,
    dpi: int = 200,
    seed: int = 0,
    vmin: float | None = None,
    vmax: float | None = None,
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
    bandwidth
        Kernel width of the density estimate, in grid bins.  This is what makes
        the plot a density rather than a histogram of delta functions and is
        not meant to be a knob.
    smoothing
        Extra smoothing, as a Gaussian standard deviation in **degrees** of
        misorientation, added in quadrature to *bandwidth*.  Off by default.

        A simulated cell is a few thousand grains at most and its grains are
        nearly perfect, so its poles arrive as very sharp spots and the peak
        MRD comes out far above anything an EBSD map of the same texture would
        report.  A few degrees of smoothing puts the intensities on a scale
        comparable with the published figure next to it.  It is a presentation
        choice, not a measurement, so the figure is annotated with the width
        that was used.
    cmap
        A matplotlib colour map name, a colour map, an ``(n, 3)`` array, or the
        path to an image strip or a text table of RGB triples.
    max_orientations
        Orientations are randomly subsampled to this many before plotting.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    from .colormap import load_colormap
    from .frames import SampleFrame

    if not isinstance(laue, LaueGroup):
        laue = get_laue_group(laue)
    if isinstance(poles, (str, tuple)) and not isinstance(poles, list):
        poles = [poles]
    frame = sample_frame or SampleFrame()
    colors = load_colormap(cmap)
    sigma = _combine(bandwidth, equal_area_sigma_bins(smoothing, resolution))

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
            centers, mrd = _density_grid(x, y, resolution, sigma)
            gx, gy = np.meshgrid(centers, centers, indexing="ij")
            contour = ax.contourf(
                gx, gy, mrd, levels=_levels(mrd, contours, vmin, vmax), cmap=colors,
                zorder=3, extend=_extend(mrd, vmin, vmax),
            )
            bar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.08)
            bar.set_label("MRD", fontsize=9)

        _draw_frame(ax, frame.label(up), frame.label(right))
        ax.set_title(_format_indices(str(pole), "{}" if plane else "<>"), fontsize=11)
        if smoothing > 0 and mode != "scatter":
            # The figure has to carry its own provenance: an MRD peak means
            # something different at 10 degrees of smoothing than at none.
            ax.text(
                0, -1.22, f"smoothed {smoothing:g}\u00b0", ha="center", va="top",
                fontsize=8, color="0.35",
            )

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
    bandwidth: float = 4.0,
    smoothing: float = 0.0,
    max_orientations: int = 200_000,
    contours: int = 12,
    cmap="magma",
    filename=None,
    dpi: int = 200,
    seed: int = 0,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Plot the density of crystal directions inside the IPF fundamental sector.

    This is the inverse pole figure that accompanies an IPF-coloured map: it
    shows how much of the sample actually sits at each colour.  *bandwidth*,
    *smoothing* and *cmap* mean what they do in :func:`pole_figure`, except
    that this plot is stereographic rather than equal-area, so the angular
    conversion uses that projection's own scale.
    """
    import matplotlib.pyplot as plt

    from .colormap import load_colormap
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
    # Stereographic: radius is tan(t/2), so one radian near the centre is half a
    # unit of the plot.  The two axes carry different bin widths here, so each
    # gets its own sigma rather than one shared value.
    span = np.array([xr[1] - xr[0], yr[1] - yr[0]])
    extra = np.radians(smoothing) * 0.5 * resolution / span if smoothing > 0 else (0.0, 0.0)
    density = _gaussian_blur(counts, [_combine(bandwidth, e) for e in np.atleast_1d(extra)])

    cx = 0.5 * (bins[0][:-1] + bins[0][1:])
    cy = 0.5 * (bins[1][:-1] + bins[1][1:])
    gx, gy = np.meshgrid(cx, cy, indexing="ij")
    inside = laue.in_sector(inverse_stereographic_grid(gx, gy)).reshape(gx.shape)
    density = np.where(inside, density, np.nan)
    mean = np.nanmean(density)
    mrd = density / mean if mean > 0 else density

    # Wide enough, with the bar pushed out, that the corner labels of the
    # sector do not run into the colour bar.
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    contour = ax.contourf(
        gx, gy, mrd, levels=_levels(mrd, contours, vmin, vmax), cmap=load_colormap(cmap),
        zorder=2, extend=_extend(mrd, vmin, vmax),
    )
    bar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.16)
    bar.set_label("MRD", fontsize=9)
    ax.plot(ex, ey, color="black", lw=1.0, zorder=3)

    place_vertex_labels(ax, laue, fontsize=10)

    ax.set_title(f"IPF {frame.label(direction)}  ({laue.name})", fontsize=11, pad=18)
    if smoothing > 0:
        # Below the sector rather than in the title: the title sits under the
        # [111] vertex label and a longer one runs into it.  The figure still
        # carries its own provenance, which is the point.
        ax.text(
            0.5 * (ex.min() + ex.max()), ey.min() - 0.09,
            f"smoothed {smoothing:g}\u00b0",
            ha="center", va="top", fontsize=8, color="0.35",
        )
    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.26)
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=dpi)
    return fig


def inverse_stereographic_grid(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    from .projections import inverse_stereographic

    return inverse_stereographic(gx.ravel(), gy.ravel())

"""Render the inverse pole figure colour key as a standalone legend.

The legend is the fundamental sector drawn in stereographic projection and
filled with the same colours used for the atoms, i.e. the familiar triangle
printed next to an EBSD orientation map.
"""

from __future__ import annotations

import numpy as np

from .colorkey import IPFColorKey
from .projections import inverse_stereographic, stereographic
from .symmetry import LaueGroup, get_laue_group

__all__ = ["ipf_legend", "sector_image"]


def _format_indices(label: str, brackets: str = "[]") -> str:
    """Turn ``2-1-10`` into ``$[2\\bar{1}\\bar{1}0]$`` for matplotlib.

    *brackets* selects the enclosing pair: ``[]`` for a direction, ``()`` for a
    plane, ``{}`` for a plane family and ``<>`` for a direction family.
    """
    out = []
    i = 0
    while i < len(label):
        if label[i] == "-" and i + 1 < len(label):
            out.append(r"\bar{" + label[i + 1] + "}")
            i += 2
        else:
            out.append(label[i])
            i += 1
    left, right = brackets
    escape = {"{": r"\{", "}": r"\}", "<": r"\langle ", ">": r"\rangle "}
    left = escape.get(left, left)
    right = escape.get(right, right)
    return "$" + left + "".join(out) + right + "$"


def sector_image(
    laue: LaueGroup, resolution: int = 600, padding: float = 0.02
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Rasterise the coloured fundamental sector in stereographic projection.

    Returns an RGBA image and its ``(xmin, xmax, ymin, ymax)`` extent.  Pixels
    outside the sector are fully transparent.
    """
    key = IPFColorKey(laue)
    vx, vy = stereographic(laue.sector_vertices)
    # Include the sector edges, not only the vertices, when sizing the image.
    edge = _sector_edge_points(laue)
    ex, ey = stereographic(edge)
    xmin, xmax = min(vx.min(), ex.min()) - padding, max(vx.max(), ex.max()) + padding
    ymin, ymax = min(vy.min(), ey.min()) - padding, max(vy.max(), ey.max()) + padding

    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    gx, gy = np.meshgrid(xs, ys)
    directions = inverse_stereographic(gx.ravel(), gy.ravel())

    inside = laue.in_sector(directions, tol=1e-12)
    image = np.zeros((directions.shape[0], 4))
    if inside.any():
        image[inside, :3] = key.sector_direction2color(directions[inside])
        image[inside, 3] = 1.0
    return image.reshape(resolution, resolution, 4), (xmin, xmax, ymin, ymax)


def _sector_edge_points(laue: LaueGroup, n: int = 200) -> np.ndarray:
    """Sample points along the great-circle edges joining the sector vertices."""
    vertices = laue.sector_vertices
    segments = []
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        t = np.linspace(0.0, 1.0, n)[:, None]
        arc = a * (1 - t) + b * t  # chord, then projected back onto the sphere
        arc /= np.linalg.norm(arc, axis=1, keepdims=True)
        segments.append(arc)
    return np.vstack(segments)


def place_vertex_labels(ax, laue: LaueGroup, fontsize: int = 11) -> None:
    """Label the sector corners, clear of the sector itself.

    Shared by the colour key and the orientation density plot so that both keep
    the labels off the data; a layout check on the density plot found them
    sitting on the contours when each drew its own.
    """
    vx, vy = stereographic(laue.sector_vertices)
    center = np.array([vx.mean(), vy.mean()])
    span = max(vx.max() - vx.min(), vy.max() - vy.min())
    gap = 0.16 * span
    for x, y, label in zip(vx, vy, laue.vertex_labels):
        offset = np.array([x, y]) - center
        norm = np.linalg.norm(offset)
        offset = offset / norm * gap if norm > 1e-9 else np.array([0.0, gap])
        ax.text(
            x + offset[0],
            y + offset[1],
            _format_indices(label),
            ha="center",
            va="center",
            fontsize=fontsize,
            zorder=3,
        )


def ipf_legend(
    laue,
    direction_label: str = "",
    structure_label: str = "",
    resolution: int = 600,
    ax=None,
    filename=None,
    dpi: int = 200,
):
    """Draw the inverse pole figure colour key.

    Parameters
    ----------
    laue
        A :class:`~ptmipf.symmetry.LaueGroup` or its name (``6/mmm``).
    direction_label
        Sample direction shown in the title, e.g. ``ND``.
    structure_label
        Structure name shown in the title, e.g. ``hcp``.
    ax
        Existing matplotlib axes to draw into.
    filename
        If given, the figure is saved here.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if not isinstance(laue, LaueGroup):
        laue = get_laue_group(laue)

    image, extent = sector_image(laue, resolution=resolution)

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
    else:
        fig = ax.figure

    ax.imshow(
        image,
        extent=(extent[0], extent[1], extent[2], extent[3]),
        origin="lower",
        interpolation="bilinear",
        zorder=1,
    )

    edge = _sector_edge_points(laue)
    ex, ey = stereographic(edge)
    ax.plot(ex, ey, color="black", linewidth=1.0, zorder=2)

    place_vertex_labels(ax, laue)

    title = " ".join(part for part in (structure_label, laue.name) if part)
    if direction_label:
        title = f"{title}\nIPF {direction_label}" if title else f"IPF {direction_label}"
    if title:
        ax.set_title(title, fontsize=11)

    ax.set_aspect("equal")
    ax.axis("off")
    # Room for the labels that now sit outside the sector.
    ax.margins(0.26)
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=dpi, transparent=True)
    return fig

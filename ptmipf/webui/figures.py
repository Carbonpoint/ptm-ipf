"""Matplotlib figures for the web UI, returned as in-memory PNG or SVG bytes.

These are thin wrappers around :mod:`ptmipf.legend` and
:mod:`ptmipf.polefigure`; the web UI adds nothing of its own to the plots, so
what the browser shows is exactly what the CLI writes to disk.
"""

from __future__ import annotations

import io

import numpy as np

from ..polefigure import IDEAL_C_OVER_A
from ..structures import get_structure
from ..symmetry import get_laue_group

__all__ = [
    "content_type",
    "figure_bytes",
    "figure_png",
    "flat_map_png",
    "ipf_density_png",
    "legend_png",
    "pole_figure_png",
]

_CONTENT_TYPES = {"png": "image/png", "svg": "image/svg+xml"}


def content_type(fmt: str) -> str:
    return _CONTENT_TYPES[normalise_format(fmt)]


def normalise_format(fmt) -> str:
    fmt = str(fmt or "png").lower().lstrip(".")
    if fmt not in _CONTENT_TYPES:
        raise ValueError(f"figures come as png or svg, not {fmt!r}")
    return fmt


#: Widest image worth producing, in pixels.  Beyond this a request is more
#: likely a typo than a plan, and the memory cost is real.
MAX_WIDTH_PX = 12000


def figure_bytes(
    fig,
    fmt: str = "png",
    dpi: int = 130,
    transparent: bool = False,
    width_px: int | None = None,
) -> bytes:
    """Serialise and close *fig*; SVG keeps the plot editable in Inkscape.

    *width_px* asks for an image of that many pixels across.  It is met by
    scaling the dots per inch rather than the figure, so the layout, the fonts
    and the line weights keep their proportions: printed at the width it was
    asked for, the figure has exactly the resolution that width implies.  A
    vector format has no pixels, so it is ignored there.
    """
    import matplotlib.pyplot as plt

    fmt = normalise_format(fmt)
    if width_px and fmt == "png":
        wanted = min(float(width_px), MAX_WIDTH_PX)
        dpi = max(20.0, wanted / max(fig.get_figwidth(), 0.1))
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format=fmt, dpi=dpi, transparent=transparent)
    finally:
        plt.close(fig)
    return buffer.getvalue()


def figure_png(fig, dpi: int = 130, transparent: bool = False) -> bytes:
    return figure_bytes(fig, "png", dpi=dpi, transparent=transparent)


def default_plot_structure(result, requested: str | None = None) -> str:
    """The structure whose orientations are plotted.

    Defaults to the colourable structure with the most identified atoms, which
    is what a user of a polycrystal almost always means.
    """
    if requested:
        return get_structure(requested).name
    candidates = [s.name for s in result.structures if s.colorable]
    if not candidates:
        raise ValueError("no colourable structure was identified")
    return max(candidates, key=lambda name: result.counts.get(name, 0))


def _rotations(result, structure: str) -> np.ndarray:
    rotations = result.rotations(structure)
    if len(rotations) == 0:
        raise ValueError(f"no atoms were identified as {structure}; nothing to plot")
    return rotations


def legend_png(
    result,
    structure: str | None = None,
    dpi: int = 130,
    fmt: str = "png",
    width_px: int | None = None,
) -> bytes:
    from ..legend import ipf_legend

    name = default_plot_structure(result, structure)
    laue = get_laue_group(get_structure(name).laue)
    fig = ipf_legend(
        laue, direction_label=result.direction_label, structure_label=name
    )
    return figure_bytes(fig, fmt, dpi=dpi, transparent=True, width_px=width_px)


def pole_figure_png(
    result,
    poles,
    structure: str | None = None,
    c_over_a: float | None = None,
    mode: str = "density",
    smoothing: float = 0.0,
    cmap="viridis",
    max_orientations: int = 200_000,
    dpi: int = 130,
    fmt: str = "png",
    up=None,
    right=None,
    width_px: int | None = None,
) -> bytes:
    """Pole figures, by default with RD up and TD right.

    *up* and *right* take any direction spec; the projection looks down their
    cross product, so choosing them chooses the whole view.
    """
    from ..polefigure import pole_figure

    name = default_plot_structure(result, structure)
    laue = get_laue_group(get_structure(name).laue)
    axes = {}
    if up or right:
        frame = result.frame
        up_v = frame.direction(up or "rd")
        right_v = frame.direction(right or "td")
        center = np.cross(right_v, up_v)
        if np.linalg.norm(center) < 1e-6:
            raise ValueError("the up and right directions of a pole figure must differ")
        # The specs, not the vectors, go through for the two labelled axes so
        # that "x" is labelled X rather than [1 0 0].
        axes = {
            "up": up or "rd",
            "right": right or "td",
            "center": center / np.linalg.norm(center),
        }
    fig = pole_figure(
        _rotations(result, name),
        list(poles),
        laue,
        sample_frame=result.frame,
        c_over_a=IDEAL_C_OVER_A if c_over_a is None else float(c_over_a),
        mode=mode,
        smoothing=smoothing,
        cmap=cmap,
        max_orientations=max_orientations,
        **axes,
    )
    return figure_bytes(fig, fmt, dpi=dpi, width_px=width_px)


def ipf_density_png(
    result,
    direction,
    structure: str | None = None,
    smoothing: float = 0.0,
    cmap="magma",
    max_orientations: int = 200_000,
    dpi: int = 130,
    fmt: str = "png",
    width_px: int | None = None,
) -> bytes:
    from ..polefigure import ipf_density

    name = default_plot_structure(result, structure)
    laue = get_laue_group(get_structure(name).laue)
    fig = ipf_density(
        _rotations(result, name),
        direction,
        laue,
        sample_frame=result.frame,
        smoothing=smoothing,
        cmap=cmap,
        max_orientations=max_orientations,
    )
    return figure_bytes(fig, fmt, dpi=dpi, width_px=width_px)


def flat_map_png(
    result,
    view: str = "z",
    slab_width: float = 10.0,
    pixel_size: float = 0.5,
    boundary_angle: float = 5.0,
    fill_unindexed: bool = True,
    structure: str | None = None,
    dpi: int = 130,
    slab_center: float | None = None,
    title: str | None = None,
    fmt: str = "png",
    width_px: int | None = None,
) -> tuple[bytes, dict]:
    """A flat orientation map of a section, plus what it found.

    Returns the image and a summary, so the page can report the grain count
    and the size of the map without decoding the image.  *slab_center* is the
    position of the section along the view direction; without it the section
    is taken through the middle of the cell.
    """
    from ..flatmap import flat_ipf_map, save_flat_map

    flat = flat_ipf_map(
        result,
        view=view,
        slab_width=slab_width,
        slab_center=slab_center,
        pixel_size=pixel_size,
        boundary_angle=boundary_angle,
        structure=structure or default_plot_structure(result, structure),
        fill_unindexed=fill_unindexed,
    )
    fig = save_flat_map(flat, None, title=title or f"IPF {result.direction_label}")
    info = {
        "n_grains": flat.n_grains,
        "rows": flat.shape[0],
        "columns": flat.shape[1],
        "slab_center": round(float(flat.slab_center), 4),
    }
    return figure_bytes(fig, fmt, dpi=dpi, width_px=width_px), info

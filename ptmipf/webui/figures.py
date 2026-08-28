"""Matplotlib figures for the web UI, returned as in-memory PNG bytes.

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

__all__ = ["figure_png", "ipf_density_png", "legend_png", "pole_figure_png"]


def figure_png(fig, dpi: int = 130, transparent: bool = False) -> bytes:
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=dpi, transparent=transparent)
    finally:
        plt.close(fig)
    return buffer.getvalue()


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


def legend_png(result, structure: str | None = None, dpi: int = 130) -> bytes:
    from ..legend import ipf_legend

    name = default_plot_structure(result, structure)
    laue = get_laue_group(get_structure(name).laue)
    fig = ipf_legend(
        laue, direction_label=result.direction_label, structure_label=name
    )
    return figure_png(fig, dpi=dpi, transparent=True)


def pole_figure_png(
    result,
    poles,
    structure: str | None = None,
    c_over_a: float | None = None,
    mode: str = "density",
    max_orientations: int = 200_000,
    dpi: int = 130,
) -> bytes:
    from ..polefigure import pole_figure

    name = default_plot_structure(result, structure)
    laue = get_laue_group(get_structure(name).laue)
    fig = pole_figure(
        _rotations(result, name),
        list(poles),
        laue,
        sample_frame=result.frame,
        c_over_a=IDEAL_C_OVER_A if c_over_a is None else float(c_over_a),
        mode=mode,
        max_orientations=max_orientations,
    )
    return figure_png(fig, dpi=dpi)


def ipf_density_png(
    result,
    direction,
    structure: str | None = None,
    max_orientations: int = 200_000,
    dpi: int = 130,
) -> bytes:
    from ..polefigure import ipf_density

    name = default_plot_structure(result, structure)
    laue = get_laue_group(get_structure(name).laue)
    fig = ipf_density(
        _rotations(result, name),
        direction,
        laue,
        sample_frame=result.frame,
        max_orientations=max_orientations,
    )
    return figure_png(fig, dpi=dpi)

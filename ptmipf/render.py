"""Render IPF-coloured configurations directly with OVITO.

This produces the picture that usually accompanies an EBSD-style figure: the
atoms coloured by orientation, optionally with the grain boundary atoms hidden
or the cell cut open so the grain interiors are visible.
"""

from __future__ import annotations

import numpy as np

from .analysis import DEFAULT_OTHER_COLOR, ipf_color_modifier
from .frames import SampleFrame
from .structures import DEFAULT_STRUCTURES, get_structure

__all__ = ["render_ipf", "render_result"]


def render_ipf(
    source,
    filename,
    direction="z",
    structures=DEFAULT_STRUCTURES,
    frame: SampleFrame | None = None,
    frame_index: int = 0,
    rmsd_cutoff: float = 0.1,
    other_color=DEFAULT_OTHER_COLOR,
    hide_other: bool = False,
    slice_normal=None,
    slice_distance: float | None = None,
    slice_width: float = 0.0,
    size=(800, 600),
    camera_dir=(-1.0, -1.0, -0.5),
    perspective: bool = True,
    radius: float | None = None,
    transparent: bool = False,
    renderer: str = "auto",
):
    """Render an IPF-coloured image of *source* to *filename*.

    Parameters
    ----------
    hide_other
        Delete atoms without a recognised orientation, which leaves only the
        grain interiors.
    slice_normal, slice_distance, slice_width
        Cut the configuration open with an OVITO slice, so grain interiors are
        visible from outside.
    renderer
        ``"opengl"``, ``"tachyon"`` or ``"auto"`` to try Tachyon (which works
        without a GPU) and fall back to OpenGL.

    Returns
    -------
    str
        The filename written.
    """
    from ovito.io import import_file
    from ovito.modifiers import (
        DeleteSelectedModifier,
        ExpressionSelectionModifier,
        PolyhedralTemplateMatchingModifier,
        SliceModifier,
    )
    from ovito.pipeline import Pipeline
    from ovito.vis import ParticlesVis, Viewport

    frame = frame or SampleFrame()
    structure_objs = tuple(get_structure(s) for s in structures)

    pipeline = source if isinstance(source, Pipeline) else import_file(str(source))
    ptm = PolyhedralTemplateMatchingModifier(
        output_orientation=True, rmsd_cutoff=rmsd_cutoff
    )
    wanted = {
        int(getattr(PolyhedralTemplateMatchingModifier.Type, s.ptm_type))
        for s in structure_objs
    }
    for structure_type in ptm.structures:
        structure_type.enabled = int(structure_type.id) in wanted
    pipeline.modifiers.append(ptm)
    pipeline.modifiers.append(
        ipf_color_modifier(
            direction=direction, frame=frame, structures=structures, other_color=other_color
        )
    )

    if hide_other:
        pipeline.modifiers.append(ExpressionSelectionModifier(expression="StructureType == 0"))
        pipeline.modifiers.append(DeleteSelectedModifier())

    if slice_normal is not None:
        normal = frame.direction(slice_normal)
        slice_modifier = SliceModifier(
            normal=tuple(float(c) for c in normal), slab_width=float(slice_width)
        )
        if slice_distance is None:
            data = pipeline.compute(frame_index)
            center = np.asarray(data.particles.positions[...]).mean(axis=0)
            slice_distance = float(np.dot(center, normal))
        slice_modifier.distance = float(slice_distance)
        pipeline.modifiers.append(slice_modifier)

    if radius is not None:
        particles = pipeline.source.data.particles
        if isinstance(particles.vis, ParticlesVis):
            particles.vis.radius = float(radius)

    pipeline.add_to_scene()
    try:
        viewport = Viewport(
            type=Viewport.Type.Perspective if perspective else Viewport.Type.Ortho,
            camera_dir=tuple(float(c) for c in camera_dir),
        )
        viewport.zoom_all(size=size)
        for name in (["tachyon", "opengl"] if renderer == "auto" else [renderer]):
            try:
                engine = _make_renderer(name)
            except Exception:
                continue
            try:
                viewport.render_image(
                    filename=str(filename),
                    size=size,
                    renderer=engine,
                    frame=frame_index,
                    alpha=transparent,
                )
                return str(filename)
            except Exception:
                if renderer != "auto":
                    raise
        raise RuntimeError(
            "no OVITO renderer could produce an image in this environment; "
            "write the coloured file with --output and render it elsewhere"
        )
    finally:
        pipeline.remove_from_scene()


def _make_renderer(name: str):
    from ovito import vis

    if name == "tachyon":
        return vis.TachyonRenderer()
    if name == "opengl":
        return vis.OpenGLRenderer()
    raise ValueError(f"unknown renderer {name!r}")


def render_result(
    result,
    filename,
    hide_other: bool = False,
    slice_normal=None,
    slice_distance: float | None = None,
    slice_width: float = 0.0,
    size=(800, 600),
    camera_dir=(-1.0, -1.0, -0.5),
    perspective: bool = True,
    radius: float | None = None,
    transparent: bool = False,
    renderer: str = "auto",
    show_cell: bool = True,
):
    """Render an :class:`~ptmipf.analysis.IPFResult` that has already been computed.

    Unlike :func:`render_ipf` this does not re-run polyhedral template matching:
    the colours in the result are drawn as they are.  That makes it the way to
    render a selection, since any subset of a result can be passed in.

    Returns
    -------
    str
        The filename written.
    """
    from ovito.data import DataCollection
    from ovito.pipeline import Pipeline, StaticSource
    from ovito.vis import Viewport

    positions = result.positions
    colors = result.colors
    if hide_other:
        keep = np.zeros(result.n_atoms, dtype=bool)
        for structure in result.structures:
            if structure.colorable:
                keep |= result.structure_types == result.type_codes[structure.name]
        positions = positions[keep]
        colors = colors[keep]

    data = DataCollection()
    particles = data.create_particles(count=len(positions))
    particles.create_property("Position", data=positions)
    particles.create_property("Color", data=colors)
    if radius is not None:
        particles.vis.radius = float(radius)

    if result.cell is not None:
        cell = data.create_cell(np.asarray(result.cell, dtype=float), pbc=(True, True, True))
        cell.vis.enabled = show_cell

    pipeline = Pipeline(source=StaticSource(data=data))

    if slice_normal is not None:
        from ovito.modifiers import SliceModifier

        normal = result.frame.direction(slice_normal)
        if slice_distance is None:
            slice_distance = float(np.dot(positions.mean(axis=0), normal))
        pipeline.modifiers.append(
            SliceModifier(
                normal=tuple(float(c) for c in normal),
                distance=float(slice_distance),
                slab_width=float(slice_width),
            )
        )

    pipeline.add_to_scene()
    try:
        viewport = Viewport(
            type=Viewport.Type.Perspective if perspective else Viewport.Type.Ortho,
            camera_dir=tuple(float(c) for c in camera_dir),
        )
        viewport.zoom_all(size=size)
        for name in (["tachyon", "opengl"] if renderer == "auto" else [renderer]):
            try:
                engine = _make_renderer(name)
            except Exception:
                continue
            try:
                viewport.render_image(
                    filename=str(filename), size=size, renderer=engine, alpha=transparent
                )
                return str(filename)
            except Exception:
                if renderer != "auto":
                    raise
        raise RuntimeError(
            "no OVITO renderer could produce an image in this environment; "
            "write the coloured file with --output and render it elsewhere"
        )
    finally:
        pipeline.remove_from_scene()

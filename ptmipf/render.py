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

__all__ = ["render_ipf", "render_result", "renderer_refusal", "TRIPOD_MARGIN"]


def renderer_refusal() -> str:
    """Why drawing must not even be attempted here, or "" when it is safe.

    A renderer that fails raises, and that is handled everywhere.  A renderer
    that crashes takes the process with it, so a combination known to crash
    has to be refused before it is asked, not after.  OVITO 3.15.5 on Windows
    and Python 3.14 raises an access violation on its first render; there is
    no OVITO 3.16 build for that Python on Windows, so the answer is Python
    3.13.  Everything that does not need a renderer keeps working.
    """
    import platform
    import sys

    if platform.system() != "Windows" or sys.version_info < (3, 14):
        return ""
    try:
        from importlib.metadata import version

        from packaging.version import Version

        if Version(version("ovito")) >= Version("3.16"):
            return ""
    except Exception:  # pragma: no cover - depends on the installed packages
        return ""
    return (
        "the 3D view is switched off: OVITO before 3.16 crashes when it draws "
        "on Windows with Python 3.14, and OVITO 3.16 has no build for that "
        "Python. Make the environment with Python 3.13 for the 3D view. The "
        "plots, the IPF map and the exports do not need it."
    )

#: Framing used for a triad when the corner is already clear.
TRIPOD_MARGIN = 1.15
#: Never pull the camera back further than this to make room for a triad.
MAX_TRIPOD_MARGIN = 1.7
#: Fraction of the image, from the lower left corner, that a triad occupies.
#: Measured from rendered images, with room for the axis labels around it.
TRIPOD_BOX = (0.23, 0.21)


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
    refusal = renderer_refusal()
    if refusal:
        raise RuntimeError(refusal)
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


def _set_if_possible(target, name: str, value) -> bool:
    """Set an optional overlay property, ignoring one this OVITO build refuses."""
    try:
        setattr(target, name, value)
        return True
    except Exception:
        return False


def _tripod_overlay(
    frame,
    labels=("rd", "td", "nd"),
    size: float = 0.11,
    offset_x: float = 0.07,
    offset_y: float = 0.05,
    names=None,
):
    """A coordinate tripod showing the sample axes, e.g. RD, TD and ND.

    Without it a rendered map does not say which way the reference direction
    points, which is the one thing a reader needs to interpret the colours.
    *labels* are direction specs (named sample axes, cell axes or vectors);
    *names* optionally replaces the text drawn for each of them.  The offsets
    are fractions of the image, measured from the lower left corner.
    """
    from ovito.vis import CoordinateTripodOverlay

    tripod = CoordinateTripodOverlay()
    # Which of these an OVITO build accepts varies between versions, and a
    # font that is not installed is refused outright, so the cosmetic ones are
    # applied one at a time: a triad drawn with the defaults beats no view.
    _set_if_possible(tripod, "style", CoordinateTripodOverlay.Style.Solid)
    _set_if_possible(tripod, "size", float(size))
    _set_if_possible(tripod, "line_width", 0.06)
    _set_if_possible(tripod, "font_size", 0.42)
    _set_if_possible(tripod, "font_family", "DejaVu Sans")
    # Enough inset that the axis labels are not clipped by the image edge.
    _set_if_possible(tripod, "offset_x", float(offset_x))
    _set_if_possible(tripod, "offset_y", float(offset_y))
    _set_if_possible(tripod, "axis4_enabled", False)
    colors = ((0.85, 0.20, 0.20), (0.20, 0.60, 0.25), (0.20, 0.35, 0.85))
    labels = [label for label in labels if str(label).strip()][:3]
    names = list(names or [])
    for index in range(1, 4):
        _set_if_possible(tripod, f"axis{index}_enabled", False)
    for index, (spec, color) in enumerate(zip(labels, colors), start=1):
        text = names[index - 1] if index - 1 < len(names) and names[index - 1] else None
        # The direction and the label are the point of the triad, so a failure
        # here is a real one and is left to propagate.
        setattr(tripod, f"axis{index}_enabled", True)
        setattr(tripod, f"axis{index}_label", text or frame.label(spec))
        setattr(tripod, f"axis{index}_dir", tuple(float(c) for c in frame.direction(spec)))
        _set_if_possible(tripod, f"axis{index}_color", color)
    return tripod


def _corner_is_clear(image_path, box=TRIPOD_BOX, threshold: float = 0.97) -> bool:
    """True when the lower left corner of a rendered image is empty.

    The triad is drawn there, so anything already in that corner would end up
    underneath it.  Checking the rendered pixels is the only reliable test:
    the silhouette of a cell depends on the camera, the aspect ratio and
    whichever atoms are hidden.  The threshold is loose enough to catch a cell
    wireframe that antialiasing has faded to pale grey in the probe.
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow ships with matplotlib
        return True
    with Image.open(image_path) as handle:
        pixels = np.asarray(handle.convert("L"), dtype=float) / 255.0
    height, width = pixels.shape
    corner = pixels[int(height * (1.0 - box[1])) :, : int(width * box[0])]
    return bool(corner.min() >= threshold)


def _margin_clearing_the_triad(render_probe, start=TRIPOD_MARGIN) -> float:
    """Smallest framing that leaves the triad corner empty.

    *render_probe* draws a small image at a given margin; a few of those cost
    far less than a figure that has to be redrawn by hand.
    """
    from .io import temporary_path

    margin = start
    while margin <= MAX_TRIPOD_MARGIN:
        with temporary_path(".png") as probe:
            render_probe(margin, probe)
            if _corner_is_clear(probe):
                return margin
        margin += 0.08
    return MAX_TRIPOD_MARGIN


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
    tripod: bool = False,
    tripod_axes=("rd", "td", "nd"),
    margin: float | None = None,
    info: dict | None = None,
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
    refusal = renderer_refusal()
    if refusal:
        raise RuntimeError(refusal)
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
        fitted_fov = viewport.fov

        scene_margin = margin or 1.0
        if tripod and margin is None:
            # Pull the camera back until the triad's corner is empty, measuring
            # small probe renders rather than guessing a fixed number: how much
            # room is needed depends on the camera, the aspect and what is hidden.
            def probe(candidate, path):
                viewport.fov = fitted_fov * candidate
                probe_width = 480
                viewport.render_image(
                    filename=path,
                    size=(probe_width, max(1, int(probe_width * size[1] / size[0]))),
                    renderer=_make_renderer("opengl"),
                )

            try:
                scene_margin = _margin_clearing_the_triad(probe)
            except Exception:
                scene_margin = TRIPOD_MARGIN

        viewport.fov = fitted_fov * float(scene_margin)
        if info is not None:
            info["margin"] = float(scene_margin)
        if tripod:
            specs, names = tripod_axes, None
            if len(tripod_axes) == 2 and isinstance(tripod_axes[0], (list, tuple)):
                specs, names = tripod_axes
            viewport.overlays.append(_tripod_overlay(result.frame, specs, names=names))
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

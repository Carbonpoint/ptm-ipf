"""Server-side 3D rendering of a cached :class:`~ptmipf.analysis.IPFResult`.

The interactive viewer re-renders on every orbit, zoom or slice change, so the
image must come from the cached result rather than from a fresh OVITO import:
a ``StaticSource`` pipeline is built directly from the stored positions and
colours, which takes milliseconds where re-running PTM would take seconds.

The camera is placed explicitly (orthographic, position, direction, up and
field of view all computed here) instead of using ``zoom_all``.  That makes
the projection reproducible in NumPy, which is what allows exact atom picking:
a click in the rendered image is mapped back to the nearest visible atom with
the same maths the renderer used.
"""

from __future__ import annotations

import numpy as np

__all__ = ["camera_basis", "pick_atom", "render_scene", "visible_mask"]

#: Colour mixed into unselected atoms in highlight mode; keeping a trace of the
#: original hue makes the context readable without competing with the selection.
_DIM_COLOR = np.array([0.78, 0.78, 0.78])
_DIM_WEIGHT = 0.85


def camera_basis(azimuth_deg: float, elevation_deg: float):
    """Right-handed (direction, right, up) triad of the orbiting camera.

    Azimuth turns about the cell's ``z`` axis, elevation lifts the camera
    towards it; ``up`` is the world ``z`` axis projected perpendicular to the
    view direction, matching OVITO's own convention for upright cameras.
    """
    az = np.radians(azimuth_deg)
    el = np.radians(np.clip(elevation_deg, -89.9, 89.9))
    # Unit vector pointing from the camera towards the scene centre.
    direction = -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    z = np.array([0.0, 0.0, 1.0])
    up = z - np.dot(z, direction) * direction
    norm = np.linalg.norm(up)
    up = np.array([0.0, 1.0, 0.0]) if norm < 1e-6 else up / norm
    right = np.cross(direction, up)
    return direction, right / np.linalg.norm(right), up


def _scene_frame(result):
    """Centre and bounding radius used to place the camera."""
    center = 0.5 * (result.positions.min(axis=0) + result.positions.max(axis=0))
    radius = float(np.linalg.norm(result.positions - center, axis=1).max())
    return center, max(radius, 1.0)


def camera_frame(
    result,
    azimuth: float,
    elevation: float,
    zoom: float = 1.0,
    pan=(0.0, 0.0),
    origin=None,
    scene_radius: float | None = None,
):
    """Everything that places the camera, shared by the render and by picking.

    *origin* is the point the camera looks at, the middle of the atoms unless
    it is given; *pan* moves that point across the view in units of the
    visible height, so a pan means the same amount at any zoom; and
    *scene_radius* fixes the size the view is scaled to, which is what keeps a
    series of frames from jumping about as the configuration changes shape.

    Returns ``(center, radius, direction, right, up, fov)``.
    """
    auto_center, auto_radius = _scene_frame(result)
    center = auto_center if origin is None else np.asarray(origin, dtype=float).reshape(3)
    radius = auto_radius if scene_radius is None else max(float(scene_radius), 1e-3)
    direction, right, up = camera_basis(azimuth, elevation)
    fov = _fov(radius, zoom)
    pan_x, pan_y = (float(pan[0]), float(pan[1])) if pan is not None else (0.0, 0.0)
    if pan_x or pan_y:
        # The visible height is 2 * fov in world units.
        center = center + right * (pan_x * 2.0 * fov) + up * (pan_y * 2.0 * fov)
    return center, radius, direction, right, up, fov


def default_radius(result) -> float:
    """Particle radius from the number density, so atoms just about touch."""
    span = result.positions.max(axis=0) - result.positions.min(axis=0)
    volume = float(np.prod(np.maximum(span, 1e-6)))
    return 0.48 * (volume / max(result.n_atoms, 1)) ** (1.0 / 3.0)


def slice_bounds(result, normal) -> tuple[float, float]:
    """Range of atom projections along the slice normal, for the UI slider."""
    normal = np.asarray(normal, dtype=float)
    projected = result.positions @ normal
    return float(projected.min()), float(projected.max())


def slice_mask(result, normal, distance: float | None, width: float = 0.0) -> np.ndarray:
    """The atoms one slice keeps.

    A width of zero keeps everything up to the plane, which is the cell cut
    open; a width keeps a slab of that thickness centred on the plane, which
    is the atomistic equivalent of an EBSD section.
    """
    normal = np.asarray(normal, dtype=float)
    projected = result.positions @ normal
    if distance is None:
        distance = float(result.positions.mean(axis=0) @ normal)
    if width > 0:
        return np.abs(projected - distance) <= width / 2.0
    return projected <= distance


def slices_mask(result, slices, mode: str = "any") -> np.ndarray:
    """The atoms a set of slices keeps, combined.

    ``any`` is the union, which is how several sections of one configuration
    are shown together; ``all`` is the intersection, which is how crossed
    slabs cut out a bar or a block.
    """
    slices = list(slices or [])
    if not slices:
        return np.ones(result.n_atoms, dtype=bool)
    masks = [
        slice_mask(result, s["normal"], s.get("distance"), s.get("width", 0.0))
        for s in slices
    ]
    combined = masks[0].copy()
    for mask in masks[1:]:
        if mode == "all":
            combined &= mask
        else:
            combined |= mask
    return combined


def visible_mask(
    result,
    hide_other: bool = False,
    slice_normal=None,
    slice_distance: float | None = None,
    slice_width: float = 0.0,
    selection: np.ndarray | None = None,
    selection_mode: str = "highlight",
    slices=None,
    slice_mode: str = "any",
) -> np.ndarray:
    """Which atoms appear in the render; picking must use the same mask.

    A single slice can be given as *slice_normal* and its distance and width,
    or several as *slices*, each a dict of ``normal``, ``distance`` and
    ``width``, combined by *slice_mode*.
    """
    visible = np.ones(result.n_atoms, dtype=bool)
    if hide_other:
        visible &= result.structure_types != 0
    if slices:
        visible &= slices_mask(result, slices, slice_mode)
    elif slice_normal is not None:
        visible &= slice_mask(result, slice_normal, slice_distance, slice_width)
    if selection is not None and selection_mode == "only":
        visible &= selection
    return visible


#: The largest 3D view the interface offers, before the machine has its say.
MAX_VIEW_PX = 12000

#: Sizes at or below this are drawn without measuring anything: every renderer
#: manages them, and ordinary use never goes near the limit.
SAFE_VIEW_PX = 1024


def safe_view_size(width, height) -> tuple[int, int]:
    """*width* and *height* brought inside what this machine can actually draw.

    A texture larger than the graphics device allows is not an error that can
    be caught: macOS fails a Metal assertion and the process is gone.  The
    limit differs from machine to machine, so it is measured rather than
    guessed, and only when a size is asked for that could exceed it.
    """
    width = int(min(max(width, 32), MAX_VIEW_PX))
    height = int(min(max(height, 32), MAX_VIEW_PX))
    if max(width, height) <= SAFE_VIEW_PX:
        return width, height
    from ..renderlimit import max_view_px

    limit = max_view_px()
    return min(width, limit), min(height, limit)


def render_scene(
    result,
    filename,
    azimuth: float = -125.0,
    elevation: float = 20.0,
    zoom: float = 1.0,
    size=(900, 700),
    hide_other: bool = False,
    slice_normal=None,
    slice_distance: float | None = None,
    slice_width: float = 0.0,
    slices=None,
    slice_mode: str = "any",
    selection: np.ndarray | None = None,
    selection_mode: str = "highlight",
    pan=(0.0, 0.0),
    origin=None,
    scene_radius: float | None = None,
    transparent: bool = False,
    radius: float | None = None,
    tripod: bool = False,
    tripod_axes=None,
    tripod_labels=None,
    tripod_size: float = 0.11,
    tripod_x: float = 0.07,
    tripod_y: float = 0.05,
    label: str | None = None,
    engine: str = "auto",
    warnings: list | None = None,
) -> str:
    """Render *result* to a PNG file and return the filename.

    Must run on the dedicated OVITO worker thread (see ``state.AppState``).
    *engine* is ``opengl``, ``tachyon`` or ``auto``, which tries OpenGL first
    and falls back; the diagnostics probe names one so it can report which
    renderer actually works here.

    The triad shows the sample axes RD, TD and ND unless *tripod_axes* names
    other directions (any spec the frame resolves), optionally captioned by
    *tripod_labels*; *tripod_size* and the offsets are fractions of the
    viewport.  *label* is stamped in the top left corner, which is how a frame
    of a trajectory series says which frame it is.  Anything that goes wrong
    with an overlay is appended to *warnings* and the image is drawn without
    it, because losing the view over a decoration helps nobody.
    """
    warnings = [] if warnings is None else warnings
    from ..render import renderer_refusal

    refusal = renderer_refusal()
    if refusal:
        raise RuntimeError(refusal)
    from ovito.data import DataCollection, Particles, SimulationCell
    from ovito.pipeline import Pipeline, StaticSource
    from ovito.vis import Viewport

    visible = visible_mask(
        result,
        hide_other=hide_other,
        slice_normal=slice_normal,
        slice_distance=slice_distance,
        slice_width=slice_width,
        slices=slices,
        slice_mode=slice_mode,
        selection=selection,
        selection_mode=selection_mode,
    )
    positions = result.positions[visible]
    colors = result.colors[visible].copy()
    if selection is not None and selection_mode == "highlight":
        dim = ~selection[visible]
        colors[dim] = (1.0 - _DIM_WEIGHT) * colors[dim] + _DIM_WEIGHT * _DIM_COLOR

    data = DataCollection()
    particles = Particles()
    particles.create_property("Position", data=positions)
    particles.create_property("Color", data=colors)
    data.objects.append(particles)
    particles.vis.radius = float(radius) if radius else default_radius(result)
    if result.cell is not None:
        cell = SimulationCell(pbc=(True, True, True))
        cell[...] = np.asarray(result.cell)
        data.objects.append(cell)

    center, radius, direction, _, up, fov = camera_frame(
        result, azimuth, elevation, zoom, pan=pan, origin=origin, scene_radius=scene_radius
    )
    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.add_to_scene()
    try:
        viewport = Viewport(type=Viewport.Type.Ortho)
        viewport.camera_dir = tuple(direction)
        viewport.camera_up = tuple(up)
        viewport.camera_pos = tuple(center - direction * radius * 3.0)
        # For an orthographic viewport, fov is half the vertical extent in
        # world units (verified against rendered pixel positions).
        viewport.fov = fov
        if tripod:
            from ..render import _tripod_overlay

            try:
                viewport.overlays.append(
                    _tripod_overlay(
                        result.frame,
                        tuple(tripod_axes) if tripod_axes else ("rd", "td", "nd"),
                        size=float(tripod_size),
                        offset_x=float(tripod_x),
                        offset_y=float(tripod_y),
                        names=tuple(tripod_labels) if tripod_labels else None,
                    )
                )
            except Exception as exc:
                # A triad this OVITO build will not draw must not cost the
                # user the whole 3D view; the caller reports what went wrong.
                warnings.append(f"the triad could not be drawn: {exc}")
        if label:
            try:
                viewport.overlays.append(_text_overlay(str(label)))
            except Exception as exc:
                warnings.append(f"the caption could not be drawn: {exc}")
        try:
            _render_with(viewport, filename, size, transparent, engine)
        except Exception as exc:
            if not viewport.overlays:
                raise
            # An overlay that builds cleanly can still break the renderer.
            warnings.append(f"drawn without the overlays: {exc}")
            del viewport.overlays[:]
            _render_with(viewport, filename, size, transparent, engine)
    finally:
        pipeline.remove_from_scene()
    return str(filename)


def _text_overlay(text: str):
    """A caption in the top left corner of the viewport."""
    from ovito.qt_compat import QtCore
    from ovito.vis import TextLabelOverlay

    overlay = TextLabelOverlay(text=text)
    # Qt's default family on a bare box measures with one font and draws
    # with another, which clips the end of the text; naming one avoids that.
    overlay.font_family = "DejaVu Sans"
    overlay.alignment = QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft
    overlay.font_size = 0.05
    overlay.offset_x = 0.02
    overlay.offset_y = -0.02
    overlay.text_color = (0.1, 0.1, 0.1)
    return overlay


def _fov(scene_radius: float, zoom: float) -> float:
    return scene_radius * 1.05 / max(float(zoom), 1e-3)


def _render_with(viewport, filename, size, transparent: bool, engine: str) -> None:
    """Draw the viewport, falling back from OpenGL to Tachyon.

    Both failures are kept and re-raised together: a machine with no GL stack
    and a machine whose OVITO build has no Tachyon fail in completely
    different ways, and the web UI can only say which when it has both
    messages.
    """
    from ovito import vis

    engines = {
        "opengl": vis.OpenGLRenderer,
        # Slower, but needs no GL stack at all.
        "tachyon": getattr(vis, "TachyonRenderer", None),
    }
    order = ("opengl", "tachyon") if engine == "auto" else (engine,)
    failures = []
    for name in order:
        factory = engines.get(name)
        if factory is None:
            failures.append(f"{name}: this OVITO build has no {name} renderer")
            continue
        try:
            viewport.render_image(
                filename=str(filename),
                size=tuple(size),
                renderer=factory(),
                alpha=transparent,
            )
            return
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    raise RuntimeError("no OVITO renderer worked here: " + "; ".join(failures))


def pick_atom(
    result,
    x: float,
    y: float,
    azimuth: float,
    elevation: float,
    zoom: float,
    size,
    tolerance_px: float = 12.0,
    pan=(0.0, 0.0),
    origin=None,
    scene_radius: float | None = None,
    **mask_kwargs,
) -> int | None:
    """Index of the atom under pixel ``(x, y)`` of the rendered image.

    Reproduces the orthographic projection of :func:`render_scene`: among the
    visible atoms whose projection lies within *tolerance_px* of the click,
    the one nearest the camera wins, which matches what the eye sees.
    """
    # The caller passes the whole view option set, which also carries camera and
    # overlay settings; only the ones that decide visibility belong here.
    mask_keys = (
        "hide_other",
        "slice_normal",
        "slice_distance",
        "slice_width",
        "slices",
        "slice_mode",
        "selection",
        "selection_mode",
    )
    visible = visible_mask(result, **{k: v for k, v in mask_kwargs.items() if k in mask_keys})
    if not visible.any():
        return None
    indices = np.flatnonzero(visible)
    positions = result.positions[indices]

    center, radius, direction, right, up, fov = camera_frame(
        result, azimuth, elevation, zoom, pan=pan, origin=origin, scene_radius=scene_radius
    )
    width, height = size
    scale = (height / 2.0) / fov  # pixels per world unit

    camera_pos = center - direction * radius * 3.0
    q = positions - camera_pos
    px = width / 2.0 + (q @ right) * scale
    py = height / 2.0 - (q @ up) * scale
    depth = q @ direction

    r2 = (px - x) ** 2 + (py - y) ** 2
    near = r2 <= tolerance_px**2
    if not near.any():
        return None
    candidates = np.flatnonzero(near)
    best = candidates[np.argmin(depth[candidates])]
    return int(indices[best])

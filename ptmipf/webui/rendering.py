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


def visible_mask(
    result,
    hide_other: bool = False,
    slice_normal=None,
    slice_distance: float | None = None,
    slice_width: float = 0.0,
    selection: np.ndarray | None = None,
    selection_mode: str = "highlight",
) -> np.ndarray:
    """Which atoms appear in the render; picking must use the same mask."""
    visible = np.ones(result.n_atoms, dtype=bool)
    if hide_other:
        visible &= result.structure_types != 0
    if slice_normal is not None:
        normal = np.asarray(slice_normal, dtype=float)
        projected = result.positions @ normal
        if slice_distance is None:
            slice_distance = float(result.positions.mean(axis=0) @ normal)
        if slice_width > 0:
            # A slab of the given thickness centred on the plane, which is the
            # atomistic equivalent of an EBSD section.
            visible &= np.abs(projected - slice_distance) <= slice_width / 2.0
        else:
            visible &= projected <= slice_distance
    if selection is not None and selection_mode == "only":
        visible &= selection
    return visible


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
    selection: np.ndarray | None = None,
    selection_mode: str = "highlight",
    transparent: bool = False,
    radius: float | None = None,
) -> str:
    """Render *result* to a PNG file and return the filename.

    Must run on the dedicated OVITO worker thread (see ``state.AppState``).
    """
    from ovito.data import DataCollection, Particles, SimulationCell
    from ovito.pipeline import Pipeline, StaticSource
    from ovito.vis import Viewport

    visible = visible_mask(
        result,
        hide_other=hide_other,
        slice_normal=slice_normal,
        slice_distance=slice_distance,
        slice_width=slice_width,
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

    center, scene_radius = _scene_frame(result)
    direction, _, up = camera_basis(azimuth, elevation)
    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.add_to_scene()
    try:
        viewport = Viewport(type=Viewport.Type.Ortho)
        viewport.camera_dir = tuple(direction)
        viewport.camera_up = tuple(up)
        viewport.camera_pos = tuple(center - direction * scene_radius * 3.0)
        # For an orthographic viewport, fov is half the vertical extent in
        # world units (verified against rendered pixel positions).
        viewport.fov = _fov(scene_radius, zoom)
        try:
            engine = _opengl_renderer()
            viewport.render_image(
                filename=str(filename), size=tuple(size), renderer=engine, alpha=transparent
            )
        except Exception:
            # Tachyon is slower but needs no GL stack at all.
            from ovito.vis import TachyonRenderer

            viewport.render_image(
                filename=str(filename),
                size=tuple(size),
                renderer=TachyonRenderer(),
                alpha=transparent,
            )
    finally:
        pipeline.remove_from_scene()
    return str(filename)


def _fov(scene_radius: float, zoom: float) -> float:
    return scene_radius * 1.05 / max(float(zoom), 1e-3)


def _opengl_renderer():
    from ovito.vis import OpenGLRenderer

    return OpenGLRenderer()


def pick_atom(
    result,
    x: float,
    y: float,
    azimuth: float,
    elevation: float,
    zoom: float,
    size,
    tolerance_px: float = 12.0,
    **mask_kwargs,
) -> int | None:
    """Index of the atom under pixel ``(x, y)`` of the rendered image.

    Reproduces the orthographic projection of :func:`render_scene`: among the
    visible atoms whose projection lies within *tolerance_px* of the click,
    the one nearest the camera wins, which matches what the eye sees.
    """
    visible = visible_mask(result, **mask_kwargs)
    if not visible.any():
        return None
    indices = np.flatnonzero(visible)
    positions = result.positions[indices]

    center, scene_radius = _scene_frame(result)
    direction, right, up = camera_basis(azimuth, elevation)
    fov = _fov(scene_radius, zoom)
    width, height = size
    scale = (height / 2.0) / fov  # pixels per world unit

    camera_pos = center - direction * scene_radius * 3.0
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

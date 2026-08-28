"""Flat, EBSD-style orientation maps.

A rendering of spheres is a picture of a simulation; an EBSD map is a picture of
a microstructure.  This module turns a section through a configuration into the
latter: a regular pixel grid coloured by local orientation, with grain
boundaries drawn as lines and unindexed points left black, which is what an
orientation map from a diffraction experiment looks like.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import SampleFrame
from .select import pairwise_misorientation_angles
from .structures import get_structure
from .symmetry import get_laue_group

__all__ = ["FlatMap", "flat_ipf_map", "save_flat_map"]


@dataclass
class FlatMap:
    """A rasterised orientation map."""

    rgb: np.ndarray  #: (rows, columns, 3) image, origin at the lower left
    extent: tuple[float, float, float, float]  #: (left, right, bottom, top) in angstrom
    pixel_size: float
    horizontal_label: str
    vertical_label: str
    view_label: str
    slab_width: float
    slab_center: float
    n_atoms: int
    boundary_fraction: float
    n_grains: int = 0
    labels: np.ndarray | None = None  #: grain label per pixel, -1 outside
    #: Rows are the in-plane horizontal and vertical axes and the view normal,
    #: in cell coordinates, so crystal directions can be projected onto the map.
    basis: np.ndarray | None = None
    #: Crystal-to-sample rotation per pixel's nearest atom, shape (rows, cols, 3, 3),
    #: only kept when grains were segmented; None otherwise.
    orientation_index: np.ndarray | None = None
    rotations: np.ndarray | None = None
    laue: str | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.rgb.shape[:2]

    @property
    def width_angstrom(self) -> float:
        return self.extent[1] - self.extent[0]

    @property
    def height_angstrom(self) -> float:
        return self.extent[3] - self.extent[2]


def _plane_basis(frame: SampleFrame, view, normal: np.ndarray):
    """In-plane axes for a view down *normal*, preferring the sample axes.

    The horizontal axis is the sample axis most perpendicular to the view, so a
    section seen down ND is laid out with RD across the page, as it would be
    printed in a paper.
    """
    candidates = [(name, frame.direction(name)) for name in ("rd", "td", "nd")]
    candidates = [c for c in candidates if abs(np.dot(c[1], normal)) < 0.99]
    if not candidates:  # pragma: no cover - only if the frame is degenerate
        candidates = [("x", np.array([1.0, 0.0, 0.0]))]
    name, vector = min(candidates, key=lambda c: abs(np.dot(c[1], normal)))
    horizontal = vector - np.dot(vector, normal) * normal
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(normal, horizontal)
    return horizontal, vertical, frame.label(name)


def _label_for(frame: SampleFrame, vector: np.ndarray) -> str:
    """Name *vector* if it is one of the frame's axes, else write it out."""
    for name, axis in frame.axes.items():
        if np.allclose(axis, vector, atol=1e-6):
            return name.upper()
        if np.allclose(-axis, vector, atol=1e-6):
            return "-" + name.upper()
    return "[" + " ".join(f"{c:.2f}" for c in vector) + "]"


def _nearest_atom(points: np.ndarray, grid: np.ndarray):
    """Index of and distance to the nearest atom for every grid point."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:  # pragma: no cover - exercised only without SciPy
        raise ImportError(
            "flat orientation maps need SciPy; install it with 'pip install scipy'"
        ) from None
    distance, index = cKDTree(points).query(grid, k=1)
    return index, distance


def flat_ipf_map(
    result,
    view="z",
    slab_width: float = 10.0,
    slab_center: float | None = None,
    pixel_size: float = 0.5,
    boundary_angle: float = 5.0,
    structure: str | None = None,
    segment: bool = True,
    min_grain_pixels: int = 24,
    smooth: int = 1,
    fill_unindexed: bool = True,
    unindexed_color=(0.0, 0.0, 0.0),
    boundary_color=(0.0, 0.0, 0.0),
    background=(1.0, 1.0, 1.0),
    max_distance: float | None = None,
) -> FlatMap:
    """Rasterise a section through *result* into a flat orientation map.

    Parameters
    ----------
    result
        An :class:`~ptmipf.analysis.IPFResult`, already coloured along the
        direction the map should show.
    view
        Normal of the section: an axis, a named sample axis or a vector.
    slab_width
        Thickness of the section in angstrom.  A few atomic layers is enough;
        a thick slab superimposes grains that lie behind one another.
    slab_center
        Position of the section along its normal.  Defaults to the middle of
        the configuration.
    pixel_size
        Edge length of a pixel in angstrom.  Roughly half an interatomic
        spacing gives a map that looks continuous without inventing detail.
    boundary_angle
        Neighbouring pixels whose orientations differ by more than this are a
        grain boundary and are drawn in *boundary_color*.  Set to 0 to draw no
        boundaries.
    segment
        Group the pixels into grains first and draw the boundary between
        grains, rather than between any two neighbouring pixels that differ.
        This is what gives clean boundary lines instead of a scribble wherever
        the orientation drifts.
    min_grain_pixels
        Grains smaller than this are absorbed into their surroundings, which
        removes the single-pixel specks that sectioning atoms produces.
    smooth
        Passes of a 3 by 3 median filter over the colours before the boundaries
        are drawn.  One pass removes the stray pixels left by atoms of a
        neighbouring grain poking into the section, without blurring anything,
        because a median filter does not average across an edge.
    fill_unindexed
        Give every pixel the colour of the nearest atom that does have an
        orientation, so grains fill the map and the boundaries between them are
        thin lines, as in an EBSD orientation map.  Set to False to paint the
        disordered grain boundary atoms in *unindexed_color* instead, which
        shows how wide they really are.
    structure
        Structure whose symmetry defines the misorientation.  Defaults to the
        colourable structure with the most atoms in the section.
    unindexed_color
        Colour of pixels whose nearest atom has no orientation, i.e. the
        equivalent of an unindexed EBSD point.  Black by default, as in EBSD.
    max_distance
        Pixels further than this from any atom are background.  Defaults to
        twice the mean atomic spacing in the section.

    Returns
    -------
    FlatMap
    """
    frame = result.frame
    normal = frame.direction(view) if isinstance(view, str) else np.asarray(view, dtype=float)
    normal = normal / np.linalg.norm(normal)
    horizontal, vertical, _ = _plane_basis(frame, view, normal)

    projected = result.positions @ normal
    if slab_center is None:
        slab_center = float(0.5 * (projected.min() + projected.max()))
    in_slab = np.abs(projected - slab_center) <= slab_width / 2.0
    if not in_slab.any():
        raise ValueError("the section contains no atoms; move it or make it thicker")

    positions = result.positions[in_slab]
    colors = result.colors[in_slab]
    structure_types = result.structure_types[in_slab]
    orientations = result.orientations[in_slab]

    # Which atoms may claim a pixel.  Filling from the indexed atoms only is
    # what makes the result look like an orientation map rather than a picture
    # of a simulation cell.
    usable = np.ones(len(positions), dtype=bool)
    if fill_unindexed:
        usable = structure_types != 0
        if not usable.any():
            raise ValueError(
                "no atom in the section has an orientation; widen the section, "
                "loosen --rmsd-cutoff, or pass fill_unindexed=False"
            )
    usable_index = np.flatnonzero(usable)

    u = positions @ horizontal
    v = positions @ vertical
    # Half a pixel of padding keeps the outermost atoms inside the image.
    left, right = u.min() - pixel_size, u.max() + pixel_size
    bottom, top = v.min() - pixel_size, v.max() + pixel_size
    columns = max(int(np.ceil((right - left) / pixel_size)), 1)
    rows = max(int(np.ceil((top - bottom) / pixel_size)), 1)

    gx = left + (np.arange(columns) + 0.5) * pixel_size
    gy = bottom + (np.arange(rows) + 0.5) * pixel_size
    mesh_x, mesh_y = np.meshgrid(gx, gy)
    grid = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])

    nearest, distance = _nearest_atom(np.column_stack([u[usable], v[usable]]), grid)
    index = usable_index[nearest]

    if max_distance is None:
        area = (right - left) * (top - bottom)
        spacing = np.sqrt(area / max(int(usable.sum()), 1))
        max_distance = 2.0 * spacing

    image = np.empty((rows * columns, 3))
    image[:] = np.asarray(background, dtype=float)
    inside = distance <= max_distance
    image[inside] = colors[index[inside]]

    unindexed = inside & (structure_types[index] == 0)
    image[unindexed] = np.asarray(unindexed_color, dtype=float)

    picture = image.reshape(rows, columns, 3)
    if smooth > 0:
        picture = _median_smooth(picture, passes=smooth)
    index_2d = index.reshape(rows, columns)
    inside_2d = inside.reshape(rows, columns)
    boundary = np.zeros((rows, columns), dtype=bool)
    labels = None
    n_grains = 0
    kept_rotations = None
    kept_laue = None

    if boundary_angle and boundary_angle > 0:
        from .analysis import quaternions_to_matrices

        dominant = _dominant_structure(result, structure_types, structure)
        laue = get_laue_group(get_structure(dominant).laue)
        rotations = quaternions_to_matrices(orientations)
        indexed = structure_types != 0
        kept_rotations = rotations
        kept_laue = laue.name
        if segment:
            labels = _segment_grains(
                index_2d, inside_2d, rotations, indexed, laue, boundary_angle
            )
            labels = _merge_small_grains(labels, picture, inside_2d, min_grain_pixels)
            n_grains = int(len(np.unique(labels[labels >= 0])))
            for shift, axis in ((-1, 0), (-1, 1)):
                neighbour = np.roll(labels, shift, axis=axis)
                differs = (labels >= 0) & (neighbour >= 0) & (labels != neighbour)
                if axis == 0:
                    differs[-1, :] = False
                else:
                    differs[:, -1] = False
                boundary |= differs
        else:
            boundary = _boundary_mask(
                result, index_2d, inside_2d, structure_types, orientations,
                structure, boundary_angle,
            )
        picture[boundary] = np.asarray(boundary_color, dtype=float)

    basis = np.stack([horizontal, vertical, normal])
    return FlatMap(
        rgb=np.clip(picture, 0.0, 1.0),
        extent=(left, right, bottom, top),
        pixel_size=pixel_size,
        horizontal_label=_label_for(frame, horizontal),
        vertical_label=_label_for(frame, vertical),
        view_label=frame.label(view) if isinstance(view, str) else _label_for(frame, normal),
        slab_width=slab_width,
        slab_center=slab_center,
        n_atoms=int(in_slab.sum()),
        boundary_fraction=float(boundary.mean()) if boundary.size else 0.0,
        n_grains=n_grains,
        labels=labels,
        basis=basis,
        orientation_index=index_2d if labels is not None else None,
        rotations=kept_rotations if labels is not None else None,
        laue=kept_laue if labels is not None else None,
    )


def _median_smooth(picture: np.ndarray, passes: int = 1) -> np.ndarray:
    """Median filter the colours, which removes specks but keeps edges sharp."""
    try:
        from scipy.ndimage import median_filter
    except ImportError:  # pragma: no cover - only without SciPy
        return picture
    for _ in range(passes):
        for channel in range(3):
            picture[:, :, channel] = median_filter(picture[:, :, channel], size=3, mode="nearest")
    return picture


def _dominant_structure(result, structure_types, structure):
    """The colourable structure with the most atoms in the section."""
    if structure is not None:
        return structure
    counts = {
        s.name: int((structure_types == result.type_codes[s.name]).sum())
        for s in result.structures
        if s.colorable
    }
    if not counts or max(counts.values()) == 0:
        raise ValueError("the section contains no atoms with an orientation")
    return max(counts, key=counts.get)


def _boundary_mask(
    result,
    index: np.ndarray,
    inside: np.ndarray,
    structure_types: np.ndarray,
    orientations: np.ndarray,
    structure: str | None,
    boundary_angle: float,
) -> np.ndarray:
    """Pixels across which the orientation jumps by more than the threshold."""
    from .analysis import quaternions_to_matrices

    if structure is None:
        counts = {
            s.name: int((structure_types == result.type_codes[s.name]).sum())
            for s in result.structures
            if s.colorable
        }
        if not counts or max(counts.values()) == 0:
            return np.zeros_like(inside)
        structure = max(counts, key=counts.get)
    laue = get_laue_group(get_structure(structure).laue)

    rotations = quaternions_to_matrices(orientations)
    indexed = structure_types != 0

    boundary = np.zeros_like(inside)
    for axis in (0, 1):
        a = index.take(np.arange(index.shape[axis] - 1), axis=axis)
        b = index.take(np.arange(1, index.shape[axis]), axis=axis)
        valid = (
            inside.take(np.arange(inside.shape[axis] - 1), axis=axis)
            & inside.take(np.arange(1, inside.shape[axis]), axis=axis)
            & indexed[a]
            & indexed[b]
        )
        if not valid.any():
            continue
        angles = pairwise_misorientation_angles(rotations[a[valid]], rotations[b[valid]], laue)
        jump = np.zeros(a.shape, dtype=bool)
        jump[valid] = angles > boundary_angle
        # Mark the pixel on each side, so the boundary is a visible line.
        boundary |= np.concatenate(
            [jump, np.zeros_like(jump.take([0], axis=axis))], axis=axis
        )
        boundary |= np.concatenate(
            [np.zeros_like(jump.take([0], axis=axis)), jump], axis=axis
        )
    return boundary


def _union_find_labels(shape, connected_pairs) -> np.ndarray:
    """Label connected pixels, given the pairs that belong to the same grain."""
    n = shape[0] * shape[1]
    parent = np.arange(n)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    for a, b in connected_pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels.reshape(shape)


def _segment_grains(index, inside, rotations, indexed, laue, tolerance):
    """Group pixels into grains, and return the labels and the neighbour pairs.

    Segmenting before drawing boundaries is what separates a clean orientation
    map from a scribble: the boundary is then the line between two grains, not
    every place where two neighbouring atoms happen to differ.
    """
    rows, columns = index.shape
    flat_ids = np.arange(rows * columns).reshape(rows, columns)
    connected = []
    for axis in (0, 1):
        take_a = np.arange(index.shape[axis] - 1)
        take_b = np.arange(1, index.shape[axis])
        a = index.take(take_a, axis=axis)
        b = index.take(take_b, axis=axis)
        valid = (
            inside.take(take_a, axis=axis)
            & inside.take(take_b, axis=axis)
            & indexed[a]
            & indexed[b]
        )
        if not valid.any():
            continue
        angles = pairwise_misorientation_angles(rotations[a[valid]], rotations[b[valid]], laue)
        same = angles <= tolerance
        ids_a = flat_ids.take(take_a, axis=axis)[valid][same]
        ids_b = flat_ids.take(take_b, axis=axis)[valid][same]
        connected.append(np.column_stack([ids_a, ids_b]))

    pairs = np.vstack(connected) if connected else np.zeros((0, 2), dtype=int)
    labels = _union_find_labels((rows, columns), pairs)
    labels[~inside] = -1
    return labels


def _merge_small_grains(labels, colors_image, inside, min_pixels: int):
    """Absorb specks into the grain around them, and flatten their colour.

    Single pixels of a stray orientation are an artefact of sectioning atoms,
    not a feature of the microstructure.
    """
    if min_pixels <= 1:
        return labels
    valid = labels >= 0
    counts = np.bincount(labels[valid], minlength=labels.max() + 1)
    small = np.flatnonzero(counts < min_pixels)
    if small.size == 0:
        return labels

    is_small = np.isin(labels, small) & valid
    rows, columns = labels.shape
    for _ in range(6):  # a few dilation passes absorb specks of any shape
        if not is_small.any():
            break
        host = np.full(labels.shape, -1)
        for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            neighbour = np.roll(labels, shift, axis=axis)
            neighbour_small = np.roll(is_small, shift, axis=axis)
            take = is_small & (host < 0) & (neighbour >= 0) & ~neighbour_small
            host[take] = neighbour[take]
            neighbour_color = np.roll(colors_image, shift, axis=axis)
            colors_image[take] = neighbour_color[take]
        absorbed = host >= 0
        labels[absorbed] = host[absorbed]
        is_small &= ~absorbed
    return labels


def save_flat_map(
    flat_map: FlatMap,
    filename,
    scale_bar: bool = True,
    axes_labels: bool = True,
    title: str | None = None,
    dpi: int = 200,
    wireframes=None,
    wireframe_linewidth: float = 1.4,
):
    """Draw *flat_map*, writing it to *filename* unless that is None.

    Returns the figure either way, so a caller that streams the image can take
    it from there.
    """
    import matplotlib.pyplot as plt

    height, width = flat_map.shape
    aspect = height / width
    figure_width = 6.0
    fig, ax = plt.subplots(figsize=(figure_width, figure_width * aspect), facecolor="white")
    fig.subplots_adjust(
        left=0.08 if axes_labels else 0.02,
        right=0.98,
        top=0.93 if title else 0.98,
        bottom=0.08 if axes_labels else 0.02,
    )
    ax.imshow(flat_map.rgb, origin="lower", extent=flat_map.extent, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    if title:
        ax.set_title(title, fontsize=12)

    if axes_labels:
        ax.set_xlabel(flat_map.horizontal_label, fontsize=11, labelpad=2)
        ax.set_ylabel(flat_map.vertical_label, fontsize=11, labelpad=2)

    if scale_bar:
        _draw_scale_bar(ax, flat_map)

    if wireframes:
        from .wireframe import draw_wireframes

        draw_wireframes(ax, wireframes, linewidth=wireframe_linewidth)

    if filename is not None:
        fig.savefig(filename, dpi=dpi)
    return fig


def _draw_scale_bar(ax, flat_map: FlatMap) -> None:
    """A scale bar of a round length, about a quarter of the image wide."""
    target = flat_map.width_angstrom / 4.0
    nice = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    length = min(nice, key=lambda n: abs(n - target))
    left, right, bottom, top = flat_map.extent
    margin = 0.05 * (right - left)
    x0 = right - margin - length
    y0 = bottom + 0.05 * (top - bottom)
    import matplotlib.patheffects as path_effects

    ax.plot([x0, x0 + length], [y0, y0], color="white", lw=5.0, solid_capstyle="butt", zorder=5)
    ax.plot([x0, x0 + length], [y0, y0], color="black", lw=2.5, solid_capstyle="butt", zorder=6)
    text = f"{length} Å" if length < 100 else f"{length / 10:g} nm"
    label = ax.text(
        x0 + length / 2,
        y0 + 0.035 * (top - bottom),
        text,
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
        zorder=7,
    )
    # An outline reads on any colour underneath, unlike a filled box.
    label.set_path_effects(
        [path_effects.Stroke(linewidth=2.5, foreground="white"), path_effects.Normal()]
    )

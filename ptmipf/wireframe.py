"""Unit cell wireframes drawn over the grains of a flat orientation map.

An inverse pole figure colour tells you one crystal direction; a wireframe of
the unit cell, rotated into the grain's orientation and projected onto the
section, shows the whole orientation at a glance, which is how EBSD software
annotates a map.  Each distinct orientation in the map gets one wireframe,
placed at the centroid of the pixels that share it and, by default, sized in
proportion to their area.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fill import average_orientations
from .select import misorientation_angles
from .symmetry import get_laue_group

__all__ = ["Wireframe", "unit_cell_edges", "grain_wireframes", "draw_wireframes"]

#: Ideal axial ratio of a close-packed hexagonal lattice.
IDEAL_C_OVER_A = np.sqrt(8.0 / 3.0)


@dataclass
class Wireframe:
    """One unit cell, ready to draw: its projected edges and where it sits."""

    label: int
    centre: tuple[float, float]  #: map coordinates, in angstrom
    area_pixels: int
    rotation: np.ndarray  #: crystal-to-sample, shape (3, 3)
    segments: np.ndarray  #: (n, 2, 2) projected edge endpoints, in angstrom
    depth: np.ndarray  #: (n,) mean depth of each edge along the view, for stacking
    color: tuple[float, float, float]


def unit_cell_edges(laue: str, c_over_a: float = IDEAL_C_OVER_A) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and edge index pairs of the unit cell, in the crystal frame.

    Both cells are centred on the origin and scaled to unit width, so a
    hexagonal prism and a cube of the same nominal size occupy similar areas.
    The hexagonal cell uses the PTM template frame, c along z and a1 along x.
    """
    group = get_laue_group(laue)
    if group.hexagonal_indices:
        angles = np.radians(np.arange(6) * 60.0)
        ring = np.column_stack([np.cos(angles), np.sin(angles)])
        half_c = 0.5 * c_over_a
        bottom = np.column_stack([ring, np.full(6, -half_c)])
        top = np.column_stack([ring, np.full(6, half_c)])
        vertices = np.vstack([bottom, top]) * 0.5
        edges = []
        for i in range(6):
            edges.append((i, (i + 1) % 6))
            edges.append((6 + i, 6 + (i + 1) % 6))
            edges.append((i, 6 + i))
        return vertices, np.array(edges)

    corners = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
    edges = [
        (i, j)
        for i in range(8)
        for j in range(i + 1, 8)
        if np.sum(np.abs(corners[i] - corners[j])) == 1.0
    ]
    return corners, np.array(edges)


def _distinct_orientations(flat, tolerance_deg: float):
    """Group the map's grains into orientation classes.

    Segmented grains that sit within *tolerance_deg* of one another are one
    orientation for the purposes of the overlay; a twin variant that appears as
    several islands gets one wireframe per island, because each island is a
    separate region of the map, but they share the same orientation class.
    """
    labels = flat.labels
    rotations = flat.rotations
    index = flat.orientation_index
    laue = get_laue_group(flat.laue)

    grains = []
    for label in np.unique(labels[labels >= 0]):
        mask = labels == label
        atom_indices = np.unique(index[mask])
        sample = rotations[atom_indices]
        if len(sample) > 400:
            rng = np.random.default_rng(int(label))
            sample = sample[rng.choice(len(sample), 400, replace=False)]
        mean = average_orientations(sample, laue, reference=sample[0])
        rows, columns = np.nonzero(mask)
        grains.append(
            {
                "label": int(label),
                "mask": mask,
                "rotation": mean,
                "area": int(mask.sum()),
                "centroid_px": (float(columns.mean()), float(rows.mean())),
            }
        )

    # Assign each grain to an orientation class by nearest existing class.
    classes: list[np.ndarray] = []
    for grain in grains:
        for k, reference in enumerate(classes):
            if misorientation_angles(grain["rotation"][None], reference, laue)[0] <= tolerance_deg:
                grain["class"] = k
                break
        else:
            grain["class"] = len(classes)
            classes.append(grain["rotation"])
    return grains, classes


def grain_wireframes(
    flat,
    tolerance_deg: float = 5.0,
    min_area_pixels: int = 200,
    size: float | None = None,
    scale: float = 1.0,
    color="invert",
    c_over_a: float = IDEAL_C_OVER_A,
    one_per_orientation: bool = False,
) -> list[Wireframe]:
    """Build the wireframes for a flat map.

    Parameters
    ----------
    flat
        A :class:`~ptmipf.flatmap.FlatMap` produced with ``segment=True``.
    tolerance_deg
        Grains whose orientations lie within this angle are the same orientation.
    min_area_pixels
        Grains smaller than this get no wireframe, so specks stay unannotated.
    size
        Edge length of the cell in angstrom.  When None, each wireframe is sized
        in proportion to the square root of its grain's area, which keeps the
        cell readable on a small grain and stops it dominating a large one.
    scale
        Multiplier on whichever size applies.
    color
        ``"invert"`` for the inverse of the grain's colour underneath, or an RGB
        triple or matplotlib colour name for one fixed colour.
    one_per_orientation
        Draw a single wireframe per orientation class, on its largest grain,
        rather than one on every grain.
    """
    if flat.labels is None or flat.rotations is None:
        raise ValueError("wireframes need a segmented map: use flat_ipf_map(..., segment=True)")

    grains, _ = _distinct_orientations(flat, tolerance_deg)
    grains = [g for g in grains if g["area"] >= min_area_pixels]
    if one_per_orientation:
        best = {}
        for grain in grains:
            if grain["class"] not in best or grain["area"] > best[grain["class"]]["area"]:
                best[grain["class"]] = grain
        grains = list(best.values())

    vertices, edges = unit_cell_edges(flat.laue, c_over_a)
    basis = flat.basis  # rows: horizontal, vertical, normal
    left, _, bottom, _ = flat.extent
    px = flat.pixel_size

    wireframes = []
    for grain in grains:
        if size is None:
            # A cell that spans about 40 percent of the grain's linear size.
            edge = 0.4 * np.sqrt(grain["area"]) * px
        else:
            edge = float(size)
        edge *= scale

        # Crystal frame -> sample frame -> map plane.
        placed = (grain["rotation"] @ (vertices * edge).T).T
        projected = placed @ basis.T  # columns: horizontal, vertical, depth
        cx = left + (grain["centroid_px"][0] + 0.5) * px
        cy = bottom + (grain["centroid_px"][1] + 0.5) * px
        points = projected[:, :2] + np.array([cx, cy])
        segments = np.stack([points[edges[:, 0]], points[edges[:, 1]]], axis=1)
        depth = 0.5 * (projected[edges[:, 0], 2] + projected[edges[:, 1], 2])

        if isinstance(color, str) and color == "invert":
            underneath = flat.rgb[grain["mask"]].mean(axis=0)
            rgb = tuple(float(1.0 - c) for c in underneath)
        else:
            from matplotlib.colors import to_rgb

            rgb = tuple(to_rgb(color))

        wireframes.append(
            Wireframe(
                label=grain["label"],
                centre=(cx, cy),
                area_pixels=grain["area"],
                rotation=grain["rotation"],
                segments=segments,
                depth=depth,
                color=rgb,
            )
        )
    return wireframes


def draw_wireframes(ax, wireframes, linewidth: float = 1.4, hidden_alpha: float = 0.35):
    """Draw wireframes on a matplotlib axes that shows the flat map.

    Edges on the far side of the cell are drawn fainter, which is what gives
    the projection its sense of depth and makes a tilted cell readable.
    """
    from matplotlib.collections import LineCollection

    for frame in wireframes:
        order = np.argsort(frame.depth)
        near = frame.depth >= np.median(frame.depth)
        colors = [
            (*frame.color, 1.0 if near[i] else hidden_alpha) for i in order
        ]
        ax.add_collection(
            LineCollection(
                frame.segments[order],
                colors=colors,
                linewidths=linewidth,
                capstyle="round",
                joinstyle="round",
                zorder=6,
            )
        )

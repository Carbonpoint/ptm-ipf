"""Run polyhedral template matching in OVITO and colour the result.

This module is the bridge between OVITO and the colour keys in
:mod:`ptmipf.colorkey`.  OVITO is imported lazily so that the crystallographic
parts of the package can be used (and tested) without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .colorkey import IPFColorKey
from .frames import SampleFrame
from .structures import DEFAULT_STRUCTURES, Structure, get_structure
from .symmetry import get_laue_group

__all__ = [
    "IPFResult",
    "analyse",
    "analyze",
    "ipf_color_modifier",
    "quaternions_to_matrices",
]

#: Colour given to atoms without a recognised lattice orientation.
DEFAULT_OTHER_COLOR = (0.35, 0.35, 0.35)


def quaternions_to_matrices(q: np.ndarray) -> np.ndarray:
    """Convert OVITO ``Orientation`` quaternions to rotation matrices.

    OVITO stores quaternions in Shoemake's ``(x, y, z, w)`` order, and the
    rotation they describe maps the **crystal** (template) frame onto the
    **sample** (simulation cell) frame.

    Parameters
    ----------
    q
        Quaternions, shape (n, 4).

    Returns
    -------
    numpy.ndarray
        Rotation matrices, shape (n, 3, 3).
    """
    q = np.atleast_2d(np.asarray(q, dtype=float))
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(norm > 0, q / norm, 0.0)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m = np.empty((len(q), 3, 3))
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y - z * w)
    m[:, 0, 2] = 2 * (x * z + y * w)
    m[:, 1, 0] = 2 * (x * y + z * w)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z - x * w)
    m[:, 2, 0] = 2 * (x * z - y * w)
    m[:, 2, 1] = 2 * (y * z + x * w)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m


@dataclass
class IPFResult:
    """Per-atom results of a PTM + IPF colouring run."""

    positions: np.ndarray
    structure_types: np.ndarray  #: OVITO PTM structure type code per atom
    orientations: np.ndarray  #: (n, 4) quaternions, (x, y, z, w)
    colors: np.ndarray  #: (n, 3) RGB in [0, 1]
    rmsd: np.ndarray
    particle_types: np.ndarray
    type_names: dict[int, str]
    direction: np.ndarray  #: projection direction in cell coordinates
    direction_label: str
    frame: SampleFrame
    structures: tuple[Structure, ...]
    type_codes: dict[str, int]
    cell: np.ndarray | None = None
    frame_index: int = 0
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return len(self.positions)

    def mask(self, structure: str) -> np.ndarray:
        """Boolean mask selecting atoms identified as *structure*."""
        return self.structure_types == self.type_codes[get_structure(structure).name]

    def rotations(self, structure: str | None = None) -> np.ndarray:
        """Crystal-to-sample rotation matrices, optionally for one structure."""
        q = self.orientations if structure is None else self.orientations[self.mask(structure)]
        return quaternions_to_matrices(q)

    def summary(self) -> str:
        components = " ".join(f"{c:.4f}" for c in self.direction)
        lines = [
            f"frame {self.frame_index}: {self.n_atoms} atoms",
            f"IPF projection direction: {self.direction_label} = [{components}]",
        ]
        for name, count in self.counts.items():
            percent = 100 * count / max(self.n_atoms, 1)
            lines.append(f"  {name:>14s}: {count:>10d}  ({percent:5.1f} %)")
        return "\n".join(lines)


def _ptm_modifier(structures, rmsd_cutoff: float):
    from ovito.modifiers import PolyhedralTemplateMatchingModifier

    modifier = PolyhedralTemplateMatchingModifier(
        output_orientation=True, rmsd_cutoff=rmsd_cutoff
    )
    type_codes = {
        s.name: int(getattr(PolyhedralTemplateMatchingModifier.Type, s.ptm_type))
        for s in structures
    }
    # Match on the numeric id: OVITO's display names ("Simple cubic") differ
    # from the names of the Type enum members ("SC").
    wanted = set(type_codes.values())
    for structure_type in modifier.structures:
        structure_type.enabled = int(structure_type.id) in wanted
    return modifier, type_codes


def _all_type_codes():
    from ovito.modifiers import PolyhedralTemplateMatchingModifier as P

    from .structures import STRUCTURES

    return {name: int(getattr(P.Type, s.ptm_type)) for name, s in STRUCTURES.items()}


def analyse(
    source,
    direction="z",
    structures=DEFAULT_STRUCTURES,
    frame: SampleFrame | None = None,
    frame_index: int = 0,
    rmsd_cutoff: float = 0.1,
    other_color=DEFAULT_OTHER_COLOR,
    only=None,
):
    """Identify structures with PTM and assign inverse pole figure colours.

    Parameters
    ----------
    source
        Path to an atomistic configuration file, or an existing OVITO pipeline.
    direction
        Sample direction the inverse pole figure is projected along: an axis
        (``+z``), a named sample axis (``rd``), or a vector (``1,1,0``).
    structures
        Structure names PTM should identify.
    frame
        Sample reference frame used to resolve named directions.
    frame_index
        Trajectory frame to analyse.
    rmsd_cutoff
        PTM RMSD cutoff; 0 disables the cutoff.
    other_color
        Colour for atoms with no recognised orientation.
    only
        Optional subset of *structures* to colour; the rest get *other_color*.

    Returns
    -------
    IPFResult
    """
    from ovito.io import import_file
    from ovito.pipeline import Pipeline

    frame = frame or SampleFrame()
    structure_objs = tuple(get_structure(s) for s in structures)
    colour_these = {get_structure(s).name for s in (only or structures)}

    pipeline = source if isinstance(source, Pipeline) else import_file(str(source))
    modifier, _ = _ptm_modifier(structure_objs, rmsd_cutoff)
    pipeline.modifiers.append(modifier)
    data = pipeline.compute(frame_index)

    type_codes = _all_type_codes()
    structure_types = np.asarray(data.particles["Structure Type"][...])
    orientations = np.asarray(data.particles["Orientation"][...])
    positions = np.asarray(data.particles.positions[...])
    rmsd = (
        np.asarray(data.particles["RMSD"][...])
        if "RMSD" in data.particles
        else np.zeros(len(positions))
    )
    if "Particle Type" in data.particles:
        particle_types = np.asarray(data.particles["Particle Type"][...])
        type_names = {
            int(t.id): (t.name or str(t.id))
            for t in data.particles["Particle Type"].types
        }
    else:
        particle_types = np.ones(len(positions), dtype=int)
        type_names = {1: "1"}

    d = frame.direction(direction)
    colors = np.tile(np.asarray(other_color, dtype=float), (len(positions), 1))
    counts = {}
    for s in structure_objs:
        selection = structure_types == type_codes[s.name]
        counts[s.name] = int(selection.sum())
        if not s.colorable or s.name not in colour_these or not selection.any():
            continue
        key = IPFColorKey(get_laue_group(s.laue))
        rotations = quaternions_to_matrices(orientations[selection])
        colors[selection] = key.orientation2color(rotations, d)
    counts["other"] = int(len(positions) - sum(counts.values()))

    return IPFResult(
        positions=positions,
        structure_types=structure_types,
        orientations=orientations,
        colors=colors,
        rmsd=rmsd,
        particle_types=particle_types,
        type_names=type_names,
        direction=d,
        direction_label=frame.label(direction),
        frame=frame,
        structures=structure_objs,
        type_codes=type_codes,
        cell=np.asarray(data.cell[...]) if data.cell is not None else None,
        frame_index=frame_index,
        counts=counts,
    )


#: American spelling, for convenience.
analyze = analyse


def ipf_color_modifier(
    direction="z",
    frame: SampleFrame | None = None,
    structures=DEFAULT_STRUCTURES,
    other_color=DEFAULT_OTHER_COLOR,
):
    """Build an OVITO modifier function that adds IPF colours to a pipeline.

    Insert it after a ``PolyhedralTemplateMatchingModifier`` that has
    ``output_orientation=True``::

        pipeline.modifiers.append(ipf_color_modifier(direction="rd"))

    The modifier sets the ``Color`` particle property, so the result can be
    rendered directly by OVITO.
    """
    frame = frame or SampleFrame()
    d = frame.direction(direction)
    structure_objs = tuple(get_structure(s) for s in structures)

    def modify(frame_index, data):  # signature required by OVITO
        type_codes = _all_type_codes()
        structure_types = np.asarray(data.particles["Structure Type"][...])
        orientations = np.asarray(data.particles["Orientation"][...])
        colors = data.particles_.create_property("Color")
        with colors:
            colors[...] = np.tile(np.asarray(other_color, dtype=float), (len(structure_types), 1))
            for s in structure_objs:
                selection = structure_types == type_codes[s.name]
                if not s.colorable or not selection.any():
                    continue
                key = IPFColorKey(get_laue_group(s.laue))
                rotations = quaternions_to_matrices(orientations[selection])
                colors[selection] = key.orientation2color(rotations, d)

    return modify

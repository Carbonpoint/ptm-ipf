"""Run polyhedral template matching in OVITO and colour the result.

This module is the bridge between OVITO and the colour keys in
:mod:`ptmipf.colorkey`.  OVITO is imported lazily so that the crystallographic
parts of the package can be used (and tested) without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .colorkey import IPFColorKey
from .frames import SampleFrame
from .structures import DEFAULT_STRUCTURES, Structure, get_structure
from .symmetry import get_laue_group

__all__ = [
    "QUATERNION_ORDERS",
    "IPFResult",
    "analyse",
    "analyse_orientations",
    "analyze",
    "list_columns",
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
    #: True for atoms whose orientation was interpolated rather than measured
    #: by PTM; see :func:`ptmipf.fill.fill_boundary_orientations`.
    interpolated: np.ndarray | None = None
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

    def subset(self, mask: np.ndarray) -> IPFResult:
        """Return a copy containing only the atoms selected by *mask*.

        The structure counts are recomputed, so the subset can be summarised,
        plotted, coloured and written out exactly like a full result.  Use it to
        restrict an IPF map or a pole figure to a selection, for example one
        grain, one phase or one region of the cell.
        """
        mask = np.asarray(mask)
        if mask.dtype != bool:
            indices = np.asarray(mask, dtype=int)
            mask = np.zeros(self.n_atoms, dtype=bool)
            mask[indices] = True
        if mask.shape != (self.n_atoms,):
            raise ValueError(
                f"mask has shape {mask.shape}, expected ({self.n_atoms},)"
            )

        counts = {}
        for structure in self.structures:
            code = self.type_codes[structure.name]
            counts[structure.name] = int((self.structure_types[mask] == code).sum())
        counts["other"] = int(mask.sum() - sum(counts.values()))

        return replace(
            self,
            positions=self.positions[mask],
            structure_types=self.structure_types[mask],
            orientations=self.orientations[mask],
            colors=self.colors[mask],
            rmsd=self.rmsd[mask],
            particle_types=self.particle_types[mask],
            counts=counts,
            interpolated=None if self.interpolated is None else self.interpolated[mask],
        )

    def recolor(self, direction, frame: SampleFrame | None = None, only=None) -> IPFResult:
        """Return a copy coloured along a different sample direction.

        Recolouring is cheap; re-running polyhedral template matching is not, so
        an interactive front end should reuse one result and call this.
        """
        from .colorkey import IPFColorKey

        frame = frame or self.frame
        d = frame.direction(direction)
        colour_these = {get_structure(s).name for s in (only or [s.name for s in self.structures])}

        other = self.colors[self.structure_types == self.type_codes.get("other", 0)]
        default = other[0] if len(other) else np.array(DEFAULT_OTHER_COLOR)
        colors = np.tile(np.asarray(default, dtype=float), (self.n_atoms, 1))
        for structure in self.structures:
            selection = self.structure_types == self.type_codes[structure.name]
            if not structure.colorable or structure.name not in colour_these or not selection.any():
                continue
            key = IPFColorKey(get_laue_group(structure.laue))
            rotations = quaternions_to_matrices(self.orientations[selection])
            colors[selection] = key.orientation2color(rotations, d)

        return replace(
            self,
            colors=colors,
            direction=d,
            direction_label=frame.label(direction),
            frame=frame,
        )

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
    progress=None,
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
    progress
        Optional ``callback(stage, n_atoms=None)`` invoked as each stage begins.
        OVITO's ``compute`` reports nothing while it runs, so a caller that
        wants a progress bar gets stage boundaries here and has to interpolate
        between them; the atom count arrives with the second stage, which is
        what makes an estimate of the third possible at all.

    Returns
    -------
    IPFResult
    """
    from ovito.io import import_file
    from ovito.pipeline import Pipeline

    frame = frame or SampleFrame()
    structure_objs = tuple(get_structure(s) for s in structures)
    colour_these = {get_structure(s).name for s in (only or structures)}

    report = progress or (lambda *args, **kwargs: None)
    report("reading the configuration")
    pipeline = source if isinstance(source, Pipeline) else import_file(str(source))
    # Evaluating the source alone reads the file and nothing else, which both
    # separates the read from the matching and yields the atom count that the
    # remaining stages are estimated from.  OVITO caches it, so the full
    # evaluation below does not read the file twice.
    source_data = pipeline.source.compute(frame_index)
    n_atoms = int(source_data.particles.count)

    report("polyhedral template matching", n_atoms=n_atoms)
    modifier, _ = _ptm_modifier(structure_objs, rmsd_cutoff)
    pipeline.modifiers.append(modifier)
    data = pipeline.compute(frame_index)

    report("colouring the orientations", n_atoms=n_atoms)
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


#: Quaternion component orders this can read.  OVITO writes (x, y, z, w).
QUATERNION_ORDERS = {"xyzw": (0, 1, 2, 3), "wxyz": (1, 2, 3, 0)}


def list_columns(source, frame_index: int = 0) -> dict:
    """The per-atom properties a file carries, without running anything on it.

    Used by the interface to offer a column mapping for a file whose
    orientations were computed elsewhere, so the mapping is chosen from what is
    actually in the file rather than typed from memory.
    """
    from ovito.io import import_file
    from ovito.pipeline import Pipeline

    pipeline = source if isinstance(source, Pipeline) else import_file(str(source))
    # The source alone, with no modifiers: this is the file as written.
    data = pipeline.source.compute(frame_index)
    columns = []
    for name in data.particles.keys():
        values = data.particles[name]
        components = list(getattr(values, "component_names", None) or [])
        count = int(values.shape[1]) if values.ndim > 1 else 1
        columns.append({"name": name, "components": count, "component_names": components})
    return {
        "columns": sorted(columns, key=lambda c: c["name"].lower()),
        "n_atoms": int(data.particles.count),
        "n_frames": int(pipeline.source.num_frames),
    }


def _resolve_column(particles, name: str) -> np.ndarray:
    """A per-atom column by name, accepting ``Property.Component`` as well.

    OVITO regroups ``Orientation.X`` and its siblings back into one
    four-component ``Orientation`` property on import, so a mapping written in
    terms of the column names a file actually shows would otherwise fail on the
    very files this is for.
    """
    if name in particles:
        return np.asarray(particles[name][...], dtype=float)
    prefix, _, component = name.rpartition(".")
    if prefix and prefix in particles:
        values = particles[prefix]
        labels = list(getattr(values, "component_names", None) or [])
        if component in labels:
            return np.asarray(values[...], dtype=float)[:, labels.index(component)]
    raise ValueError(f"the file has no column named {name!r}")


def _gather_quaternions(particles, spec, n_atoms: int) -> np.ndarray:
    """Pull an (n, 4) quaternion array out of whatever columns hold it."""
    columns = spec.get("quaternion")
    if isinstance(columns, str):
        columns = [columns]
    if not columns:
        raise ValueError("choose the column or columns that hold the orientation quaternion")

    if len(columns) == 1:
        name = columns[0]
        values = _resolve_column(particles, name)
        if values.ndim != 2 or values.shape[1] != 4:
            found = 1 if values.ndim == 1 else values.shape[1]
            raise ValueError(
                f"column {name!r} has {found} components; a quaternion needs four.  "
                "Pick four scalar columns instead."
            )
    elif len(columns) == 4:
        gathered = []
        for name in columns:
            component = _resolve_column(particles, name)
            if component.ndim != 1:
                raise ValueError(f"column {name!r} is not a single number per atom")
            gathered.append(component)
        values = np.stack(gathered, axis=1)
    else:
        raise ValueError("give either one four-component column or four scalar columns")

    if len(values) != n_atoms:
        raise ValueError("the orientation columns do not cover every atom")

    order = str(spec.get("order", "xyzw")).lower()
    if order not in QUATERNION_ORDERS:
        raise ValueError(f"order must be one of {sorted(QUATERNION_ORDERS)}, got {order!r}")
    values = values[:, list(QUATERNION_ORDERS[order])]
    if spec.get("conjugate"):
        # A file that stores the sample to crystal rotation is the transpose of
        # what the colour key wants, which for a unit quaternion is its conjugate.
        values = values * np.array([-1.0, -1.0, -1.0, 1.0])
    return values


def analyse_orientations(
    source,
    columns: dict,
    direction="z",
    structures=DEFAULT_STRUCTURES,
    frame: SampleFrame | None = None,
    frame_index: int = 0,
    other_color=DEFAULT_OTHER_COLOR,
    only=None,
):
    """Colour a file whose orientations were computed somewhere else.

    Skips polyhedral template matching entirely and reads the quaternions the
    file already carries, which is what a configuration exported from another
    OVITO session with PTM applied looks like.  Everything downstream, the
    colour key, the pole figures, the flat maps and the exports, is the same.

    Parameters
    ----------
    columns
        Which columns hold what.  ``quaternion`` is one four-component column
        name or a list of four scalar column names; ``order`` is ``xyzw``
        (OVITO's own, the default) or ``wxyz``; ``conjugate`` inverts the sense
        for a file that stores the sample to crystal rotation.  ``structure``
        names the phase when the file has no structure column;
        ``structure_type`` names that column when it has one, and its codes are
        read as OVITO's PTM codes.  ``rmsd`` names an RMSD column if there is one.

    Notes
    -----
    Nothing in the file says which convention its quaternions follow, and the
    two that matter differ by a transpose, which turns an IPF map into a
    plausible looking but wrong one.  The mapping is therefore explicit rather
    than guessed, and the interface shows the colour key so the result can be
    sanity checked against a known grain.
    """
    from ovito.io import import_file
    from ovito.pipeline import Pipeline

    frame = frame or SampleFrame()
    structure_objs = tuple(get_structure(s) for s in structures)
    colour_these = {get_structure(s).name for s in (only or structures)}

    pipeline = source if isinstance(source, Pipeline) else import_file(str(source))
    data = pipeline.source.compute(frame_index)
    particles = data.particles
    positions = np.asarray(particles.positions[...])
    n_atoms = len(positions)

    orientations = _gather_quaternions(particles, columns, n_atoms)
    norms = np.linalg.norm(orientations, axis=1)
    indexed = norms > 1e-6

    type_codes = _all_type_codes()
    name_of_column = columns.get("structure_type")
    if name_of_column:
        structure_types = _resolve_column(particles, name_of_column).astype(int).reshape(-1)
        if len(structure_types) != n_atoms:
            raise ValueError(f"column {name_of_column!r} does not cover every atom")
    else:
        # No structure column: every atom carrying an orientation is the one
        # phase the caller named, and the rest are "other".
        single = get_structure(columns.get("structure") or structure_objs[0].name)
        structure_objs = (single,)
        colour_these = {single.name}
        structure_types = np.where(indexed, type_codes[single.name], 0)
    structure_types = np.where(indexed, structure_types, 0)

    rmsd_column = columns.get("rmsd")
    if rmsd_column:
        rmsd = _resolve_column(particles, rmsd_column).reshape(-1)
    else:
        rmsd = np.zeros(n_atoms)

    if "Particle Type" in particles:
        particle_types = np.asarray(particles["Particle Type"][...])
        type_names = {
            int(t.id): (t.name or str(t.id)) for t in particles["Particle Type"].types
        }
    else:
        particle_types = np.ones(n_atoms, dtype=int)
        type_names = {1: "1"}

    d = frame.direction(direction)
    colors = np.tile(np.asarray(other_color, dtype=float), (n_atoms, 1))
    counts = {}
    for s in structure_objs:
        selection = structure_types == type_codes[s.name]
        counts[s.name] = int(selection.sum())
        if not s.colorable or s.name not in colour_these or not selection.any():
            continue
        key = IPFColorKey(get_laue_group(s.laue))
        colors[selection] = key.orientation2color(
            quaternions_to_matrices(orientations[selection]), d
        )
    counts["other"] = int(n_atoms - sum(counts.values()))

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

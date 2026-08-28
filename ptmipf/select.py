"""Select subsets of atoms by orientation, structure, type, position or fit.

Every function here returns a boolean mask over an
:class:`~ptmipf.analysis.IPFResult`, so criteria can be combined freely and the
result passed to :meth:`~ptmipf.analysis.IPFResult.subset` to recolour, plot or
export just those atoms.  This is what makes it possible to ask questions like
"what is the basal pole figure of the grains whose c axis lies within 15
degrees of ND?".
"""

from __future__ import annotations

import numpy as np

from .frames import SampleFrame, parse_vector
from .polefigure import IDEAL_C_OVER_A, miller_to_cartesian, symmetry_equivalents
from .structures import get_structure
from .symmetry import LaueGroup, get_laue_group

__all__ = [
    "combine",
    "invert",
    "misorientation_angles",
    "select_by_ipf_direction",
    "select_by_misorientation",
    "select_by_region",
    "select_by_rmsd",
    "select_by_structure",
    "select_by_type",
]


def _empty(result) -> np.ndarray:
    return np.zeros(result.n_atoms, dtype=bool)


def _laue_of(result, structure: str) -> LaueGroup:
    return get_laue_group(get_structure(structure).laue)


def _as_rotation(reference, result=None, structure=None) -> np.ndarray:
    """Coerce a reference orientation to a crystal-to-sample rotation matrix.

    Accepts a (3, 3) matrix, a (4,) quaternion in OVITO's (x, y, z, w) order, or
    an atom index into *result*.
    """
    from .analysis import quaternions_to_matrices

    if isinstance(reference, (int, np.integer)):
        if result is None:
            raise ValueError("an atom index needs a result to look the orientation up in")
        return quaternions_to_matrices(result.orientations[int(reference)][None])[0]

    reference = np.asarray(reference, dtype=float)
    if reference.shape == (4,):
        return quaternions_to_matrices(reference[None])[0]
    if reference.shape == (3, 3):
        return reference
    raise ValueError(
        "reference orientation must be a (3, 3) rotation matrix, a (4,) quaternion "
        f"or an atom index, got shape {reference.shape}"
    )


def misorientation_angles(
    rotations: np.ndarray, reference, laue, degrees: bool = True
) -> np.ndarray:
    """Symmetry-reduced misorientation angle between *rotations* and *reference*.

    Parameters
    ----------
    rotations
        Crystal-to-sample rotation matrices, shape (n, 3, 3).
    reference
        A single crystal-to-sample rotation matrix or quaternion.
    laue
        Laue group, or its name, whose symmetry the misorientation is reduced by.
    degrees
        Return degrees rather than radians.

    Returns
    -------
    numpy.ndarray
        The disorientation angle of each rotation, shape (n,).

    Notes
    -----
    The disorientation is the smallest angle of ``S1 M S2`` over all pairs of
    symmetry operators, where ``M`` is the misorientation.  Because the trace is
    cyclic, ``tr(S1 M S2) = tr(M S2 S1)``, and both crystals share one symmetry
    group, so minimising over the single product ``S2 S1`` gives the same answer
    with ``|G|`` terms instead of ``|G|^2``.
    """
    if not isinstance(laue, LaueGroup):
        laue = get_laue_group(laue)
    rotations = np.atleast_3d(np.asarray(rotations, dtype=float))
    reference = _as_rotation(reference)

    misorientation = np.einsum("ji,njk->nik", reference, rotations)
    # tr(M S) summed without forming the products.
    traces = np.einsum("nab,kba->nk", misorientation, laue.operators)
    cosine = np.clip((traces.max(axis=1) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cosine)
    return np.degrees(angle) if degrees else angle


def select_by_structure(result, structures) -> np.ndarray:
    """Select atoms identified as any of *structures*."""
    if isinstance(structures, str):
        structures = [structures]
    mask = _empty(result)
    for name in structures:
        mask |= result.mask(get_structure(name).name)
    return mask


def select_by_type(result, types) -> np.ndarray:
    """Select atoms by particle type, given as names (``"Mg"``) or numeric ids."""
    if isinstance(types, (str, int, np.integer)):
        types = [types]
    by_name = {name: identifier for identifier, name in result.type_names.items()}
    wanted = set()
    for entry in types:
        if isinstance(entry, (int, np.integer)):
            wanted.add(int(entry))
        elif entry in by_name:
            wanted.add(by_name[entry])
        elif entry.isdigit():
            wanted.add(int(entry))
        else:
            known = ", ".join(sorted(by_name)) or "none"
            raise KeyError(f"unknown particle type {entry!r}; the file contains: {known}")
    return np.isin(result.particle_types, list(wanted))


def select_by_rmsd(result, maximum: float | None = None, minimum: float | None = None):
    """Select atoms whose PTM template fit lies within the given bounds.

    A tighter RMSD keeps only the better-ordered atoms, which is a simple way to
    drop the distorted first shell around a grain boundary.
    """
    mask = np.ones(result.n_atoms, dtype=bool)
    if maximum is not None:
        mask &= result.rmsd <= maximum
    if minimum is not None:
        mask &= result.rmsd >= minimum
    return mask


def select_by_region(
    result,
    axis="z",
    minimum: float | None = None,
    maximum: float | None = None,
    frame: SampleFrame | None = None,
) -> np.ndarray:
    """Select a slab of atoms between two planes perpendicular to *axis*.

    *axis* may be a Cartesian axis, a named sample axis of the result's frame, or
    an explicit vector; the bounds are distances along it in cell units.
    """
    frame = frame or result.frame
    direction = frame.direction(axis) if isinstance(axis, str) else parse_vector(axis)
    projection = result.positions @ direction
    mask = np.ones(result.n_atoms, dtype=bool)
    if minimum is not None:
        mask &= projection >= minimum
    if maximum is not None:
        mask &= projection <= maximum
    return mask


def select_by_ipf_direction(
    result,
    crystal_direction,
    sample_direction,
    tolerance_deg: float = 15.0,
    structure: str = "hcp",
    c_over_a: float = IDEAL_C_OVER_A,
    plane: bool = True,
    frame: SampleFrame | None = None,
) -> np.ndarray:
    """Select atoms whose *crystal_direction* points near *sample_direction*.

    This is the "select the basal-oriented grains" query: with
    ``crystal_direction="0001"`` and ``sample_direction="nd"`` it selects every
    atom whose c axis lies within *tolerance_deg* of the sheet normal.

    Parameters
    ----------
    crystal_direction
        Miller or Miller-Bravais indices (``"0001"``, ``"10-10"``, ``"111"``) or
        a vector in the crystal frame.
    sample_direction
        An axis, a named sample axis or a vector, as elsewhere in the package.
    tolerance_deg
        Angular tolerance.  All symmetry equivalents of the crystal direction are
        considered, and so are both senses of it, as an axis has no sign.
    structure
        Only atoms of this structure can be selected; the rest are excluded.
    """
    frame = frame or result.frame
    laue = _laue_of(result, structure)
    structure_mask = result.mask(get_structure(structure).name)
    mask = _empty(result)
    if not structure_mask.any():
        return mask

    if isinstance(crystal_direction, str):
        crystal = miller_to_cartesian(crystal_direction, laue, c_over_a, plane=plane)
    else:
        crystal = parse_vector(crystal_direction)
    equivalents = symmetry_equivalents(crystal, laue)
    if isinstance(sample_direction, str):
        target = frame.direction(sample_direction)
    else:
        target = parse_vector(sample_direction)

    rotations = result.rotations(get_structure(structure).name)
    # Angle between every equivalent pole and the target direction, in the sample frame.
    poles = np.einsum("nij,mj->nmi", rotations, equivalents)
    cosine = np.clip(np.abs(poles @ target), 0.0, 1.0)
    angle = np.degrees(np.arccos(cosine.max(axis=1)))

    mask[structure_mask] = angle <= tolerance_deg
    return mask


def select_by_misorientation(
    result,
    reference,
    tolerance_deg: float = 10.0,
    structure: str = "hcp",
) -> np.ndarray:
    """Select atoms whose full orientation is close to a reference orientation.

    Unlike :func:`select_by_ipf_direction`, which fixes only one crystal
    direction, this constrains the whole orientation, so it isolates a single
    grain rather than a fibre of orientations.  *reference* may be a rotation
    matrix, a quaternion, or the index of an atom to take the orientation from.
    """
    laue = _laue_of(result, structure)
    name = get_structure(structure).name
    structure_mask = result.mask(name)
    mask = _empty(result)
    if not structure_mask.any():
        return mask

    reference_rotation = _as_rotation(reference, result=result, structure=structure)
    angles = misorientation_angles(result.rotations(name), reference_rotation, laue)
    mask[structure_mask] = angles <= tolerance_deg
    return mask


def combine(*masks: np.ndarray, mode: str = "and") -> np.ndarray:
    """Combine boolean masks with ``"and"`` or ``"or"``."""
    masks = [np.asarray(m, dtype=bool) for m in masks if m is not None]
    if not masks:
        raise ValueError("combine needs at least one mask")
    result = masks[0].copy()
    for mask in masks[1:]:
        if mode == "and":
            result &= mask
        elif mode == "or":
            result |= mask
        else:
            raise ValueError(f"mode must be 'and' or 'or', got {mode!r}")
    return result


def invert(mask: np.ndarray) -> np.ndarray:
    """Invert a selection."""
    return ~np.asarray(mask, dtype=bool)

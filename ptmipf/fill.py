"""Give the disordered boundary atoms an interpolated orientation.

Polyhedral template matching leaves grain boundary, surface and defect atoms
unindexed, which shows up as black or empty bands in an orientation map.  An
EBSD workflow would clean those points up by extrapolating from their
neighbours; this module does the same, averaging the orientations of the
indexed atoms within a chosen radius.

The result is an interpolation, not a measurement: an atom in the middle of a
high angle boundary has neighbours in two grains and its average lies in
neither.  Filled atoms are therefore marked in ``IPFResult.interpolated`` so
they can always be told apart from what PTM actually identified.
"""

from __future__ import annotations

import numpy as np

from .analysis import quaternions_to_matrices
from .colorkey import IPFColorKey
from .structures import get_structure
from .symmetry import LaueGroup, get_laue_group

__all__ = ["average_orientations", "fill_boundary_orientations"]


def _matrices_to_quaternions(matrices: np.ndarray) -> np.ndarray:
    """Rotation matrices to (x, y, z, w) quaternions, OVITO's convention."""
    m = np.asarray(matrices, dtype=float)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = np.empty((len(m), 4))
    # The branch with the largest denominator is the numerically stable one.
    big = trace > 0
    if big.any():
        s = np.sqrt(trace[big] + 1.0) * 2.0
        q[big, 3] = 0.25 * s
        q[big, 0] = (m[big, 2, 1] - m[big, 1, 2]) / s
        q[big, 1] = (m[big, 0, 2] - m[big, 2, 0]) / s
        q[big, 2] = (m[big, 1, 0] - m[big, 0, 1]) / s
    rest = np.flatnonzero(~big)
    for i in rest:
        diagonal = np.array([m[i, 0, 0], m[i, 1, 1], m[i, 2, 2]])
        k = int(np.argmax(diagonal))
        a, b = (k + 1) % 3, (k + 2) % 3
        s = np.sqrt(1.0 + m[i, k, k] - m[i, a, a] - m[i, b, b]) * 2.0
        q[i, 3] = (m[i, b, a] - m[i, a, b]) / s
        q[i, k] = 0.25 * s
        q[i, a] = (m[i, a, k] + m[i, k, a]) / s
        q[i, b] = (m[i, b, k] + m[i, k, b]) / s
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _align_to_reference(rotations: np.ndarray, references: np.ndarray, laue: LaueGroup):
    """Replace each rotation by the symmetry equivalent closest to its reference.

    Orientations cannot be averaged before this is done: two atoms in the same
    grain may be described by quaternions a symmetry operation apart, and the
    naive mean of those is meaningless.
    """
    misorientation = np.einsum("nji,njk->nik", references, rotations)
    traces = np.einsum("nab,kba->nk", misorientation, laue.operators)
    best = np.argmax(traces, axis=1)
    return np.einsum("nij,njk->nik", rotations, laue.operators[best])


def average_orientations(rotations: np.ndarray, laue, reference=None) -> np.ndarray:
    """Symmetry-aware mean of crystal-to-sample rotations, shape (3, 3).

    Each orientation is first moved to the symmetry equivalent nearest the
    reference, then the matrices are averaged and projected back onto a proper
    rotation.
    """
    if not isinstance(laue, LaueGroup):
        laue = get_laue_group(laue)
    rotations = np.asarray(rotations, dtype=float)
    if reference is None:
        reference = rotations[0]
    references = np.broadcast_to(np.asarray(reference, float), rotations.shape)
    aligned = _align_to_reference(rotations, references, laue)
    return _nearest_rotation(aligned.mean(axis=0)[None])[0]


def _nearest_rotation(matrices: np.ndarray) -> np.ndarray:
    """Closest proper rotation to each matrix, by polar decomposition."""
    u, _, vt = np.linalg.svd(matrices)
    rotation = u @ vt
    flip = np.linalg.det(rotation) < 0
    if flip.any():
        u[flip, :, -1] *= -1.0
        rotation[flip] = u[flip] @ vt[flip]
    return rotation


def _neighbour_lists(positions, sources, targets, radius, cell):
    """For every target, the source atoms within *radius*, honouring periodicity."""
    from scipy.spatial import cKDTree

    boxsize = None
    if cell is not None:
        matrix = np.asarray(cell, dtype=float)[:3, :3]
        off_diagonal = matrix - np.diag(np.diag(matrix))
        if np.abs(off_diagonal).max() < 1e-8 and np.all(np.diag(matrix) > 0):
            boxsize = np.diag(matrix)

    source_positions = positions[sources]
    target_positions = positions[targets]
    if boxsize is not None:
        origin = np.asarray(cell, dtype=float)[:3, 3] if np.shape(cell)[1] > 3 else 0.0
        source_positions = np.mod(source_positions - origin, boxsize)
        target_positions = np.mod(target_positions - origin, boxsize)
    tree = cKDTree(source_positions, boxsize=boxsize)
    return tree.query_ball_point(target_positions, radius), tree, target_positions


def fill_boundary_orientations(
    result,
    radius: float = 6.0,
    min_neighbours: int = 3,
    structure: str | None = None,
    recolor: bool = True,
):
    """Interpolate an orientation for every atom PTM left unindexed.

    Each unindexed atom takes the symmetry-aware average orientation of the
    indexed atoms within *radius*, and is coloured with the same inverse pole
    figure key as the rest.  This closes the gaps a grain boundary leaves in an
    orientation map, in the way EBSD software extrapolates unindexed points.

    Parameters
    ----------
    result
        An :class:`~ptmipf.analysis.IPFResult`.
    radius
        Neighbourhood radius in angstrom.  About two atomic shells, 6 A for
        magnesium, is enough to reach into the neighbouring grains.
    min_neighbours
        Atoms with fewer indexed neighbours than this are left unindexed, which
        keeps free surfaces and voids from being filled with noise.
    structure
        Structure whose symmetry is used.  Defaults to the colourable structure
        with the most atoms.
    recolor
        Colour the filled atoms with the result's own IPF key and direction.

    Returns
    -------
    IPFResult
        A copy in which the filled atoms carry an orientation, a structure type
        and a colour, and are flagged in ``interpolated``.

    Notes
    -----
    The average of orientations from two different grains is an orientation in
    neither of them, so atoms in the middle of a wide boundary get a colour that
    interpolates between the grains.  That is what makes the map continuous, and
    it is why the filled atoms stay flagged.
    """
    from dataclasses import replace

    counts = {
        s.name: int((result.structure_types == result.type_codes[s.name]).sum())
        for s in result.structures
        if s.colorable
    }
    if structure is None:
        if not counts or max(counts.values()) == 0:
            raise ValueError("no atoms with an orientation to interpolate from")
        structure = max(counts, key=counts.get)
    structure_object = get_structure(structure)
    laue = get_laue_group(structure_object.laue)
    code = result.type_codes[structure_object.name]

    indexed = np.flatnonzero(result.structure_types == code)
    unindexed = np.flatnonzero(result.structure_types == 0)
    if len(indexed) == 0 or len(unindexed) == 0:
        return replace(result, interpolated=np.zeros(result.n_atoms, dtype=bool))

    neighbours, tree, targets = _neighbour_lists(
        result.positions, indexed, unindexed, radius, result.cell
    )
    sizes = np.array([len(n) for n in neighbours])
    keep = sizes >= max(1, min_neighbours)
    if not keep.any():
        return replace(result, interpolated=np.zeros(result.n_atoms, dtype=bool))

    # Owner indices address the compacted arrays below, which hold one entry per
    # atom that is actually being filled.
    owners = np.repeat(np.arange(int(keep.sum())), sizes[keep])
    flat = np.concatenate([np.asarray(n, dtype=int) for n, k in zip(neighbours, keep) if k])
    rotations = quaternions_to_matrices(result.orientations[indexed])

    # Reference orientation per filled atom: its nearest indexed neighbour.
    _, nearest = tree.query(targets[keep], k=1)
    references = rotations[nearest]

    aligned = _align_to_reference(rotations[flat], references[owners], laue)
    accumulated = np.zeros((int(keep.sum()), 3, 3))
    np.add.at(accumulated, owners, aligned)
    averaged = _nearest_rotation(accumulated / sizes[keep][:, None, None])

    orientations = result.orientations.copy()
    structure_types = result.structure_types.copy()
    colors = result.colors.copy()
    filled = unindexed[keep]
    orientations[filled] = _matrices_to_quaternions(averaged)
    structure_types[filled] = code
    if recolor:
        key = IPFColorKey(laue)
        colors[filled] = key.orientation2color(averaged, result.direction)

    interpolated = np.zeros(result.n_atoms, dtype=bool)
    interpolated[filled] = True

    new_counts = dict(result.counts)
    new_counts[structure_object.name] = int((structure_types == code).sum())
    new_counts["other"] = int((structure_types == 0).sum())

    return replace(
        result,
        orientations=orientations,
        structure_types=structure_types,
        colors=colors,
        counts=new_counts,
        interpolated=interpolated,
    )

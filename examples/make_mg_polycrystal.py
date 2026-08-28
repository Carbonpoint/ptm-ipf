"""Build a synthetic textured magnesium polycrystal for trying out ptm-ipf.

Grains are Voronoi cells filled with hcp magnesium.  Their orientations are
drawn with a basal texture, i.e. the c axes cluster around ND, which is what a
rolled or extruded magnesium sheet looks like.

Usage::

    python examples/make_mg_polycrystal.py mg_polycrystal.xyz
"""

from __future__ import annotations

import sys

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write

A_MG, C_MG = 3.2094, 5.2108


def random_rotation(rng, basal_spread_deg=25.0):
    """A rotation whose c axis sits within roughly *basal_spread_deg* of z."""
    # Uniform azimuth, polar angle from a half-normal: a simple fibre texture.
    polar = np.abs(rng.normal(0.0, np.radians(basal_spread_deg)))
    azimuth = rng.uniform(0, 2 * np.pi)
    c_axis = np.array(
        [np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)]
    )
    # Complete to a frame with a random rotation about the c axis.
    helper = np.array([1.0, 0.0, 0.0])
    if abs(helper @ c_axis) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    a1 = np.cross(helper, c_axis)
    a1 /= np.linalg.norm(a1)
    a2 = np.cross(c_axis, a1)
    spin = rng.uniform(0, 2 * np.pi)
    a1r = np.cos(spin) * a1 + np.sin(spin) * a2
    a2r = np.cross(c_axis, a1r)
    return np.stack([a1r, a2r, c_axis])  # rows: crystal axes in sample coordinates


def make_polycrystal(box=120.0, n_grains=12, seed=1, basal_spread_deg=25.0):
    rng = np.random.default_rng(seed)
    seeds = rng.uniform(0, box, size=(n_grains, 3))
    rotations = [random_rotation(rng, basal_spread_deg) for _ in range(n_grains)]

    unit = bulk("Mg", "hcp", a=A_MG, c=C_MG)
    # Enough repeats to cover the box in any orientation.
    repeat = int(np.ceil(box * np.sqrt(3) / min(A_MG, C_MG))) + 2
    block = unit.repeat((repeat, repeat, repeat))
    block.positions -= block.positions.mean(axis=0)

    positions = []
    for index, (center, rotation) in enumerate(zip(seeds, rotations)):
        # rotation rows are the crystal axes in sample coordinates, so the
        # crystal-to-sample rotation matrix is its transpose.
        p = block.positions @ rotation + center
        keep = np.all((p > -1e-9) & (p < box), axis=1)
        p = p[keep]
        if len(p) == 0:
            continue
        distances = np.linalg.norm(p[:, None, :] - seeds[None, :, :], axis=2)
        positions.append(p[np.argmin(distances, axis=1) == index])

    positions = np.vstack(positions)
    atoms = Atoms("Mg" + str(len(positions)), positions=positions, cell=[box] * 3, pbc=True)

    # Drop atoms that ended up unphysically close across a grain boundary.
    from scipy.spatial import cKDTree

    tree = cKDTree(atoms.positions, boxsize=box)
    pairs = tree.query_pairs(0.8 * A_MG, output_type="ndarray")
    drop = set(pairs[:, 1].tolist())
    keep = np.array([i for i in range(len(atoms)) if i not in drop])
    return atoms[keep]


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "mg_polycrystal.xyz"
    atoms = make_polycrystal()
    write(path, atoms, format="extxyz")
    print(f"wrote {path}: {len(atoms)} atoms")

import numpy as np
import pytest


def rotation_matrix(axis, degrees):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    t = np.radians(degrees)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


@pytest.fixture
def write_crystal(tmp_path):
    """Write a rotated single crystal and return its path.

    The extended XYZ format is used because it stores an arbitrary cell
    orientation, which the LAMMPS formats do not.
    """
    ase_build = pytest.importorskip("ase.build")
    ase_io = pytest.importorskip("ase.io")

    def _write(kind="hcp", rotation=None, repeat=6, name="crystal.xyz"):
        if kind == "hcp":
            atoms = ase_build.bulk("Mg", "hcp", a=3.2094, c=5.2108)
        elif kind == "fcc":
            atoms = ase_build.bulk("Al", "fcc", a=4.05, cubic=True)
        elif kind == "bcc":
            atoms = ase_build.bulk("Fe", "bcc", a=2.87, cubic=True)
        elif kind == "sc":
            atoms = ase_build.bulk("Po", "sc", a=3.0)
        elif kind == "cubic_diamond":
            atoms = ase_build.bulk("Si", "diamond", a=5.43, cubic=True)
            repeat = min(repeat, 4)
        elif kind == "graphene":
            atoms = ase_build.graphene(formula="C2", a=2.46, size=(12, 12, 1), vacuum=8.0)
            repeat = 1
        else:  # pragma: no cover
            raise ValueError(kind)
        atoms = atoms.repeat((repeat, repeat, repeat))
        if rotation is not None:
            atoms.set_cell(np.array(atoms.cell) @ np.asarray(rotation).T, scale_atoms=True)
        path = tmp_path / name
        ase_io.write(str(path), atoms, format="extxyz")
        return str(path)

    return _write

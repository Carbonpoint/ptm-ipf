import os

import numpy as np
import pytest

_RENDERER_STATE = {}


def _renderer_available() -> bool:
    """Whether OVITO can produce an image here, probed once per session.

    GitHub's headless runners have no working GL at all, so every render test
    must skip there rather than fail. PTMIPF_NO_RENDER=1 forces the negative
    path, which is how the skip logic itself is tested.
    """
    if "ok" not in _RENDERER_STATE:
        if os.environ.get("PTMIPF_NO_RENDER"):
            _RENDERER_STATE["ok"] = False
            return False
        try:
            import tempfile

            from ptmipf.frames import SampleFrame
            from ptmipf.render import render_result
            from ptmipf.structures import get_structure

            class _Probe:
                positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
                colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
                structure_types = np.array([1, 1])
                n_atoms = 2
                frame = SampleFrame()
                structures = (get_structure("fcc"),)
                type_codes = {"fcc": 1}
                cell = None

            from ptmipf.render import renderer_refusal

            if renderer_refusal():
                raise RuntimeError("this combination must not be asked to draw")
            with tempfile.NamedTemporaryFile(suffix=".png") as handle:
                render_result(_Probe(), handle.name, size=(32, 32))
            _RENDERER_STATE["ok"] = True
        except Exception:
            _RENDERER_STATE["ok"] = False
    return _RENDERER_STATE["ok"]


@pytest.fixture(scope="session")
def renderer():
    """Tests that draw with OVITO take this fixture and skip without a renderer."""
    if not _renderer_available():
        pytest.skip("no OVITO renderer available in this environment")



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


@pytest.fixture
def write_bicrystal(tmp_path):
    """Two hcp grains meeting on a plane, for grain and boundary tests."""
    ase_build = pytest.importorskip("ase.build")
    ase_io = pytest.importorskip("ase.io")

    def _write(misorientation_deg=40.0, repeat=10, name="bicrystal.xyz"):
        from ase import Atoms

        # Build a block big enough that a rotated copy still covers the box.
        block = ase_build.bulk("Mg", "hcp", a=3.2094, c=5.2108).repeat(
            (2 * repeat, 2 * repeat, repeat)
        )
        positions = block.positions - block.positions.mean(axis=0)
        side = 0.45 * float(np.ptp(positions[:, 0]))
        height = 0.45 * float(np.ptp(positions[:, 2]))

        turn = rotation_matrix([0.0, 0.0, 1.0], misorientation_deg)
        rotated = positions @ turn.T

        def clip(points, x_low, x_high):
            keep = (
                (points[:, 0] >= x_low)
                & (points[:, 0] < x_high)
                & (np.abs(points[:, 1]) < side)
                & (np.abs(points[:, 2]) < height)
            )
            return points[keep]

        grain_a = clip(positions, -side, 0.0)
        grain_b = clip(rotated, 0.0, side)
        merged = np.vstack([grain_a, grain_b])
        merged -= merged.min(axis=0)

        span = merged.max(axis=0) + 2.0
        atoms = Atoms(
            "Mg" + str(len(merged)), positions=merged, cell=np.diag(span), pbc=False
        )
        path = tmp_path / name
        ase_io.write(str(path), atoms, format="extxyz")
        return str(path)

    return _write

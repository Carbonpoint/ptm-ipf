"""Boundary filling against real PTM output."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("ovito")
pytest.importorskip("scipy")

from ptmipf.analysis import analyse  # noqa: E402
from ptmipf.fill import fill_boundary_orientations  # noqa: E402
from ptmipf.select import misorientation_angles  # noqa: E402
from ptmipf.symmetry import get_laue_group  # noqa: E402


@pytest.fixture(scope="module")
def crystal(tmp_path_factory):
    ase_build = pytest.importorskip("ase.build")
    ase_io = pytest.importorskip("ase.io")
    atoms = ase_build.bulk("Mg", "hcp", a=3.2094, c=5.2108).repeat((8, 8, 8))
    path = tmp_path_factory.mktemp("fill") / "mg.xyz"
    ase_io.write(str(path), atoms, format="extxyz")
    return analyse(str(path), direction="z", structures=("hcp",))


def _blank_out(result, fraction=0.2, seed=0):
    """Pretend PTM failed on some atoms, so there is something to fill."""
    rng = np.random.default_rng(seed)
    types = result.structure_types.copy()
    orientations = result.orientations.copy()
    victims = rng.choice(result.n_atoms, int(fraction * result.n_atoms), replace=False)
    types[victims] = 0
    orientations[victims] = 0.0
    return replace(result, structure_types=types, orientations=orientations), victims


def test_filled_atoms_recover_the_surrounding_orientation(crystal):
    """In a single crystal the interpolated orientation must be the right one."""
    blanked, victims = _blank_out(crystal)
    filled = fill_boundary_orientations(blanked, radius=6.0, min_neighbours=3)

    assert filled.interpolated.sum() > 0.9 * len(victims)
    reference = crystal.rotations("hcp")[0]
    angles = misorientation_angles(
        filled.rotations()[filled.interpolated], reference, get_laue_group("hexagonal")
    )
    assert angles.max() < 1.0


def test_filled_atoms_get_the_right_colour(crystal):
    blanked, _ = _blank_out(crystal)
    filled = fill_boundary_orientations(blanked, radius=6.0)
    assert np.allclose(filled.colors[filled.interpolated], [1.0, 0.0, 0.0], atol=0.02)


def test_counts_and_flag_are_updated(crystal):
    blanked, victims = _blank_out(crystal)
    filled = fill_boundary_orientations(blanked, radius=6.0)
    assert filled.counts["other"] == int((filled.structure_types == 0).sum())
    # Counts are recomputed from the structure types, not carried over.
    assert filled.counts["hcp"] > int((blanked.structure_types != 0).sum())
    assert filled.counts["hcp"] == int((filled.structure_types != 0).sum())
    assert filled.interpolated.dtype == bool


def test_min_neighbours_can_refuse_to_fill(crystal):
    blanked, _ = _blank_out(crystal)
    filled = fill_boundary_orientations(blanked, radius=2.0, min_neighbours=500)
    assert filled.interpolated.sum() == 0


def test_nothing_to_fill_is_harmless(crystal):
    filled = fill_boundary_orientations(crystal, radius=6.0)
    assert filled.interpolated.sum() == 0
    assert filled.counts["hcp"] == crystal.counts["hcp"]


def test_subset_carries_the_flag(crystal):
    blanked, _ = _blank_out(crystal)
    filled = fill_boundary_orientations(blanked, radius=6.0)
    subset = filled.subset(np.arange(100))
    assert subset.interpolated is not None
    assert subset.interpolated.shape == (100,)
    assert np.array_equal(subset.interpolated, filled.interpolated[:100])

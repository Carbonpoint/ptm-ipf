"""The polycrystal builders.

The number these tests care about is the atom density at the grain boundaries.
A builder that trims the boundaries back looks fine in a picture and is wrong in
a stress strain curve, which is what happened to the iron runs of the showcase
campaign, so the density is asserted rather than eyeballed.
"""

import numpy as np
import pytest

from ptmipf.polycrystal import (
    Polycrystal,
    find_atomsk,
    random_rotations,
    voronoi_polycrystal,
    xyz_angles,
)


def _axis_rotation(axis, degrees):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    t = np.radians(degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def test_random_rotations_are_rotations():
    matrices = random_rotations(64, np.random.default_rng(0))
    assert matrices.shape == (64, 3, 3)
    for m in matrices:
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(m), 1.0)


def test_random_rotations_are_uniform_on_the_group():
    """Uniform on SO(3), not uniform in three Euler angles.

    Euler sampling crowds the poles; the test is that a fixed crystal direction
    lands uniformly on the sphere, so the mean cosine to any axis is near zero.
    """
    matrices = random_rotations(4000, np.random.default_rng(3))
    poles = matrices[:, :, 2]
    assert np.abs(poles.mean(axis=0)).max() < 0.05
    # Uniform on the sphere means the z component is uniform on [-1, 1].
    fractions = [np.mean(np.abs(poles[:, 2]) < t) for t in (0.25, 0.5, 0.75)]
    assert np.allclose(fractions, [0.25, 0.5, 0.75], atol=0.05)


@pytest.mark.parametrize("angles", [(0, 0, 0), (17, -32, 51), (90, 0, 0), (10, 90, 0)])
def test_xyz_angles_invert_the_atomsk_convention(angles):
    """R = Rz(tz) Ry(ty) Rx(tx), which is what atomsk's node file means."""
    tx, ty, tz = angles
    rotation = (
        _axis_rotation([0, 0, 1], tz)
        @ _axis_rotation([0, 1, 0], ty)
        @ _axis_rotation([1, 0, 0], tx)
    )
    got = xyz_angles(rotation)
    rebuilt = (
        _axis_rotation([0, 0, 1], got[2])
        @ _axis_rotation([0, 1, 0], got[1])
        @ _axis_rotation([1, 0, 0], got[0])
    )
    assert np.allclose(rebuilt, rotation, atol=1e-9)


@pytest.mark.parametrize("structure,a0", [("fcc", 3.615), ("bcc", 2.8553)])
def test_the_fallback_builder_keeps_the_boundaries_dense(structure, a0):
    crystal = voronoi_polycrystal(
        "X", box=40.0, n_grains=6, structure=structure, a0=a0, seed=2
    )
    # The failure this guards against showed up as 0.85 and below.
    assert crystal.density > 0.95, crystal.summary()
    assert crystal.n_atoms == len(crystal.positions)
    assert crystal.builder == "voronoi"


def test_the_fallback_builder_leaves_no_atoms_on_top_of_each_other():
    crystal = voronoi_polycrystal("Cu", box=40.0, n_grains=6, a0=3.615, seed=5)
    nearest = 3.615 * 2.0**-0.5
    assert 0.4 * nearest < crystal.min_separation < nearest


def test_the_cell_is_periodic_in_every_direction():
    """Atoms must not pile up or thin out at the faces of the box.

    A tessellation that is not periodic leaves a slab of open boundary running
    through the cell, which a deformation run would find immediately.
    """
    crystal = voronoi_polycrystal("Cu", box=48.0, n_grains=8, a0=3.615, seed=7)
    positions = crystal.positions
    assert positions.min() >= 0.0 and positions.max() < crystal.box
    edge = 4.0
    for axis in range(3):
        near_face = (positions[:, axis] < edge) | (positions[:, axis] > crystal.box - edge)
        middle = np.abs(positions[:, axis] - crystal.box / 2) < edge
        # Same slab thickness, so the counts should be within noise of each other.
        assert 0.8 < near_face.sum() / max(middle.sum(), 1) < 1.25


def test_the_box_is_snapped_to_whole_lattice_cells():
    crystal = voronoi_polycrystal("Cu", box=50.0, n_grains=1, a0=3.615, seed=0)
    assert abs(crystal.box / 3.615 - round(crystal.box / 3.615)) < 1e-9


def test_an_unrotated_single_grain_is_a_perfect_crystal():
    """The control case: no rotation, no boundary, nothing removed.

    A *rotated* single grain is not this.  It still meets its own periodic
    image at a seam, because a rotated lattice is not commensurate with the
    cube, so a few atoms are removed there.  That seam is a real grain boundary
    and every Voronoi builder produces it.
    """
    crystal = voronoi_polycrystal(
        "Cu", box=36.0, n_grains=1, a0=3.615, seed=0, rotations=np.eye(3)[None]
    )
    assert crystal.removed == 0
    assert crystal.density == pytest.approx(1.0, abs=1e-9)


def test_a_rotated_single_grain_has_only_its_own_seam():
    crystal = voronoi_polycrystal("Cu", box=36.0, n_grains=1, a0=3.615, seed=0)
    assert 0 < crystal.removed < 0.03 * crystal.ideal_count


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError, match="structure"):
        voronoi_polycrystal("Cu", structure="hcp")
    with pytest.raises(ValueError, match="at least one grain"):
        voronoi_polycrystal("Cu", n_grains=0)


def test_summary_names_the_builder():
    crystal = voronoi_polycrystal("Cu", box=30.0, n_grains=3, a0=3.615, seed=1)
    assert "voronoi" in crystal.summary()
    assert isinstance(crystal, Polycrystal)


def test_find_atomsk_accepts_an_explicit_path(tmp_path):
    fake = tmp_path / "atomsk"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    assert find_atomsk(str(fake)) == str(fake)
    assert find_atomsk(str(tmp_path / "nowhere")) == find_atomsk()


atomsk = pytest.mark.skipif(find_atomsk() is None, reason="atomsk is not installed")


@atomsk
def test_atomsk_builds_at_least_as_densely_as_the_fallback(tmp_path):
    from ptmipf.polycrystal import atomsk_polycrystal

    built = atomsk_polycrystal("Cu", 40.0, 6, tmp_path, a0=3.615, seed=2)
    assert built.builder == "atomsk"
    assert built.density > 0.95
    assert (tmp_path / built.files["data"]).is_file()

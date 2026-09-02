"""Tests for rigid rotations of a configuration and the slab mask they feed."""

import numpy as np
import pytest

from ptmipf.analysis import slab_mask
from ptmipf.transform import (
    parse_rotation,
    rotate_positions,
    rotate_quaternions,
    rotation_center,
    rotation_matrix,
)


def test_rotation_matrix_is_right_handed_and_orthonormal():
    r = rotation_matrix([0, 0, 1], 90)
    assert np.allclose(r @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(r @ r.T, np.eye(3))
    assert np.isclose(np.linalg.det(r), 1.0)
    # The axis need not be a unit vector.
    assert np.allclose(rotation_matrix([0, 0, 7], 90), r)


def test_rotation_matrix_refuses_a_zero_axis():
    with pytest.raises(ValueError):
        rotation_matrix([0, 0, 0], 10)


def test_parse_rotation_splits_axis_and_angle():
    assert parse_rotation("z:45") == ("z", 45.0)
    assert parse_rotation(" 1,1,0 : -90 ") == ("1,1,0", -90.0)
    assert parse_rotation("nd:12.5") == ("nd", 12.5)
    for bad in ("z", ":45", "z:lots", ""):
        with pytest.raises(ValueError):
            parse_rotation(bad)


def test_rotation_center_is_the_cell_centre_or_the_atoms_centre():
    cell = np.array([[10.0, 0, 0, 1.0], [0, 20.0, 0, 2.0], [0, 0, 30.0, 3.0]])
    assert np.allclose(rotation_center(cell), [6.0, 12.0, 18.0])
    positions = np.array([[0, 0, 0], [4, 2, 6]], dtype=float)
    assert np.allclose(rotation_center(None, positions), [2, 1, 3])
    assert np.allclose(rotation_center(None, None), 0)


def test_positions_turn_about_the_centre():
    r = rotation_matrix([0, 0, 1], 90)
    moved = rotate_positions([[2, 0, 0], [1, 1, 1]], r, [1, 1, 1])
    assert np.allclose(moved, [[2, 2, 0], [1, 1, 1]])


def test_quaternions_turn_with_the_matrix():
    """A crystal-to-sample quaternion rotated by R is R applied after it."""
    from ptmipf.analysis import quaternions_to_matrices

    q = np.array([[0.1, 0.2, 0.3, 0.9]])
    q /= np.linalg.norm(q)
    r = rotation_matrix([1, 2, 3], 37)
    turned = rotate_quaternions(q, r)
    assert np.allclose(np.linalg.norm(turned, axis=1), 1.0)
    before = quaternions_to_matrices(q)[0]
    after = quaternions_to_matrices(turned)[0]
    assert np.allclose(after, r @ before, atol=1e-9)


def test_rotate_result_turns_atoms_orientations_and_cell_together():
    from ptmipf.analysis import IPFResult
    from ptmipf.frames import SampleFrame
    from ptmipf.structures import get_structure
    from ptmipf.transform import rotate_result

    positions = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
    cell = np.array([[10.0, 0, 0, 0], [0, 10.0, 0, 0], [0, 0, 10.0, 0]])
    result = IPFResult(
        positions=positions,
        structure_types=np.array([1, 1]),
        orientations=np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
        colors=np.zeros((2, 3)),
        rmsd=np.zeros(2),
        particle_types=np.array([1, 1]),
        type_names={1: "Al"},
        direction=np.array([0.0, 0.0, 1.0]),
        direction_label="z",
        frame=SampleFrame(),
        structures=(get_structure("fcc"),),
        type_codes={"fcc": 1},
        cell=cell,
    )
    turned = rotate_result(result, rotation_matrix([0, 0, 1], 90))
    # The cell centre (5, 5, 5) is fixed; the atom at the origin swings round it.
    assert np.allclose(turned.positions[0], [10, 0, 0])
    assert np.allclose(turned.positions[1], [10, 10, 0])
    assert np.allclose(turned.cell[:, 0], [0, 10, 0])
    assert np.allclose(turned.cell[:, 3], [10, 0, 0])
    assert np.allclose(np.linalg.norm(turned.orientations, axis=1), 1.0)
    assert not np.allclose(turned.orientations, result.orientations)


def test_slab_mask_selects_a_band_or_a_half_space():
    z = np.arange(10, dtype=float)
    positions = np.column_stack([np.zeros(10), np.zeros(10), z])
    assert slab_mask(positions, [0, 0, 1], 2.0, 4.0).sum() == 3
    assert slab_mask(positions, [0, 0, 1], None, 4.0).sum() == 5
    assert slab_mask(positions, [0, 0, 1], 6.0, None).sum() == 4


def test_slab_mask_margin_keeps_periodic_neighbours():
    """Atoms across the periodic face of the slab count as its neighbours."""
    z = np.arange(10, dtype=float)
    positions = np.column_stack([np.zeros(10), np.zeros(10), z])
    cell = np.diag([10.0, 10.0, 10.0])
    cell = np.column_stack([cell, np.zeros(3)])
    plain = slab_mask(positions, [0, 0, 1], 0.0, 2.0)
    widened = slab_mask(positions, [0, 0, 1], 0.0, 2.0, margin=1.5, cell=cell)
    assert plain.sum() == 3
    # z = 3 by the margin, and z = 9 as the periodic image of z = -1.
    assert widened.sum() == 5
    assert widened[9] and widened[3]
    without_pbc = slab_mask(
        positions, [0, 0, 1], 0.0, 2.0, margin=1.5, cell=cell, pbc=(True, True, False)
    )
    assert without_pbc.sum() == 4


def test_slab_mask_can_be_defined_on_the_rotated_system():
    positions = np.array([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]])
    r = rotation_matrix([0, 0, 1], 90)
    turned = slab_mask(positions, [0, 1, 0], 4.0, 6.0, rotation=(r, np.zeros(3)))
    # After turning by 90 degrees about z, the x atom sits at y = 5.
    assert turned.tolist() == [True, False, False]

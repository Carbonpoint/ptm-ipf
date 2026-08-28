"""Interpolating orientations for the atoms PTM leaves unindexed."""

import numpy as np

from ptmipf.fill import _matrices_to_quaternions, average_orientations
from ptmipf.select import misorientation_angles
from ptmipf.symmetry import get_laue_group

from .conftest import rotation_matrix

HEX = get_laue_group("hexagonal")
CUBIC = get_laue_group("cubic")


def test_average_of_identical_orientations_is_that_orientation():
    r = rotation_matrix([1, 2, 3], 27.0)
    average = average_orientations(np.tile(r, (5, 1, 1)), HEX)
    # arccos near 1 loses half the significant digits, so 1e-4 degrees is exact.
    assert np.isclose(misorientation_angles(average[None], r, HEX)[0], 0.0, atol=1e-4)


def test_symmetry_equivalents_average_to_the_same_orientation():
    """Averaging without symmetry reduction would give nonsense here."""
    r = rotation_matrix([0.3, -0.7, 0.2], 33.0)
    equivalents = np.stack([r @ op for op in HEX.operators])
    average = average_orientations(equivalents, HEX, reference=r)
    assert np.isclose(misorientation_angles(average[None], r, HEX)[0], 0.0, atol=1e-6)


def test_average_lies_between_two_orientations():
    a = np.eye(3)
    b = rotation_matrix([0, 0, 1], 10.0)
    average = average_orientations(np.stack([a, b]), CUBIC, reference=a)
    to_a = misorientation_angles(average[None], a, CUBIC)[0]
    to_b = misorientation_angles(average[None], b, CUBIC)[0]
    assert np.isclose(to_a, 5.0, atol=0.1) and np.isclose(to_b, 5.0, atol=0.1)


def test_average_is_a_proper_rotation():
    rng = np.random.default_rng(17)
    q, _ = np.linalg.qr(rng.normal(size=(8, 3, 3)))
    rotations = q * np.linalg.det(q)[:, None, None]
    average = average_orientations(rotations, CUBIC)
    assert np.allclose(average @ average.T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(average), 1.0)


def test_quaternion_round_trip():
    from ptmipf.analysis import quaternions_to_matrices

    rng = np.random.default_rng(18)
    q, _ = np.linalg.qr(rng.normal(size=(50, 3, 3)))
    rotations = q * np.linalg.det(q)[:, None, None]
    round_tripped = quaternions_to_matrices(_matrices_to_quaternions(rotations))
    assert np.allclose(round_tripped, rotations, atol=1e-9)


def test_quaternion_conversion_handles_all_branches():
    """The four numerical branches must all be exercised and agree."""
    from ptmipf.analysis import quaternions_to_matrices

    rotations = np.stack(
        [
            np.eye(3),
            rotation_matrix([1, 0, 0], 180.0),
            rotation_matrix([0, 1, 0], 180.0),
            rotation_matrix([0, 0, 1], 180.0),
        ]
    )
    round_tripped = quaternions_to_matrices(_matrices_to_quaternions(rotations))
    assert np.allclose(round_tripped, rotations, atol=1e-9)

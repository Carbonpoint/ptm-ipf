"""Tests for orientation, structure, type and region selection."""

import numpy as np
import pytest

from ptmipf.select import (
    combine,
    invert,
    misorientation_angles,
)
from ptmipf.symmetry import get_laue_group

from .conftest import rotation_matrix

HEX = get_laue_group("hexagonal")
CUBIC = get_laue_group("cubic")


def test_identical_orientations_have_zero_misorientation():
    rotations = np.tile(np.eye(3), (5, 1, 1))
    assert np.allclose(misorientation_angles(rotations, np.eye(3), CUBIC), 0.0, atol=1e-6)


@pytest.mark.parametrize("angle", [3.0, 17.0, 41.0])
def test_small_rotation_gives_back_its_angle(angle):
    """Below the symmetry-folding limit the disorientation is the rotation angle."""
    rotation = rotation_matrix([1.0, 0.4, -0.2], angle)
    measured = misorientation_angles(rotation[None], np.eye(3), CUBIC)[0]
    assert np.isclose(measured, angle, atol=1e-6)


def test_symmetry_equivalent_orientations_are_not_misoriented():
    for laue, axis, angle in [(CUBIC, [0, 0, 1], 90.0), (HEX, [0, 0, 1], 60.0)]:
        rotation = rotation_matrix(axis, angle)
        assert np.isclose(misorientation_angles(rotation[None], np.eye(3), laue)[0], 0.0, atol=1e-6)


def test_misorientation_is_symmetric_in_its_arguments():
    a = rotation_matrix([1, 2, 3], 33.0)
    b = rotation_matrix([-2, 1, 0.5], 51.0)
    assert np.isclose(
        misorientation_angles(a[None], b, HEX)[0],
        misorientation_angles(b[None], a, HEX)[0],
        atol=1e-9,
    )


@pytest.mark.parametrize("laue,limit", [(CUBIC, 62.8), (HEX, 93.9)])
def test_disorientation_never_exceeds_the_known_maximum(laue, limit):
    """The largest disorientation of a Laue group is a textbook constant."""
    rng = np.random.default_rng(21)
    q, _ = np.linalg.qr(rng.normal(size=(2000, 3, 3)))
    rotations = q * np.linalg.det(q)[:, None, None]
    angles = misorientation_angles(rotations, np.eye(3), laue)
    assert angles.min() >= 0.0
    assert angles.max() <= limit + 1e-6


def test_misorientation_accepts_a_quaternion_reference():
    angle = np.radians(24.0)
    axis = np.array([0.0, 0.0, 1.0])
    quaternion = np.concatenate([axis * np.sin(angle / 2), [np.cos(angle / 2)]])
    rotation = rotation_matrix(axis, 24.0)
    assert np.isclose(misorientation_angles(rotation[None], quaternion, CUBIC)[0], 0.0, atol=1e-6)


def test_combine_and_invert():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    assert combine(a, b, mode="and").tolist() == [True, False, False, False]
    assert combine(a, b, mode="or").tolist() == [True, True, True, False]
    assert invert(a).tolist() == [False, False, True, True]
    with pytest.raises(ValueError):
        combine(a, b, mode="xor")
    with pytest.raises(ValueError):
        combine()

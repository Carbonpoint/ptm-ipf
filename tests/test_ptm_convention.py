"""Pin down the PTM conventions this package depends on.

These tests fail loudly if a future OVITO release changes the reference frame
of a template or the meaning of the orientation quaternion, which would
silently rotate every colour map this package produces.
"""

import numpy as np
import pytest

pytest.importorskip("ovito")

from ptmipf.analysis import analyse, quaternions_to_matrices

from .conftest import rotation_matrix

RED, GREEN, BLUE = np.eye(3)


def _rotations(path, structure):
    result = analyse(path, direction="z", structures=(structure,))
    mask = result.mask(structure)
    assert mask.sum() > 0.5 * result.n_atoms, "PTM did not identify the crystal"
    return result, quaternions_to_matrices(result.orientations[mask])


@pytest.mark.parametrize("kind", ["fcc", "bcc", "hcp"])
def test_aligned_crystal_has_the_identity_orientation(write_crystal, kind):
    """A crystal built on the Cartesian axes must match the template exactly."""
    _, rotations = _rotations(write_crystal(kind), kind)
    assert np.allclose(rotations, np.eye(3), atol=1e-4)


@pytest.mark.parametrize("kind", ["fcc", "bcc"])
def test_quaternion_is_the_crystal_to_sample_rotation(write_crystal, kind):
    """Rotating the crystal by R must multiply the stored rotation by R."""
    applied = rotation_matrix([1, 2, 3], 37.0)
    _, rotations = _rotations(write_crystal(kind, rotation=applied), kind)
    assert np.allclose(rotations, applied, atol=1e-4)


def test_hcp_template_has_c_along_z_and_a_along_x(write_crystal):
    """The hcp template frame, which fixes what the IPF colours mean."""
    _, rotations = _rotations(write_crystal("hcp"), "hcp")
    # Crystal axes expressed in the sample frame are the columns of R.
    assert np.allclose(rotations[:, :, 2], [0, 0, 1], atol=1e-4)  # c
    assert np.allclose(rotations[:, :, 0], [1, 0, 0], atol=1e-4)  # a1


@pytest.mark.parametrize(
    "rotation,expected,label",
    [
        (None, RED, "c parallel to z"),
        (rotation_matrix([0, 1, 0], -90), GREEN, "a parallel to z"),
        (
            rotation_matrix([0, 1, 0], -90) @ rotation_matrix([0, 0, 1], -30),
            BLUE,
            "m parallel to z",
        ),
    ],
)
def test_hcp_ipf_colors_of_known_orientations(write_crystal, rotation, expected, label):
    path = write_crystal("hcp", rotation=rotation)
    result = analyse(path, direction="z", structures=("hcp",))
    colors = result.colors[result.mask("hcp")]
    assert np.allclose(colors.mean(axis=0), expected, atol=0.02), label


def test_color_is_invariant_when_crystal_and_direction_rotate_together(write_crystal):
    """Rotating sample and reference direction alike must not change colours."""
    applied = rotation_matrix([2, -1, 3], 53.0)
    reference = analyse(write_crystal("hcp", name="a.xyz"), direction="z", structures=("hcp",))
    rotated = analyse(
        write_crystal("hcp", rotation=applied, name="b.xyz"),
        direction=applied @ np.array([0.0, 0.0, 1.0]),
        structures=("hcp",),
    )
    assert np.allclose(
        reference.colors[reference.mask("hcp")].mean(axis=0),
        rotated.colors[rotated.mask("hcp")].mean(axis=0),
        atol=0.02,
    )


@pytest.mark.parametrize("kind", ["sc", "cubic_diamond", "graphene"])
def test_less_common_structures_are_enabled_and_aligned(write_crystal, kind):
    """OVITO's display names differ from its Type enum names; enabling these
    structures must not depend on the display name."""
    result, rotations = _rotations(write_crystal(kind), kind)
    assert np.allclose(rotations, np.eye(3), atol=1e-3)
    # z is [001] for the cubic templates and [0001] for graphene: both red.
    assert np.allclose(result.colors[result.mask(kind)].mean(axis=0), RED, atol=0.02)


def test_fcc_ipf_colors_of_known_orientations(write_crystal):
    result = analyse(write_crystal("fcc"), direction="z", structures=("fcc",))
    assert np.allclose(result.colors[result.mask("fcc")].mean(axis=0), RED, atol=0.02)

    tilted = rotation_matrix([0, 1, 0], -45)  # [101] onto z
    result = analyse(
        write_crystal("fcc", rotation=tilted, name="c.xyz"),
        direction="z",
        structures=("fcc",),
    )
    assert np.allclose(result.colors[result.mask("fcc")].mean(axis=0), GREEN, atol=0.02)

import numpy as np
import pytest

from ptmipf.polefigure import (
    IDEAL_C_OVER_A,
    miller_to_cartesian,
    parse_miller,
    pole_directions,
    symmetry_equivalents,
)
from ptmipf.projections import (
    equal_area,
    inverse_stereographic,
    stereographic,
    upper_hemisphere,
)
from ptmipf.symmetry import get_laue_group

HEX = get_laue_group("hexagonal")
CUBIC = get_laue_group("cubic")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0001", (0, 0, 1)),
        ("10-10", (1, 0, 0)),
        ("2-1-10", (2, -1, 0)),
        ("111", (1, 1, 1)),
        ("(1 1 1)", (1, 1, 1)),
        ("[1,-1,0]", (1, -1, 0)),
        ("{10-11}", (1, 0, 1)),
    ],
)
def test_parse_miller(text, expected):
    assert parse_miller(text) == expected


def test_parse_miller_checks_the_bravais_index():
    with pytest.raises(ValueError):
        parse_miller("1010")  # h + k + i != 0


@pytest.mark.parametrize(
    "indices,n_equivalents",
    [("0001", 2), ("10-10", 6), ("2-1-10", 6), ("10-11", 12)],
)
def test_hexagonal_pole_multiplicity(indices, n_equivalents):
    v = miller_to_cartesian(indices, HEX, c_over_a=1.6236)
    assert len(symmetry_equivalents(v, HEX)) == n_equivalents


@pytest.mark.parametrize("indices,n_equivalents", [("100", 6), ("110", 12), ("111", 8)])
def test_cubic_pole_multiplicity(indices, n_equivalents):
    v = miller_to_cartesian(indices, CUBIC)
    assert len(symmetry_equivalents(v, CUBIC)) == n_equivalents


def test_basal_and_prismatic_normals():
    assert np.allclose(miller_to_cartesian("0001", HEX), [0, 0, 1])
    # The (10-10) plane normal lies 30 degrees from a1, independent of c/a.
    m = miller_to_cartesian("10-10", HEX)
    assert np.isclose(np.degrees(np.arctan2(m[1], m[0])), 30.0)
    assert np.isclose(m[2], 0.0)


def test_pyramidal_normal_depends_on_c_over_a():
    ideal = miller_to_cartesian("10-11", HEX, c_over_a=IDEAL_C_OVER_A)
    magnesium = miller_to_cartesian("10-11", HEX, c_over_a=1.6236)
    assert not np.allclose(ideal, magnesium)
    assert np.isclose(np.linalg.norm(magnesium), 1.0)


def test_direction_versus_plane_normal():
    """For hexagonal lattices [10-10] and (10-10) are not parallel."""
    plane = miller_to_cartesian("10-10", HEX, plane=True)
    direction = miller_to_cartesian("10-10", HEX, plane=False)
    assert not np.allclose(plane, direction)
    # ...but for cubic lattices they are.
    assert np.allclose(
        miller_to_cartesian("110", CUBIC, plane=True),
        miller_to_cartesian("110", CUBIC, plane=False),
    )


def test_pole_directions_shape_and_frame():
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(20, 3, 3)))
    rotations = q * np.sign(np.linalg.det(q))[:, None, None]
    poles = pole_directions(rotations, "0001", HEX)
    assert poles.shape == (20 * 2, 3)
    assert np.allclose(np.linalg.norm(poles, axis=1), 1.0)


def test_equal_area_preserves_solid_angle():
    """Uniform directions must project to a uniform disc, or MRD is wrong."""
    rng = np.random.default_rng(12)
    v = rng.normal(size=(200_000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    x, y = equal_area(upper_hemisphere(v))
    radius = np.hypot(x, y)
    assert radius.max() <= 1.0 + 1e-9
    # For a uniform disc the fraction inside radius r is r^2.
    for r in (0.25, 0.5, 0.75):
        assert np.isclose((radius < r).mean(), r**2, atol=0.01)


def test_stereographic_round_trip():
    rng = np.random.default_rng(13)
    v = rng.normal(size=(1000, 3))
    v = upper_hemisphere(v / np.linalg.norm(v, axis=1, keepdims=True))
    x, y = stereographic(v)
    assert np.allclose(inverse_stereographic(x, y), v, atol=1e-10)

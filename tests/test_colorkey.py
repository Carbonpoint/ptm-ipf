import numpy as np
import pytest

from ptmipf.colorkey import IPFColorKey
from ptmipf.symmetry import LAUE_GROUPS, get_laue_group

# The three sector vertices must come out pure red, green and blue, which is
# what makes the key readable next to an EBSD map.
VERTEX_COLORS = np.eye(3)


# In -3m the two rim vertices of the sector are symmetrically equivalent, so
# they necessarily share a colour and the sector has no separate blue corner.
THREE_CORNER_GROUPS = [n for n in LAUE_GROUPS if n != "-3m"]


@pytest.mark.parametrize("name", THREE_CORNER_GROUPS)
def test_vertices_are_red_green_blue(name):
    laue = get_laue_group(name)
    colors = IPFColorKey(laue).direction2color(laue.sector_vertices)
    assert np.allclose(colors, VERTEX_COLORS, atol=0.02)


def test_trigonal_rim_vertices_are_equivalent():
    laue = get_laue_group("-3m")
    colors = IPFColorKey(laue).direction2color(laue.sector_vertices)
    assert np.allclose(colors[0], [1.0, 0.0, 0.0], atol=0.02)
    assert np.allclose(colors[1], colors[2], atol=1e-9)


def test_cubic_corner_directions():
    key = IPFColorKey(get_laue_group("cubic"))
    colors = key.direction2color(np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=float))
    assert np.allclose(colors, VERTEX_COLORS, atol=0.02)


def test_hexagonal_corner_directions():
    key = IPFColorKey(get_laue_group("hexagonal"))
    m = [np.cos(np.pi / 6), np.sin(np.pi / 6), 0.0]
    colors = key.direction2color(np.array([[0, 0, 1], [1, 0, 0], m], dtype=float))
    assert np.allclose(colors, VERTEX_COLORS, atol=0.02)


@pytest.mark.parametrize("name", list(LAUE_GROUPS))
def test_barycentre_is_pale(name):
    laue = get_laue_group(name)
    color = IPFColorKey(laue).direction2color(laue.center[None])[0]
    assert color.min() > 0.85  # close to white


@pytest.mark.parametrize("name", list(LAUE_GROUPS))
def test_symmetrically_equivalent_directions_share_a_color(name):
    laue = get_laue_group(name)
    key = IPFColorKey(laue)
    rng = np.random.default_rng(7)
    v = rng.normal(size=(200, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    reference = key.direction2color(v)
    for op in laue.operators:
        assert np.allclose(key.direction2color(v @ op.T), reference, atol=1e-6)
    # A pole figure colours axes, so -v must match v as well.
    assert np.allclose(key.direction2color(-v), reference, atol=1e-6)


@pytest.mark.parametrize("name", list(LAUE_GROUPS))
def test_colors_are_finite_and_in_range(name):
    key = IPFColorKey(get_laue_group(name))
    rng = np.random.default_rng(8)
    colors = key.direction2color(rng.normal(size=(5000, 3)))
    assert np.isfinite(colors).all()
    assert colors.min() >= 0.0 and colors.max() <= 1.0
    # Every colour should be saturated somewhere: the key uses full value.
    assert np.allclose(colors.max(axis=1), 1.0, atol=1e-6)


def test_orientation2color_uses_the_transpose():
    """R maps crystal to sample, so the IPF direction is R^T d."""
    key = IPFColorKey(get_laue_group("hexagonal"))
    angle = np.radians(35.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    d = np.array([0.3, -0.4, 0.866])
    expected = key.direction2color((rotation.T @ d)[None])
    assert np.allclose(key.orientation2color(rotation[None], d), expected)


def test_hsv_to_rgb_matches_matplotlib():
    matplotlib_colors = pytest.importorskip("matplotlib.colors")
    from ptmipf.colorkey import hsv_to_rgb

    rng = np.random.default_rng(9)
    hsv = rng.uniform(size=(500, 3))
    assert np.allclose(hsv_to_rgb(hsv), matplotlib_colors.hsv_to_rgb(hsv), atol=1e-12)

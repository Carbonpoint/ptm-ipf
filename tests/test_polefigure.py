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


# ----------------------------------------------------------------------
# smoothing and colour scales
# ----------------------------------------------------------------------
def _sharp_rotations(n_grains=12, per_grain=300, seed=0):
    """A handful of perfect grains, which is what an MD cell looks like."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n_grains, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    m = np.empty((n_grains, 3, 3))
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y - z * w)
    m[:, 0, 2] = 2 * (x * z + y * w)
    m[:, 1, 0] = 2 * (x * y + z * w)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z - x * w)
    m[:, 2, 0] = 2 * (x * z - y * w)
    m[:, 2, 1] = 2 * (y * z + x * w)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return np.repeat(m, per_grain, axis=0)


def test_angular_smoothing_converts_through_the_projection():
    from ptmipf.polefigure import equal_area_sigma_bins

    assert equal_area_sigma_bins(0.0, 300) == 0.0
    # Lambert: one radian at the centre is 1/sqrt(2) of the disc radius.
    expected = np.radians(10.0) / np.sqrt(2.0) * 150
    assert equal_area_sigma_bins(10.0, 300) == pytest.approx(expected)
    # Twice the resolution, twice the bins, for the same angle.
    assert equal_area_sigma_bins(10.0, 600) == pytest.approx(2 * expected)


def test_smoothing_brings_the_peak_down_and_leaves_the_normalisation_alone():
    """The reason this option exists.

    A simulated cell of a few perfect grains gives peak intensities far above
    anything an EBSD map reports; a few degrees puts them on a comparable
    scale.  What must not change is that MRD still means multiples of random,
    so the mean over the disc stays one.
    """
    from ptmipf.polefigure import (
        _combine,
        _density_grid,
        equal_area_sigma_bins,
        pole_directions,
    )
    from ptmipf.projections import equal_area, upper_hemisphere
    from ptmipf.symmetry import get_laue_group

    laue = get_laue_group("m-3m")
    directions = pole_directions(_sharp_rotations(), "111", laue, plane=True)
    x, y = equal_area(upper_hemisphere(directions))

    peaks = []
    for degrees in (0.0, 3.0, 5.0, 10.0):
        sigma = _combine(4.0, equal_area_sigma_bins(degrees, 300))
        _, mrd = _density_grid(x, y, 300, sigma)
        peaks.append(float(np.nanmax(mrd)))
        assert np.nanmean(mrd) == pytest.approx(1.0, abs=1e-6)
    assert peaks == sorted(peaks, reverse=True), peaks
    # Unsmoothed, a dozen perfect grains give an MRD no measured texture shows.
    assert peaks[0] > 20 and peaks[2] < 10


def test_the_blur_takes_one_sigma_per_axis():
    """The IPF sector grid has different bin widths on the two axes."""
    from ptmipf.polefigure import _gaussian_blur

    image = np.zeros((41, 41))
    image[20, 20] = 1.0
    wide = _gaussian_blur(image, (6.0, 0.0))
    # The kernel is cut off at three standard deviations, so a fraction of a
    # percent of the mass is lost rather than none of it.
    assert wide[:, 20].sum() == pytest.approx(1.0, abs=0.01)
    # Blurred along the first axis only, so a row still holds a single pixel.
    assert np.count_nonzero(wide[20] > 1e-9) == 1
    assert np.count_nonzero(wide[:, 20] > 1e-9) > 10
    assert np.allclose(_gaussian_blur(image, 0.0), image)


def test_a_smoothed_figure_says_so_on_itself():
    """An MRD peak means something different at 10 degrees than at none."""
    from ptmipf.polefigure import pole_figure
    from ptmipf.symmetry import get_laue_group

    rotations = _sharp_rotations(n_grains=4, per_grain=50)
    laue = get_laue_group("m-3m")

    plain = pole_figure(rotations, "111", laue)
    assert not any("smoothed" in t.get_text() for t in plain.axes[0].texts)

    smoothed = pole_figure(rotations, "111", laue, smoothing=7.5)
    labels = [t.get_text() for t in smoothed.axes[0].texts]
    assert any("smoothed 7.5" in text for text in labels), labels


def test_pole_figures_accept_a_colour_map_by_name_or_by_table():
    from matplotlib.colors import ListedColormap

    from ptmipf.polefigure import pole_figure
    from ptmipf.symmetry import get_laue_group

    rotations = _sharp_rotations(n_grains=4, per_grain=50)
    laue = get_laue_group("m-3m")
    for cmap in ("jet", "rainbow", ListedColormap([[1, 0, 0], [0, 0, 1]])):
        assert pole_figure(rotations, "111", laue, cmap=cmap) is not None
    with pytest.raises(ValueError, match="unknown colour map"):
        pole_figure(rotations, "111", laue, cmap="not-a-colour-map")


def test_the_ipf_density_plot_takes_the_same_two_options():
    from ptmipf.polefigure import ipf_density
    from ptmipf.symmetry import get_laue_group

    rotations = _sharp_rotations(n_grains=4, per_grain=50)
    figure = ipf_density(rotations, "z", get_laue_group("m-3m"), cmap="jet", smoothing=6.0)
    labels = [t.get_text() for t in figure.axes[0].texts]
    assert any("smoothed 6" in text for text in labels), labels

"""Colouring grain boundaries by misorientation."""

import numpy as np
import pytest

pytest.importorskip("ovito")
pytest.importorskip("scipy")

from ptmipf import boundaries  # noqa: E402
from ptmipf.analysis import analyse  # noqa: E402
from ptmipf.flatmap import flat_ipf_map  # noqa: E402


@pytest.fixture
def bicrystal_map(write_bicrystal):
    """Two grains 40 degrees apart about c, so the boundary angle is known."""
    result = analyse(write_bicrystal(misorientation_deg=40.0), direction="z", structures=("hcp",))
    return flat_ipf_map(result, view="z", slab_width=8.0, pixel_size=0.6, boundary_angle=5.0)


def test_boundary_angle_is_recorded(bicrystal_map):
    flat = bicrystal_map
    assert flat.boundary is not None and flat.boundary_angle_map is not None
    on = flat.boundary & ~np.isnan(flat.boundary_angle_map)
    assert on.sum() > 0
    # 40 degrees about c is 20 degrees after the 60 degree symmetry folds it.
    assert np.isclose(np.nanmedian(flat.boundary_angle_map[on]), 20.0, atol=1.5)


def test_axis_angle_for_the_rotation_axis_is_zero(bicrystal_map):
    """A rotation about c leaves the c axis where it was."""
    tilt = boundaries.boundary_axis_angles(bicrystal_map, "0001")
    on = ~np.isnan(tilt)
    assert on.sum() > 0
    assert np.nanmedian(tilt[on]) < 1.0


def test_axis_angle_for_an_in_plane_axis_matches_the_rotation(bicrystal_map):
    """A 40 degree turn about c moves the m axis 20 degrees from its nearest
    equivalent (six of them, 60 apart), and the a axis under 2 degrees (twelve
    equivalents, 30 apart).  The pole family matters, and this pins it."""
    m_tilt = boundaries.boundary_axis_angles(bicrystal_map, "10-10", plane=False)
    assert np.isclose(np.nanmedian(m_tilt[~np.isnan(m_tilt)]), 20.0, atol=1.5)
    a_tilt = boundaries.boundary_axis_angles(bicrystal_map, "2-1-10", plane=False)
    assert np.nanmedian(a_tilt[~np.isnan(a_tilt)]) < 2.5


def test_scale_colouring_paints_only_the_boundary(bicrystal_map):
    flat = bicrystal_map
    rgb = boundaries.color_boundaries_by_angle(flat, 0.0, 90.0, "viridis")
    changed = np.any(np.abs(rgb - flat.rgb) > 1e-9, axis=2)
    assert changed.any()
    assert not changed[~flat.boundary].any()


def test_scale_range_clips(bicrystal_map):
    import matplotlib

    flat = bicrystal_map
    # With vmax below the actual angle, the boundary sits at the top of the scale.
    rgb = boundaries.color_boundaries_by_angle(flat, 0.0, 10.0, "viridis")
    top = np.asarray(matplotlib.colormaps["viridis"](1.0)[:3])
    on = flat.boundary & ~np.isnan(flat.boundary_angle_map)
    assert np.allclose(rgb[on], top, atol=1e-6)


def test_threshold_splits_high_and_low(bicrystal_map):
    flat = bicrystal_map
    on = flat.boundary & ~np.isnan(flat.boundary_angle_map)
    above = boundaries.color_boundaries_by_threshold(flat, 15.0, "black", "white")
    assert np.allclose(above[on], [0, 0, 0], atol=1e-6)  # 20 degrees is above 15
    below = boundaries.color_boundaries_by_threshold(flat, 30.0, "black", "white")
    assert np.allclose(below[on], [1, 1, 1], atol=1e-6)  # and below 30


def test_hidden_low_angle_boundary_takes_the_grain_colour(bicrystal_map):
    flat = bicrystal_map
    on = flat.boundary & ~np.isnan(flat.boundary_angle_map)
    hidden = boundaries.color_boundaries_by_threshold(flat, 30.0, "black", None)
    # No longer black, and no longer any single fixed colour.
    assert not np.allclose(hidden[on], [0, 0, 0], atol=0.05)


def test_width_thickens_the_line(bicrystal_map):
    flat = bicrystal_map
    thin = boundaries.color_boundaries_by_angle(flat, 0, 90, width=1)
    thick = boundaries.color_boundaries_by_angle(flat, 0, 90, width=3)
    changed = lambda rgb: np.any(np.abs(rgb - flat.rgb) > 1e-9, axis=2).sum()  # noqa: E731
    assert changed(thick) > changed(thin)


def test_unsegmented_map_is_refused(write_bicrystal):
    result = analyse(write_bicrystal(name="b2.xyz"), direction="z", structures=("hcp",))
    flat = flat_ipf_map(result, view="z", slab_width=8.0, pixel_size=0.8, boundary_angle=0.0)
    with pytest.raises(ValueError):
        boundaries.color_boundaries_by_angle(flat)

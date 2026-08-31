"""Slab sections and orthographic axis views.

A slab of a few atomic layers viewed down its normal is the atomistic
equivalent of an EBSD orientation map, so it needs to behave predictably.
"""

import numpy as np
import pytest

from ptmipf.webui.rendering import camera_basis, visible_mask


class FakeSlabResult:
    """Atoms on a line along z, so slab membership is easy to reason about."""

    def __init__(self):
        self.positions = np.zeros((21, 3))
        self.positions[:, 2] = np.arange(-10.0, 11.0)
        self.structure_types = np.ones(21, dtype=int)
        self.structure_types[0] = 0  # one unidentified atom
        self.n_atoms = 21


def test_zero_width_cuts_the_cell_in_half():
    result = FakeSlabResult()
    visible = visible_mask(result, slice_normal=[0, 0, 1], slice_distance=0.0, slice_width=0.0)
    assert visible.sum() == 11  # z <= 0
    assert not visible[result.positions[:, 2] > 0].any()


@pytest.mark.parametrize("width,expected", [(2.0, 3), (10.0, 11), (0.5, 1)])
def test_slab_keeps_only_the_requested_thickness(width, expected):
    result = FakeSlabResult()
    visible = visible_mask(result, slice_normal=[0, 0, 1], slice_distance=0.0, slice_width=width)
    assert visible.sum() == expected
    kept = result.positions[visible, 2]
    assert np.abs(kept).max() <= width / 2 + 1e-9


def test_slab_is_centred_on_the_plane():
    result = FakeSlabResult()
    visible = visible_mask(result, slice_normal=[0, 0, 1], slice_distance=5.0, slice_width=4.0)
    kept = result.positions[visible, 2]
    assert kept.min() >= 3.0 and kept.max() <= 7.0


def test_slab_combines_with_hide_other():
    result = FakeSlabResult()
    visible = visible_mask(
        result, hide_other=True, slice_normal=[0, 0, 1], slice_distance=-10.0, slice_width=4.0
    )
    assert not visible[0]  # the unidentified atom is dropped


@pytest.mark.parametrize(
    "azimuth,elevation,expected",
    [(0.0, 0.0, [-1, 0, 0]), (90.0, 0.0, [0, -1, 0]), (-90.0, 89.9, [0, 0, -1])],
)
def test_axis_views_look_straight_down_an_axis(azimuth, elevation, expected):
    """The X/Y/Z buttons of the web UI must give exactly axial views."""
    direction, _, _ = camera_basis(azimuth, elevation)
    assert np.allclose(direction, expected, atol=2e-3)


def test_cli_exposes_the_section_options():
    from ptmipf.cli import build_parser

    args = build_parser().parse_args(
        ["f.dump", "--slice", "z", "--slice-width", "10", "--view", "z"]
    )
    assert args.slice_width == 10.0
    assert args.view == "z"
    assert args.perspective is False


def test_cli_renders_a_section(write_crystal, tmp_path, renderer):
    pytest.importorskip("ovito")
    from ptmipf.cli import main

    out = tmp_path / "section.png"
    code = main(
        [
            write_crystal("hcp", repeat=8),
            "--structures", "hcp", "--direction", "z", "-q",
            "--slice", "z", "--slice-width", "6", "--view", "z",
            "--hide-other", "--render", str(out), "--render-size", "200x200",
        ]
    )
    assert code == 0 and out.exists() and out.stat().st_size > 0

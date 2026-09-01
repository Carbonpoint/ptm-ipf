"""Colour-coding keys and colour bars for OVITO.

The point of these columns is that OVITO's own Color coding modifier can
repaint the atoms from them, so the tests that matter run the real modifier
and compare its output with the colours the analysis produced.
"""

import numpy as np
import pytest

from ptmipf.colormap import (
    color_keys,
    direction_column,
    gradient_keys,
    keys_from_indices,
    quantise_colors,
    write_color_map,
)


def _ramp(n=500):
    """Colours spread over the RGB cube, standing in for an IPF map."""
    rng = np.random.default_rng(0)
    return rng.random((n, 3))


def test_column_names_are_identifiers():
    assert direction_column("Z") == "ipf_z"
    assert direction_column("RD") == "ipf_rd"
    # A leading minus must not collapse onto the positive axis.
    assert direction_column("-Z") != direction_column("+Z")
    assert direction_column("[1 1 0]") == "ipf_1_1_0"


def test_column_names_are_made_unique():
    assert direction_column("Z", taken={"ipf_z"}) == "ipf_z_2"


def test_palette_round_trips_the_colours():
    colors = _ramp()
    palette, indices, error = quantise_colors([colors])
    assert error <= 0.5 / 255 + 1e-9
    recovered = palette[indices[0]] / 255.0
    assert np.abs(recovered - colors).max() == pytest.approx(error)


def test_directions_share_one_palette():
    a, b = _ramp(300), _ramp(200)
    palette, indices, _ = quantise_colors([a, b])
    assert [len(i) for i in indices] == [300, 200]
    assert indices[0].max() < len(palette) and indices[1].max() < len(palette)


def test_palette_is_capped_and_says_what_it_cost():
    colors = _ramp(20000)
    palette, indices, error = quantise_colors([colors], max_entries=256)
    assert len(palette) <= 256
    # Snapping to a coarser lattice is the price of the cap, and it is reported.
    assert error > 0.5 / 255
    assert np.abs(palette[indices[0]] / 255.0 - colors).max() == pytest.approx(error)


def test_keys_address_the_centre_of_their_entry():
    keys = keys_from_indices(np.arange(4), 4)
    assert list(keys) == [0.125, 0.375, 0.625, 0.875]


def test_color_map_png_has_one_pixel_per_entry(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    palette = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    path = write_color_map(palette, tmp_path / "map.png", height=4)
    with Image.open(path) as image:
        assert image.size == (3, 4)
        assert np.array_equal(np.array(image)[0], palette)


# ----------------------------------------------------------------------
# the round trip through OVITO itself
# ----------------------------------------------------------------------
ovito = pytest.importorskip("ovito", reason="OVITO is unavailable")


def test_ovito_reads_the_colour_bar_by_nearest_pixel(tmp_path):
    """The keys assume nearest-pixel sampling; pin that down.

    If OVITO ever interpolated between entries instead, a palette of unrelated
    colours would blend at the edges and the exported map would drift, so this
    assumption deserves a test of its own rather than a comment.
    """
    from ovito.modifiers import ColorCodingModifier

    palette = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]], np.uint8)
    path = write_color_map(palette, tmp_path / "map.png")
    gradient = ColorCodingModifier.Image(str(path))
    for index, expected in enumerate(palette / 255.0):
        got = gradient.value_to_color(float(keys_from_indices(index, len(palette))))
        assert np.allclose(got, expected, atol=1 / 255)


def test_color_keys_reproduce_the_ipf_map_inside_ovito(tmp_path, write_crystal):
    """The whole point: OVITO repaints the exported file and gets the same map."""
    from ovito.io import import_file
    from ovito.modifiers import ColorCodingModifier

    from ptmipf.analysis import analyse
    from ptmipf.io import write_result
    from tests.conftest import rotation_matrix

    path = write_crystal("fcc", rotation=rotation_matrix([1, 2, 3], 37.0), repeat=4)
    result = analyse(path, structures=("fcc",), direction="z")
    keys, palette, info = color_keys(result, ("x", "y", "z"))
    assert set(keys) == {"ipf_x", "ipf_y", "ipf_z"}

    dump = tmp_path / "colored.dump"
    write_result(result, dump, keys=keys)
    write_color_map(palette, tmp_path / "map.png")

    pipeline = import_file(str(dump))
    exported = np.array(pipeline.compute().particles["Color"])
    pipeline.modifiers.append(
        ColorCodingModifier(
            property="ipf_z",
            start_value=0.0,
            end_value=1.0,
            gradient=ColorCodingModifier.Image(str(tmp_path / "map.png")),
        )
    )
    recoloured = np.array(pipeline.compute().particles["Color"])
    # Both sides are written to a file at 8-bit depth, so one part in 255 is
    # the honest tolerance here.
    assert np.abs(recoloured - exported).max() < 2 / 255

    # And a different direction really is a different map, not a copy.
    pipeline.modifiers[0].property = "ipf_x"
    along_x = np.array(pipeline.compute().particles["Color"])
    assert np.abs(along_x - exported).max() > 0.05


def test_builtin_gradients_are_reported_as_the_approximation_they_are():
    """A stock colour bar is a curve; the IPF colours are a surface on it."""
    colors = _ramp(400)
    keys, worst, mean = gradient_keys([colors], "jet", samples=256)
    assert keys[0].shape == (400,)
    assert 0.0 <= keys[0].min() and keys[0].max() <= 1.0
    assert mean > 0.05 and worst >= mean


def test_unknown_gradient_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="jet"):
        gradient_keys([_ramp(10)], "not-a-colour-bar")


# ----------------------------------------------------------------------
# colour maps for the density plots
# ----------------------------------------------------------------------
def test_named_colormaps_resolve():
    from ptmipf.colormap import PLOT_COLORMAPS, load_colormap

    for name in PLOT_COLORMAPS:
        assert load_colormap(name)(0.5) is not None
    # jet and rainbow are here because the texture literature uses them.
    assert "jet" in PLOT_COLORMAPS and "rainbow" in PLOT_COLORMAPS


def test_an_unknown_colormap_says_what_is_on_offer():
    from ptmipf.colormap import load_colormap

    with pytest.raises(ValueError, match="viridis"):
        load_colormap("not-a-colour-map")


def test_a_colormap_array_is_used_as_given():
    from ptmipf.colormap import load_colormap

    colors = load_colormap(np.array([[1.0, 0, 0], [0, 0, 1.0]]))
    assert np.allclose(colors(0.0)[:3], [1, 0, 0])
    assert np.allclose(colors(1.0)[:3], [0, 0, 1])
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        load_colormap(np.zeros((4, 2)))


@pytest.mark.parametrize(
    "text,expected",
    [
        ("255 0 0\n0 0 255\n", [[1.0, 0, 0], [0, 0, 1.0]]),
        ("1.0,0,0\n0,0,1.0\n", [[1.0, 0, 0], [0, 0, 1.0]]),
        ("# a comment\n0 255 0 0\n1 0 0 255\n", [[1.0, 0, 0], [0, 0, 1.0]]),
    ],
)
def test_colour_tables_are_read_in_the_shapes_people_write_them(text, expected):
    """Whitespace or commas, 0 to 1 or 0 to 255, with or without an index column."""
    from ptmipf.colormap import read_colormap_table

    assert np.allclose(read_colormap_table(text), expected)


@pytest.mark.parametrize(
    "text,message",
    [
        ("1 0\n0 1\n", "three numbers"),
        ("nope nope nope\n", "cannot read"),
        ("1 0 0\n", "at least two"),
        ("-1 0 0\n0 0 1\n", "0 to 1 or 0 to 255"),
    ],
)
def test_a_broken_colour_table_says_why(text, message):
    from ptmipf.colormap import read_colormap_table

    with pytest.raises(ValueError, match=message):
        read_colormap_table(text)


def test_an_image_strip_is_read_left_to_right(tmp_path):
    """Any height works, so a screenshot of a colour bar is usable."""
    Image = pytest.importorskip("PIL.Image")
    from ptmipf.colormap import load_colormap

    strip = np.zeros((6, 3, 3), np.uint8)
    strip[:, 0] = [255, 0, 0]
    strip[:, 1] = [0, 255, 0]
    strip[:, 2] = [0, 0, 255]
    path = tmp_path / "bar.png"
    Image.fromarray(strip).save(path)
    colors = load_colormap(str(path))
    assert np.allclose(colors(0.0)[:3], [1, 0, 0], atol=1 / 255)
    assert np.allclose(colors(1.0)[:3], [0, 0, 1], atol=1 / 255)


def test_the_colour_bar_written_for_ovito_is_a_valid_plot_colormap(tmp_path):
    """The same file serves both, which is the point of it being a strip."""
    from ptmipf.colormap import load_colormap

    pytest.importorskip("PIL.Image")
    palette, _, _ = quantise_colors([_ramp(200)])
    path = write_color_map(palette, tmp_path / "ovito.png")
    assert load_colormap(path).N == len(palette)

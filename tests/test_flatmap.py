"""Flat, EBSD-style orientation maps."""

import numpy as np
import pytest

pytest.importorskip("ovito")
pytest.importorskip("scipy")

from ptmipf.analysis import analyse  # noqa: E402
from ptmipf.flatmap import flat_ipf_map, save_flat_map  # noqa: E402


@pytest.fixture(scope="module")
def single_crystal(tmp_path_factory):
    ase_build = pytest.importorskip("ase.build")
    ase_io = pytest.importorskip("ase.io")
    atoms = ase_build.bulk("Mg", "hcp", a=3.2094, c=5.2108).repeat((10, 10, 10))
    path = tmp_path_factory.mktemp("flat") / "mg.xyz"
    ase_io.write(str(path), atoms, format="extxyz")
    return analyse(str(path), direction="z", structures=("hcp",))


def test_single_crystal_is_one_uniform_grain(single_crystal):
    flat = flat_ipf_map(single_crystal, view="z", slab_width=8.0, pixel_size=0.6)
    assert flat.n_grains == 1
    assert flat.boundary_fraction == 0.0
    inside = flat.labels >= 0
    # A hexagonal cell is a parallelogram, so its footprint covers about
    # sin(60) of the rectangular image and the corners stay background.
    assert inside.mean() > 0.75
    assert np.allclose(flat.rgb[inside], [1.0, 0.0, 0.0], atol=0.02)  # c axis along z


def test_map_geometry_matches_the_request(single_crystal):
    flat = flat_ipf_map(single_crystal, view="z", slab_width=8.0, pixel_size=0.5)
    rows, columns = flat.shape
    assert np.isclose(flat.width_angstrom / columns, 0.5, atol=0.05)
    assert np.isclose(flat.height_angstrom / rows, 0.5, atol=0.05)
    assert flat.slab_width == 8.0
    assert flat.labels is not None and flat.labels.shape == flat.shape


def test_axes_are_named_from_the_sample_frame(single_crystal):
    flat = flat_ipf_map(single_crystal, view="z", slab_width=8.0, pixel_size=1.0)
    assert {flat.horizontal_label, flat.vertical_label} == {"RD", "TD"}
    assert flat.view_label == "Z"


def test_a_thin_section_takes_fewer_atoms(single_crystal):
    thin = flat_ipf_map(single_crystal, view="z", slab_width=4.0, pixel_size=1.0)
    thick = flat_ipf_map(single_crystal, view="z", slab_width=16.0, pixel_size=1.0)
    assert thin.n_atoms < thick.n_atoms


def test_an_empty_section_is_rejected(single_crystal):
    with pytest.raises(ValueError):
        flat_ipf_map(single_crystal, view="z", slab_center=1e6, slab_width=1.0)


def test_bicrystal_is_two_grains_with_a_boundary(write_bicrystal):
    result = analyse(write_bicrystal(misorientation_deg=40.0), direction="z", structures=("hcp",))
    flat = flat_ipf_map(result, view="z", slab_width=8.0, pixel_size=0.6, boundary_angle=5.0)
    assert flat.n_grains == 2
    assert flat.boundary_fraction > 0
    # The boundary is a line, not a smear: it should cover a small part of the map.
    assert flat.boundary_fraction < 0.15
    # Two grains means two distinct colours in the mapped area.
    colours = np.unique(flat.rgb[flat.labels >= 0].round(2), axis=0)
    assert len(colours) >= 2


def test_no_boundaries_when_the_angle_is_zero(write_bicrystal):
    result = analyse(write_bicrystal(), direction="z", structures=("hcp",))
    flat = flat_ipf_map(result, view="z", slab_width=8.0, pixel_size=0.8, boundary_angle=0.0)
    assert flat.boundary_fraction == 0.0
    assert flat.n_grains == 0


def test_saving_produces_a_png(single_crystal, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    flat = flat_ipf_map(single_crystal, view="z", slab_width=8.0, pixel_size=1.0)
    out = tmp_path / "map.png"
    save_flat_map(flat, out, title="test")
    assert out.exists() and out.stat().st_size > 0

    from ovito.io import import_file  # noqa: F401  (kept for the ovito import guard)
    from PIL import Image

    with Image.open(out) as image:
        assert image.format == "PNG"

"""Unit cell wireframes over a flat orientation map."""

import numpy as np
import pytest

from ptmipf.wireframe import unit_cell_edges

pytest.importorskip("scipy")


def test_hexagonal_cell_has_eighteen_edges_and_the_right_shape():
    vertices, edges = unit_cell_edges("6/mmm", c_over_a=1.6)
    assert vertices.shape == (12, 3) and edges.shape == (18, 2)
    # Basal ring radius 0.5, c axis along z, height 0.5 * c/a.
    assert np.allclose(np.linalg.norm(vertices[:6, :2], axis=1), 0.5)
    assert np.allclose(vertices[:6, 2], -0.4) and np.allclose(vertices[6:, 2], 0.4)
    # a1 along x: the first vertex sits on +x.
    assert np.allclose(vertices[0], [0.5, 0.0, -0.4])


def test_cubic_cell_has_twelve_unit_edges():
    vertices, edges = unit_cell_edges("m-3m")
    assert vertices.shape == (8, 3) and edges.shape == (12, 2)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    assert np.allclose(lengths, 1.0)


@pytest.fixture(scope="module")
def flat_single(tmp_path_factory):
    pytest.importorskip("ovito")
    ase_build = pytest.importorskip("ase.build")
    ase_io = pytest.importorskip("ase.io")
    from ptmipf.analysis import analyse
    from ptmipf.flatmap import flat_ipf_map

    atoms = ase_build.bulk("Mg", "hcp", a=3.2094, c=5.2108).repeat((12, 12, 8))
    path = tmp_path_factory.mktemp("wf") / "mg.xyz"
    ase_io.write(str(path), atoms, format="extxyz")
    result = analyse(str(path), direction="z", structures=("hcp",))
    return {
        view: flat_ipf_map(result, view=view, slab_width=8.0, pixel_size=0.5)
        for view in ("z", "x")
    }


def test_wireframe_projection_follows_the_crystal(flat_single):
    """c along z: the hexagon is face-on down z and edge-on down x."""
    from ptmipf.wireframe import grain_wireframes

    down_z = grain_wireframes(flat_single["z"], size=10.0, color="black")[0]
    down_x = grain_wireframes(flat_single["x"], size=10.0, color="black")[0]
    lengths_z = np.linalg.norm(down_z.segments[:, 1] - down_z.segments[:, 0], axis=1)
    lengths_x = np.linalg.norm(down_x.segments[:, 1] - down_x.segments[:, 0], axis=1)
    # Down z the twelve ring edges keep their length and the six c edges vanish.
    assert np.isclose(np.sort(lengths_z)[-1], 5.0, atol=0.05)
    assert (lengths_z < 1e-6).sum() == 6
    # Down x the c edges are fully visible at 0.5 * size * c/a.
    assert np.isclose(lengths_x.max(), 5.0 * np.sqrt(8 / 3), atol=0.05)


def test_one_grain_gives_one_wireframe_sized_by_area(flat_single):
    from ptmipf.wireframe import grain_wireframes

    flat = flat_single["z"]
    proportional = grain_wireframes(flat, color="black")
    assert len(proportional) == 1
    doubled = grain_wireframes(flat, color="black", scale=2.0)
    span = lambda w: np.ptp(w.segments.reshape(-1, 2), axis=0).max()  # noqa: E731
    assert np.isclose(span(doubled[0]), 2 * span(proportional[0]), rtol=1e-6)
    fixed = grain_wireframes(flat, color="black", size=7.0)
    assert np.isclose(span(fixed[0]), 7.0, atol=0.05)


def test_inverted_colour_is_the_complement(flat_single):
    from ptmipf.wireframe import grain_wireframes

    wire = grain_wireframes(flat_single["z"], color="invert")[0]
    # The grain is red, so the wireframe is cyan.
    assert np.allclose(wire.color, [0.0, 1.0, 1.0], atol=0.05)
    fixed = grain_wireframes(flat_single["z"], color="black")[0]
    assert fixed.color == (0.0, 0.0, 0.0)


def test_unsegmented_map_is_refused(flat_single):
    from ptmipf.flatmap import flat_ipf_map
    from ptmipf.wireframe import grain_wireframes

    pytest.importorskip("ovito")
    # boundary_angle=0 skips segmentation, so there is nothing to annotate.
    # Rebuild from the same result via the fixture's map settings.
    flat = flat_single["z"]
    unsegmented = flat_ipf_map.__wrapped__ if hasattr(flat_ipf_map, "__wrapped__") else None
    assert unsegmented is None  # no decorator; keep the guard honest
    flat_no = type(flat)(**{**flat.__dict__, "labels": None, "rotations": None})
    with pytest.raises(ValueError):
        grain_wireframes(flat_no)


def test_draw_writes_a_png(flat_single, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from ptmipf.flatmap import save_flat_map
    from ptmipf.wireframe import grain_wireframes

    flat = flat_single["z"]
    out = tmp_path / "wf.png"
    save_flat_map(flat, out, wireframes=grain_wireframes(flat, color="black"))
    assert out.exists() and out.stat().st_size > 0

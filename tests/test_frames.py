import numpy as np
import pytest

from ptmipf.frames import SampleFrame, parse_vector


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("z", [0, 0, 1]),
        ("+z", [0, 0, 1]),
        ("-x", [-1, 0, 0]),
        ("Y", [0, 1, 0]),
        ("1,1,0", [np.sqrt(0.5), np.sqrt(0.5), 0]),
        ("[1 1 0]", [np.sqrt(0.5), np.sqrt(0.5), 0]),
        ("0,0,-2", [0, 0, -1]),
        ("1e0,0,0", [1, 0, 0]),
    ],
)
def test_parse_vector(spec, expected):
    assert np.allclose(parse_vector(spec), expected)


@pytest.mark.parametrize("spec", ["", "q", "1,2", "1,2,3,4", "0,0,0"])
def test_parse_vector_rejects_nonsense(spec):
    with pytest.raises(ValueError):
        parse_vector(spec)


def test_default_frame_is_the_cell_frame():
    frame = SampleFrame()
    assert np.allclose(frame.direction("rd"), [1, 0, 0])
    assert np.allclose(frame.direction("td"), [0, 1, 0])
    assert np.allclose(frame.direction("nd"), [0, 0, 1])


def test_third_axis_completes_a_right_handed_frame():
    frame = SampleFrame({"rd": "1,1,0", "nd": "0,0,1"})
    rd, td, nd = (frame.direction(a) for a in ("rd", "td", "nd"))
    assert np.allclose(np.cross(rd, td), nd)
    assert np.isclose(np.linalg.det(np.stack([rd, td, nd])), 1.0)


def test_single_axis_is_completed():
    frame = SampleFrame({"nd": "1,1,1"})
    axes = np.stack([frame.direction(a) for a in ("rd", "td", "nd")])
    assert np.allclose(axes @ axes.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(axes), 1.0)


def test_non_orthogonal_axes_are_orthogonalised_with_a_warning():
    frame = SampleFrame({"rd": "1,0,0", "td": "1,1,0"})
    assert frame.warnings
    assert np.isclose(np.dot(frame.direction("rd"), frame.direction("td")), 0.0, atol=1e-12)


def test_parallel_axes_are_rejected():
    with pytest.raises(ValueError):
        SampleFrame({"rd": "1,0,0", "td": "2,0,0"})


def test_extra_named_axis_is_kept():
    frame = SampleFrame({"ed": "1,1,0"})
    assert np.allclose(frame.direction("ed"), [np.sqrt(0.5), np.sqrt(0.5), 0])
    assert frame.label("ed") == "ED"
    assert frame.label("-z") == "-z"

"""The coordinate triad must not be drawn on top of the scene.

The overlap is measured in rendered pixels rather than judged by eye: render
the same scene with and without the triad at the same framing, and any pixel
the triad changed must not be a pixel that had something in it.
"""

import numpy as np
import pytest

pytest.importorskip("ovito")
PIL = pytest.importorskip("PIL.Image")

from ptmipf.analysis import analyse  # noqa: E402
from ptmipf.render import MAX_TRIPOD_MARGIN, TRIPOD_MARGIN, render_result  # noqa: E402


@pytest.fixture(scope="module")
def result(tmp_path_factory, renderer):
    ase_build = pytest.importorskip("ase.build")
    ase_io = pytest.importorskip("ase.io")
    atoms = ase_build.bulk("Mg", "hcp", a=3.2094, c=5.2108).repeat((9, 9, 9))
    path = tmp_path_factory.mktemp("tripod") / "mg.xyz"
    ase_io.write(str(path), atoms, format="extxyz")
    return analyse(str(path), direction="z", structures=("hcp",))


def _image(path):
    with PIL.open(path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=float) / 255.0


def _overlap_pixels(result, tmp_path, size, **kwargs):
    """Triad pixels that land on something already drawn."""
    info = {}
    with_triad = tmp_path / "on.png"
    without = tmp_path / "off.png"
    render_result(result, with_triad, tripod=True, size=size, info=info, **kwargs)
    # Same framing, so the only difference between the two is the triad itself.
    render_result(result, without, tripod=False, size=size, margin=info["margin"], **kwargs)

    on, off = _image(with_triad), _image(without)
    triad = np.abs(on - off).max(axis=2) > 0.02
    content = (off.min(axis=2) < 0.97) & ~triad
    assert triad.sum() > 100, "the triad was not drawn at all"
    return int((triad & content).sum()), info["margin"]


@pytest.mark.parametrize(
    "name,size,kwargs",
    [
        ("perspective", (500, 420), {}),
        ("down z", (500, 420), {"camera_dir": (0, 0, -1), "perspective": False}),
        ("along x", (500, 420), {"camera_dir": (-1, 0, 0), "perspective": False}),
        ("wide", (700, 300), {}),
    ],
)
def test_triad_does_not_sit_on_the_scene(result, tmp_path, name, size, kwargs):
    overlap, margin = _overlap_pixels(result, tmp_path, size, hide_other=True, **kwargs)
    assert overlap == 0, f"{name}: {overlap} triad pixels land on the scene"
    assert TRIPOD_MARGIN <= margin <= MAX_TRIPOD_MARGIN


def test_framing_is_only_widened_when_it_has_to_be(result, tmp_path):
    """A view that fills the frame needs more room than one that does not."""
    loose, tight = {}, {}
    render_result(result, tmp_path / "a.png", tripod=True, size=(500, 420), info=loose)
    render_result(
        result,
        tmp_path / "b.png",
        tripod=True,
        size=(500, 420),
        camera_dir=(0, 0, -1),
        perspective=False,
        info=tight,
    )
    assert loose["margin"] <= tight["margin"]


def test_margin_can_be_set_by_hand(result, tmp_path):
    info = {}
    render_result(result, tmp_path / "c.png", tripod=True, size=(300, 260), margin=1.6, info=info)
    assert info["margin"] == 1.6


def test_no_triad_means_no_extra_framing(result, tmp_path):
    info = {}
    render_result(result, tmp_path / "d.png", tripod=False, size=(300, 260), info=info)
    assert info["margin"] == 1.0

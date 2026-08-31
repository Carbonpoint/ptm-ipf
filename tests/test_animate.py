"""Animating a section over a trajectory."""

from pathlib import Path

import numpy as np
import pytest

from ptmipf.animate import _pad_to_common_size, frame_files, write_video


def test_frame_files_sort_numerically(tmp_path):
    for step in (0, 10000, 5000, 100000):
        (tmp_path / f"run.{step}.dump").write_text("")
    names = [p.name for p in frame_files(tmp_path / "run.*.dump")]
    assert names == ["run.0.dump", "run.5000.dump", "run.10000.dump", "run.100000.dump"]


def test_padding_gives_every_frame_the_same_size(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    paths = []
    for i, size in enumerate([(40, 30), (50, 20), (35, 35)]):
        p = tmp_path / f"{i}.png"
        PIL.new("RGBA", size, (255, 0, 0, 255)).save(p)
        paths.append(str(p))
    _pad_to_common_size(paths)
    sizes = {PIL.open(p).size for p in paths}
    modes = {PIL.open(p).mode for p in paths}
    # 35 is rounded up to 36: H.264 cannot encode an odd dimension.
    assert sizes == {(50, 36)}
    assert modes == {"RGB"}


def test_odd_sized_frames_still_encode(tmp_path):
    """libx264 with yuv420p rejects odd dimensions, so padding must round up."""
    PIL = pytest.importorskip("PIL.Image")
    pytest.importorskip("imageio_ffmpeg")
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.png"
        PIL.new("RGB", (33, 25), (i * 80, 0, 0)).save(p)
        paths.append(str(p))
    mp4 = write_video(paths, tmp_path / "odd.mp4", fps=2)
    assert Path(mp4).stat().st_size > 0


def test_gif_and_mp4_are_written(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.png"
        PIL.new("RGB", (32, 24), (i * 80, 0, 0)).save(p)
        paths.append(str(p))
    gif = write_video(paths, tmp_path / "a.gif", fps=2)
    assert Path(gif).stat().st_size > 0
    pytest.importorskip("imageio_ffmpeg")
    mp4 = write_video(paths, tmp_path / "a.mp4", fps=2)
    assert Path(mp4).stat().st_size > 0


def test_flat_map_animation_end_to_end(tmp_path, write_crystal):
    pytest.importorskip("ovito")
    pytest.importorskip("scipy")
    from ptmipf.animate import animate_flat_map

    # Two "frames" of the same crystal stand in for a trajectory.
    for s in (0, 5000):
        write_crystal("hcp", repeat=6, name=f"c.{s}.xyz")
    pngs = animate_flat_map(
        frame_files(tmp_path / "c.*.xyz"), tmp_path / "flat.gif", direction="z", view="z",
        structures=("hcp",), pixel_size=1.0, fill=None, rate=0.001, fps=2,
    )
    assert len(pngs) == 2 and (tmp_path / "flat.gif").exists()
    assert np.all([Path(p).exists() for p in pngs])

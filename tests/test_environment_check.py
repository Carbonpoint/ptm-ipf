"""What the environment is missing, said before it costs anybody time.

A series can run for a quarter of an hour.  Finding out at the end that no
encoder is installed, and losing every rendered frame with it, is the failure
these tests exist to prevent.
"""

from __future__ import annotations

import sys

import pytest

from ptmipf import animate
from ptmipf.webui.series import SeriesJob


class _Stub:
    """Just enough of the application state for the checks in __init__."""

    def resolve(self, path):  # pragma: no cover - never reached in these tests
        raise AssertionError("the job should have been refused before this")


def test_a_format_that_is_not_a_movie_is_named_as_such():
    assert "not a movie format" in animate.video_support(".avi")


def test_a_missing_encoder_names_the_package_and_the_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    reason = animate.video_support(".mp4")
    assert "imageio-ffmpeg" in reason
    assert "pip install 'imageio[ffmpeg]'" in reason


def test_a_series_is_refused_before_any_frame_is_matched(monkeypatch):
    monkeypatch.setattr(animate, "video_support", lambda ext: "mp4 output needs imageio")
    series = {"stem": "dump", "kind": "files", "items": [{"path": "a.xyz", "frame_index": 0}]}
    with pytest.raises(ValueError, match="mp4 output needs imageio"):
        SeriesJob(_Stub(), series, {"outputs": ["poles:mp4"], "start": 0, "stop": 0})


def test_an_encoder_that_fails_leaves_the_stills_and_says_so(monkeypatch, tmp_path):
    """The frames are already rendered, so the run is done, with a note."""

    def explode(pngs, out, fps=4):
        raise ImportError("no encoder here")

    monkeypatch.setattr(animate, "write_video", explode)
    job = SeriesJob.__new__(SeriesJob)
    job.movies = {"poles": {"mp4"}}
    job.series = {"stem": "dump"}
    job.out_dir = tmp_path
    job.seconds = 0.5
    job.files = []
    job.notes = []
    job.stage = ""
    job._write_movies({"poles": [tmp_path / "poles_00000.png"]})
    assert job.files == []
    assert job.notes == ["the mp4 movie could not be written: no encoder here"]


def test_the_environment_check_names_git_and_how_to_install_it(monkeypatch):
    """Most people arrive here because 'uv pip install git+https://...' failed."""
    from ptmipf.webui import state

    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("platform.system", lambda: "Windows")
    ok, detail = state._git_check()
    assert not ok
    assert "winget install Git.Git" in detail
    monkeypatch.setattr("platform.system", lambda: "Linux")
    assert "sudo apt install git" in state._git_check()[1]


def test_the_environment_check_finds_git_when_it_is_there(monkeypatch):
    from ptmipf.webui import state

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/git")
    assert state._git_check() == (True, "/usr/bin/git")

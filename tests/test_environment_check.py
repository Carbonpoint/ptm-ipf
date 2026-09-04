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


def _version(major, minor):
    """A stand-in for sys.version_info that answers .major and .minor too."""
    import collections

    fields = collections.namedtuple(
        "version_info", "major minor micro releaselevel serial"
    )
    return fields(major, minor, 0, "final", 0)


def test_a_windows_dll_failure_names_the_python_version_as_the_cause(monkeypatch):
    """OVITO 3.16 has no Windows build for Python 3.14, and 3.15.5 breaks there."""
    from ptmipf.webui import state

    monkeypatch.setattr("sys.version_info", _version(3, 14))
    message = state._explain_ovito(
        ImportError("DLL load failed while importing ovito_bindings: "
                    "The specified module could not be found.")
    )
    assert "Python 3.13" in message
    assert "uv venv --python 3.13" in message


def test_a_dll_failure_on_a_supported_python_talks_about_qt(monkeypatch):
    from ptmipf.webui import state

    monkeypatch.setattr("sys.version_info", _version(3, 13))
    message = state._explain_ovito(ImportError("DLL load failed while importing x"))
    assert "PySide6" in message and "3.13" not in message


def _fake_metadata(monkeypatch, requires, installed):
    """Stand in for the installed OVITO and PySide6, without installing either."""
    import importlib.metadata as metadata

    from ptmipf.webui import state

    versions = {"PySide6": installed, "ovito": "3.15.5"}

    def version(name):
        try:
            return versions[name]
        except KeyError:
            raise metadata.PackageNotFoundError(name) from None

    monkeypatch.setattr(metadata, "requires", lambda name: requires)
    monkeypatch.setattr(metadata, "version", version)
    return state


def test_a_qt_that_this_ovito_cannot_use_is_named_with_the_fix(monkeypatch):
    """The import error names neither package, so the check has to."""
    state = _fake_metadata(
        monkeypatch, ["numpy>=2", "PySide6~=6.10.3"], installed="6.11.2"
    )
    ok, detail = state._qt_pairing_check()
    assert not ok
    assert "6.11.2" in detail and "~=6.10.3" in detail
    assert 'uv pip install "PySide6~=6.10.3"' in detail


def test_a_matching_qt_passes_quietly(monkeypatch):
    state = _fake_metadata(
        monkeypatch, ["numpy>=2", "PySide6~=6.10.3"], installed="6.10.3"
    )
    assert state._qt_pairing_check() == (True, "PySide6 6.10.3")


def test_an_ovito_that_does_not_constrain_qt_is_not_second_guessed(monkeypatch):
    """OVITO 3.15.5 asks for PySide6>=6.8.3 and means it, as far as we know."""
    state = _fake_metadata(monkeypatch, ["numpy>=2", "PySide6>=6.8.3"], installed="6.11.2")
    ok, detail = state._qt_pairing_check()
    assert ok and "6.11.2" in detail


def test_the_qt_check_accepts_the_package_that_is_really_installed(monkeypatch):
    """OVITO asks for PySide6 but pulls in PySide6-Essentials; that is the one there."""
    import importlib.metadata as metadata

    from ptmipf.webui import state

    versions = {"PySide6-Essentials": "6.10.3", "ovito": "3.16.0"}

    def version(name):
        try:
            return versions[name]
        except KeyError:
            raise metadata.PackageNotFoundError(name) from None

    monkeypatch.setattr(metadata, "requires", lambda name: ["PySide6~=6.10.3"])
    monkeypatch.setattr(metadata, "version", version)
    assert state._qt_pairing_check() == (True, "PySide6 6.10.3")


def test_a_renderer_known_to_crash_is_refused_rather_than_asked(monkeypatch):
    """A raised error is survivable; an access violation takes the server with it."""
    from ptmipf import render

    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr("sys.version_info", _version(3, 14))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "3.15.5")
    refusal = render.renderer_refusal()
    assert "Python 3.13" in refusal
    with pytest.raises(RuntimeError, match="3D view is switched off"):
        render.render_result(object(), "nowhere.png")


def test_the_renderer_is_not_refused_where_it_works(monkeypatch):
    from ptmipf import render

    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr("sys.version_info", _version(3, 14))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "3.16.0")
    assert render.renderer_refusal() == ""
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("importlib.metadata.version", lambda name: "3.15.5")
    assert render.renderer_refusal() == ""

"""How large a picture this machine's renderer can actually draw.

OVITO draws the 3D view through the platform's graphics stack, and a size one
machine takes without complaint aborts the process on another: not an
exception that can be caught, but the process gone.  Each size is therefore
drawn in a child process, so a machine that cannot manage one is reported as
a failed size rather than as a killed test run.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("ovito")

#: The sizes the interface itself can ask for: the screen, a 1080p export, a
#: 4K one, and the widest view this machine says it can draw.  The server
#: clamps to that same measurement, so nothing here should ever die.
from ptmipf.renderlimit import max_view_px  # noqa: E402


def _sizes():
    return [(320, 240), (1920, 1080), (3840, 2160), (max_view_px(), 64)]


SIZES = _sizes()

PROBE = """
import sys
from ptmipf.frames import SampleFrame
from ptmipf.structures import get_structure
from ptmipf.webui import rendering
from ptmipf.io import temporary_path
import numpy as np

class Probe:
    positions = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.4, 1.0]])
    structure_types = np.array([1, 1])
    rmsd = np.zeros(2)
    n_atoms = 2
    frame = SampleFrame()
    structures = (get_structure("fcc"),)
    type_codes = {"fcc": 1}
    cell = None

width, height = int(sys.argv[1]), int(sys.argv[2])
with temporary_path(".png") as out:
    rendering.render_scene(Probe(), out, size=(width, height))
    print("drew", width, height, flush=True)
"""


@pytest.mark.parametrize("width,height", SIZES)
def test_the_renderer_survives_the_sizes_the_interface_offers(width, height, renderer):
    outcome = subprocess.run(
        [sys.executable, "-c", PROBE, str(width), str(height)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert outcome.returncode == 0, (
        f"the renderer died on {width}x{height} with return code "
        f"{outcome.returncode}: {outcome.stdout[-2000:]} {outcome.stderr[-2000:]}"
    )


def test_the_measurement_reads_what_the_child_managed(monkeypatch):
    """The child dies part way through on purpose; what it printed still counts."""
    import subprocess

    from ptmipf import renderlimit

    class _Died:
        stdout = "drew 1024\ndrew 2048\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Died())
    monkeypatch.setattr(renderlimit, "_measured", None)
    assert renderlimit.max_view_px() == 2048


def test_a_machine_that_cannot_measure_gets_the_floor(monkeypatch):
    import subprocess

    from ptmipf import renderlimit

    def explode(*args, **kwargs):
        raise OSError("no child processes here")

    monkeypatch.setattr(subprocess, "run", explode)
    monkeypatch.setattr(renderlimit, "_measured", None)
    assert renderlimit.max_view_px() == renderlimit.FLOOR

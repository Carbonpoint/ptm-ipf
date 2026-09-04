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
#: 4K one, and the widest view the server will render.  Anything wider is
#: clamped there, because macOS aborts on a texture past the device limit.
from ptmipf.webui.rendering import MAX_VIEW_PX  # noqa: E402

SIZES = [(320, 240), (1920, 1080), (3840, 2160), (MAX_VIEW_PX, 64)]

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

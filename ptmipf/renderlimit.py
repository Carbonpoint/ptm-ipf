"""How large a picture this machine's renderer can draw, asked once.

The 3D view is drawn into a graphics texture, and a texture larger than the
device allows is not an error that can be caught: macOS fails a Metal
assertion and the process is gone.  The limit differs from machine to machine
(8192 is common, 4096 is not rare, some devices allow far more), so it is
measured here rather than guessed, in a child process, where a crash costs
one child and nothing else.

The measurement is one child that draws a tiny scene at each candidate size in
turn and reports the ones that worked.  It runs at most once per session.
"""

from __future__ import annotations

import subprocess
import sys

__all__ = ["CANDIDATES", "FLOOR", "max_view_px", "main"]

#: The sizes tried, smallest first.  The largest is the widest the interface
#: offers; there is no point measuring past it.
CANDIDATES = (1024, 2048, 4096, 8192, 12000)

#: Used when the measurement itself cannot run.  Every renderer manages this.
FLOOR = 1024

_measured: int | None = None


def max_view_px(timeout: float = 180.0) -> int:
    """The widest 3D view this machine can draw, measured once and remembered."""
    global _measured
    if _measured is None:
        _measured = _measure(timeout)
    return _measured


def _measure(timeout: float) -> int:
    try:
        outcome = subprocess.run(
            [sys.executable, "-m", "ptmipf.renderlimit"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:  # pragma: no cover - a machine that cannot spawn anything
        return FLOOR
    # The child prints one line per size it managed.  It may well die part way
    # through, which is the point: what it printed before that still counts.
    widths = [
        int(line.split()[1])
        for line in outcome.stdout.splitlines()
        if line.startswith("drew ")
    ]
    return max(widths) if widths else FLOOR


def _probe_scene():
    """Two atoms, which is enough to make the renderer allocate its buffer."""
    import numpy as np

    from .frames import SampleFrame
    from .structures import get_structure

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

    return Probe()


def main(argv=None) -> int:
    """The child: draw at each candidate size and say which worked."""
    from .io import temporary_path
    from .webui import rendering

    scene = _probe_scene()
    for width in CANDIDATES:
        try:
            with temporary_path(".png") as out:
                rendering.render_scene(scene, out, size=(width, 64))
        except Exception:
            break
        # Flushed one at a time: the next size may take the process with it.
        print("drew", width, flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point of the child
    raise SystemExit(main())

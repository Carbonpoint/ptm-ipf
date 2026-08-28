"""Local web interface for ptm-ipf.

Start it with ``python -m ptmipf.webui`` (or the ``ptmipf-ui`` script) and
open the printed address in a browser.  The interface wraps the same analysis,
plotting and rendering code as the ``ptmipf`` command line tool, and can print
the equivalent command line for any state it displays.
"""

from __future__ import annotations

__all__ = ["main"]


def __getattr__(name):
    """Import the server lazily; it pulls in matplotlib at import time."""
    if name == "main":
        from .server import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

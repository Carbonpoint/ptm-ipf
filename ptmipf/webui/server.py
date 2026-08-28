"""A local web UI for ptm-ipf, built on the standard library only.

Run it with ``python -m ptmipf.webui`` (or the ``ptmipf-ui`` entry point).
The server binds to 127.0.0.1 by default and only ever reads files below an
explicit root directory, so it can be pointed at a directory of large MD
trajectories without exposing the rest of the file system.

Design notes
------------
* ``http.server`` keeps the package free of web-framework dependencies; the
  API is small enough that a framework would only add weight.
* PTM runs once per (file, structures, cutoff, frame) combination and the
  result is cached; everything else — projection direction, sample frame,
  selections, orbiting the 3D view — reuses the cache and responds in well
  under a second even for millions of atoms.
* Every plot in the UI is produced by the same functions the CLI uses, and
  the UI can emit the equivalent ``ptmipf`` command line, so an interactive
  session can always be turned back into a reproducible script.
"""

from __future__ import annotations

import argparse
import json
import shlex
import tempfile
import threading
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .. import __version__
from ..analysis import DEFAULT_OTHER_COLOR
from ..polefigure import IDEAL_C_OVER_A
from ..structures import DEFAULT_STRUCTURES, STRUCTURES
from . import figures, rendering
from .state import AppState, SelectionUnavailableError

__all__ = ["main", "make_server"]

STATIC_DIR = Path(__file__).parent / "static"
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}
#: File extensions offered by the server-side browser; everything OVITO reads.
_DATA_SUFFIXES = (
    ".xyz", ".extxyz", ".dump", ".lammpstrj", ".data", ".lmp", ".cfg", ".poscar",
    ".cif", ".pdb", ".gsd", ".nc", ".dcd", ".gz",
)


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _selection_active(state: AppState, mode: str) -> np.ndarray | None:
    if mode == "off":
        return None
    return state.selection_mask


def _flag(query: dict, name: str, default: bool = False) -> bool:
    value = query.get(name, [None])[0]
    if value is None:
        return default
    return value not in ("0", "false", "no", "")


def _number(query: dict, name: str, default: float) -> float:
    value = query.get(name, [None])[0]
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        raise ApiError(f"{name} must be a number, got {value!r}") from None


class Handler(BaseHTTPRequestHandler):
    """Routes requests to the JSON API, the figure endpoints and the assets."""

    server_version = f"ptm-ipf/{__version__}"
    protocol_version = "HTTP/1.1"

    # The handler is instantiated per request; shared state lives on the server.
    @property
    def state(self) -> AppState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        if self.server.verbose:  # type: ignore[attr-defined]
            super().log_message(format, *args)

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------
    def _send(self, body: bytes, content_type: str, status: int = 200, download: str | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status: int = 200):
        self._send(json.dumps(payload).encode(), "application/json; charset=utf-8", status)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode())
        except (ValueError, UnicodeDecodeError):
            raise ApiError("request body is not valid JSON") from None

    def _dispatch(self, handler, *args):
        try:
            handler(*args)
        except ApiError as exc:
            self._json({"error": str(exc)}, exc.status)
        except SelectionUnavailableError as exc:
            self._json({"error": str(exc)}, HTTPStatus.NOT_IMPLEMENTED)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except PermissionError as exc:
            self._json({"error": str(exc)}, HTTPStatus.FORBIDDEN)
        except BrokenPipeError:
            pass  # the browser cancelled an in-flight image; nothing to do
        except Exception as exc:  # keep the server alive, report the failure
            traceback.print_exc()
            self._json({"error": f"internal error: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    # ------------------------------------------------------------------
    # GET
    # ------------------------------------------------------------------
    def do_GET(self):  # noqa: N802 - stdlib signature
        url = urlparse(self.path)
        query = parse_qs(url.query)
        routes = {
            "/api/meta": self._get_meta,
            "/api/status": self._get_status,
            "/api/browse": self._get_browse,
            "/api/render": self._get_render,
            "/api/figure/legend": self._get_legend,
            "/api/figure/poles": self._get_poles,
            "/api/figure/ipfdensity": self._get_ipf_density,
            "/api/export": self._get_export,
            "/api/atom": self._get_atom,
            "/api/slicebounds": self._get_slice_bounds,
        }
        if url.path in routes:
            self._dispatch(routes[url.path], query)
        elif url.path == "/" or url.path.startswith("/static/"):
            self._dispatch(self._get_static, url.path)
        else:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _get_static(self, path: str):
        name = "index.html" if path == "/" else Path(path).name
        target = (STATIC_DIR / name).resolve()
        if STATIC_DIR.resolve() not in target.parents or not target.is_file():
            raise ApiError("not found", HTTPStatus.NOT_FOUND)
        content_type = _CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        self._send(target.read_bytes(), content_type)

    def _get_meta(self, query):
        state = self.state
        try:
            from .. import select  # noqa: F401
            has_select = True
        except ImportError:
            has_select = False
        self._json(
            {
                "version": __version__,
                "root": str(state.root),
                "initial_path": self.server.initial_path,  # type: ignore[attr-defined]
                "structures": [
                    {
                        "name": s.name,
                        "description": s.description,
                        "colorable": s.colorable,
                        "default": s.name in DEFAULT_STRUCTURES,
                    }
                    for s in STRUCTURES.values()
                ],
                "defaults": {
                    "rmsd_cutoff": 0.1,
                    "other_color": list(DEFAULT_OTHER_COLOR),
                    "c_over_a": round(IDEAL_C_OVER_A, 6),
                    "poles": ["0001", "10-10", "11-20"],
                },
                "selection_available": has_select,
            }
        )

    def _get_status(self, query):
        self._json(self.state.status())

    def _get_browse(self, query):
        state = self.state
        rel = query.get("path", [""])[0]
        directory = state.resolve(rel or ".")
        if not directory.is_dir():
            raise ApiError(f"not a directory: {rel}")
        entries = []
        for entry in sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                entries.append({"name": entry.name, "dir": True})
            elif entry.suffix.lower() in _DATA_SUFFIXES:
                entries.append({"name": entry.name, "dir": False, "size": entry.stat().st_size})
        relative = "" if directory == state.root else str(directory.relative_to(state.root))
        self._json({"path": relative, "at_root": directory == state.root, "entries": entries})

    # -- images -------------------------------------------------------
    def _result_or_409(self):
        result = self.state.result
        if result is None:
            raise ApiError("no analysis result yet", HTTPStatus.CONFLICT)
        return result

    def _view_options(self, query) -> dict:
        """Decode the viewer options shared by the render and pick endpoints."""
        state = self.state
        result = self._result_or_409()
        options = {
            "azimuth": _number(query, "az", -125.0),
            "elevation": _number(query, "el", 20.0),
            "zoom": _number(query, "zoom", 1.0),
            "size": (
                int(_number(query, "w", 900)),
                int(_number(query, "h", 700)),
            ),
            "hide_other": _flag(query, "hide_other"),
        }
        axis = query.get("slice_axis", [""])[0]
        if axis:
            normal = result.frame.direction(axis)
            low, high = rendering.slice_bounds(result, normal)
            fraction = np.clip(_number(query, "slice_frac", 1.0), 0.0, 1.0)
            options["slice_normal"] = normal
            options["slice_distance"] = low + fraction * (high - low)
        mode = query.get("highlight", ["highlight"])[0]
        options["selection_mode"] = mode
        options["selection"] = _selection_active(state, mode)
        return options

    def _get_render(self, query):
        state = self.state
        with state.lock:
            result = self._result_or_409()
            options = self._view_options(query)
            transparent = _flag(query, "transparent")
            with tempfile.NamedTemporaryFile(suffix=".png") as handle:
                state.ovito.submit(
                    rendering.render_scene,
                    result,
                    handle.name,
                    transparent=transparent,
                    **options,
                ).result()
                body = Path(handle.name).read_bytes()
        download = "ipf_view.png" if _flag(query, "download") else None
        self._send(body, "image/png", download=download)

    def _figure_result(self, query):
        """The full result, or the current selection when ``selection=1``."""
        if _flag(query, "selection"):
            return self.state.selection_subset()
        return self._result_or_409()

    def _get_legend(self, query):
        state = self.state
        with state.lock:
            result = self._result_or_409()
            body = figures.legend_png(result, query.get("structure", [None])[0])
        download = None
        if _flag(query, "download"):
            download = f"ipf_key_{result.direction_label.lower()}.png"
        self._send(body, "image/png", download=download)

    def _get_poles(self, query):
        state = self.state
        with state.lock:
            result = self._figure_result(query)
            poles = [p for p in query.get("poles", ["0001"])[0].split(",") if p.strip()]
            if not poles:
                raise ApiError("at least one pole family is required")
            body = figures.pole_figure_png(
                result,
                poles,
                structure=query.get("structure", [None])[0],
                c_over_a=_number(query, "c_over_a", IDEAL_C_OVER_A),
                mode=query.get("mode", ["density"])[0],
            )
        download = "pole_figures.png" if _flag(query, "download") else None
        self._send(body, "image/png", download=download)

    def _get_ipf_density(self, query):
        state = self.state
        with state.lock:
            result = self._figure_result(query)
            direction = self.state.colour_params.get("direction", "z")
            body = figures.ipf_density_png(
                result, direction, structure=query.get("structure", [None])[0]
            )
        download = "ipf_density.png" if _flag(query, "download") else None
        self._send(body, "image/png", download=download)

    def _get_export(self, query):
        from ..io import write_result

        state = self.state
        fmt = query.get("format", ["extxyz"])[0]
        suffix = ".dump" if fmt in ("dump", "lammps-dump") else ".xyz"
        with state.lock:
            if _flag(query, "selection"):
                result = state.selection_subset()
                stem = "ipf_selection"
            else:
                result = self._result_or_409()
                stem = "ipf_colored"
            with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
                write_result(result, handle.name, fmt)
                body = Path(handle.name).read_bytes()
        self._send(body, "application/octet-stream", download=stem + suffix)

    def _get_atom(self, query):
        index = int(_number(query, "index", -1))
        self._json(self.state.atom_info(index))

    def _get_slice_bounds(self, query):
        result = self._result_or_409()
        axis = query.get("axis", ["z"])[0]
        normal = result.frame.direction(axis)
        low, high = rendering.slice_bounds(result, normal)
        self._json({"min": round(low, 3), "max": round(high, 3)})

    # ------------------------------------------------------------------
    # POST
    # ------------------------------------------------------------------
    def do_POST(self):  # noqa: N802 - stdlib signature
        url = urlparse(self.path)
        routes = {
            "/api/analyse": self._post_analyse,
            "/api/selection": self._post_selection,
            "/api/pick": self._post_pick,
            "/api/command": self._post_command,
        }
        if url.path in routes:
            self._dispatch(routes[url.path])
        else:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _post_analyse(self):
        body = self._body()
        if not body.get("path"):
            raise ApiError("a file path is required")
        self._json(self.state.submit_analysis(body))

    def _post_selection(self):
        body = self._body()
        outcome = self.state.apply_selection(
            body.get("criteria") or [], body.get("mode", "and")
        )
        self._json(outcome)

    def _post_pick(self):
        body = self._body()
        state = self.state
        with state.lock:
            result = self._result_or_409()
            query = {
                k: ["1" if v is True else "0" if v is False else str(v)]
                for k, v in body.items()
                if not isinstance(v, (dict, list))
            }
            options = self._view_options(query)
            options.pop("size")
            index = rendering.pick_atom(
                result,
                float(body.get("x", -1)),
                float(body.get("y", -1)),
                options.pop("azimuth"),
                options.pop("elevation"),
                options.pop("zoom"),
                (int(body.get("w", 900)), int(body.get("h", 700))),
                **options,
            )
            if index is None:
                self._json({"atom": None})
            else:
                self._json({"atom": state.atom_info(index)})

    def _post_command(self):
        self._json({"command": build_command(self.state, self._body())})


def build_command(state: AppState, ui: dict) -> str:
    """The ``ptmipf`` command line reproducing the current session.

    Options equal to the CLI defaults are omitted, so the command stays as
    short as the one an experienced user would type.
    """
    with state.lock:
        analysis = state.analysis_params
        colour = state.colour_params
        if analysis is None:
            raise ApiError("run an analysis first", HTTPStatus.CONFLICT)
        parts = ["ptmipf", shlex.quote(analysis["path"])]
        if tuple(analysis["structures"]) != tuple(DEFAULT_STRUCTURES):
            parts.append("--structures " + ",".join(analysis["structures"]))
        if analysis["rmsd_cutoff"] != 0.1:
            parts.append(f"--rmsd-cutoff {analysis['rmsd_cutoff']:g}")
        if analysis["frame_index"]:
            parts.append(f"--frame {analysis['frame_index']}")
        for axis, vector in colour.get("axes", {}).items():
            parts.append(f"--{axis} {shlex.quote(str(vector))}")
        if str(colour.get("direction", "z")) != "z":
            parts.append(f"--direction {shlex.quote(str(colour['direction']))}")
        if colour.get("color_only"):
            parts.append("--color-only " + ",".join(colour["color_only"]))
        if tuple(colour.get("other_color", DEFAULT_OTHER_COLOR)) != DEFAULT_OTHER_COLOR:
            parts.append("--other-color " + ",".join(f"{c:g}" for c in colour["other_color"]))

        parts.append("-o ipf_colored.xyz")
        parts.append("--legend ipf_key.png")
        for pole in ui.get("poles") or []:
            parts.append(f"--pole-figure {shlex.quote(str(pole))}")
        if ui.get("poles"):
            parts.append("--pole-figure-file pole_figures.png")
            if ui.get("pole_mode") and ui["pole_mode"] != "density":
                parts.append(f"--pole-figure-mode {ui['pole_mode']}")
            if ui.get("pole_structure"):
                parts.append(f"--pole-figure-structure {ui['pole_structure']}")
            c_over_a = ui.get("c_over_a")
            if c_over_a and abs(float(c_over_a) - IDEAL_C_OVER_A) > 1e-4:
                parts.append(f"--c-over-a {float(c_over_a):g}")
        parts.append("--ipf-density ipf_density.png")
        parts.append("--render ipf_view.png")
        size = ui.get("render_size")
        if size and tuple(size) != (800, 600):
            parts.append(f"--render-size {int(size[0])}x{int(size[1])}")
        if ui.get("hide_other"):
            parts.append("--hide-other")
        if ui.get("slice_axis"):
            parts.append(f"--slice {shlex.quote(str(ui['slice_axis']))}")
            if ui.get("slice_distance") is not None:
                parts.append(f"--slice-distance {float(ui['slice_distance']):g}")
        note = ""
        if state.selection_mask is not None:
            flags, exact = _selection_flags(state.selection_criteria, state.selection_mode)
            if exact:
                parts.extend(flags)
                parts.append("--from-selection")
                parts.append("--selection-output ipf_selection.xyz")
            else:
                note = (
                    "\n# note: the active selection uses options (per-criterion inversion"
                    "\n# or a quaternion reference) the CLI cannot express; see the"
                    "\n# selection options in `ptmipf --help` for the closest equivalent."
                )
        return " \\\n    ".join(parts) + note


def _selection_flags(criteria: list, mode: str) -> tuple[list[str], bool]:
    """Translate the UI's selection criteria into ``ptmipf`` options.

    Returns the flag list and whether the translation is exact; per-criterion
    inversion and quaternion references have no CLI counterpart.
    """
    flags: list[str] = []
    structure = None
    exact = True
    rmsd_below = rmsd_above = grain = None
    for criterion in criteria:
        if criterion.get("invert"):
            exact = False
        kind = criterion.get("kind")
        if kind == "structure":
            flags.append("--select-structure " + ",".join(criterion["structures"]))
        elif kind == "type":
            flags.append("--select-type " + ",".join(str(t) for t in criterion["types"]))
        elif kind == "rmsd":
            if criterion.get("max") is not None:
                exact = exact and rmsd_below is None
                rmsd_below = criterion["max"]
            if criterion.get("min") is not None:
                exact = exact and rmsd_above is None
                rmsd_above = criterion["min"]
        elif kind == "region":
            low = "" if criterion.get("min") is None else f"{float(criterion['min']):g}"
            high = "" if criterion.get("max") is None else f"{float(criterion['max']):g}"
            flags.append(
                "--select-region " + shlex.quote(f"{criterion.get('axis', 'z')}|{low}|{high}")
            )
        elif kind == "ipf":
            spec = (
                f"{criterion['crystal']}|{criterion['sample']}"
                f"|{float(criterion.get('tolerance', 10)):g}"
            )
            flags.append("--select-orientation " + shlex.quote(spec))
            structure = criterion.get("structure") or structure
        elif kind == "misorientation":
            reference = criterion.get("reference")
            if not isinstance(reference, dict) or "atom" not in reference or grain is not None:
                exact = False
                continue
            grain = int(reference["atom"])
            flags.append(f"--select-grain {grain}")
            tolerance = float(criterion.get("tolerance", 5))
            if tolerance != 10.0:
                flags.append(f"--select-grain-tolerance {tolerance:g}")
            structure = criterion.get("structure") or structure
    if rmsd_below is not None:
        flags.append(f"--select-rmsd-below {float(rmsd_below):g}")
    if rmsd_above is not None:
        flags.append(f"--select-rmsd-above {float(rmsd_above):g}")
    if structure:
        flags.append(f"--orientation-structure {structure}")
    if mode != "and":
        flags.append(f"--select-mode {mode}")
    return flags, exact


def make_server(root, host: str = "127.0.0.1", port: int = 0, initial_path: str = ""):
    """Build (but do not start) the HTTP server; used by ``main`` and the tests."""
    import matplotlib

    matplotlib.use("Agg")
    server = ThreadingHTTPServer((host, port), Handler)
    server.state = AppState(Path(root))  # type: ignore[attr-defined]
    server.initial_path = initial_path  # type: ignore[attr-defined]
    server.verbose = False  # type: ignore[attr-defined]
    return server


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ptmipf-ui",
        description="Local web interface for ptm-ipf: IPF maps, pole figures and "
        "atom selections in the browser.",
    )
    parser.add_argument(
        "input", nargs="?", help="configuration file to preload into the interface"
    )
    parser.add_argument(
        "--root",
        help="directory the file browser is confined to (default: the input file's "
        "directory, or the current directory)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8465, help="port (default: %(default)s)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("-v", "--verbose", action="store_true", help="log every request")
    args = parser.parse_args(argv)

    root = args.root
    initial = ""
    if args.input:
        input_path = Path(args.input).resolve()
        if not input_path.is_file():
            parser.error(f"no such file: {args.input}")
        root = root or str(input_path.parent)
        initial = str(input_path)
    root = Path(root or ".").resolve()

    server = make_server(root, host=args.host, port=args.port, initial_path=initial)
    server.verbose = args.verbose  # type: ignore[attr-defined]
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"ptm-ipf web UI serving files under {root}")
    print(f"listening on {url}  (Ctrl-C to quit)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        server.server_close()
    return 0

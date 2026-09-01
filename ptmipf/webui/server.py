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
  result is cached; everything else (projection direction, sample frame,
  selections, orbiting the 3D view) reuses the cache and responds in well
  under a second even for millions of atoms.
* Every plot in the UI is produced by the same functions the CLI uses, and
  the UI can emit the equivalent ``ptmipf`` command line, so an interactive
  session can always be turned back into a reproducible script.
"""

from __future__ import annotations

import argparse
import json
import shlex
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
from ..colormap import PLOT_COLORMAPS
from ..io import temporary_path
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


#: Column names that usually hold what, lower case, most specific first.
_ORIENTATION_HINTS = ("orientation", "quat", "q")
_STRUCTURE_HINTS = ("structure type", "structuretype", "structure", "phase", "ptm")
_RMSD_HINTS = ("rmsd", "interatomic distance")


def _guess_orientation_columns(columns: list) -> dict:
    """A first guess at the column mapping, for the interface to show.

    Only a guess: nothing in a file says which convention its quaternions
    follow, so the interface presents this as a starting point and leaves the
    choice with the person who made the file.
    """
    guess = {"quaternion": [], "structure_type": "", "rmsd": ""}
    by_name = {c["name"].lower(): c for c in columns}

    for hint in _ORIENTATION_HINTS:
        four = [c for c in columns if c["components"] == 4 and hint in c["name"].lower()]
        if four:
            guess["quaternion"] = [four[0]["name"]]
            break
        scalars = sorted(
            c["name"] for c in columns if c["components"] == 1 and hint in c["name"].lower()
        )
        if len(scalars) == 4:
            guess["quaternion"] = scalars
            break
    if not guess["quaternion"]:
        four = [c for c in columns if c["components"] == 4]
        if len(four) == 1:
            guess["quaternion"] = [four[0]["name"]]

    for hint in _STRUCTURE_HINTS:
        match = next((n for n in by_name if hint in n), None)
        if match:
            guess["structure_type"] = by_name[match]["name"]
            break
    for hint in _RMSD_HINTS:
        match = next((n for n in by_name if hint in n), None)
        if match:
            guess["rmsd"] = by_name[match]["name"]
            break
    return guess


def _export_keys(result, query: dict):
    """Colour-coding columns for an export request.

    The UI asks for x, y and z by default so that the three IPF maps can be
    switched inside OVITO; ``directions=`` overrides the set and
    ``keys=0`` leaves them out.
    """
    if not _flag(query, "keys", True):
        return {}, None, None
    from ..colormap import color_keys

    # Semicolons, not commas: a direction may itself be a vector like 1,1,0.
    raw = query.get("directions", ["x;y;z"])[0]
    directions = [d.strip() for d in raw.split(";") if d.strip()]
    gradient = query.get("gradient", [""])[0] or None
    return color_keys(result, directions or ["x", "y", "z"], gradient=gradient)


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
    def _send(
        self,
        body: bytes,
        content_type: str,
        status: int = 200,
        download: str | None = None,
        extra_headers: dict | None = None,
    ):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
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
            "/api/figure/flatmap": self._get_flat_map,
            "/api/export": self._get_export,
            "/api/colormap": self._get_colormap,
            "/api/diagnostics": self._get_diagnostics,
            "/api/columns": self._get_columns,
            "/api/examples": self._get_examples,
            "/api/atom": self._get_atom,
            "/api/slicebounds": self._get_slice_bounds,
        }
        if url.path in routes:
            self._dispatch(routes[url.path], query)
        elif url.path in ("/", "/examples") or url.path.startswith("/static/"):
            self._dispatch(self._get_static, url.path)
        else:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _get_static(self, path: str):
        name = {"/": "index.html", "/examples": "examples.html"}.get(path) or Path(path).name
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
                "colormaps": list(PLOT_COLORMAPS),
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
    def _result_or_409(self, query=None):
        """The cached result, optionally with the grain boundaries filled in."""
        radius = _number(query, "fill_radius", 0.0) if query else 0.0
        minimum = int(_number(query, "fill_min_neighbours", 3)) if query else 3
        result = self.state.view_result(radius or None, minimum)
        if result is None:
            raise ApiError("no analysis result yet", HTTPStatus.CONFLICT)
        return result

    def _view_options(self, query) -> dict:
        """Decode the viewer options shared by the render and pick endpoints."""
        state = self.state
        result = self._result_or_409(query)
        options = {
            "azimuth": _number(query, "az", -125.0),
            "elevation": _number(query, "el", 20.0),
            "zoom": _number(query, "zoom", 1.0),
            "size": (
                int(_number(query, "w", 900)),
                int(_number(query, "h", 700)),
            ),
            "hide_other": _flag(query, "hide_other"),
            "tripod": _flag(query, "tripod"),
        }
        axis = query.get("slice_axis", [""])[0]
        if axis:
            normal = result.frame.direction(axis)
            low, high = rendering.slice_bounds(result, normal)
            fraction = np.clip(_number(query, "slice_frac", 1.0), 0.0, 1.0)
            options["slice_normal"] = normal
            options["slice_distance"] = low + fraction * (high - low)
            options["slice_width"] = max(0.0, _number(query, "slice_width", 0.0))
        mode = query.get("highlight", ["highlight"])[0]
        options["selection_mode"] = mode
        options["selection"] = _selection_active(state, mode)
        return options

    def _get_render(self, query):
        state = self.state
        with state.lock:
            result = self._result_or_409(query)
            options = self._view_options(query)
            transparent = _flag(query, "transparent")
            with temporary_path(".png") as scratch:
                state.ovito.submit(
                    rendering.render_scene,
                    result,
                    scratch,
                    transparent=transparent,
                    **options,
                ).result()
                body = Path(scratch).read_bytes()
        download = "ipf_view.png" if _flag(query, "download") else None
        self._send(body, "image/png", download=download)

    def _figure_result(self, query):
        """The full result, or the current selection when ``selection=1``.

        Filling happens before any subsetting: an atom needs its neighbours to
        borrow an orientation from, and a selection may have removed them.
        """
        result = self._result_or_409(query)
        if _flag(query, "selection"):
            from .state import subset_result

            mask = self.state.selection_mask
            if mask is None:
                raise ApiError("no selection has been applied")
            return subset_result(result, mask)
        return result

    def _get_legend(self, query):
        state = self.state
        with state.lock:
            result = self._result_or_409(query)
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
                smoothing=max(0.0, _number(query, "smoothing", 0.0)),
                cmap=self._colormap(query, "viridis"),
            )
        download = "pole_figures.png" if _flag(query, "download") else None
        self._send(body, "image/png", download=download)

    def _get_ipf_density(self, query):
        state = self.state
        with state.lock:
            result = self._figure_result(query)
            direction = self.state.colour_params.get("direction", "z")
            body = figures.ipf_density_png(
                result,
                direction,
                structure=query.get("structure", [None])[0],
                smoothing=max(0.0, _number(query, "smoothing", 0.0)),
                cmap=self._colormap(query, "magma"),
            )
        download = "ipf_density.png" if _flag(query, "download") else None
        self._send(body, "image/png", download=download)

    def _get_flat_map(self, query):
        """A flat, EBSD-style orientation map of a section."""
        state = self.state
        with state.lock:
            result = self._figure_result(query)
            body, info = figures.flat_map_png(
                result,
                view=query.get("view", ["z"])[0] or "z",
                slab_width=_number(query, "slab_width", 10.0),
                pixel_size=_number(query, "pixel_size", 0.5),
                boundary_angle=_number(query, "boundary_angle", 5.0),
                fill_unindexed=not _flag(query, "raw"),
                structure=query.get("structure", [None])[0],
            )
        download = "ipf_flat_map.png" if _flag(query, "download") else None
        self._send(body, "image/png", download=download, extra_headers={
            "X-Grain-Count": str(info["n_grains"]),
            "X-Map-Size": f"{info['columns']}x{info['rows']}",
        })

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
            keys, _, info = _export_keys(result, query)
            with temporary_path(suffix) as scratch:
                write_result(result, scratch, fmt, keys=keys)
                body = Path(scratch).read_bytes()
        headers = {}
        if info:
            headers["X-Color-Columns"] = ",".join(info["directions"])
            headers["X-Color-Entries"] = str(info["entries"])
        self._send(
            body, "application/octet-stream", download=stem + suffix, extra_headers=headers
        )

    def _get_colormap(self, query):
        """The colour bar the exported colour-coding columns index."""
        from ..colormap import write_color_map

        state = self.state
        with state.lock:
            result = self._result_or_409()
            _, palette, info = _export_keys(result, query)
            if palette is None:
                raise ApiError(
                    "a built-in OVITO colour bar was requested, so there is no "
                    "custom colour map to download"
                )
            with temporary_path(".png") as scratch:
                write_color_map(palette, scratch, height=16)
                body = Path(scratch).read_bytes()
        self._send(
            body,
            "image/png",
            download="ipf_colormap.png",
            extra_headers={
                "X-Color-Entries": str(info["entries"]),
                "X-Color-Error": f"{info['max_error']:.5f}",
                "X-Color-Columns": ",".join(info["directions"]),
            },
        )

    def _get_diagnostics(self, query):
        """What this installation can and cannot do, for a first-run check.

        The 3D view is rendered by OVITO on the server, so when it stays blank
        the answer is almost always here rather than in the browser.
        """
        self._json(self.state.diagnostics())

    def _get_columns(self, query):
        """The per-atom columns a file carries, for the orientation import.

        Reads the file but runs nothing on it, so a mapping is chosen from what
        is really there rather than from what the column was called last time.
        """
        from ..analysis import list_columns

        state = self.state
        path = query.get("path", [""])[0]
        if not path:
            raise ApiError("a file path is required")
        target = state.resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f"no such file: {path}")
        frame_index = int(_number(query, "frame", 0))
        info = state.ovito.submit(list_columns, str(target), frame_index).result()
        info["path"] = str(target)
        info["guess"] = _guess_orientation_columns(info["columns"])
        self._json(info)

    def _get_examples(self, query):
        """The starter examples on offer, and whether atomsk is here to build them."""
        from ..examples import DEFAULTS
        from ..lammps import estimate_cost
        from ..polycrystal import ATOMSK_HELP, find_atomsk
        from ..potentials import POTENTIALS

        atomsk = find_atomsk()
        entries = []
        for name, spec in DEFAULTS.items():
            potential = POTENTIALS[spec.element]
            # Close enough to size the promise before anything is built.
            per_cell = 4 if potential.structure == "fcc" else 2
            atoms = int(0.97 * per_cell * (spec.box / potential.a0) ** 3)
            steps = int(spec.strain / spec.strain_rate / 0.002) + 2000
            entries.append(
                {
                    "name": name,
                    "element": spec.element,
                    "structure": potential.structure,
                    "citation": potential.citation,
                    "url": potential.entry_url,
                    "a0": potential.a0,
                    "atoms_per_cell": per_cell,
                    "spec": {
                        "element": spec.element,
                        "box": spec.box,
                        "n_grains": spec.n_grains,
                        "strain": spec.strain,
                        "strain_rate": spec.strain_rate,
                        "temperature": spec.temperature,
                        "seed": spec.seed,
                    },
                    "estimate": {
                        "n_atoms": atoms,
                        **{
                            k: (round(v, 2) if isinstance(v, float) else v)
                            for k, v in estimate_cost(atoms, steps).items()
                        },
                    },
                }
            )
        self._json(
            {
                "examples": entries,
                "root": str(self.state.root),
                "atomsk": atomsk,
                "atomsk_help": ATOMSK_HELP,
            }
        )

    def _post_build_example(self):
        """Build one example under the served root.

        Synchronous: the download is under a megabyte and atomsk builds a cell
        this size in a second or two, so a job queue would be more machinery
        than the wait deserves.
        """
        from ..examples import ExampleSpec, build_example

        body = self._body()
        fields = {
            k: v
            for k, v in body.items()
            if k in ExampleSpec.__dataclass_fields__ and v is not None
        }
        try:
            report = self.state.ovito.submit(
                build_example, self.state.root, ExampleSpec(**fields)
            ).result()
        except (RuntimeError, ValueError) as exc:
            raise ApiError(str(exc)) from None
        report["relative_xyz"] = str(
            Path(report["relative"]) / next(
                (f for f in report["files"] if f.endswith(".xyz")), "structure.xyz"
            )
        )
        self._json(report)

    def _colormap(self, query, default: str):
        """The colour map a figure request asks for.

        ``cmap=custom`` means the one uploaded in this session, which is held
        as an array rather than a file so nothing an upload contains is ever
        written to disk.
        """
        from ..colormap import load_colormap

        name = query.get("cmap", [default])[0] or default
        if name == "custom":
            table = self.state.custom_colormap
            if table is None:
                raise ApiError("no colour map has been uploaded in this session")
            return load_colormap(table)
        try:
            return load_colormap(name)
        except ValueError as exc:
            raise ApiError(str(exc)) from None

    def _post_colormap(self):
        """Take a colour map uploaded from the browser, as base64.

        An image strip or a text table of RGB triples.  It is parsed here and
        kept as an array in memory: the server never writes an uploaded file
        anywhere, and it lasts as long as the session does.
        """
        import base64
        import binascii

        from ..colormap import read_colormap_image, read_colormap_table

        body = self._body()
        try:
            raw = base64.b64decode(body.get("data", ""), validate=True)
        except (binascii.Error, ValueError):
            raise ApiError("the uploaded colour map could not be decoded") from None
        if not raw:
            raise ApiError("the uploaded colour map is empty")
        if len(raw) > 8 << 20:
            raise ApiError("that file is far too large to be a colour map")

        name = str(body.get("name") or "custom")
        reader = read_colormap_image if raw[:8] in (b"\x89PNG\r\n\x1a\n",) or raw[:2] in (
            b"\xff\xd8",
            b"BM",
        ) else read_colormap_table
        try:
            table = reader(raw)
        except Exception as exc:
            # A wrong guess at the format is the likely cause, so try the other.
            other = read_colormap_table if reader is read_colormap_image else read_colormap_image
            try:
                table = other(raw)
            except Exception:
                raise ApiError(f"{name} is not a colour map this can read: {exc}") from None

        with self.state.lock:
            self.state.custom_colormap = table
        self._json({"name": name, "entries": int(len(table))})

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
            "/api/command/parse": self._post_parse_command,
            "/api/examples/build": self._post_build_example,
            "/api/colormap/upload": self._post_colormap,
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
        """The equivalent command line, in both the wrapped and the one-line form.

        The wrapped form is what belongs in a script; the one-line form is what
        survives being pasted into a shell that does not honour backslash
        continuations, PowerShell above all.  The explanatory note is kept out
        of the one-line form, where a ``#`` would comment out everything after
        it.
        """
        parts, note = build_command(self.state, self._body())
        wrapped = " \\\n    ".join(parts)
        self._json(
            {
                "command": wrapped + note,
                "one_line": " ".join(parts),
                "note": note.strip(),
            }
        )

    def _post_parse_command(self):
        """Turn a saved command line back into the settings the interface holds.

        Parsing goes through the real ``ptmipf`` parser rather than a second
        implementation here, so anything the CLI accepts can be imported and
        the two can never drift apart.
        """
        self._json(parse_command(self._body().get("command", "")))


def build_command(state: AppState, ui: dict) -> tuple[list[str], str]:
    """The ``ptmipf`` command line reproducing the current session.

    Returns the argument list and an explanatory note, so the caller can join
    them with continuations for a script or with spaces for a paste.  Options
    equal to the CLI defaults are omitted, so the command stays as short as the
    one an experienced user would type.
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

        radius = ui.get("fill_radius")
        if radius:
            parts.append(f"--fill-boundaries {float(radius):g}")
            neighbours = int(ui.get("fill_min_neighbours") or 3)
            if neighbours != 3:
                parts.append(f"--fill-min-neighbours {neighbours}")
        directions = [str(d) for d in (ui.get("export_directions") or [])]
        if directions != ["x", "y", "z"]:  # the CLI default, so worth no flags
            for spec in directions:
                parts.append(f"--export-direction {shlex.quote(spec)}")
            if not directions:
                parts.append("--no-export-directions")
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
            if ui.get("slice_width"):
                parts.append(f"--slice-width {float(ui['slice_width']):g}")
        if ui.get("view"):
            parts.append(f"--view {shlex.quote(str(ui['view']))}")
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
        return parts, note


def parse_command(text: str) -> dict:
    """Read a ``ptmipf`` command line back into the interface's settings.

    Accepts what the command dialog offers: the wrapped form with its
    backslash continuations, the one-line form, and either with or without the
    leading program name.  Comment lines are dropped, so a command saved with
    its explanatory note can be imported unchanged.
    """
    from ..cli import build_parser

    cleaned = []
    for line in (text or "").replace("\\\n", " ").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            cleaned.append(stripped)
    joined = " ".join(cleaned).strip()
    if not joined:
        raise ApiError("paste a ptmipf command line first")
    try:
        tokens = shlex.split(joined)
    except ValueError as exc:
        raise ApiError(f"cannot read the command: {exc}") from None
    if tokens and Path(tokens[0]).stem in ("ptmipf", "ptmipf-ui", "cli"):
        tokens = tokens[1:]
    if tokens and tokens[0] in ("-m", "python", "python3"):
        raise ApiError(
            "paste the ptmipf command itself, without the python interpreter in front"
        )

    parser = build_parser()
    # argparse exits the process on a bad command line; a pasted one is data.
    def _fail(message):
        raise ApiError(f"cannot read the command: {message}")

    parser.error = _fail  # type: ignore[method-assign]
    try:
        args = parser.parse_args(tokens)
    except SystemExit:
        raise ApiError("cannot read the command; check it against `ptmipf --help`") from None

    axes = {name: getattr(args, name) for name in ("rd", "td", "nd", "ed")}
    return {
        "analysis": {
            "path": args.input or "",
            "structures": [s for s in (args.structures or "").split(",") if s],
            "rmsd_cutoff": args.rmsd_cutoff,
            "frame_index": args.frame,
        },
        "colour": {
            "direction": args.direction,
            "axes": {k: v for k, v in axes.items() if v},
            "color_only": [s for s in (args.color_only or "").split(",") if s],
            "other_color": args.other_color,
        },
        "ui": {
            "poles": list(args.pole_figures),
            "pole_mode": args.pole_figure_mode,
            "pole_structure": args.pole_figure_structure or "",
            "c_over_a": args.c_over_a,
            "hide_other": bool(args.hide_other),
            "slice_axis": args.slice_normal or "",
            "slice_width": args.slice_width,
            "view": args.view or "",
            "fill_radius": args.fill_boundaries,
            "fill_min_neighbours": args.fill_min_neighbours,
            "export_directions": list(args.export_direction or []),
            "flat_map": {
                "pixel_size": args.pixel_size,
                "slab_width": args.slice_width or None,
                "view": args.view or "",
            },
        },
    }


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


def _print_diagnostics() -> int:
    """``ptmipf-ui --check``: the same probe the interface runs, on the terminal."""
    import matplotlib

    matplotlib.use("Agg")
    from .state import _probe_environment

    report = _probe_environment()
    print(f"ptm-ipf {report['ptmipf']} on Python {report['python']}, {report['platform']}")
    for check in report["checks"]:
        mark = "ok  " if check["ok"] else "FAIL"
        detail = f"  {check['detail']}" if check["detail"] else ""
        print(f"  [{mark}] {check['name']}{detail}")
    if report["ok"]:
        print("The 3D view will work here.")
        return 0
    print(
        "The interface will start, but the 3D view will stay empty.  The plots, the "
        "flat orientation map and the exports do not need a renderer and still work."
    )
    return 1


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
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether this installation can render the 3D view, then exit",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log every request")
    args = parser.parse_args(argv)

    if args.check:
        return _print_diagnostics()

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

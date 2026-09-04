"""Trajectory series: finding the frames, and rendering them in a batch.

A trajectory arrives in one of two shapes.  Either every frame is its own
file and the frame number sits in the name (``dump_0.cfg``, ``dump_100.cfg``,
``dump_120.cfg``), or one file holds many frames.  :func:`detect_series`
recognises both from the file the interface has open, so the user can step
through the frames with the analysis settings they have, and the batch
renderer can turn any range of them into stills or a movie.

The batch runs in its own thread but does its PTM and OVITO rendering on the
interface's single OVITO worker, one frame at a time, so the interactive view
keeps working (slowly) while a series renders.  The batch uses the analysis
and colouring the interface currently shows; a selection is left out because
a selection is a set of atom indices of one frame and means nothing on the
others.
"""

from __future__ import annotations

import re
import shutil
import threading
import time
from pathlib import Path

from .. import animate
from . import rendering
from .state import derive_result, run_ptm

__all__ = ["OUTPUTS", "content_type_of", "detect_series", "start_series"]

#: The views a series can produce, with the stills they can be written as.
#: The 3D view is an OVITO raster; the plots come from matplotlib and can be
#: vector too.  Movies are built from PNG frames of any of them.
OUTPUTS = {
    "view": {"label": "3D IPF map", "formats": ("png",)},
    "ipfmap": {"label": "IPF map", "formats": ("png", "svg")},
    "poles": {"label": "Pole figures", "formats": ("png", "svg")},
    "density": {"label": "IPF density", "formats": ("png", "svg")},
    "legend": {"label": "Colour key", "formats": ("png", "svg")},
}
MOVIE_FORMATS = ("gif", "mp4")
_CONTENT_TYPES = {
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
}
_NUMBER = re.compile(r"(\d+)(?!.*\d)")


def content_type_of(path) -> str:
    return _CONTENT_TYPES.get(Path(path).suffix.lower(), "application/octet-stream")


def _relative(state, path: Path) -> str:
    try:
        return str(path.relative_to(state.root))
    except ValueError:
        return str(path)


def detect_series(state, target: Path) -> dict:
    """The frames *target* belongs to.

    Sibling files whose names differ from *target* only in the last run of
    digits form a file series, sorted by that number (so 100 comes after 20).
    Failing that, a file with several frames inside is a series of its own.
    Runs on the OVITO worker because looking inside the file needs OVITO.
    """
    target = Path(target)
    match = _NUMBER.search(target.stem)
    if match is not None:
        head, tail = target.stem[: match.start()], target.stem[match.end() :]
        pattern = re.compile(
            "^" + re.escape(head) + r"(\d+)" + re.escape(tail + target.suffix) + "$"
        )
        found = []
        for sibling in target.parent.iterdir():
            hit = pattern.match(sibling.name)
            if hit and sibling.is_file():
                found.append((int(hit.group(1)), sibling))
        found.sort(key=lambda pair: (pair[0], pair[1].name))
        if len(found) > 1:
            items = [
                {
                    "path": _relative(state, path),
                    "label": path.stem,
                    "step": number,
                    "frame_index": 0,
                }
                for number, path in found
            ]
            current = next(
                (i for i, (_, path) in enumerate(found) if path == target), 0
            )
            stem = head.rstrip("_-. ") or target.stem
            return {"kind": "files", "items": items, "current": current, "stem": stem}
    from ..analysis import list_columns

    n_frames = int(list_columns(target)["n_frames"])
    if n_frames > 1:
        items = [
            {
                "path": _relative(state, target),
                "label": f"{target.stem} frame {i}",
                "step": i,
                "frame_index": i,
            }
            for i in range(n_frames)
        ]
        return {"kind": "frames", "items": items, "current": 0, "stem": target.stem}
    return {"kind": "none", "items": [], "current": 0, "stem": target.stem}


def _query(spec) -> dict:
    """A JSON object of scalars as the ``parse_qs`` shape the figure code reads."""
    out = {}
    for key, value in (spec or {}).items():
        if value is None or value is False or value == "":
            continue
        out[str(key)] = ["1" if value is True else str(value)]
    out.pop("selection", None)  # a selection belongs to one frame only
    return out


def _wanted(spec) -> tuple[dict, dict]:
    """Split ``kind:ext`` output names into stills and movies per kind."""
    stills: dict[str, set] = {}
    movies: dict[str, set] = {}
    for name in spec or ():
        kind, _, ext = str(name).partition(":")
        ext = (ext or "png").lower()
        if kind not in OUTPUTS:
            raise ValueError(f"unknown output {kind!r}")
        if ext in MOVIE_FORMATS:
            movies.setdefault(kind, set()).add(ext)
        elif ext in OUTPUTS[kind]["formats"]:
            stills.setdefault(kind, set()).add(ext)
        else:
            raise ValueError(f"{OUTPUTS[kind]['label']} cannot be written as {ext}")
    if not stills and not movies:
        raise ValueError("choose at least one output")
    return stills, movies


class SeriesJob(threading.Thread):
    """Renders a range of frames with the interface's current settings."""

    def __init__(self, state, series: dict, body: dict):
        super().__init__(name="ptmipf-series", daemon=True)
        self.state = state
        self.series = series
        items = series["items"]
        start = int(body.get("start", 0))
        stop = int(body.get("stop", len(items) - 1))
        step = max(1, int(body.get("step", 1)))
        if not items:
            raise ValueError("this file is not part of a series")
        if not 0 <= start < len(items) or not 0 <= stop < len(items) or stop < start:
            raise ValueError("the frame range is outside the series")
        self.items = items[start : stop + 1 : step]
        self.stills, self.movies = _wanted(body.get("outputs"))
        # Refuse an encoder we do not have now, before a quarter of an hour of
        # matching is spent on frames that could never be joined up.
        for formats in self.movies.values():
            for ext in sorted(formats):
                missing = animate.video_support(ext)
                if missing:
                    raise ValueError(missing)
        self.seconds = float(body.get("seconds_per_frame", 0.5))
        if not 0.01 <= self.seconds <= 60:
            raise ValueError("seconds per frame must be between 0.01 and 60")
        self.stamp = bool(body.get("label", True))
        self.queries = {kind: _query(body.get(f"{kind}_query")) for kind in OUTPUTS}
        out = body.get("out_dir") or f"{series['stem']}_series"
        self.out_dir = state.resolve(str(out))
        with state.lock:
            if state.analysis_params is None:
                raise ValueError("analyse a frame first; the series uses those settings")
            self.analysis = dict(state.analysis_params)
            self.colour = dict(state.colour_params)
        if self.analysis.get("columns") and not any(
            state.resolve(item["path"]) == state.resolve(self.analysis["path"])
            for item in self.items
        ):
            # Orientation columns were mapped for another file; the frames of
            # this series are matched with PTM like any other file.
            self.analysis["columns"] = None
        self.cancel = threading.Event()
        self.started = time.time()
        self.finished: float | None = None
        self.state_name = "running"
        self.item = 0
        self.stage = ""
        self.files: list[str] = []
        self.error = ""
        self.notes: list[str] = []
        self.seconds_per_item: float | None = None

    # -- status -----------------------------------------------------------
    def status(self) -> dict:
        n = len(self.items)
        done = self.item if self.state_name == "running" else n
        return {
            "state": self.state_name,
            "item": self.item,
            "n_items": n,
            "label": self.items[min(self.item, n - 1)]["label"] if n else "",
            "stage": self.stage,
            "progress": (done / n) if n else 1.0,
            "files": list(self.files),
            "error": self.error,
            "notes": list(self.notes),
            "elapsed": round((self.finished or time.time()) - self.started, 1),
            "seconds_per_item": self.seconds_per_item,
            "out_dir": _relative(self.state, self.out_dir),
        }

    # -- work -------------------------------------------------------------
    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:  # surfaced through /api/series/status
            self.state_name = "error"
            self.error = str(exc)
        finally:
            self.finished = time.time()
            shutil.rmtree(self.out_dir / "_frames", ignore_errors=True)

    def _run(self) -> None:
        state = self.state
        self.out_dir.mkdir(parents=True, exist_ok=True)
        scratch = self.out_dir / "_frames"
        scratch.mkdir(exist_ok=True)
        frames: dict[str, list[Path]] = {kind: [] for kind in self.movies}
        for i, item in enumerate(self.items):
            self.item = i
            if self.cancel.is_set():
                self.state_name = "cancelled"
                return
            tick = time.time()
            self.stage = "matching"
            path = state.resolve(item["path"])
            base = state.ovito.submit(
                run_ptm, self.analysis, self.colour, path=path, frame_index=item["frame_index"]
            ).result()
            # run_ptm has already coloured along the requested direction, so
            # only a rotation needs the colours worked out again: this was a
            # second full colouring pass on every frame of the series.
            result = derive_result(base, self.colour) if self.colour.get("rotations") else base
            result = self._filled(result)
            for kind in sorted(set(self.stills) | set(self.movies)):
                self.stage = f"rendering {OUTPUTS[kind]['label']}"
                self._render(kind, result, item, scratch, frames)
            self.seconds_per_item = round(time.time() - tick, 2)
        self.item = len(self.items)
        self._write_movies(frames)
        self.stage = ""
        self.state_name = "done"

    def _write_movies(self, frames: dict) -> None:
        """Join the rendered frames into each movie that was asked for."""
        for kind, formats in self.movies.items():
            self.stage = f"writing {OUTPUTS[kind]['label']} movie"
            pngs = frames[kind]
            if not pngs:
                continue
            for ext in sorted(formats):
                out = self.out_dir / f"{self.series['stem']}_{kind}.{ext}"
                try:
                    animate.write_video(pngs, out, fps=1.0 / self.seconds)
                except Exception as exc:
                    # Every frame is already rendered.  One encoder that will
                    # not run must not throw the whole series away.
                    self.notes.append(f"the {ext} movie could not be written: {exc}")
                    continue
                self.files.append(out.name)

    def _filled(self, result):
        query = self.queries["view"]
        radius = float(query.get("fill_radius", ["0"])[0] or 0)
        if radius <= 0:
            return result
        from ..fill import fill_boundary_orientations

        minimum = int(float(query.get("fill_min_neighbours", ["3"])[0] or 3))
        return fill_boundary_orientations(result, radius=radius, min_neighbours=minimum)

    def _render(self, kind: str, result, item: dict, scratch: Path, frames: dict) -> None:
        query = dict(self.queries[kind])
        label = item["label"]
        stem = f"{Path(item['path']).stem}"
        if self.series["kind"] == "frames":
            stem = f"{stem}_{item['frame_index']:05d}"
        wanted = set(self.stills.get(kind, ()))
        if kind in self.movies:
            wanted.add("png")
        for ext in sorted(wanted):
            if kind == "view":
                body = self._render_view(result, query, label)
            else:
                query["format"] = [ext]
                body = self._render_figure(kind, result, query)
            keep = ext in self.stills.get(kind, ())
            out = self.out_dir / f"{stem}_{kind}.{ext}"
            if keep:
                out.write_bytes(body)
                self.files.append(out.name)
            if ext == "png" and kind in self.movies:
                frame = scratch / f"{kind}_{len(frames[kind]):05d}.png"
                frame.write_bytes(body)
                if self.stamp and kind != "view":
                    animate._stamp(frame, label)
                frames[kind].append(frame)

    def _render_view(self, result, query: dict, label: str) -> bytes:
        from ..io import temporary_path

        state = self.state
        from .server import view_options

        options = view_options(state, result, query)
        if self.stamp:
            options["label"] = label
        with temporary_path(".png") as scratch:
            state.ovito.submit(
                rendering.render_scene,
                result,
                scratch,
                transparent=query.get("transparent", ["0"])[0] == "1",
                **options,
            ).result()
            return Path(scratch).read_bytes()

    def _render_figure(self, kind: str, result, query: dict) -> bytes:
        from .server import FIGURES, figure_result

        state = self.state
        with state.lock:  # matplotlib is not safe to share between threads
            drawn = figure_result(state, query, result=result, slice_atoms=kind != "ipfmap")
            body, _fmt, _name, _headers = FIGURES[kind](state, drawn, query)
        return body


def start_series(state, body: dict) -> dict:
    """Start a batch render; refuses while one is running."""
    path = str(body.get("path") or "")
    if not path:
        raise ValueError("a file path is required")
    target = state.resolve(path)
    if not target.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    with state.lock:
        job = state.series_job
        if job is not None and job.state_name == "running":
            raise ValueError("a series is already rendering; cancel it first")
    series = state.ovito.submit(detect_series, state, target).result()
    job = SeriesJob(state, series, body)
    with state.lock:
        state.series_job = job
    job.start()
    return {"accepted": True, "n_items": len(job.items), "out_dir": _relative(state, job.out_dir)}

"""Server-side state for the web UI.

The expensive step is polyhedral template matching, so the :class:`AppState`
keeps the most recent :class:`~ptmipf.analysis.IPFResult` in memory and
distinguishes settings that require a new PTM run (input file, structure set,
RMSD cutoff, trajectory frame) from settings that only change the colouring
(projection direction, sample frame, colour-only set, rigid rotations of the
system).  The latter are applied by deriving a new result from the cached
orientation quaternions: rotate, then recolour.

Restricting PTM to a slab is an analysis setting, but a cheap one when the
full cell has already been matched: the slab is then cut from the cached full
result instead of being matched again.

All OVITO work is funnelled through a single worker thread: OVITO's scene is
global and its Qt internals bind to the first thread that runs a pipeline, so
the many threads of a threading HTTP server must never touch it directly.
For the same reason the server should live in a process where no other thread
has used OVITO; ``python -m ptmipf.webui`` guarantees that, and the tests run
the server as a subprocess.
"""

from __future__ import annotations

import dataclasses
import functools
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..analysis import DEFAULT_OTHER_COLOR, IPFResult, quaternions_to_matrices
from ..colorkey import IPFColorKey
from ..frames import SampleFrame
from ..io import temporary_path
from ..structures import DEFAULT_STRUCTURES, get_structure
from ..symmetry import get_laue_group
from ..transform import (
    parse_rotation,
    rotate_positions,
    rotate_result,
    rotation_center,
    rotation_matrix,
)
from .worker import AnalysisWorker, Cancelled, WorkerUnavailable

__all__ = ["AppState", "SelectionUnavailableError", "derive_result", "rotation_of", "run_ptm"]

#: Atoms beyond the slab that PTM still sees, so that atoms at the slab faces
#: keep their full neighbour shell; see :func:`ptmipf.analysis.analyse`.
SLAB_MARGIN = 8.0


class SelectionUnavailableError(RuntimeError):
    """Raised when the optional :mod:`ptmipf.select` module is missing."""


def _select_module():
    try:
        from .. import select
    except ImportError as exc:
        raise SelectionUnavailableError(
            "the ptmipf.select module is not available in this installation"
        ) from exc
    return select


def subset_result(result: IPFResult, mask: np.ndarray) -> IPFResult:
    """Restrict *result* to the atoms selected by *mask*.

    Prefers :meth:`IPFResult.subset` and falls back to an equivalent local
    implementation, so the figure endpoints keep working against older
    versions of the analysis module.
    """
    if hasattr(result, "subset"):
        return result.subset(mask)
    mask = np.asarray(mask, dtype=bool)
    counts = {}
    structure_types = result.structure_types[mask]
    for s in result.structures:
        counts[s.name] = int((structure_types == result.type_codes[s.name]).sum())
    counts["other"] = int(mask.sum() - sum(counts.values()))
    return dataclasses.replace(
        result,
        positions=result.positions[mask],
        structure_types=structure_types,
        orientations=result.orientations[mask],
        colors=result.colors[mask],
        rmsd=result.rmsd[mask],
        particle_types=result.particle_types[mask],
        counts=counts,
    )


def _parse_reference(spec, result: IPFResult):
    """Turn the JSON reference of a misorientation criterion into the contract's form.

    The contract accepts a rotation matrix, a quaternion or an atom index; the
    UI offers an atom index (from picking) or an explicit quaternion.
    """
    if isinstance(spec, dict):
        if "atom" in spec:
            index = int(spec["atom"])
            if not 0 <= index < result.n_atoms:
                raise ValueError(f"atom index {index} out of range")
            return index
        if "quaternion" in spec:
            q = np.asarray(spec["quaternion"], dtype=float).reshape(4)
            return q
        raise ValueError("misorientation reference needs an 'atom' or 'quaternion' key")
    if isinstance(spec, (int, np.integer)):
        return int(spec)
    return np.asarray(spec, dtype=float)


def _clean_columns(spec) -> dict | None:
    """Normalise the orientation column mapping arriving from the browser."""
    if not spec:
        return None
    quaternion = spec.get("quaternion")
    if isinstance(quaternion, str):
        quaternion = [quaternion]
    quaternion = [str(name) for name in (quaternion or []) if str(name).strip()]
    cleaned = {
        "quaternion": quaternion,
        "order": str(spec.get("order") or "xyzw").lower(),
        "conjugate": bool(spec.get("conjugate")),
    }
    for name in ("structure_type", "rmsd", "structure"):
        value = str(spec.get(name) or "").strip()
        if value:
            cleaned[name] = value
    return cleaned


def _clean_slab(spec) -> dict | None:
    """Normalise the slab a PTM run is restricted to.

    The browser sends the slice it is showing: a normal (any direction spec),
    an absolute distance along it in angstroms and a width; a zero width means
    everything up to the plane, as in the 3D view.
    """
    if not spec:
        return None
    axis = str(spec.get("axis") or "").strip()
    if not axis:
        return None
    return {
        "axis": axis,
        "distance": round(float(spec.get("distance") or 0.0), 4),
        "width": round(max(0.0, float(spec.get("width") or 0.0)), 4),
    }


def _clean_rotations(spec) -> tuple:
    """Normalise the list of rigid rotations, as ``(axis, degrees)`` pairs.

    Accepts the dialog's ``{"axis": .., "angle": ..}`` objects, ``[axis, angle]``
    pairs and the CLI's ``AXIS:DEGREES`` strings.  Zero-angle entries are
    dropped, so that they do not force a needless recolouring.
    """
    rotations = []
    for item in spec or []:
        if isinstance(item, str):
            axis, angle = parse_rotation(item)
        elif isinstance(item, dict):
            axis, angle = str(item.get("axis") or ""), float(item.get("angle") or 0.0)
        else:
            axis, angle = str(item[0]), float(item[1])
        axis = axis.strip()
        if axis and angle:
            rotations.append((axis, round(angle, 6)))
    return tuple(rotations)


def rotation_of(colour: dict, frame: SampleFrame | None = None) -> np.ndarray | None:
    """The composed rotation matrix of the colour settings, or None without one.

    Each rotation turns the system about an axis fixed in the sample frame,
    in the order given, which is what the CLI's repeated ``--rotate`` does.
    """
    rotations = colour.get("rotations") or ()
    if not rotations:
        return None
    frame = frame or SampleFrame(colour.get("axes") or {})
    matrix = np.eye(3)
    for axis, angle in rotations:
        matrix = rotation_matrix(frame.direction(axis), angle) @ matrix
    return matrix


def derive_result(base: IPFResult, colour: dict) -> IPFResult:
    """The result the interface shows: *base* rotated, then recoloured.

    This mirrors the colouring block of :func:`ptmipf.analysis.analyse` and
    is what makes changing the projection direction, or turning the whole
    system, instant.  *base* is never modified.
    """
    frame = SampleFrame(colour["axes"])
    matrix = rotation_of(colour, frame)
    result = base if matrix is None else rotate_result(base, matrix)
    d = frame.direction(colour["direction"])
    colour_these = {
        get_structure(s).name
        for s in (colour["color_only"] or [s.name for s in result.structures])
    }
    colors = np.tile(np.asarray(colour["other_color"], dtype=float), (result.n_atoms, 1))
    for s in result.structures:
        selection = result.structure_types == result.type_codes[s.name]
        if not s.colorable or s.name not in colour_these or not selection.any():
            continue
        key = IPFColorKey(get_laue_group(s.laue))
        rotations = quaternions_to_matrices(result.orientations[selection])
        colors[selection] = key.orientation2color(rotations, d)
    return dataclasses.replace(
        result,
        colors=colors,
        direction=d,
        direction_label=frame.label(colour["direction"]),
        frame=frame,
    )


def _slab_mask_of(result: IPFResult, slab: dict, colour: dict) -> np.ndarray:
    """Which atoms of the unrotated *result* lie in *slab*.

    The slab is what the 3D view shows, so it is defined on the rotated
    positions with the same rule as :func:`ptmipf.webui.rendering.visible_mask`.
    """
    frame = SampleFrame(colour["axes"])
    matrix = rotation_of(colour, frame)
    positions = result.positions
    if matrix is not None:
        center = rotation_center(result.cell, positions)
        positions = rotate_positions(positions, matrix, center)
    projected = positions @ frame.direction(slab["axis"])
    if slab["width"] > 0:
        return np.abs(projected - slab["distance"]) <= slab["width"] / 2.0
    return projected <= slab["distance"]


def _slab_for_analysis(slab: dict, colour: dict) -> dict:
    """The ``slab=`` argument of :func:`ptmipf.analysis.analyse` for *slab*."""
    frame = SampleFrame(colour["axes"])
    distance, width = slab["distance"], slab["width"]
    low, high = (None, distance) if width <= 0 else (distance - width / 2, distance + width / 2)
    spec = {"normal": frame.direction(slab["axis"]), "low": low, "high": high,
            "margin": SLAB_MARGIN}
    matrix = rotation_of(colour, frame)
    if matrix is not None:
        spec["rotation"] = (matrix, None)
    return spec


@functools.lru_cache(maxsize=1)
def movie_support() -> dict:
    """Which movie formats this environment can write, and why not.

    The answer cannot change while the server runs, so it is worked out once
    and sent with every status, which is how the interface knows to grey out
    a format instead of letting somebody wait for a run that cannot finish.
    """
    from .. import animate

    out = {}
    for suffix in animate.VIDEO_NEEDS:
        ext = suffix.lstrip(".")
        out[ext] = animate.video_support(suffix)
    return out


def run_ptm(analysis: dict, colour: dict, progress=None, path=None, frame_index=None):
    """Match and colour one configuration with the interface's settings.

    Must run on the OVITO worker thread.  *path* and *frame_index* override
    the ones in *analysis*, which is how a trajectory series is stepped
    through with the settings of the current analysis.  The result is what
    :func:`ptmipf.analysis.analyse` returns: unrotated, coloured along the
    requested direction; :func:`derive_result` applies any rotation.
    """
    from ..analysis import analyse, analyse_orientations

    frame = SampleFrame(colour["axes"])
    common = {
        "direction": colour["direction"],
        "structures": analysis["structures"],
        "frame": frame,
        "frame_index": analysis["frame_index"] if frame_index is None else int(frame_index),
        "other_color": colour["other_color"],
        "only": colour["color_only"],
    }
    path = str(path or analysis["path"])
    slab = _slab_for_analysis(analysis["slab"], colour) if analysis.get("slab") else None
    if analysis.get("columns"):
        if progress:
            progress(STAGES[0][0])
        result = analyse_orientations(path, analysis["columns"], slab=slab, **common)
        if progress:
            progress(STAGES[2][0])
    else:
        result = analyse(
            path, rmsd_cutoff=analysis["rmsd_cutoff"], progress=progress, slab=slab, **common
        )
    if result.n_atoms == 0:
        raise ValueError("the slice contains no atoms; move it first")
    return result


def _without_slab(analysis: dict | None) -> dict | None:
    if analysis is None:
        return None
    return {k: v for k, v in analysis.items() if k != "slab"}


#: What each stage of an analysis covers on the progress bar, in order, with
#: the quantity its duration scales with.  OVITO reports nothing while it is
#: working, so the bar is interpolated inside a stage from elapsed time against
#: a throughput estimate.  That makes it an estimate and it is labelled as one.
STAGES = (
    ("reading the configuration", 0.00, 0.15, "read"),
    ("polyhedral template matching", 0.15, 0.90, "ptm"),
    ("colouring the orientations", 0.90, 1.00, "colour"),
)

#: Deliberately pessimistic starting throughputs: bytes a second for the read,
#: atoms a second for the other two.  A bar that runs behind and then completes
#: reads better than one that sits at 99 per cent, and the first real run
#: replaces these anyway.
_SEED_RATES = {"read": 4.0e7, "ptm": 2.5e5, "colour": 5.0e5}


class FigureCache:
    """The last few figures that were drawn, keyed on what they were drawn from.

    Nothing about a figure changes unless its settings or the result change,
    and both are in the key, so a figure that has been drawn once can be
    served again as it stands.  That is what makes going back to a previous
    setting, or asking for the same figure in a second format, instant instead
    of a second full computation.

    Bounded by total size rather than by entry count, because the figures range
    from a few tens of kilobytes on screen to several megabytes at 600 dpi.
    """

    def __init__(self, budget: int = 192 << 20) -> None:
        self.budget = int(budget)
        self.entries: OrderedDict = OrderedDict()
        self.size = 0
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            item = self.entries.get(key)
            if item is None:
                self.misses += 1
                return None
            self.entries.move_to_end(key)
            self.hits += 1
            return item

    def put(self, key, value) -> None:
        body = value[0]
        cost = len(body) if body is not None else 0
        if cost > self.budget:  # one enormous figure must not empty the cache
            return
        with self._lock:
            if key in self.entries:
                self.size -= len(self.entries[key][0])
            self.entries[key] = value
            self.entries.move_to_end(key)
            self.size += cost
            while self.size > self.budget and len(self.entries) > 1:
                _, dropped = self.entries.popitem(last=False)
                self.size -= len(dropped[0])

    def clear(self) -> None:
        with self._lock:
            self.entries.clear()
            self.size = 0

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self.entries),
                "bytes": self.size,
                "hits": self.hits,
                "misses": self.misses,
            }


class AppState:
    """Cached analysis result, selection and job status behind one lock."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.lock = threading.RLock()
        # A single thread for everything that touches OVITO (see module docstring).
        self.ovito = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ovito")

        self.result: IPFResult | None = None  # what the interface shows: derived
        self.base_result: IPFResult | None = None  # the PTM output it derives from
        self.analysis_params: dict | None = None  # settings that forced the last PTM run
        self.colour_params: dict = {}
        # The last whole-cell PTM result and its settings, kept so that a slab
        # can be cut from it, and the whole cell restored, without matching again.
        self.full_result: IPFResult | None = None
        self.full_params: dict | None = None
        self.job = {"state": "idle", "stage": "", "error": "", "started": 0.0}
        self.selection_mask: np.ndarray | None = None
        self.selection_criteria: list = []
        self.selection_mode: str = "and"
        self.generation = 0  # bumped on every visible change, used for cache busting
        self._fill_cache: tuple | None = None  # (radius, min_neighbours, generation)
        self._diagnostics: dict | None = None  # probed once, on first request
        # A colour map uploaded this session, as an (n, 3) array.  Kept in
        # memory rather than written anywhere, and gone when the server stops.
        self.custom_colormap = None
        # Bumped when a colour map is uploaded, so that a figure cached with
        # the previous one under the same name is not served again.
        self.colormap_generation = 0
        # Finished figures, keyed on their settings and the generation they
        # were drawn from; see FigureCache.
        self.figures = FigureCache()
        # The batch render of a trajectory series, if one has been started.
        self.series_job = None
        # The child process the analysis runs in, so that it can be stopped;
        # see ptmipf.webui.worker.  Started on the first analysis.
        self.worker = AnalysisWorker()
        self.worker_note = ""  # why it is not being used, if it is not
        # Set while an analysis is being stopped, so that the run that is
        # unwinding does not overwrite the state with its own outcome.
        self.stopping = False
        # Throughput used to turn stage boundaries into a percentage, refined
        # from every run this process completes; see _stage_progress.
        self._rates = dict(_SEED_RATES)

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------
    def set_root(self, path: str) -> dict:
        """Serve a different folder, so any folder on the machine is reachable.

        The interface starts in one folder, which is rarely the only one worth
        looking at.  The server is bound to the loopback address and is the
        person's own account, so the new folder is only checked for being a
        folder, not for where it is.
        """
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve()
        if not candidate.is_dir():
            raise ValueError(f"not a folder: {path}")
        with self.lock:
            self.root = candidate
        return {"root": str(candidate)}

    def home_folders(self) -> list[dict]:
        """Somewhere to start from when the typed path is a guess."""
        import platform

        places = [("home", Path.home()), ("here", Path.cwd())]
        if platform.system() == "Windows":  # pragma: no cover - not run on Linux
            places += [
                (f"{letter}:", Path(f"{letter}:/"))
                for letter in "CDEFG"
                if Path(f"{letter}:/").is_dir()
            ]
        seen, out = set(), []
        for label, place in places:
            text = str(place)
            if place.is_dir() and text not in seen:
                seen.add(text)
                out.append({"label": label, "path": text})
        return out

    def resolve(self, path: str) -> Path:
        """Resolve *path* against the served root, refusing escapes."""
        candidate = (self.root / path).resolve() if not Path(path).is_absolute() else (
            Path(path).resolve()
        )
        if candidate != self.root and self.root not in candidate.parents:
            raise PermissionError(f"{path!r} is outside the served root directory")
        return candidate

    # ------------------------------------------------------------------
    # analysis
    # ------------------------------------------------------------------
    @staticmethod
    def _split_params(params: dict) -> tuple[dict, dict]:
        analysis = {
            "path": str(params["path"]),
            "structures": tuple(params.get("structures") or DEFAULT_STRUCTURES),
            "rmsd_cutoff": float(params.get("rmsd_cutoff", 0.1)),
            "frame_index": int(params.get("frame_index", 0)),
            # A column mapping means the orientations are already in the file
            # and PTM is skipped; it belongs with the settings that force a
            # re-read, because changing the mapping changes every orientation.
            "columns": _clean_columns(params.get("columns")),
            # Restricting PTM to a slab changes which atoms exist in the result.
            "slab": _clean_slab(params.get("slab")),
        }
        colour = {
            "direction": params.get("direction", "z"),
            "axes": {k: v for k, v in (params.get("axes") or {}).items() if v},
            "color_only": tuple(params["color_only"]) if params.get("color_only") else None,
            "other_color": tuple(params.get("other_color") or DEFAULT_OTHER_COLOR),
            # Rigid rotations of the whole system only move orientations
            # relative to the sample frame, so they are a colouring setting.
            "rotations": _clean_rotations(params.get("rotations")),
        }
        return analysis, colour

    def submit_analysis(self, params: dict) -> dict:
        """Run PTM in the background, or just recolour when nothing costly changed."""
        analysis, colour = self._split_params(params)
        path = self.resolve(analysis["path"])
        if not path.is_file():
            raise FileNotFoundError(f"no such file: {analysis['path']}")
        analysis["path"] = str(path)
        for name in analysis["structures"]:
            get_structure(name)  # fail early on typos
        if analysis["columns"] and not analysis["columns"].get("quaternion"):
            raise ValueError("choose the column or columns that hold the quaternion")

        force = bool(params.get("force"))
        with self.lock:
            if self.job["state"] == "running":
                return {"accepted": False, "reason": "an analysis is already running"}
            if force:
                # "Run again" means exactly that: matched again from the file,
                # not rebuilt from anything this session has cached.  The
                # current result stays on screen until the new one lands, so
                # only the caches that would skip the run are dropped.
                self.full_result, self.full_params = None, None
            if not force and self.base_result is not None and analysis == self.analysis_params:
                self._derive(colour)
                return {"accepted": True, "recoloured": True}
            # A slab of a cell that has already been matched, or the whole
            # cell again after a slab, comes straight from the cached result.
            if (
                not force
                and self.full_result is not None
                and _without_slab(analysis) == _without_slab(self.full_params)
            ):
                base = self.full_result
                if analysis["slab"]:
                    keep = _slab_mask_of(base, analysis["slab"], colour)
                    if not keep.any():
                        raise ValueError("the slice contains no atoms; move it first")
                    base = base.subset(keep)
                self._install(base, analysis, colour)
                return {"accepted": True, "recoloured": True, "subset": True}
            now = time.time()
            self.job = {
                "state": "running",
                "stage": STAGES[0][0],
                "error": "",
                "started": now,
                "stage_index": 0,
                "stage_started": now,
                "stage_expected": max(0.2, path.stat().st_size / self._rates["read"]),
                "file_bytes": path.stat().st_size,
                "n_atoms": 0,
                "progress": 0.0,
            }
            self.stopping = False
            self.ovito.submit(self._run_analysis, analysis, colour)
            return {"accepted": True, "recoloured": False}

    def _run_analysis(self, analysis: dict, colour: dict) -> None:
        with self.lock:
            if self.stopping:  # stopped while it was still queued
                self._finish_stopped()
                return
        try:
            result = self._match(analysis, colour)
        except Cancelled:
            with self.lock:
                self._finish_stopped()
            return
        except Exception as exc:  # surfaced through /api/status, not a traceback
            with self.lock:
                if self.stopping:
                    self._finish_stopped()
                    return
                self.job = {
                    "state": "error", "stage": "", "error": str(exc),
                    "started": 0.0, "progress": 0.0,
                }
            return
        with self.lock:
            # analyse() has already coloured along the requested direction; a
            # rotation is the one colour setting it knows nothing about.
            self._install(result, analysis, colour, derive=bool(colour["rotations"]))
            self.job = {
                "state": "done", "stage": "", "error": "", "started": 0.0, "progress": 1.0
            }

    def _match(self, analysis: dict, colour: dict) -> IPFResult:
        """Run PTM, in the child process when there is one.

        The child is what makes *Stop* work, but the interface must not depend
        on it: if it cannot be started or dies of its own accord, the analysis
        runs here instead and simply cannot be interrupted mid-stage.
        """
        with temporary_path(".pkl") as scratch:
            try:
                return self.worker.run(
                    analysis, colour, Path(scratch), on_stage=self._stage_started
                )
            except WorkerUnavailable as exc:
                with self.lock:
                    self.worker_note = str(exc)
        return run_ptm(analysis, colour, progress=self._stage_started)

    def _finish_stopped(self) -> None:
        """Leave the job stopped; the caller holds the lock."""
        self.stopping = False
        self.job = {
            "state": "cancelled", "stage": "", "error": "", "started": 0.0, "progress": 0.0,
        }

    def cancel_analysis(self) -> dict:
        """Stop the running analysis, killing the process it runs in.

        Returns what happened, so the interface can say whether it stopped
        something or there was nothing to stop.
        """
        with self.lock:
            running = self.job.get("state") == "running"
            if running:
                self.stopping = True
        killed = self.worker.stop()
        with self.lock:
            if running and not killed and self.job.get("state") == "running":
                # It is between stages, or running in this process because the
                # child is unavailable; either way it stops at the next
                # checkpoint.  Say so now rather than leaving the page busy.
                self._finish_stopped()
            return {"stopped": running, "killed": killed, "state": self.job["state"]}

    def _install(self, base: IPFResult, analysis: dict, colour: dict, derive=True) -> None:
        """Make *base* the current PTM result and derive the shown one from it."""
        with self.lock:
            self.base_result = base
            self.analysis_params = analysis
            self.colour_params = colour
            if analysis["slab"] is None:
                self.full_result, self.full_params = base, analysis
            self.result = derive_result(base, colour) if derive else base
            self.selection_mask = None
            self.selection_criteria = []
            self._fill_cache = None
            self.generation += 1

    def _stage_started(self, stage: str, n_atoms: int | None = None) -> None:
        """Record a stage boundary, and learn from the stage that just ended."""
        with self.lock:
            job = self.job
            if job.get("state") != "running":
                return
            now = time.time()
            index = next((i for i, s in enumerate(STAGES) if s[0] == stage), 0)
            previous = job.get("stage_index")
            if previous is not None and previous < index:
                self._learn_rate(STAGES[previous][3], now - job["stage_started"], job)
            expected = 0.2
            if n_atoms:
                key = STAGES[index][3]
                expected = max(0.2, n_atoms / self._rates.get(key, 1e6))
            job.update(
                stage=stage,
                stage_index=index,
                stage_started=now,
                stage_expected=expected,
                n_atoms=int(n_atoms or job.get("n_atoms") or 0),
                progress=STAGES[index][1],
            )

    def _learn_rate(self, key: str, seconds: float, job: dict) -> None:
        """Fold one completed stage into the throughput estimate.

        A running mean rather than a replacement, so one unlucky stage (a cold
        file cache, another job on the machine) does not throw the next bar off.
        """
        # The read is sized in bytes and the rest in atoms; the rate carries
        # whichever unit its own stage is measured in.
        amount = job.get("file_bytes", 0) if key == "read" else (job.get("n_atoms") or 0)
        if amount <= 0 or seconds <= 0.05:
            return
        observed = amount / seconds
        self._rates[key] = 0.5 * self._rates.get(key, observed) + 0.5 * observed

    def _stage_progress(self, job: dict, now: float) -> float:
        """Interpolate inside the current stage from elapsed time.

        Capped just short of the end of its band so the bar never claims a stage
        is finished while it is still running.
        """
        index = job.get("stage_index", 0)
        _, base, top, _ = STAGES[min(index, len(STAGES) - 1)]
        expected = max(job.get("stage_expected", 1.0), 1e-6)
        fraction = min((now - job.get("stage_started", now)) / expected, 0.97)
        return round(min(1.0, base + (top - base) * max(fraction, 0.0)), 3)

    def _derive(self, colour: dict) -> None:
        """Re-derive the shown result for new colour settings; PTM is untouched."""
        with self.lock:
            self.result = derive_result(self.base_result, colour)
            self.colour_params = colour
            self.generation += 1

    def view_result(self, fill_radius=None, fill_min_neighbours: int = 3) -> IPFResult | None:
        """The cached result, with the boundary atoms filled in if asked.

        Filling is cached against the generation counter, so it is paid for once
        and thrown away as soon as the colours or the analysis change.
        """
        with self.lock:
            result = self.result
            if result is None or not fill_radius:
                return result
            key = (float(fill_radius), int(fill_min_neighbours), self.generation)
            if self._fill_cache is not None and self._fill_cache[0] == key:
                return self._fill_cache[1]

        from ..fill import fill_boundary_orientations

        filled = fill_boundary_orientations(
            result, radius=float(fill_radius), min_neighbours=int(fill_min_neighbours)
        )
        with self.lock:
            self._fill_cache = (key, filled)
        return filled

    def status(self) -> dict:
        with self.lock:
            payload = dict(self.job)
            if payload["state"] == "running":
                now = time.time()
                payload["progress"] = self._stage_progress(self.job, now)
                payload["elapsed"] = round(now - payload.pop("started"), 1)
                remaining = payload["stage_expected"] - (now - payload["stage_started"])
                payload["stage_remaining"] = round(max(remaining, 0.0), 1)
            else:
                payload.pop("started")
            for key in ("stage_started", "stage_expected", "stage_index", "file_bytes"):
                payload.pop(key, None)
            payload["generation"] = self.generation
            payload["movies"] = movie_support()
            if self.result is not None:
                result = self.result
                payload["result"] = {
                    "path": self.analysis_params["path"],
                    "n_atoms": result.n_atoms,
                    "counts": result.counts,
                    "direction_label": result.direction_label,
                    "direction": [round(float(c), 6) for c in result.direction],
                    "cell": None
                    if result.cell is None
                    else np.asarray(result.cell).round(6).tolist(),
                    "type_names": {str(k): v for k, v in result.type_names.items()},
                    "structures": [s.name for s in result.structures],
                    "colorable": [s.name for s in result.structures if s.colorable],
                    "frame_axes": {
                        k: np.round(v, 6).tolist() for k, v in result.frame.axes.items()
                    },
                    "summary": result.summary(),
                    "slab": self.analysis_params.get("slab"),
                    "rotations": [list(r) for r in self.colour_params.get("rotations", ())],
                    "full_n_atoms": (
                        self.full_result.n_atoms
                        if self.full_result is not None
                        and _without_slab(self.full_params) == _without_slab(self.analysis_params)
                        else None
                    ),
                }
                if self.selection_mask is not None:
                    payload["selection"] = {
                        "count": int(self.selection_mask.sum()),
                        "criteria": self.selection_criteria,
                        "mode": self.selection_mode,
                    }
            return payload

    # ------------------------------------------------------------------
    # selection
    # ------------------------------------------------------------------
    def apply_selection(self, criteria: list, mode: str = "and") -> dict:
        select = _select_module()
        with self.lock:
            result = self.result
            if result is None:
                raise ValueError("no analysis result yet")
            if not criteria:
                self.selection_mask = None
                self.selection_criteria = []
                self.generation += 1
                return {"count": None}
            masks = []
            for criterion in criteria:
                mask = self._one_mask(select, result, criterion)
                if criterion.get("invert"):
                    mask = select.invert(mask)
                masks.append(mask)
            mode = "or" if mode == "or" else "and"
            combined = masks[0] if len(masks) == 1 else select.combine(*masks, mode=mode)
            self.selection_mask = np.asarray(combined, dtype=bool)
            self.selection_criteria = criteria
            self.selection_mode = mode
            self.generation += 1
            return {"count": int(self.selection_mask.sum())}

    @staticmethod
    def _one_mask(select, result: IPFResult, criterion: dict) -> np.ndarray:
        kind = criterion.get("kind")
        if kind == "structure":
            return select.select_by_structure(result, tuple(criterion["structures"]))
        if kind == "type":
            types = [int(t) if str(t).lstrip("-").isdigit() else t for t in criterion["types"]]
            return select.select_by_type(result, tuple(types))
        if kind == "rmsd":
            return select.select_by_rmsd(
                result,
                maximum=criterion.get("max"),
                minimum=criterion.get("min"),
            )
        if kind == "region":
            return select.select_by_region(
                result,
                criterion.get("axis", "z"),
                minimum=criterion.get("min"),
                maximum=criterion.get("max"),
            )
        if kind == "ipf":
            return select.select_by_ipf_direction(
                result,
                criterion["crystal"],
                criterion["sample"],
                float(criterion.get("tolerance", 10.0)),
                criterion["structure"],
            )
        if kind == "misorientation":
            return select.select_by_misorientation(
                result,
                _parse_reference(criterion["reference"], result),
                float(criterion.get("tolerance", 5.0)),
                criterion["structure"],
            )
        raise ValueError(f"unknown selection criterion {kind!r}")

    def selection_subset(self) -> IPFResult:
        with self.lock:
            if self.result is None:
                raise ValueError("no analysis result yet")
            if self.selection_mask is None:
                raise ValueError("no selection is active")
            return subset_result(self.result, self.selection_mask)

    # ------------------------------------------------------------------
    # atom info (used by picking and misorientation references)
    # ------------------------------------------------------------------
    def atom_info(self, index: int) -> dict:
        with self.lock:
            result = self.result
            if result is None or not 0 <= index < result.n_atoms:
                raise ValueError(f"atom index {index} out of range")
            code = int(result.structure_types[index])
            name = next(
                (n for n, c in result.type_codes.items() if c == code), "other"
            ) if code else "other"
            return {
                "index": index,
                "position": [round(float(c), 4) for c in result.positions[index]],
                "structure": name,
                "rmsd": round(float(result.rmsd[index]), 5),
                "type": result.type_names.get(int(result.particle_types[index]), ""),
                "orientation": [round(float(c), 6) for c in result.orientations[index]],
                "color": [round(float(c), 4) for c in result.colors[index]],
            }

    # ------------------------------------------------------------------
    # first-run diagnostics
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # trajectory series
    # ------------------------------------------------------------------
    def series_status(self) -> dict:
        with self.lock:
            job = self.series_job
        if job is None:
            return {"state": "idle", "files": [], "item": 0, "n_items": 0, "progress": 0.0}
        return job.status()

    def series_output(self, name: str) -> Path:
        """One file the series job wrote; nothing outside its folder is served."""
        with self.lock:
            job = self.series_job
        if job is None:
            raise FileNotFoundError("no series has been rendered")
        target = (job.out_dir / name).resolve()
        if job.out_dir.resolve() not in target.parents or not target.is_file():
            raise FileNotFoundError(f"no such series output: {name}")
        return target

    def cancel_series(self) -> dict:
        with self.lock:
            job = self.series_job
        if job is None or job.state_name != "running":
            return {"cancelled": False}
        job.cancel.set()
        return {"cancelled": True}

    def diagnostics(self) -> dict:
        """Probe everything the interface needs, and say what is missing.

        The 3D view is drawn by OVITO in this process and arrives in the
        browser as a PNG, so a blank viewer is almost always a server-side
        renderer problem rather than a browser one.  The probe runs on the
        OVITO worker thread, like every other OVITO call, and caches its
        verdict: building a scene costs little but is not free.
        """
        with self.lock:
            if self._diagnostics is None:
                self._diagnostics = self.ovito.submit(_probe_environment).result()
            probe = dict(self._diagnostics)
        probe["root"] = str(self.root)
        # Whether an analysis can be stopped once it is inside OVITO, which
        # depends on the child process being usable here.
        probe["figure_cache"] = self.figures.stats()
        probe["stoppable"] = bool(self.worker.available)
        probe["stop_note"] = self.worker_note or self.worker.reason
        return probe


def _probe_environment() -> dict:
    """Import and render probes.

    Inside the server this runs on the OVITO worker thread, like every other
    OVITO call; ``ptmipf-ui --check`` calls it on the main thread instead,
    because a one-shot check has no HTTP threads to protect and OVITO crashes
    at interpreter shutdown if its objects outlive the thread that made them.
    """
    import platform
    import sys

    import numpy as np

    from .. import __version__
    from ..io import temporary_path

    report = {
        "ptmipf": __version__,
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "checks": [],
        "renderer": None,
        "ok": True,
    }

    def record(name: str, ok: bool, detail: str = "", fatal: bool = False):
        report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        if fatal and not ok:
            report["ok"] = False

    # Git first: it is not needed to run anything, but "uv pip install
    # git+https://..." fails without it, which is how most people get here,
    # and the failure does not say so.  Reported before OVITO because the
    # OVITO check returns early when it fails.
    record("git", *_git_check())
    record("mp4 movies", *_movie_check())

    try:
        import ovito

        record("ovito", True, f"version {ovito.version_string}")
    except Exception as exc:  # the interface cannot do anything without it
        record("ovito", False, _explain_ovito(exc), fatal=True)
        return report

    for name, module in (("matplotlib", "matplotlib"), ("pillow", "PIL"), ("scipy", "scipy")):
        try:
            __import__(module)
            record(name, True)
        except Exception as exc:
            record(name, False, str(exc), fatal=name != "scipy")

    try:
        from .. import select  # noqa: F401

        record("selection", True)
    except Exception as exc:
        record("selection", False, str(exc))

    from ..frames import SampleFrame
    from ..structures import get_structure

    class _Probe:
        positions = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.4, 1.0]])
        structure_types = np.array([1, 1])
        rmsd = np.zeros(2)
        n_atoms = 2
        frame = SampleFrame()
        structures = (get_structure("fcc"),)
        type_codes = {"fcc": 1}
        cell = None

    from . import rendering

    errors = []
    for engine in ("opengl", "tachyon"):
        try:
            with temporary_path(".png") as scratch:
                rendering.render_scene(_Probe(), scratch, size=(48, 48), engine=engine)
                size = Path(scratch).stat().st_size
            if size == 0:
                raise RuntimeError("the renderer wrote an empty file")
            report["renderer"] = engine
            record("3D view", True, f"rendered with the {engine} renderer")
            break
        except Exception as exc:
            errors.append(f"{engine}: {exc}")
    else:
        record("3D view", False, "; ".join(errors) or "no renderer worked", fatal=True)
    return report


def _git_check() -> tuple[bool, str]:
    """Whether git is on the PATH, and the command that installs it if not."""
    import platform
    import shutil

    found = shutil.which("git")
    if found:
        return True, found
    how = {
        "Windows": "winget install Git.Git",
        "Darwin": "xcode-select --install",
    }.get(platform.system(), "sudo apt install git")
    return False, (
        "git is not on the PATH, so installing or updating with "
        f"'uv pip install git+https://...' will fail: {how}. Without git, "
        "download the repository as a ZIP and run 'uv pip install .' in it."
    )


def _movie_check() -> tuple[bool, str]:
    """Whether an MP4 can be written here. GIF needs only Pillow, checked above."""
    from .. import animate

    missing = animate.video_support(".mp4")
    if missing:
        return False, missing + " (GIF movies still work)"
    return True, ""


def _explain_ovito(exc: Exception) -> str:
    """Turn the usual OVITO import failures into the fix for them."""
    text = str(exc)
    if "libOpenGL" in text or "libEGL" in text:
        return (
            f"{text}. On Linux install the OpenGL runtime "
            "(sudo apt install libopengl0 libegl1), or symlink libGL.so.1 to "
            "libOpenGL.so.0 in a directory on LD_LIBRARY_PATH."
        )
    return text

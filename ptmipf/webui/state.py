"""Server-side state for the web UI.

The expensive step is polyhedral template matching, so the :class:`AppState`
keeps the most recent :class:`~ptmipf.analysis.IPFResult` in memory and
distinguishes settings that require a new PTM run (input file, structure set,
RMSD cutoff, trajectory frame) from settings that only change the colouring
(projection direction, sample frame, colour-only set).  The latter are applied
in place by recomputing colours from the cached orientation quaternions.

All OVITO work is funnelled through a single worker thread: OVITO's scene is
global and its Qt internals bind to the first thread that runs a pipeline, so
the many threads of a threading HTTP server must never touch it directly.
For the same reason the server should live in a process where no other thread
has used OVITO; ``python -m ptmipf.webui`` guarantees that, and the tests run
the server as a subprocess.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..analysis import DEFAULT_OTHER_COLOR, IPFResult, quaternions_to_matrices
from ..colorkey import IPFColorKey
from ..frames import SampleFrame
from ..structures import DEFAULT_STRUCTURES, get_structure
from ..symmetry import get_laue_group

__all__ = ["AppState", "SelectionUnavailableError"]


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


class AppState:
    """Cached analysis result, selection and job status behind one lock."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.lock = threading.RLock()
        # A single thread for everything that touches OVITO (see module docstring).
        self.ovito = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ovito")

        self.result: IPFResult | None = None
        self.analysis_params: dict | None = None  # settings that forced the last PTM run
        self.colour_params: dict = {}
        self.job = {"state": "idle", "stage": "", "error": "", "started": 0.0}
        self.selection_mask: np.ndarray | None = None
        self.selection_criteria: list = []
        self.selection_mode: str = "and"
        self.generation = 0  # bumped on every visible change, used for cache busting
        self._fill_cache: tuple | None = None  # (radius, min_neighbours, generation)
        self._diagnostics: dict | None = None  # probed once, on first request
        # Throughput used to turn stage boundaries into a percentage, refined
        # from every run this process completes; see _stage_progress.
        self._rates = dict(_SEED_RATES)

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------
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
        }
        colour = {
            "direction": params.get("direction", "z"),
            "axes": {k: v for k, v in (params.get("axes") or {}).items() if v},
            "color_only": tuple(params["color_only"]) if params.get("color_only") else None,
            "other_color": tuple(params.get("other_color") or DEFAULT_OTHER_COLOR),
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

        with self.lock:
            if self.job["state"] == "running":
                return {"accepted": False, "reason": "an analysis is already running"}
            if self.result is not None and analysis == self.analysis_params:
                self._recolour(colour)
                return {"accepted": True, "recoloured": True}
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
            self.ovito.submit(self._run_analysis, analysis, colour)
            return {"accepted": True, "recoloured": False}

    def _run_analysis(self, analysis: dict, colour: dict) -> None:
        from ..analysis import analyse, analyse_orientations

        try:
            frame = SampleFrame(colour["axes"])
            common = {
                "direction": colour["direction"],
                "structures": analysis["structures"],
                "frame": frame,
                "frame_index": analysis["frame_index"],
                "other_color": colour["other_color"],
                "only": colour["color_only"],
            }
            if analysis["columns"]:
                self._stage_started(STAGES[0][0])
                result = analyse_orientations(
                    analysis["path"], analysis["columns"], **common
                )
                self._stage_started(STAGES[2][0])
            else:
                result = analyse(
                    analysis["path"],
                    rmsd_cutoff=analysis["rmsd_cutoff"],
                    progress=self._stage_started,
                    **common,
                )
        except Exception as exc:  # surfaced through /api/status, not a traceback
            with self.lock:
                self.job = {
                    "state": "error", "stage": "", "error": str(exc),
                    "started": 0.0, "progress": 0.0,
                }
            return
        with self.lock:
            self.result = result
            self.analysis_params = analysis
            self.colour_params = colour
            self.selection_mask = None
            self.selection_criteria = []
            self.job = {
                "state": "done", "stage": "", "error": "", "started": 0.0, "progress": 1.0
            }
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

    def _recolour(self, colour: dict) -> None:
        """Recompute colours in place from the cached orientations.

        This mirrors the colouring block of :func:`ptmipf.analysis.analyse`
        and is what makes changing the projection direction instant.
        """
        result = self.result
        frame = SampleFrame(colour["axes"])
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
        result.colors = colors
        result.direction = d
        result.direction_label = frame.label(colour["direction"])
        result.frame = frame
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

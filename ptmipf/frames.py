"""Sample reference frames and direction parsing.

Users describe the projection direction of an inverse pole figure map the way
they would for an EBSD map: either as a Cartesian axis of the simulation cell
(``+z``), as a named sample axis (``rd``, ``td``, ``nd``, ``ed``), or as an
explicit vector (``1,1,0``).
"""

from __future__ import annotations

import re

import numpy as np

__all__ = ["DEFAULT_AXIS_NAMES", "SampleFrame", "parse_vector"]

#: Axis names understood out of the box.  ``ed`` (extrusion direction) is a
#: common alias for ``rd`` in extruded-magnesium work, but is kept separate so
#: that it can be given its own vector.
DEFAULT_AXIS_NAMES = ("rd", "td", "nd", "ed")

_AXES = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}


def parse_vector(spec: str | np.ndarray) -> np.ndarray:
    """Parse a direction given as ``+z``, ``-x``, ``1,1,0`` or ``[1 1 0]``.

    Returns a unit vector of shape (3,).
    """
    if not isinstance(spec, str):
        v = np.asarray(spec, dtype=float).reshape(3)
        return v / np.linalg.norm(v)

    text = spec.strip().lower()
    sign = 1.0
    if text.startswith(("+", "-")) and len(text) > 1 and text[1] in _AXES:
        sign = -1.0 if text[0] == "-" else 1.0
        text = text[1:]
    if text in _AXES:
        return sign * np.array(_AXES[text])

    numbers = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", spec)
    if len(numbers) != 3:
        raise ValueError(
            f"cannot parse direction {spec!r}; use an axis (+z, -x), a named "
            "sample axis (rd, td, nd, ed) or three components (1,1,0)"
        )
    v = np.array([float(n) for n in numbers])
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError(f"direction {spec!r} is the zero vector")
    return v / norm


class SampleFrame:
    """A named, right-handed orthonormal frame attached to the simulation cell.

    Parameters
    ----------
    axes
        Mapping of axis name to vector in simulation-cell coordinates.  Any
        subset of ``rd``, ``td`` and ``nd`` may be given; missing axes are
        completed to a right-handed orthonormal triad.  ``ed`` and any other
        name is stored as an extra direction and only normalised.
    tol
        Directions whose mutual dot product exceeds this value are reported as
        non-orthogonal.
    """

    def __init__(
        self, axes: dict[str, str | np.ndarray] | None = None, tol: float = 1e-6
    ):
        given = {k.lower(): parse_vector(v) for k, v in (axes or {}).items()}
        self.warnings: list[str] = []

        triad = {k: given[k] for k in ("rd", "td", "nd") if k in given}
        if not triad:
            triad = {
                "rd": np.array([1.0, 0.0, 0.0]),
                "td": np.array([0.0, 1.0, 0.0]),
                "nd": np.array([0.0, 0.0, 1.0]),
            }
        else:
            triad = self._complete_triad(triad, tol)

        self.axes: dict[str, np.ndarray] = dict(triad)
        for name, vector in given.items():
            if name not in self.axes:
                self.axes[name] = vector

    def _complete_triad(self, triad: dict[str, np.ndarray], tol: float) -> dict[str, np.ndarray]:
        order = ["rd", "td", "nd"]
        names = [n for n in order if n in triad]

        for a, b in zip(names, names[1:]):
            dot = float(np.dot(triad[a], triad[b]))
            if abs(dot) > tol:
                self.warnings.append(
                    f"{a.upper()} and {b.upper()} are not orthogonal "
                    f"(cos = {dot:.4f}); {b.upper()} was orthogonalised against {a.upper()}"
                )
                v = triad[b] - dot * triad[a]
                norm = np.linalg.norm(v)
                if norm < 1e-8:
                    raise ValueError(f"{a.upper()} and {b.upper()} are parallel")
                triad[b] = v / norm

        if len(names) == 3:
            return triad
        if len(names) == 1:
            # Complete with an arbitrary but stable perpendicular direction.
            first = triad[names[0]]
            helper = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(helper, first)) > 0.9:
                helper = np.array([0.0, 1.0, 0.0])
            second = np.cross(first, helper)
            second /= np.linalg.norm(second)
            missing = [n for n in order if n not in triad]
            triad[missing[0]] = second
            names = [n for n in order if n in triad]

        # Two axes known: the third follows from a right-handed frame.
        missing = next(n for n in order if n not in triad)
        rd, td, nd = triad.get("rd"), triad.get("td"), triad.get("nd")
        if missing == "nd":
            triad["nd"] = np.cross(rd, td)
        elif missing == "td":
            triad["td"] = np.cross(nd, rd)
        else:
            triad["rd"] = np.cross(td, nd)
        triad[missing] /= np.linalg.norm(triad[missing])
        return triad

    def direction(self, spec: str | np.ndarray) -> np.ndarray:
        """Resolve *spec* against this frame, returning a unit vector."""
        if isinstance(spec, str) and spec.strip().lower() in self.axes:
            return self.axes[spec.strip().lower()]
        return parse_vector(spec)

    def label(self, spec: str | np.ndarray) -> str:
        """A short human-readable label for *spec*, used in plot titles."""
        if isinstance(spec, str):
            key = spec.strip().lower()
            if key in self.axes:
                return key.upper()
            return spec.strip()
        v = np.asarray(spec, dtype=float)
        return "[" + " ".join(f"{c:g}" for c in v) + "]"

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={np.round(v, 4).tolist()}" for k, v in self.axes.items())
        return f"SampleFrame({items})"

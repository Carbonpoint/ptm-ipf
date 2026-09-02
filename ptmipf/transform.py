"""Rigid rotations of a whole configuration.

Rotating the system is the atomistic counterpart of re-mounting a specimen in
the microscope: the atoms, their crystal orientations and the cell all turn
together about the cell centre, while the sample frame (RD, TD, ND) stays
where it is.  That is what changes the IPF colours, which is the point of
doing it: a rolled sheet whose normal happens to lie along ``x`` can be turned
so that it lies along ``z`` and the map reads like a textbook one.

Orientations are OVITO's crystal-to-sample quaternions ``(x, y, z, w)``, so a
rotation ``R`` of the sample turns each of them into ``R @ R_cs``, which in
quaternion form is ``q_R * q``.
"""

from __future__ import annotations

import dataclasses

import numpy as np

__all__ = [
    "parse_rotation",
    "rotate_positions",
    "rotate_quaternions",
    "rotate_result",
    "rotation_center",
    "rotation_matrix",
]


def rotation_matrix(axis, angle_deg: float) -> np.ndarray:
    """Right-handed rotation of *angle_deg* about the unit vector *axis*."""
    axis = np.asarray(axis, dtype=float).reshape(3)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise ValueError("the rotation axis must not be the zero vector")
    x, y, z = axis / norm
    angle = np.radians(float(angle_deg))
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def parse_rotation(spec: str) -> tuple[str, float]:
    """Split an ``AXIS:DEGREES`` rotation spec, as the CLI takes it.

    The axis is anything a direction accepts (``z``, ``nd``, ``1,1,0``), so it
    is returned unresolved; the caller resolves it against its sample frame.
    """
    text = str(spec).strip()
    axis, sep, angle = text.rpartition(":")
    if not sep or not axis.strip():
        raise ValueError(f"a rotation looks like AXIS:DEGREES, e.g. z:45, got {spec!r}")
    try:
        return axis.strip(), float(angle)
    except ValueError:
        raise ValueError(f"the rotation angle in {spec!r} is not a number") from None


def rotation_center(cell, positions=None) -> np.ndarray:
    """Where the system turns about: the cell centre, or the atoms' centre without one."""
    if cell is not None:
        cell = np.asarray(cell, dtype=float)
        return cell[:, 3] + 0.5 * cell[:, :3].sum(axis=1)
    if positions is None or len(positions) == 0:
        return np.zeros(3)
    positions = np.asarray(positions, dtype=float)
    return 0.5 * (positions.min(axis=0) + positions.max(axis=0))


def rotate_positions(positions, matrix, center) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    center = np.asarray(center, dtype=float)
    return (positions - center) @ np.asarray(matrix, dtype=float).T + center


def _matrix_to_quaternion(matrix) -> np.ndarray:
    """Unit quaternion ``(x, y, z, w)`` of a rotation matrix (Shepperd's method)."""
    m = np.asarray(matrix, dtype=float)
    trace = np.trace(m)
    if trace > 0:
        s = 2.0 * np.sqrt(1.0 + trace)
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def rotate_quaternions(quaternions, matrix) -> np.ndarray:
    """Left-multiply crystal-to-sample quaternions ``(x, y, z, w)`` by *matrix*."""
    q = np.asarray(quaternions, dtype=float).reshape(-1, 4)
    rx, ry, rz, rw = _matrix_to_quaternion(matrix)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # Hamilton product r * q with the scalar part last.
    out = np.empty_like(q)
    out[:, 3] = rw * w - rx * x - ry * y - rz * z
    out[:, 0] = rw * x + rx * w + ry * z - rz * y
    out[:, 1] = rw * y - rx * z + ry * w + rz * x
    out[:, 2] = rw * z + rx * y - ry * x + rz * w
    return out


def rotate_result(result, matrix, center=None):
    """A copy of *result* rotated rigidly by *matrix* about *center*.

    Positions, orientations and the cell turn together; the colours are left
    as they are, because which direction they refer to is the caller's
    decision (a recolouring usually follows).
    """
    matrix = np.asarray(matrix, dtype=float)
    if center is None:
        center = rotation_center(result.cell, result.positions)
    center = np.asarray(center, dtype=float)
    cell = None
    if result.cell is not None:
        cell = np.asarray(result.cell, dtype=float).copy()
        cell[:, :3] = matrix @ cell[:, :3]
        cell[:, 3] = matrix @ (cell[:, 3] - center) + center
    return dataclasses.replace(
        result,
        positions=rotate_positions(result.positions, matrix, center),
        orientations=rotate_quaternions(result.orientations, matrix),
        cell=cell,
    )

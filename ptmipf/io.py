"""Writers for IPF-coloured configurations.

Both writers keep the per-atom colour as three extra columns, so the result can
be re-read and rendered by OVITO, VMD or any other viewer without re-running
the analysis.

They also take *keys*: one scalar column per projection direction, produced by
:mod:`ptmipf.colormap`, which is what OVITO's own Color coding modifier needs
to repaint the atoms along IPF-X, IPF-Y or IPF-Z without a re-export.
"""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

import numpy as np

__all__ = [
    "SUPPORTED_FORMATS",
    "temporary_path",
    "write_extxyz",
    "write_lammps_dump",
    "write_result",
]

SUPPORTED_FORMATS = ("extxyz", "xyz", "lammps-dump", "dump")


@contextlib.contextmanager
def temporary_path(suffix: str = ""):
    """Yield a temporary path for a library that opens the file by name itself.

    ``tempfile.NamedTemporaryFile`` cannot serve here.  On Windows the handle it
    holds open locks the name, so OVITO or matplotlib reopening the same path
    fails with a permission error; that is why the web UI's 3D view and its
    exports came back empty there while working on Linux.  A temporary
    directory carries no such rule on any platform, and still cleans up.
    """
    with tempfile.TemporaryDirectory(prefix="ptmipf-") as directory:
        yield str(Path(directory) / ("scratch" + suffix))


def _lattice_string(cell: np.ndarray | None) -> str:
    if cell is None:
        return ""
    matrix = np.asarray(cell)[:3, :3].T  # OVITO stores column vectors
    return " ".join(f"{v:.8f}" for v in matrix.reshape(-1))


def _key_columns(result, keys) -> dict[str, np.ndarray]:
    """Validate the optional colour-coding columns against the atom count."""
    if not keys:
        return {}
    columns = {}
    for name, values in keys.items():
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) != result.n_atoms:
            raise ValueError(
                f"colour key {name!r} has {len(values)} values, "
                f"expected {result.n_atoms}"
            )
        columns[str(name)] = values
    return columns


def write_extxyz(result, path, keys=None) -> None:
    """Write an extended XYZ file with structure type, RMSD and IPF colour."""
    n = result.n_atoms
    names = np.array([result.type_names.get(int(t), str(t)) for t in result.particle_types])
    extra = _key_columns(result, keys)
    columns = [
        names,
        *result.positions.T,
        result.structure_types,
        result.rmsd,
        *result.colors.T,
        *extra.values(),
    ]
    header_props = "species:S:1:pos:R:3:structure_type:I:1:rmsd:R:1:color:R:3" + "".join(
        f":{name}:R:1" for name in extra
    )
    comment = [f"Properties={header_props}"]
    lattice = _lattice_string(result.cell)
    if lattice:
        comment.insert(0, f'Lattice="{lattice}"')
    comment.append(f'ipf_direction="{" ".join(f"{c:.6f}" for c in result.direction)}"')
    comment.append(f'ipf_direction_label="{result.direction_label}"')

    body = "{} {:.5f} {:.5f} {:.5f} {:d} {:.5f} {:.5f} {:.5f} {:.5f}" + " {:.8f}" * len(extra)
    with open(path, "w") as handle:
        handle.write(f"{n}\n")
        handle.write(" ".join(comment) + "\n")
        for row in zip(*columns):
            handle.write(
                body.format(row[0], *row[1:4], int(row[4]), row[5], *row[6:]) + "\n"
            )


def write_lammps_dump(result, path, keys=None) -> None:
    """Write a LAMMPS dump file whose colours bind to the atoms on reload.

    The column names matter: OVITO maps ``Color.R/G/B`` and ``StructureType``
    onto its standard particle properties, so the file opens already coloured by
    orientation.  Plain names such as ``r g b`` would come back as three
    unrelated per-atom values that have to be mapped by hand.  The colour-key
    columns are meant to arrive as user-defined properties under their own
    names, which is exactly what an unrecognised column name gives.
    """
    cell = np.asarray(result.cell) if result.cell is not None else None
    extra = _key_columns(result, keys)
    with open(path, "w") as handle:
        handle.write("ITEM: TIMESTEP\n")
        handle.write(f"{result.frame_index}\n")
        handle.write("ITEM: NUMBER OF ATOMS\n")
        handle.write(f"{result.n_atoms}\n")
        if cell is not None:
            origin = cell[:, 3]
            handle.write("ITEM: BOX BOUNDS pp pp pp\n")
            for axis in range(3):
                handle.write(f"{origin[axis]:.8f} {origin[axis] + cell[axis, axis]:.8f}\n")
        else:
            lo = result.positions.min(axis=0)
            hi = result.positions.max(axis=0)
            handle.write("ITEM: BOX BOUNDS pp pp pp\n")
            for axis in range(3):
                handle.write(f"{lo[axis]:.8f} {hi[axis]:.8f}\n")
        handle.write(
            "ITEM: ATOMS id type x y z StructureType rmsd Color.R Color.G Color.B"
            + "".join(f" {name}" for name in extra)
            + "\n"
        )
        values = list(extra.values())
        for i in range(result.n_atoms):
            x, y, z = result.positions[i]
            r, g, b = result.colors[i]
            line = (
                f"{i + 1} {int(result.particle_types[i])} {x:.5f} {y:.5f} {z:.5f} "
                f"{int(result.structure_types[i])} {result.rmsd[i]:.5f} "
                f"{r:.5f} {g:.5f} {b:.5f}"
            )
            for column in values:
                line += f" {column[i]:.8f}"
            handle.write(line + "\n")


def write_result(result, path, fmt: str | None = None, keys=None) -> str:
    """Write *result* to *path*, guessing the format from the extension."""
    path = str(path)
    if fmt is None:
        lowered = path.lower()
        if lowered.endswith((".dump", ".lammpstrj")):
            fmt = "lammps-dump"
        else:
            fmt = "extxyz"
    fmt = fmt.lower()
    if fmt in ("extxyz", "xyz"):
        write_extxyz(result, path, keys=keys)
    elif fmt in ("lammps-dump", "dump"):
        write_lammps_dump(result, path, keys=keys)
    else:
        raise ValueError(f"unsupported output format {fmt!r}; use one of {SUPPORTED_FORMATS}")
    return fmt

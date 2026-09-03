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
import io
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

    formats = ["%.5f", "%.5f", "%.5f", "%d", "%.5f", "%.5f", "%.5f", "%.5f"]
    formats += ["%.8f"] * len(extra)
    with open(path, "w") as handle:
        handle.write(f"{n}\n")
        handle.write(" ".join(comment) + "\n")
        # The numbers are formatted a block at a time by NumPy and the species
        # names put in front of the finished lines: formatting per atom in
        # Python is several times slower, and these files run to millions of
        # lines.
        _write_columns(handle, columns[1:], formats, labels=columns[0])


def _write_columns(handle, columns, formats, labels=None) -> None:
    """Write per-atom numeric columns, formatting them a block at a time.

    ``savetxt`` formats a whole block in one call, which is far quicker than a
    Python loop over atoms, but it holds the formatted text in memory, so the
    blocks keep that bounded for a configuration of any size.  *labels* is an
    optional column of strings written in front of each line, which is how the
    extended XYZ species column is carried without turning the whole table
    into strings.
    """
    fmt = " ".join(formats)
    n = len(columns[0])
    block = 200_000
    for start in range(0, n, block):
        stop = start + block
        table = np.column_stack([np.asarray(c[start:stop]) for c in columns])
        if labels is None:
            np.savetxt(handle, table, fmt=fmt)
            continue
        text = io.StringIO()
        np.savetxt(text, table, fmt=fmt)
        handle.write(
            "".join(
                f"{name} {line}\n"
                for name, line in zip(labels[start:stop], text.getvalue().splitlines())
            )
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
        columns = [
            np.arange(1, result.n_atoms + 1),
            result.particle_types,
            *result.positions.T,
            result.structure_types,
            result.rmsd,
            *result.colors.T,
            *extra.values(),
        ]
        formats = ["%d", "%d", "%.5f", "%.5f", "%.5f", "%d", "%.5f", "%.5f", "%.5f", "%.5f"]
        formats += ["%.8f"] * len(extra)
        _write_columns(handle, columns, formats)


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

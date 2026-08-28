"""Writers for IPF-coloured configurations.

Both writers keep the per-atom colour as three extra columns, so the result can
be re-read and rendered by OVITO, VMD or any other viewer without re-running
the analysis.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SUPPORTED_FORMATS", "write_extxyz", "write_lammps_dump", "write_result"]

SUPPORTED_FORMATS = ("extxyz", "xyz", "lammps-dump", "dump")


def _lattice_string(cell: np.ndarray | None) -> str:
    if cell is None:
        return ""
    matrix = np.asarray(cell)[:3, :3].T  # OVITO stores column vectors
    return " ".join(f"{v:.8f}" for v in matrix.reshape(-1))


def write_extxyz(result, path) -> None:
    """Write an extended XYZ file with structure type, RMSD and IPF colour."""
    n = result.n_atoms
    names = np.array([result.type_names.get(int(t), str(t)) for t in result.particle_types])
    columns = [
        names,
        *result.positions.T,
        result.structure_types,
        result.rmsd,
        *result.colors.T,
    ]
    header_props = (
        "species:S:1:pos:R:3:structure_type:I:1:rmsd:R:1:color:R:3"
    )
    comment = [f"Properties={header_props}"]
    lattice = _lattice_string(result.cell)
    if lattice:
        comment.insert(0, f'Lattice="{lattice}"')
    comment.append(f'ipf_direction="{" ".join(f"{c:.6f}" for c in result.direction)}"')
    comment.append(f'ipf_direction_label="{result.direction_label}"')

    with open(path, "w") as handle:
        handle.write(f"{n}\n")
        handle.write(" ".join(comment) + "\n")
        for row in zip(*columns):
            handle.write(
                "{} {:.5f} {:.5f} {:.5f} {:d} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(
                    row[0], *row[1:4], int(row[4]), row[5], *row[6:9]
                )
            )


def write_lammps_dump(result, path) -> None:
    """Write a LAMMPS dump file whose colours bind to the atoms on reload.

    The column names matter: OVITO maps ``Color.R/G/B`` and ``StructureType``
    onto its standard particle properties, so the file opens already coloured by
    orientation.  Plain names such as ``r g b`` would come back as three
    unrelated per-atom values that have to be mapped by hand.
    """
    cell = np.asarray(result.cell) if result.cell is not None else None
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
            "ITEM: ATOMS id type x y z StructureType rmsd Color.R Color.G Color.B\n"
        )
        for i in range(result.n_atoms):
            x, y, z = result.positions[i]
            r, g, b = result.colors[i]
            handle.write(
                f"{i + 1} {int(result.particle_types[i])} {x:.5f} {y:.5f} {z:.5f} "
                f"{int(result.structure_types[i])} {result.rmsd[i]:.5f} "
                f"{r:.5f} {g:.5f} {b:.5f}\n"
            )


def write_result(result, path, fmt: str | None = None) -> str:
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
        write_extxyz(result, path)
    elif fmt in ("lammps-dump", "dump"):
        write_lammps_dump(result, path)
    else:
        raise ValueError(f"unsupported output format {fmt!r}; use one of {SUPPORTED_FORMATS}")
    return fmt

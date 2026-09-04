"""Build a random polycrystal to deform.

This is the starting structure for the built-in examples: a cube of randomly
oriented grains, periodic in all three directions, ready for LAMMPS and ready
for PTM.

The geometry is done by `atomsk <https://atomsk.univ-lille.fr>`_, which builds
Voronoi polycrystals at full density.  That matters more than it sounds.  A
builder that places rotated lattice blocks and trims the overlaps back loses
several percent of its atoms at the boundaries, and the loss is not cosmetic:
the boundary region ends up too open, which softens the elastic response and
brings yield forward.  Rebuilding the iron runs of the showcase campaign on
atomsk moved the peak stress of a random cell from 4.99 to 5.94 GPa, with a
visibly steeper elastic slope, purely from closing that gap.

atomsk's node file takes three angles per grain.  They are not Euler angles:
they are extrinsic rotations about the Cartesian x, y and z axes, composing as
``R = Rz(tz) @ Ry(ty) @ Rx(tx)``, crystal to sample.  That was established by
building single grains at known angles and measuring them back with PTM, and
:func:`xyz_angles` is checked against its own matrices before anything is
written.

:func:`voronoi_polycrystal` is a fallback for machines with no atomsk, in pure
NumPy.  It is honest about what it costs: it reports its density, and the
examples label a structure built that way, because it is not the same thing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "Polycrystal",
    "atomsk_polycrystal",
    "build_polycrystal",
    "find_atomsk",
    "random_rotations",
    "voronoi_polycrystal",
    "xyz_angles",
]

#: Fractional cell coordinates of the conventional cubic cells.
_MOTIFS = {
    "fcc": np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),
    "bcc": np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
}

#: Nearest neighbour distance in units of the lattice parameter.
_NEIGHBOUR = {"fcc": 2.0**-0.5, "bcc": 3.0**0.5 / 2.0}


@dataclass
class Polycrystal:
    """A periodic polycrystal and the record of how it was built."""

    positions: np.ndarray  #: (n, 3) in angstrom, inside the box
    rotations: np.ndarray  #: (n_grains, 3, 3) crystal to sample, per grain
    seeds: np.ndarray  #: (n_grains, 3) Voronoi seed points
    box: float  #: cube edge in angstrom
    a0: float
    structure: str
    element: str
    builder: str  #: "atomsk" or "voronoi"
    files: dict[str, str] = field(default_factory=dict)
    removed: int = 0  #: atoms the fallback builder deleted as overlaps
    min_separation: float = 0.0  #: closest pair in the cell, in angstrom

    @property
    def n_atoms(self) -> int:
        return len(self.positions)

    @property
    def n_grains(self) -> int:
        return len(self.rotations)

    @property
    def ideal_count(self) -> int:
        """Atoms a perfect single crystal of the same volume would hold."""
        per_cell = len(_MOTIFS[self.structure])
        return int(round(per_cell * (self.box / self.a0) ** 3))

    @property
    def density(self) -> float:
        """Atom count as a fraction of the perfect single crystal.

        A deficit of a couple of percent is real at these grain sizes: the
        boundaries carry excess volume and, with grains a few nanometres
        across, they are a large fraction of the cell.  A deficit of ten
        percent is not, and is what the module docstring is about.
        """
        return self.n_atoms / max(self.ideal_count, 1)

    @property
    def mean_grain_size(self) -> float:
        """Diameter of a sphere of the mean grain volume, in angstrom."""
        volume = self.box**3 / max(self.n_grains, 1)
        return float((6.0 * volume / np.pi) ** (1.0 / 3.0))

    def summary(self) -> str:
        return (
            f"{self.n_atoms} atoms of {self.element} in a {self.box:.2f} A cube, "
            f"{self.n_grains} grains averaging {self.mean_grain_size:.1f} A across, "
            f"{100 * self.density:.1f} % of single-crystal density "
            f"(built with {self.builder})"
        )


# ----------------------------------------------------------------------
# orientations
# ----------------------------------------------------------------------
def random_rotations(n: int, rng: np.random.Generator) -> np.ndarray:
    """*n* rotation matrices drawn uniformly from SO(3).

    Uniform on the group, not uniform in three Euler angles, which would crowd
    the poles and give a texture nobody asked for.  Shoemake's quaternion
    construction is the usual way to get it right.
    """
    u1, u2, u3 = rng.random((3, n))
    a, b = np.sqrt(1.0 - u1), np.sqrt(u1)
    x = a * np.sin(2 * np.pi * u2)
    y = a * np.cos(2 * np.pi * u2)
    z = b * np.sin(2 * np.pi * u3)
    w = b * np.cos(2 * np.pi * u3)
    matrices = np.empty((n, 3, 3))
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - z * w)
    matrices[:, 0, 2] = 2 * (x * z + y * w)
    matrices[:, 1, 0] = 2 * (x * y + z * w)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - x * w)
    matrices[:, 2, 0] = 2 * (x * z - y * w)
    matrices[:, 2, 1] = 2 * (y * z + x * w)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


def xyz_angles(rotation: np.ndarray) -> tuple[float, float, float]:
    """Decompose a crystal to sample rotation into atomsk's three angles.

    Returns degrees ``(tx, ty, tz)`` with ``R = Rz(tz) @ Ry(ty) @ Rx(tx)``.
    """
    r = np.asarray(rotation, dtype=float)
    ty = np.degrees(np.arcsin(np.clip(-r[2, 0], -1.0, 1.0)))
    if abs(r[2, 0]) < 1.0 - 1e-10:
        tx = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
        tz = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    else:
        # Gimbal lock, ty = +-90: only tx - tz (or tx + tz) is determined, so
        # everything is folded into tx.  Which of the two combinations it is
        # depends on the sign of ty, which is what r[2, 0] carries.
        tx = np.degrees(np.arctan2(-r[2, 0] * r[0, 1], r[1, 1]))
        tz = 0.0
    return float(tx), float(ty), float(tz)


def _axis_rotation(axis, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    t = np.radians(degrees)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def _check_angles(rotations: np.ndarray) -> None:
    """Reproduce every matrix from its angles before writing the node file."""
    for rotation in rotations:
        tx, ty, tz = xyz_angles(rotation)
        rebuilt = (
            _axis_rotation([0, 0, 1], tz)
            @ _axis_rotation([0, 1, 0], ty)
            @ _axis_rotation([1, 0, 0], tx)
        )
        if np.abs(rebuilt - rotation).max() > 1e-9:
            raise RuntimeError("the atomsk angle decomposition did not reproduce its matrix")


# ----------------------------------------------------------------------
# atomsk
# ----------------------------------------------------------------------
def find_atomsk(explicit: str | None = None) -> str | None:
    """Path to the atomsk executable, or None.

    Looks at an explicit path, then ``PTMIPF_ATOMSK``, then the PATH.
    """
    for candidate in (explicit, os.environ.get("PTMIPF_ATOMSK")):
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which("atomsk")


ATOMSK_HELP = (
    "atomsk was not found.  Install it from https://atomsk.univ-lille.fr/dl.php "
    "(binaries for Linux, macOS and Windows), put it on the PATH or point "
    "PTMIPF_ATOMSK at it, and build again.  Without it the fallback Voronoi "
    "builder can be used instead, at a few percent lower boundary density."
)


def _run_atomsk(binary: str, arguments: list[str], directory: Path) -> None:
    result = subprocess.run(
        [binary, *arguments],
        cwd=str(directory),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-6:]
        raise RuntimeError(
            "atomsk failed: " + " / ".join(tail) if tail else "atomsk failed with no output"
        )


def _read_lammps_data(path: Path) -> tuple[np.ndarray, float]:
    """Positions and the cube edge from a LAMMPS data file atomsk wrote."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    bounds, start, count = {}, None, 0
    for index, line in enumerate(lines):
        stripped = line.split("#")[0].strip()
        if stripped.endswith(("xlo xhi", "ylo yhi", "zlo zhi")):
            fields = stripped.split()
            bounds[fields[-1][0]] = (float(fields[0]), float(fields[1]))
        elif stripped.endswith("atoms") and len(stripped.split()) == 2:
            count = int(stripped.split()[0])
        elif stripped.split("#")[0].strip() == "Atoms" or stripped.startswith("Atoms"):
            start = index + 1
    if start is None or not bounds:
        raise RuntimeError(f"{path.name} is not a LAMMPS data file this can read")

    positions = []
    for line in lines[start:]:
        fields = line.split()
        if not fields:
            continue
        if not fields[0].lstrip("-").isdigit():
            break  # the next section header
        # id type x y z, which is what atom_style atomic writes.
        positions.append([float(fields[2]), float(fields[3]), float(fields[4])])
        if len(positions) == count:
            break
    origin = np.array([bounds[axis][0] for axis in "xyz"])
    edge = float(bounds["x"][1] - bounds["x"][0])
    return np.asarray(positions, dtype=float) - origin, edge


def atomsk_polycrystal(
    element: str,
    box: float,
    n_grains: int,
    directory,
    structure: str = "fcc",
    a0: float = 3.615,
    seed: int | None = None,
    stem: str = "structure",
    binary: str | None = None,
    rotations: np.ndarray | None = None,
) -> Polycrystal:
    """Build a Voronoi polycrystal with atomsk, into *directory*.

    Writes ``<stem>.lmp`` for LAMMPS, ``<stem>.xyz`` for ptm-ipf, the unit cell
    and the node file that produced them, so the build is reproducible from the
    directory alone.
    """
    binary = find_atomsk(binary)
    if binary is None:
        raise RuntimeError(ATOMSK_HELP)
    if structure not in _MOTIFS:
        raise ValueError(f"structure must be one of {sorted(_MOTIFS)}, got {structure!r}")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # A whole number of cells so that an unrotated cell is seamless.  A rotated
    # grain still meets its own periodic image at a seam, which is a real grain
    # boundary like any other and is what every Voronoi builder produces.
    box = a0 * max(1, int(round(box / a0)))

    if rotations is None:
        rotations = random_rotations(n_grains, rng)
    rotations = np.asarray(rotations, dtype=float).reshape(-1, 3, 3)
    _check_angles(rotations)
    seeds = rng.random((len(rotations), 3)) * box

    nodes = directory / f"{stem}.nodes"
    with open(nodes, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"box {box:.6f} {box:.6f} {box:.6f}\n")
        for point, rotation in zip(seeds, rotations):
            tx, ty, tz = xyz_angles(rotation)
            handle.write(
                f"node {point[0]:.4f} {point[1]:.4f} {point[2]:.4f} "
                f"{tx:.6f} {ty:.6f} {tz:.6f}\n"
            )

    unit = f"{stem}_unit.xsf"
    _run_atomsk(
        binary,
        ["--create", structure, f"{a0:.6f}", element, unit, "-overwrite"],
        directory,
    )
    _run_atomsk(
        binary,
        ["--polycrystal", unit, nodes.name, stem, "lmp", "exyz", "-wrap", "-overwrite"],
        directory,
    )

    data = directory / f"{stem}.lmp"
    if not data.is_file():
        raise RuntimeError(f"atomsk did not write {data.name}")
    positions, edge = _read_lammps_data(data)

    from scipy.spatial import cKDTree

    separation = float(
        cKDTree(np.mod(positions, edge), boxsize=edge).query(np.mod(positions, edge), k=2)[0][
            :, 1
        ].min()
    )
    return Polycrystal(
        positions=positions,
        rotations=rotations,
        seeds=seeds,
        box=edge,
        a0=a0,
        structure=structure,
        element=element,
        builder="atomsk",
        files={
            "data": data.name,
            "xyz": f"{stem}.xyz",
            "nodes": nodes.name,
            "unit_cell": unit,
        },
        min_separation=separation,
    )


# ----------------------------------------------------------------------
# the fallback builder
# ----------------------------------------------------------------------
def _lattice_filling(box: float, a0: float, structure: str, rotation: np.ndarray) -> np.ndarray:
    """Rotated lattice points covering the whole box, centred on it."""
    motif = _MOTIFS[structure]
    reach = int(np.ceil(box * np.sqrt(3.0) / (2.0 * a0))) + 1
    grid = np.arange(-reach, reach + 1)
    cells = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3)
    points = (cells[:, None, :] + motif[None, :, :]).reshape(-1, 3) * a0
    return points @ rotation.T + 0.5 * box


def _nearest_seed(points: np.ndarray, seeds: np.ndarray, box: float) -> np.ndarray:
    """Index of the nearest seed to each point, under the minimum image rule.

    The tessellation has to be periodic, or the grains at the faces would not
    meet their own images and a slab of open boundary would run through the
    middle of the deformation.
    """
    best = np.zeros(len(points), dtype=np.int32)
    closest = np.full(len(points), np.inf)
    for index, seed in enumerate(seeds):
        delta = points - seed
        delta -= box * np.round(delta / box)
        distance = np.einsum("ij,ij->i", delta, delta)
        closer = distance < closest
        closest[closer] = distance[closer]
        best[closer] = index
    return best


def _drop_overlaps(positions: np.ndarray, box: float, minimum: float):
    """Remove one atom of every pair closer than *minimum*, periodically.

    The cutoff has to stay well under the nearest neighbour distance: across a
    general boundary the two lattices meet at random offsets, so a cutoff
    anywhere near the bulk spacing finds a partner for almost every boundary
    atom and empties the boundary out, which is the failure this module's
    docstring is about.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(positions, boxsize=box)
    doomed = np.zeros(len(positions), dtype=bool)
    for low, high in sorted(tree.query_pairs(minimum)):
        # A chain of overlaps thins to a single survivor rather than being wiped.
        if not doomed[low] and not doomed[high]:
            doomed[high] = True
    survivors = positions[~doomed]
    separation = float(cKDTree(survivors, boxsize=box).query(survivors, k=2)[0][:, 1].min())
    return survivors, int(doomed.sum()), separation


def voronoi_polycrystal(
    element: str = "Cu",
    box: float = 48.0,
    n_grains: int = 8,
    structure: str = "fcc",
    a0: float = 3.615,
    seed: int | None = None,
    overlap_factor: float = 0.50,
    rotations: np.ndarray | None = None,
) -> Polycrystal:
    """A periodic cube of randomly oriented grains, in pure NumPy.

    The fallback for machines with no atomsk.  Every Voronoi cell is filled
    completely and only then are the atoms that overlap across a boundary
    removed, one of each too-close pair rather than a whole layer, which keeps
    the density at about 97 per cent of the perfect crystal.  That is close but
    not equal to what atomsk produces, so the caller should say which builder
    ran.
    """
    if structure not in _MOTIFS:
        raise ValueError(f"structure must be one of {sorted(_MOTIFS)}, got {structure!r}")
    if n_grains < 1:
        raise ValueError("a polycrystal needs at least one grain")

    rng = np.random.default_rng(seed)
    box = a0 * max(1, int(round(box / a0)))
    if rotations is None:
        rotations = random_rotations(n_grains, rng)
    rotations = np.asarray(rotations, dtype=float).reshape(-1, 3, 3)
    seeds = rng.random((len(rotations), 3)) * box

    kept = []
    for index, rotation in enumerate(rotations):
        points = _lattice_filling(box, a0, structure, rotation)
        points = points[np.all((points >= 0.0) & (points < box), axis=1)]
        kept.append(points[_nearest_seed(points, seeds, box) == index])

    positions = np.concatenate(kept) if kept else np.zeros((0, 3))
    minimum = overlap_factor * _NEIGHBOUR[structure] * a0
    positions, removed, separation = _drop_overlaps(positions, box, minimum)

    return Polycrystal(
        positions=positions,
        rotations=rotations,
        seeds=seeds,
        box=box,
        a0=a0,
        structure=structure,
        element=element,
        builder="voronoi",
        removed=removed,
        min_separation=separation,
    )


def build_polycrystal(
    element: str,
    box: float,
    n_grains: int,
    directory,
    structure: str = "fcc",
    a0: float = 3.615,
    seed: int | None = None,
    stem: str = "structure",
    builder: str = "atomsk",
    binary: str | None = None,
) -> Polycrystal:
    """Build with atomsk, or with the NumPy fallback when asked.

    *builder* is ``atomsk``, ``voronoi`` or ``auto``.  ``auto`` prefers atomsk
    and falls back silently, which is convenient but hides which one ran, so
    the caller should report :attr:`Polycrystal.builder` either way.
    """
    if builder not in ("atomsk", "voronoi", "auto"):
        raise ValueError(f"builder must be atomsk, voronoi or auto, got {builder!r}")
    if builder == "voronoi" or (builder == "auto" and find_atomsk(binary) is None):
        crystal = voronoi_polycrystal(
            element, box, n_grains, structure=structure, a0=a0, seed=seed
        )
        crystal.files = write_structure_files(crystal, directory, stem)
        return crystal
    return atomsk_polycrystal(
        element,
        box,
        n_grains,
        directory,
        structure=structure,
        a0=a0,
        seed=seed,
        stem=stem,
        binary=binary,
    )


def write_structure_files(crystal: Polycrystal, directory, stem: str = "structure") -> dict:
    """Write the fallback builder's output as LAMMPS data and extended XYZ."""
    from .lammps import write_data_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = directory / f"{stem}.lmp"
    xyz = directory / f"{stem}.xyz"
    write_data_file(crystal, data)

    with open(xyz, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{crystal.n_atoms}\n")
        lattice = f"{crystal.box:.8f} 0.0 0.0 0.0 {crystal.box:.8f} 0.0 0.0 0.0 {crystal.box:.8f}"
        handle.write(
            f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        )
        for x, y, z in crystal.positions:
            handle.write(f"{crystal.element} {x:.5f} {y:.5f} {z:.5f}\n")
    return {"data": data.name, "xyz": xyz.name}

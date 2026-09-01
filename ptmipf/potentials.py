"""EAM potentials from the NIST Interatomic Potentials Repository.

The starter examples need a potential file, and the honest place to get one is
the repository that curates it.  Each entry here names a specific version of a
specific NIST entry, together with the lattice parameter, mass and pair style
that potential was fitted for, so the generated LAMMPS input is consistent with
the file it reads rather than with a remembered number.

Downloads are pinned: the URL names an immutable version directory and the
content is checked against a recorded SHA-256.  A file that arrives changed is
refused rather than quietly used, because a silently different potential is a
silently different result.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

__all__ = ["POTENTIALS", "Potential", "download_potential", "potential_for"]

_BASE = "https://www.ctcms.nist.gov/potentials"


@dataclass(frozen=True)
class Potential:
    """One EAM potential, and everything the input script needs to use it."""

    element: str
    structure: str  #: lattice the potential's ground state is, as PTM names it
    a0: float  #: lattice parameter in angstrom, from the potential file itself
    mass: float  #: atomic mass in g/mol, from the potential file itself
    pair_style: str  #: LAMMPS pair style; eam/fs and eam/alloy are not the same
    filename: str
    entry: str  #: NIST entry id
    version: int
    sha256: str
    citation: str

    @property
    def url(self) -> str:
        return f"{_BASE}/Download/{self.entry}/{self.version}/{self.filename}"

    @property
    def entry_url(self) -> str:
        return f"{_BASE}/entry/{self.entry}/"


#: The starter set: three fcc metals and one bcc, all with well known
#: potentials and all cheap enough to deform on a laptop.
POTENTIALS = {
    "Cu": Potential(
        element="Cu",
        structure="fcc",
        a0=3.615,
        mass=63.546,
        pair_style="eam/alloy",
        filename="Cu01.eam.alloy",
        entry="2001--Mishin-Y-Mehl-M-J-Papaconstantopoulos-D-A-et-al--Cu-1",
        version=2,
        sha256="5e374260448eea5c26de59432d75e702c1778cc6999d2527a789e5387ce9dc9e",
        citation="Y. Mishin, M. J. Mehl, D. A. Papaconstantopoulos, A. F. Voter and "
        "J. D. Kress, Phys. Rev. B 63, 224106 (2001)",
    ),
    "Al": Potential(
        element="Al",
        structure="fcc",
        a0=4.050,
        mass=26.982,
        pair_style="eam/alloy",
        filename="Al99.eam.alloy",
        entry="1999--Mishin-Y-Farkas-D-Mehl-M-J-Papaconstantopoulos-D-A--Al",
        version=2,
        sha256="60c8a085be79d273324ab421f5b1447578fef55c1acfc6492c0999f15ee8a284",
        citation="Y. Mishin, D. Farkas, M. J. Mehl and D. A. Papaconstantopoulos, "
        "Phys. Rev. B 59, 3393 (1999)",
    ),
    "Ni": Potential(
        element="Ni",
        structure="fcc",
        a0=3.520,
        mass=58.710,
        pair_style="eam/alloy",
        filename="Ni99.eam.alloy",
        entry="1999--Mishin-Y-Farkas-D-Mehl-M-J-Papaconstantopoulos-D-A--Ni",
        version=2,
        sha256="fb84d3dc0ed9d68136ee3eaa638e3dd178f3057e7de549fe0029069e749a9876",
        citation="Y. Mishin, D. Farkas, M. J. Mehl and D. A. Papaconstantopoulos, "
        "Phys. Rev. B 59, 3393 (1999)",
    ),
    "Fe": Potential(
        element="Fe",
        structure="bcc",
        a0=2.855312,
        mass=55.850,
        # eam/fs, not eam/alloy: the file carries separate density functions and
        # reading it with the wrong style is silently wrong rather than an error.
        pair_style="eam/fs",
        filename="Fe_2.eam.fs",
        entry="2003--Mendelev-M-I-Han-S-Srolovitz-D-J-et-al--Fe-2",
        version=3,
        sha256="7b000d2bb2b654f9af7890b18e9d0ad06c15635077169d9822cb6e44393496c8",
        citation="M. I. Mendelev, S. Han, D. J. Srolovitz, G. J. Ackland, D. Y. Sun "
        "and M. Asta, Philos. Mag. 83, 3977 (2003)",
    ),
}


def potential_for(element: str) -> Potential:
    """The catalogued potential for *element*, by symbol."""
    key = element.strip().capitalize()
    if key not in POTENTIALS:
        raise ValueError(
            f"no catalogued potential for {element!r}; "
            f"available: {', '.join(sorted(POTENTIALS))}"
        )
    return POTENTIALS[key]


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_potential(element: str, directory, timeout: float = 120.0) -> Path:
    """Fetch the potential for *element* into *directory* and verify it.

    A file already there with the right digest is kept, so building the same
    example twice costs one download.  Returns the path to the potential file.
    """
    import urllib.error
    import urllib.request

    potential = potential_for(element)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / potential.filename

    if target.is_file() and _digest(target) == potential.sha256:
        return target

    request = urllib.request.Request(
        potential.url, headers={"User-Agent": "ptm-ipf potential fetch"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(
            f"could not download the {potential.element} potential from NIST "
            f"({exc}).  Fetch {potential.url} by hand and put "
            f"{potential.filename} in {directory}, then build again."
        ) from None

    got = hashlib.sha256(body).hexdigest()
    if got != potential.sha256:
        raise RuntimeError(
            f"the {potential.element} potential downloaded from NIST does not match "
            f"the recorded checksum (expected {potential.sha256[:12]}, got "
            f"{got[:12]}).  The repository entry may have been revised; nothing was "
            "written."
        )
    target.write_bytes(body)
    return target

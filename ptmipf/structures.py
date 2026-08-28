"""Mapping between OVITO's PTM structure types and crystallographic settings.

The reference frame of each polyhedral template matching template was
determined empirically (see ``tests/test_ptm_convention.py``): the orientation
quaternion stored by OVITO is the **crystal-to-sample** rotation, cubic
templates have their crystal axes along the Cartesian axes, and the hexagonal
templates have ``c`` along ``z`` and ``a1`` along ``x``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["STRUCTURES", "Structure", "get_structure", "structure_names"]


@dataclass(frozen=True)
class Structure:
    """A crystal structure that PTM can identify and that can be IPF coloured."""

    name: str
    #: Name of the corresponding attribute of ``PolyhedralTemplateMatchingModifier.Type``.
    ptm_type: str
    #: Laue group used for the inverse pole figure colour key.
    laue: str
    #: Human readable description, shown by ``--list-structures``.
    description: str
    #: False for structures without a crystallographic orientation (icosahedral).
    colorable: bool = True


STRUCTURES: dict[str, Structure] = {
    s.name: s
    for s in (
        Structure("fcc", "FCC", "cubic", "Face-centred cubic (Al, Cu, Ni, gamma-Fe)"),
        Structure("hcp", "HCP", "hexagonal", "Hexagonal close-packed (Mg, Ti, Zr, Zn)"),
        Structure("bcc", "BCC", "cubic", "Body-centred cubic (Fe, W, Mo, Nb)"),
        Structure("sc", "SC", "cubic", "Simple cubic"),
        Structure("cubic_diamond", "CUBIC_DIAMOND", "cubic", "Cubic diamond (Si, Ge, C)"),
        Structure("hex_diamond", "HEX_DIAMOND", "hexagonal", "Hexagonal diamond (lonsdaleite)"),
        Structure("graphene", "GRAPHENE", "hexagonal", "Graphene"),
        Structure("ico", "ICO", "cubic", "Icosahedral (no lattice orientation)", False),
    )
}

#: Structures enabled by default, matching OVITO's own defaults.
DEFAULT_STRUCTURES = ("fcc", "hcp", "bcc")


def structure_names() -> list[str]:
    return list(STRUCTURES)


def get_structure(name: str) -> Structure:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"diamond": "cubic_diamond", "dia": "cubic_diamond", "lonsdaleite": "hex_diamond"}
    key = aliases.get(key, key)
    try:
        return STRUCTURES[key]
    except KeyError:
        raise KeyError(
            f"unknown structure {name!r}; available: " + ", ".join(structure_names())
        ) from None

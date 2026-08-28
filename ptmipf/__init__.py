"""ptm-ipf: EBSD-style inverse pole figure colouring for atomistic simulations.

The package runs OVITO's polyhedral template matching (PTM) on an atomistic
configuration, converts the per-atom orientation quaternions into inverse pole
figure colours using the standard EDAX/TSL colour key, and produces the
matching colour key, pole figures and orientation density plots.

Typical use::

    from ptmipf import analyse, ipf_legend

    result = analyse("mg.dump", direction="z", structures=("hcp",))
    ipf_legend("6/mmm", direction_label="ND", filename="key.png")
"""

from __future__ import annotations

__version__ = "0.1.0"

from .colorkey import IPFColorKey
from .frames import SampleFrame, parse_vector
from .structures import STRUCTURES, get_structure, structure_names
from .symmetry import LAUE_GROUPS, LaueGroup, get_laue_group

__all__ = [
    "LAUE_GROUPS",
    "STRUCTURES",
    "IPFColorKey",
    "LaueGroup",
    "SampleFrame",
    "__version__",
    "analyse",
    "analyze",
    "get_laue_group",
    "get_structure",
    "ipf_color_modifier",
    "ipf_density",
    "ipf_legend",
    "parse_vector",
    "pole_figure",
    "quaternions_to_matrices",
    "structure_names",
    "write_result",
]


def __getattr__(name):
    """Import the OVITO- and matplotlib-backed helpers lazily."""
    if name in ("analyse", "analyze", "ipf_color_modifier", "quaternions_to_matrices"):
        from . import analysis

        return getattr(analysis, name)
    if name == "ipf_legend":
        from .legend import ipf_legend

        return ipf_legend
    if name in ("pole_figure", "ipf_density"):
        from . import polefigure

        return getattr(polefigure, name)
    if name == "write_result":
        from .io import write_result

        return write_result
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

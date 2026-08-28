"""Selection tests that run real polyhedral template matching."""

import numpy as np
import pytest

pytest.importorskip("ovito")

from ptmipf.analysis import analyse  # noqa: E402
from ptmipf.select import (  # noqa: E402
    combine,
    select_by_ipf_direction,
    select_by_misorientation,
    select_by_region,
    select_by_rmsd,
    select_by_structure,
    select_by_type,
)

from .conftest import rotation_matrix  # noqa: E402


@pytest.fixture(scope="module")
def hcp_result(tmp_path_factory):
    ase_build = pytest.importorskip("ase.build")
    ase_io = pytest.importorskip("ase.io")
    # A crystal tilted 20 degrees, so a 15 degree basal query must reject it and
    # a 25 degree one must accept it.
    atoms = ase_build.bulk("Mg", "hcp", a=3.2094, c=5.2108).repeat((7, 7, 7))
    tilt = rotation_matrix([1.0, 0.0, 0.0], 20.0)
    atoms.set_cell(np.array(atoms.cell) @ tilt.T, scale_atoms=True)
    path = tmp_path_factory.mktemp("sel") / "tilted.xyz"
    ase_io.write(str(path), atoms, format="extxyz")
    return analyse(str(path), direction="z", structures=("hcp",))


def test_basal_selection_respects_the_tolerance(hcp_result):
    tight = select_by_ipf_direction(hcp_result, "0001", "z", 15.0, "hcp")
    loose = select_by_ipf_direction(hcp_result, "0001", "z", 25.0, "hcp")
    hcp = hcp_result.mask("hcp")
    assert tight.sum() == 0
    assert loose.sum() == hcp.sum()
    assert not loose[~hcp].any()  # atoms without an orientation are never selected


def test_selection_of_a_perpendicular_direction(hcp_result):
    """The c axis is 20 degrees from z, so it is 70 degrees from the x-z plane."""
    assert select_by_ipf_direction(hcp_result, "0001", "y", 25.0, "hcp").sum() == 0
    assert select_by_ipf_direction(hcp_result, "0001", "0,-1,2.7475", 5.0, "hcp").sum() > 0


def test_misorientation_selection_picks_the_whole_single_crystal(hcp_result):
    hcp = np.flatnonzero(hcp_result.mask("hcp"))
    selected = select_by_misorientation(hcp_result, int(hcp[0]), 5.0, "hcp")
    assert selected.sum() == len(hcp)


def test_subset_recomputes_counts_and_keeps_colours(hcp_result):
    mask = select_by_region(hcp_result, "z", maximum=float(np.median(hcp_result.positions[:, 2])))
    subset = hcp_result.subset(mask)
    assert subset.n_atoms == mask.sum()
    assert subset.counts["hcp"] == int(hcp_result.mask("hcp")[mask].sum())
    assert np.allclose(subset.colors, hcp_result.colors[mask])
    assert subset.summary()


def test_subset_accepts_indices(hcp_result):
    subset = hcp_result.subset(np.array([0, 5, 9]))
    assert subset.n_atoms == 3
    with pytest.raises(ValueError):
        hcp_result.subset(np.ones(3, dtype=bool))


def test_recolor_matches_a_fresh_analysis(hcp_result):
    recoloured = hcp_result.recolor("x")
    assert recoloured.direction_label == "X"
    assert not np.allclose(recoloured.colors, hcp_result.colors)
    # Recolouring must agree with having asked for that direction in the first place.
    assert np.allclose(hcp_result.recolor("z").colors, hcp_result.colors, atol=1e-9)


def test_structure_type_and_rmsd_selection(hcp_result):
    assert select_by_structure(hcp_result, "hcp").sum() == hcp_result.mask("hcp").sum()
    assert select_by_type(hcp_result, "Mg").all()
    with pytest.raises(KeyError):
        select_by_type(hcp_result, "Fe")
    tight = select_by_rmsd(hcp_result, maximum=0.01)
    assert tight.sum() <= hcp_result.n_atoms
    assert combine(tight, select_by_structure(hcp_result, "hcp")).sum() <= tight.sum()

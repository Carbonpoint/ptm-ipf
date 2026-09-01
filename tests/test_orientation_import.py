"""Colouring a file whose orientations were computed somewhere else.

The point of these tests is that the imported map is the *same* map: the
reference is a real PTM run on the same configuration, not a plausible looking
picture.
"""

import numpy as np
import pytest

pytest.importorskip("ovito")

from ptmipf.analysis import (  # noqa: E402
    QUATERNION_ORDERS,
    analyse,
    analyse_orientations,
    list_columns,
)


@pytest.fixture(scope="module")
def matched(tmp_path_factory):
    """A PTM result and the same data written out with generic column names.

    Generic names are the realistic case: a file from another session rarely
    calls its columns what this one would.
    """
    from .conftest import rotation_matrix

    ase_build = pytest.importorskip("ase.build")
    ase_io = pytest.importorskip("ase.io")

    directory = tmp_path_factory.mktemp("orientations")
    atoms = ase_build.bulk("Al", "fcc", a=4.05, cubic=True).repeat((5, 5, 5))
    atoms.set_cell(np.array(atoms.cell) @ rotation_matrix([1, 2, 3], 29.0).T, scale_atoms=True)
    source = directory / "crystal.xyz"
    ase_io.write(str(source), atoms, format="extxyz")

    result = analyse(str(source), structures=("fcc",), direction="z")
    q = result.orientations
    foreign = directory / "foreign.dump"
    with open(foreign, "w") as handle:
        handle.write("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n")
        handle.write(f"{result.n_atoms}\nITEM: BOX BOUNDS pp pp pp\n")
        low, high = result.positions.min(axis=0), result.positions.max(axis=0)
        for axis in range(3):
            handle.write(f"{low[axis] - 1:.6f} {high[axis] + 1:.6f}\n")
        # Scalar part first, which is the other common convention.
        handle.write("ITEM: ATOMS id type x y z phase quat_w quat_x quat_y quat_z\n")
        for i in range(result.n_atoms):
            x, y, z = result.positions[i]
            handle.write(
                f"{i + 1} 1 {x:.5f} {y:.5f} {z:.5f} {int(result.structure_types[i])} "
                f"{q[i, 3]:.8f} {q[i, 0]:.8f} {q[i, 1]:.8f} {q[i, 2]:.8f}\n"
            )
    return result, str(foreign)


def test_columns_are_listed_without_running_anything(matched):
    _, foreign = matched
    info = list_columns(foreign)
    names = {c["name"]: c["components"] for c in info["columns"]}
    assert names["Position"] == 3
    assert names["quat_w"] == 1 and names["phase"] == 1
    # No PTM ran, so there is no structure type property beyond the one in the file.
    assert "Orientation" not in names
    assert info["n_atoms"] > 0


def test_imported_orientations_reproduce_the_ptm_map(matched):
    reference, foreign = matched
    got = analyse_orientations(
        foreign,
        {
            "quaternion": ["quat_w", "quat_x", "quat_y", "quat_z"],
            "order": "wxyz",
            "structure_type": "phase",
        },
        structures=("fcc",),
        direction="z",
    )
    assert got.counts == reference.counts
    # The only difference is the text precision the file was written at.
    assert np.abs(got.colors - reference.colors).max() < 1e-5


def test_the_wrong_component_order_gives_a_different_map(matched):
    """The hazard this feature has to be honest about.

    Nothing in a file states its convention, and reading w-first data as
    x-first produces a map that looks entirely plausible and is wrong.
    """
    reference, foreign = matched
    wrong = analyse_orientations(
        foreign,
        {"quaternion": ["quat_w", "quat_x", "quat_y", "quat_z"], "order": "xyzw",
         "structure_type": "phase"},
        structures=("fcc",),
        direction="z",
    )
    assert np.abs(wrong.colors - reference.colors).max() > 0.1


def test_inverting_the_sense_is_not_the_same_as_not(matched):
    reference, foreign = matched
    spec = {"quaternion": ["quat_w", "quat_x", "quat_y", "quat_z"], "order": "wxyz",
            "structure_type": "phase"}
    flipped = analyse_orientations(
        foreign, {**spec, "conjugate": True}, structures=("fcc",), direction="z"
    )
    assert np.abs(flipped.colors - reference.colors).max() > 0.1


def test_a_four_component_column_works_too(matched):
    """The straightforward case: a file exported from OVITO with PTM applied."""
    from ovito.io import export_file, import_file
    from ovito.modifiers import PolyhedralTemplateMatchingModifier as Ptm

    reference, foreign = matched
    directory = __import__("pathlib").Path(foreign).parent
    pipeline = import_file(str(directory / "crystal.xyz"))
    pipeline.modifiers.append(Ptm(output_orientation=True, rmsd_cutoff=0.1))
    exported = directory / "ovito.dump"
    export_file(
        pipeline,
        str(exported),
        "lammps/dump",
        columns=[
            "Particle Identifier", "Particle Type",
            "Position.X", "Position.Y", "Position.Z", "Structure Type",
            "Orientation.X", "Orientation.Y", "Orientation.Z", "Orientation.W",
        ],
    )
    got = analyse_orientations(
        str(exported),
        {"quaternion": "Orientation", "structure_type": "Structure Type"},
        structures=("fcc",),
        direction="z",
    )
    assert np.abs(got.colors - reference.colors).max() < 1e-5

    # Component names resolve as well, because OVITO regroups them on import.
    by_component = analyse_orientations(
        str(exported),
        {"quaternion": ["Orientation.X", "Orientation.Y", "Orientation.Z", "Orientation.W"],
         "structure_type": "Structure Type"},
        structures=("fcc",),
        direction="z",
    )
    assert np.abs(by_component.colors - reference.colors).max() < 1e-5


def test_without_a_structure_column_one_phase_is_assumed(matched):
    reference, foreign = matched
    got = analyse_orientations(
        foreign,
        {"quaternion": ["quat_w", "quat_x", "quat_y", "quat_z"], "order": "wxyz",
         "structure": "fcc"},
        structures=("fcc",),
        direction="z",
    )
    # Atoms PTM left unindexed carry a zero quaternion and stay "other".
    assert got.counts["fcc"] == reference.counts["fcc"]


def test_the_result_supports_everything_downstream(matched):
    """An imported result has to be a full result, not a colour array."""
    reference, foreign = matched
    got = analyse_orientations(
        foreign,
        {"quaternion": ["quat_w", "quat_x", "quat_y", "quat_z"], "order": "wxyz",
         "structure_type": "phase"},
        structures=("fcc",),
        direction="z",
    )
    assert got.recolor("x").direction_label == "X"
    assert got.subset(got.mask("fcc")).n_atoms == got.counts["fcc"]
    assert got.rotations("fcc").shape == (got.counts["fcc"], 3, 3)
    assert "atoms" in got.summary()


@pytest.mark.parametrize(
    "spec,message",
    [
        ({}, "quaternion"),
        ({"quaternion": "Position"}, "four"),
        ({"quaternion": ["quat_w", "quat_x"]}, "four scalar columns"),
        ({"quaternion": "nope"}, "no column named"),
        ({"quaternion": "Position", "order": "abcd"}, "four"),
    ],
)
def test_a_bad_mapping_says_what_is_wrong(matched, spec, message):
    _, foreign = matched
    with pytest.raises(ValueError, match=message):
        analyse_orientations(foreign, spec, structures=("fcc",))


def test_the_orders_on_offer_are_both_permutations():
    for order in QUATERNION_ORDERS.values():
        assert sorted(order) == [0, 1, 2, 3]

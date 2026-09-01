from pathlib import Path

import numpy as np
import pytest

from ptmipf.frames import SampleFrame
from ptmipf.io import write_result
from ptmipf.structures import get_structure


class FakeResult:
    """A minimal stand-in for IPFResult, so the writers need no OVITO."""

    def __init__(self):
        self.positions = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        self.structure_types = np.array([2, 0])
        self.orientations = np.tile([0.0, 0.0, 0.0, 1.0], (2, 1))
        self.colors = np.array([[1.0, 0.0, 0.0], [0.35, 0.35, 0.35]])
        self.rmsd = np.array([0.05, 0.0])
        self.particle_types = np.array([1, 1])
        self.type_names = {1: "Mg"}
        self.direction = np.array([0.0, 0.0, 1.0])
        self.direction_label = "ND"
        self.frame = SampleFrame()
        self.structures = (get_structure("hcp"),)
        self.cell = np.array([[10.0, 0, 0, 0], [0, 10.0, 0, 0], [0, 0, 10.0, 0]])
        self.frame_index = 0
        self.n_atoms = 2


@pytest.mark.parametrize("name,expected", [("out.xyz", "extxyz"), ("out.dump", "lammps-dump")])
def test_format_is_guessed_from_the_extension(tmp_path, name, expected):
    assert write_result(FakeResult(), tmp_path / name) == expected


def test_extxyz_is_readable_and_keeps_colours(tmp_path):
    path = tmp_path / "out.xyz"
    write_result(FakeResult(), path)
    lines = path.read_text().splitlines()
    assert lines[0] == "2"
    assert "Properties=species:S:1:pos:R:3:structure_type:I:1:rmsd:R:1:color:R:3" in lines[1]
    assert 'ipf_direction_label="ND"' in lines[1]
    first = lines[2].split()
    assert first[0] == "Mg"
    assert np.allclose([float(v) for v in first[6:9]], [1.0, 0.0, 0.0])

    ase_io = pytest.importorskip("ase.io")
    atoms = ase_io.read(str(path), format="extxyz")
    assert len(atoms) == 2
    assert np.allclose(atoms.get_array("color")[0], [1.0, 0.0, 0.0])


def test_lammps_dump_has_the_expected_columns(tmp_path):
    path = tmp_path / "out.dump"
    write_result(FakeResult(), path)
    text = path.read_text().splitlines()
    assert text[3] == "2"
    # The column names are what make OVITO bind the colours to the atoms.
    assert text[8] == "ITEM: ATOMS id type x y z StructureType rmsd Color.R Color.G Color.B"
    assert len(text[9].split()) == 10


def test_dump_colours_bind_to_atoms_in_ovito(tmp_path):
    """Reopening the dump must give coloured atoms, not loose r/g/b columns."""
    pytest.importorskip("ovito")
    from ovito.io import import_file

    path = tmp_path / "out.dump"
    write_result(FakeResult(), path)
    data = import_file(str(path)).compute()
    assert "Color" in data.particles
    assert "Structure Type" in data.particles
    assert np.allclose(data.particles["Color"][0], [1.0, 0.0, 0.0])


def test_extxyz_colours_bind_to_atoms_in_ovito(tmp_path):
    pytest.importorskip("ovito")
    from ovito.io import import_file

    path = tmp_path / "out.xyz"
    write_result(FakeResult(), path)
    data = import_file(str(path)).compute()
    assert "Color" in data.particles
    assert np.allclose(data.particles["Color"][0], [1.0, 0.0, 0.0])


def test_unknown_format_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_result(FakeResult(), tmp_path / "out.foo", fmt="cif")


def test_extxyz_carries_the_colour_coding_columns(tmp_path):
    path = tmp_path / "keys.xyz"
    keys = {"ipf_x": np.array([0.125, 0.875]), "ipf_z": np.array([0.375, 0.625])}
    write_result(FakeResult(), path, keys=keys)
    lines = path.read_text().splitlines()
    assert ":color:R:3:ipf_x:R:1:ipf_z:R:1" in lines[1]
    first = lines[2].split()
    assert len(first) == 11
    assert float(first[9]) == pytest.approx(0.125)
    assert float(first[10]) == pytest.approx(0.375)


def test_dump_names_the_colour_coding_columns(tmp_path):
    path = tmp_path / "keys.dump"
    write_result(FakeResult(), path, keys={"ipf_y": np.array([0.25, 0.75])})
    text = path.read_text()
    assert "ITEM: ATOMS id type x y z StructureType rmsd Color.R Color.G Color.B ipf_y" in text
    assert text.strip().splitlines()[-1].split()[-1] == "0.75000000"


def test_a_key_of_the_wrong_length_is_refused(tmp_path):
    with pytest.raises(ValueError, match="expected 2"):
        write_result(FakeResult(), tmp_path / "bad.xyz", keys={"ipf_x": np.zeros(3)})


def test_temporary_path_can_be_written_by_name():
    """The web UI hands this path to OVITO, which opens it itself.

    NamedTemporaryFile cannot serve that purpose: on Windows its open handle
    locks the name and the second open fails, which is what left the 3D view
    and the exports empty there.
    """
    from ptmipf.io import temporary_path

    with temporary_path(".png") as path:
        assert path.endswith(".png")
        with open(path, "w") as handle:
            handle.write("written by name")
        with open(path) as handle:
            assert handle.read() == "written by name"
    assert not Path(path).exists()

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
    assert text[8] == "ITEM: ATOMS id type x y z structure rmsd r g b"
    assert len(text[9].split()) == 10


def test_unknown_format_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_result(FakeResult(), tmp_path / "out.foo", fmt="cif")

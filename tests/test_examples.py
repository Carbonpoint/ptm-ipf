"""The starter examples: potentials, the LAMMPS input, and what gets written.

The network is not assumed.  Everything that would download is exercised
against a file already on disk, which is also the path a second build takes.
"""

import json

import numpy as np
import pytest

from ptmipf.examples import DEFAULTS, ExampleSpec, build_example, example_directory
from ptmipf.lammps import compression_script, estimate_cost, write_data_file
from ptmipf.polycrystal import voronoi_polycrystal
from ptmipf.potentials import POTENTIALS, download_potential, potential_for


# ----------------------------------------------------------------------
# the catalogue
# ----------------------------------------------------------------------
def test_every_potential_names_a_pinned_version():
    for element, potential in POTENTIALS.items():
        assert potential.element == element
        assert potential.structure in ("fcc", "bcc")
        assert potential.a0 > 1.0 and potential.mass > 1.0
        assert len(potential.sha256) == 64
        # A version directory, not "latest": the file must not move under us.
        assert f"/{potential.version}/{potential.filename}" in potential.url
        assert potential.url.startswith("https://www.ctcms.nist.gov/potentials/Download/")


def test_iron_is_read_with_the_pair_style_its_file_needs():
    """eam/fs and eam/alloy are different formats, and the wrong one is silent."""
    assert potential_for("Fe").pair_style == "eam/fs"
    assert potential_for("Fe").filename.endswith(".eam.fs")
    assert potential_for("Cu").pair_style == "eam/alloy"


def test_unknown_elements_list_the_ones_that_exist():
    with pytest.raises(ValueError, match="Cu"):
        potential_for("Unobtainium")


def test_a_potential_already_on_disk_is_not_downloaded_again(tmp_path, monkeypatch):
    import dataclasses
    import hashlib
    import urllib.request

    import ptmipf.potentials as module

    potential = potential_for("Cu")
    body = b"a pretend potential file"
    (tmp_path / potential.filename).write_bytes(body)
    monkeypatch.setitem(
        module.POTENTIALS,
        "Cu",
        dataclasses.replace(potential, sha256=hashlib.sha256(body).hexdigest()),
    )

    def refuse(*args, **kwargs):
        raise AssertionError("the network must not be touched for a cached file")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    assert download_potential("Cu", tmp_path).read_bytes() == body


def test_a_changed_download_is_refused(tmp_path, monkeypatch):
    """A silently different potential is a silently different result."""
    import io
    import urllib.request

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: _Response(b"not the potential")
    )
    with pytest.raises(RuntimeError, match="checksum"):
        download_potential("Cu", tmp_path)
    assert not (tmp_path / "Cu01.eam.alloy").exists()


def test_a_download_failure_says_what_to_do_by_hand(tmp_path, monkeypatch):
    import urllib.error
    import urllib.request

    def fail(*args, **kwargs):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError, match="Fetch https://"):
        download_potential("Cu", tmp_path)


# ----------------------------------------------------------------------
# the LAMMPS input
# ----------------------------------------------------------------------
@pytest.fixture
def crystal():
    return voronoi_polycrystal("Cu", box=25.0, n_grains=3, a0=3.615, seed=1)


def test_data_file_round_trips_through_the_reader(tmp_path, crystal):
    from ptmipf.polycrystal import _read_lammps_data

    path = tmp_path / "structure.lmp"
    write_data_file(crystal, path)
    positions, edge = _read_lammps_data(path)
    assert len(positions) == crystal.n_atoms
    assert edge == pytest.approx(crystal.box)
    assert np.abs(positions - crystal.positions).max() < 1e-5
    assert "1 63.5460" in path.read_text()


def test_the_barostat_leaves_the_loading_axis_to_fix_deform(crystal):
    """Both driving z would be a conflict; only the transverse axes relax."""
    script, _ = compression_script(potential_for("Cu"), "structure.lmp", crystal.n_atoms)
    load = next(line for line in script.splitlines() if line.startswith("fix             load"))
    assert " z " not in load
    assert "x 0.0 0.0" in script and "y 0.0 0.0" in script
    assert "fix             squash all deform 1 z erate" in script


def test_the_reference_length_is_captured_once(crystal):
    """"variable lz0 equal lz" would re-read lz and the strain would be zero."""
    script, _ = compression_script(potential_for("Cu"), "structure.lmp", crystal.n_atoms)
    assert "variable        lz0 equal $(lz)" in script
    assert "variable        strain equal (v_lz0-lz)/v_lz0" in script


def test_the_stress_curve_is_averaged_not_sampled(crystal):
    """fix print can substitute its variables once, at parse time."""
    script, settings = compression_script(potential_for("Cu"), "structure.lmp", crystal.n_atoms)
    assert "fix             curve all ave/time" in script
    commands = [line.split() for line in script.splitlines() if line.startswith("fix ")]
    assert not any("print" in words for words in commands)
    line = next(
        line
        for line in script.splitlines()
        if line.startswith("fix ") and "ave/time" in line
    )
    every, repeat, freq = (int(f) for f in line.split()[4:7])
    # ave/time requires every * (repeat - 1) <= freq.
    assert every * (repeat - 1) <= freq == settings["dump_every"]


def test_the_potential_is_read_with_its_own_pair_style():
    script, _ = compression_script(potential_for("Fe"), "structure.lmp", 1000)
    assert "pair_style      eam/fs" in script
    assert "pair_coeff      * * Fe_2.eam.fs Fe" in script


def test_the_estimate_scales_with_the_work():
    small = estimate_cost(1000, 1000)
    big = estimate_cost(2000, 2000)
    assert big["atom_steps"] == 4 * small["atom_steps"]
    assert big["minutes_one_core"] > big["minutes_four_cores"] > 0


# ----------------------------------------------------------------------
# putting it together
# ----------------------------------------------------------------------
def test_the_spec_refuses_what_would_not_run():
    for bad in ({"box": 5.0}, {"n_grains": 0}, {"strain": 2.0}, {"temperature": 5000.0}):
        with pytest.raises(ValueError):
            ExampleSpec(**bad).validate()


def test_the_catalogue_entries_are_all_valid():
    for name, spec in DEFAULTS.items():
        spec.validate()
        assert spec.element in POTENTIALS
        assert name.startswith(spec.element.lower())


def test_build_example_writes_a_runnable_directory(tmp_path, monkeypatch):
    import ptmipf.potentials as module

    potential = potential_for("Cu")
    directory = example_directory(tmp_path, ExampleSpec(element="Cu", n_grains=3))
    directory.mkdir(parents=True)
    (directory / potential.filename).write_text("stand-in for the real potential\n")
    # The download is the one step that needs the network; the rest is local.
    monkeypatch.setattr(
        module, "download_potential", lambda element, where, **kw: where / potential.filename
    )

    report = build_example(
        tmp_path, ExampleSpec(element="Cu", box=40.0, n_grains=3, builder="voronoi")
    )
    written = set(report["files"])
    assert {"in.compression", "structure.lmp", "structure.xyz", "grains.json",
            "README.md", potential.filename} <= written
    assert report["density"] > 0.95
    assert report["run"]["atom_steps"] > 0
    assert report["builder"] == "voronoi"

    grains = json.loads((directory / "grains.json").read_text())
    assert len(grains["rotations"]) == 3
    for matrix in np.asarray(grains["rotations"]):
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-9)

    readme = (directory / "README.md").read_text()
    assert "lmp -in in.compression" in readme
    assert potential.citation in readme
    # The fallback builder must say so, so nobody quotes it as the reference.
    assert "not atomsk" in readme

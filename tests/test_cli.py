"""Tests for the command line interface."""

import pytest

from ptmipf.cli import _optional_float, _split_fields, build_parser, main


def test_parser_defaults():
    args = build_parser().parse_args(["file.dump"])
    assert args.direction == "z"
    assert args.structures == "fcc,hcp,bcc"
    assert args.select_mode == "and"
    assert args.select_orientation == []


def test_split_fields_pads_and_keeps_commas():
    assert _split_fields("0001|nd|15", 3, "--x") == ["0001", "nd", "15"]
    assert _split_fields("0001|1,1,0", 3, "--x") == ["0001", "1,1,0", ""]
    assert _split_fields("z||60", 3, "--x") == ["z", "", "60"]
    with pytest.raises(SystemExit):
        _split_fields("a|b|c|d", 3, "--x")


def test_optional_float():
    assert _optional_float("", "--x") is None
    assert _optional_float("2.5", "--x") == 2.5
    with pytest.raises(SystemExit):
        _optional_float("nope", "--x")


def test_list_structures_exits_cleanly(capsys):
    assert main(["--list-structures"]) == 0
    out = capsys.readouterr().out
    assert "hcp" in out and "6/mmm" in out


def test_input_is_required():
    with pytest.raises(SystemExit):
        main([])


@pytest.fixture
def crystal(write_crystal):
    pytest.importorskip("ovito")
    from .conftest import rotation_matrix

    return write_crystal("hcp", rotation=rotation_matrix([1, 0, 0], 20.0), repeat=5)


def test_selection_output_and_plots(crystal, tmp_path, capsys):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    out = tmp_path / "sel.xyz"
    code = main(
        [
            crystal,
            "--structures",
            "hcp",
            "--direction",
            "z",
            "--select-orientation",
            "0001|z|25",
            "--selection-output",
            str(out),
            "--from-selection",
            "--legend",
            str(tmp_path / "key.png"),
            "--pole-figure",
            "0001",
            "--pole-figure-file",
            str(tmp_path / "pf.png"),
        ]
    )
    assert code == 0
    assert out.exists() and (tmp_path / "key.png").exists() and (tmp_path / "pf.png").exists()
    assert "selection:" in capsys.readouterr().out


def test_an_empty_selection_is_reported(crystal, tmp_path):
    with pytest.raises(SystemExit):
        main(
            [crystal, "--structures", "hcp", "--select-orientation", "0001|z|5", "--from-selection"]
        )


def test_selection_flags_need_criteria(crystal, tmp_path):
    with pytest.raises(SystemExit):
        main([crystal, "--structures", "hcp", "--from-selection"])


def test_invert_selection_is_the_complement(crystal, tmp_path, capsys):
    def selected(*extra):
        main([crystal, "--structures", "hcp", "--select-orientation", "0001|z|25", *extra])
        line = [x for x in capsys.readouterr().out.splitlines() if x.startswith("selection:")][0]
        return int(line.split()[1])

    total = int(
        [x for x in _summary(crystal, capsys).splitlines() if "atoms" in x][0].split()[2]
    )
    assert selected() + selected("--invert-selection") == total


def _summary(path, capsys):
    main([path, "--structures", "hcp"])
    return capsys.readouterr().out


def test_bad_region_and_grain_arguments(crystal):
    with pytest.raises(SystemExit):
        main([crystal, "--structures", "hcp", "--select-region", "z|abc|1"])
    with pytest.raises(SystemExit):
        main([crystal, "--structures", "hcp", "--select-grain", "999999999"])


def test_render_size_is_validated(crystal, tmp_path):
    with pytest.raises(SystemExit):
        main([crystal, "--structures", "hcp", "--render", str(tmp_path / "r.png"),
              "--render-size", "big"])


def test_export_directions_are_written_with_a_colour_map(crystal, tmp_path, capsys):
    """The default output carries x, y and z keys and the bar that decodes them."""
    out = tmp_path / "out.dump"
    assert main([crystal, "--structures", "hcp", "-o", str(out)]) == 0
    header = next(
        line for line in out.read_text().splitlines() if line.startswith("ITEM: ATOMS")
    )
    assert header.endswith("Color.R Color.G Color.B ipf_x ipf_y ipf_z")
    colour_map = tmp_path / "out_colormap.png"
    assert colour_map.exists()
    printed = capsys.readouterr().out
    assert "colour-coding columns: ipf_x (X), ipf_y (Y), ipf_z (Z)" in printed
    assert "Load custom color map" in printed


def test_export_directions_can_be_chosen_or_switched_off(crystal, tmp_path):
    named = tmp_path / "named.xyz"
    assert main(
        [crystal, "--structures", "hcp", "-o", str(named),
         "--export-direction", "nd", "--export-direction", "1,1,0"]
    ) == 0
    assert ":ipf_nd:R:1:ipf_1_1_0:R:1" in named.read_text().splitlines()[1]

    plain = tmp_path / "plain.xyz"
    assert main(
        [crystal, "--structures", "hcp", "-o", str(plain), "--no-export-directions"]
    ) == 0
    properties = plain.read_text().splitlines()[1].split("Properties=")[1].split()[0]
    assert properties.endswith("color:R:3")
    assert not (tmp_path / "plain_colormap.png").exists()


def test_a_builtin_colour_bar_reports_how_far_it_misses(crystal, tmp_path, capsys):
    out = tmp_path / "jet.xyz"
    assert main(
        [crystal, "--structures", "hcp", "-o", str(out), "--color-map-gradient", "jet"]
    ) == 0
    printed = capsys.readouterr().out
    assert "built-in jet colour bar" in printed
    assert "approximation error" in printed
    # A built-in bar carries no file of its own to write.
    assert not (tmp_path / "jet_colormap.png").exists()


def test_pole_figure_colour_map_and_smoothing_flags(crystal, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    assert main([
        crystal, "--structures", "hcp", "-q", "--no-export-directions",
        "--pole-figure", "0001",
        "--pole-figure-file", str(tmp_path / "pf.png"),
        "--pole-figure-cmap", "jet",
        "--pole-figure-smoothing", "6",
        "--ipf-density", str(tmp_path / "density.png"),
        "--ipf-density-cmap", "rainbow",
        "--ipf-density-smoothing", "4",
    ]) == 0
    assert (tmp_path / "pf.png").stat().st_size > 0
    assert (tmp_path / "density.png").stat().st_size > 0


def test_a_colour_map_file_is_accepted_by_the_cli(crystal, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    table = tmp_path / "scale.txt"
    table.write_text("255 255 255\n255 0 0\n0 0 0\n")
    assert main([
        crystal, "--structures", "hcp", "-q", "--no-export-directions",
        "--pole-figure", "0001",
        "--pole-figure-file", str(tmp_path / "pf.png"),
        "--pole-figure-cmap", str(table),
    ]) == 0
    assert (tmp_path / "pf.png").stat().st_size > 0


def test_an_unknown_colour_map_stops_the_run(crystal, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    with pytest.raises(ValueError, match="unknown colour map"):
        main([
            crystal, "--structures", "hcp", "-q", "--no-export-directions",
            "--pole-figure", "0001",
            "--pole-figure-file", str(tmp_path / "pf.png"),
            "--pole-figure-cmap", "not-a-colour-map",
        ])


def _atoms_written(path):
    ase_io = pytest.importorskip("ase.io")
    return ase_io.read(str(path), format="extxyz")


def test_rotate_turns_the_written_system_and_its_colours(crystal, tmp_path, capsys):
    """A crystal seen along z, turned 90 degrees about x, is seen along -y."""
    plain = tmp_path / "plain.xyz"
    turned = tmp_path / "turned.xyz"
    assert main([crystal, "--structures", "hcp", "-q", "--no-export-directions",
                 "-o", str(plain)]) == 0
    assert main([crystal, "--structures", "hcp", "-q", "--no-export-directions",
                 "--rotate", "x:90", "-o", str(turned)]) == 0
    before, after = _atoms_written(plain), _atoms_written(turned)
    assert len(before) == len(after)
    # The cell has turned with the atoms: its old y edge now runs along z.
    assert before.cell[1][1] > 1 and abs(after.cell[1][2] - before.cell[1][1]) < 1e-6
    # And the colours are of a different direction through the crystal.
    assert not (before.arrays["color"] == after.arrays["color"]).all()


def test_rotate_rejects_a_malformed_spec(crystal, tmp_path):
    with pytest.raises(ValueError):
        main([crystal, "--structures", "hcp", "-q", "--rotate", "z"])


def test_ptm_slice_writes_only_the_slab(crystal, tmp_path):
    whole = tmp_path / "whole.xyz"
    slab = tmp_path / "slab.xyz"
    assert main([crystal, "--structures", "hcp", "-q", "--no-export-directions",
                 "-o", str(whole)]) == 0
    assert main([crystal, "--structures", "hcp", "-q", "--no-export-directions",
                 "--ptm-slice", "--slice", "z", "--slice-distance", "6",
                 "--slice-width", "4", "-o", str(slab)]) == 0
    before, after = _atoms_written(whole), _atoms_written(slab)
    assert 0 < len(after) < len(before)
    assert (abs(after.positions[:, 2] - 6.0) <= 2.0 + 1e-6).all()
    # The slab is matched properly: its atoms are hcp, not "other" at the faces.
    assert (after.arrays["structure_type"] > 0).all()


def test_ptm_slice_needs_a_slice_and_a_distance(crystal):
    with pytest.raises(SystemExit, match="--ptm-slice needs --slice"):
        main([crystal, "--structures", "hcp", "-q", "--ptm-slice"])
    with pytest.raises(SystemExit, match="--slice-distance"):
        main([crystal, "--structures", "hcp", "-q", "--ptm-slice", "--slice", "z"])


def test_tripod_axes_are_split_into_directions_and_labels():
    from ptmipf.cli import _tripod_axes

    assert _tripod_axes("rd;td;nd") == (["rd", "td", "nd"], ["", "", ""])
    assert _tripod_axes("1,1,0=loading; nd ;td=T") == (["1,1,0", "nd", "td"], ["loading", "", "T"])
    assert _tripod_axes("") == (["rd", "td", "nd"], ["", "", ""])

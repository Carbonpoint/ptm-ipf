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

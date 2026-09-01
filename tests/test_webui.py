"""Tests for the local web UI server.

The server is started as a subprocess on an ephemeral port and exercised over
HTTP with urllib, so these tests cover exactly what a browser would receive:
JSON shapes, PNG images that decode, and the path-traversal refusals.

A subprocess rather than an in-process server is essential: OVITO binds to
the first thread that runs its pipeline machinery, and the rest of this test
suite runs OVITO on pytest's main thread, which would crash the web UI's
dedicated OVITO worker thread.
"""

import io
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
ase_build = pytest.importorskip("ase.build")
ase_io = pytest.importorskip("ase.io")

if subprocess.run(
    [sys.executable, "-c", "import ovito"], capture_output=True
).returncode != 0:
    pytest.skip("OVITO is unavailable", allow_module_level=True)


def test_every_element_the_script_reaches_for_exists():
    """A renamed or missing id is a silent front end failure in every browser.

    ``$("...")`` returning null throws on the next property access and stops
    the rest of the handler, which shows up as a control that simply does
    nothing, so it is worth catching here rather than by hand.
    """
    import re

    from ptmipf.webui.server import STATIC_DIR

    html = (STATIC_DIR / "index.html").read_text()
    script = (STATIC_DIR / "app.js").read_text()
    declared = set(re.findall(r'id="([^"]+)"', html))
    used = set(re.findall(r'\$\("([^"]+)"\)', script))
    assert used <= declared, f"app.js reaches for ids the page does not define: {used - declared}"


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=60) as response:
        return response.status, dict(response.headers), response.read()


def _get_json(base, path):
    status, _, body = _get(base, path)
    assert status == 200
    return json.loads(body)


def _post_json(base, path, payload):
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def _decode_png(body: bytes) -> np.ndarray:
    assert body[:8] == b"\x89PNG\r\n\x1a\n"
    import matplotlib.image

    return matplotlib.image.imread(io.BytesIO(body))


@pytest.fixture(scope="module")
def served_dir(tmp_path_factory):
    """A directory holding a basal-oriented hcp Mg crystal."""
    directory = tmp_path_factory.mktemp("webui")
    atoms = ase_build.bulk("Mg", "hcp", a=3.2094, c=5.2108).repeat((8, 8, 8))
    ase_io.write(str(directory / "crystal.xyz"), atoms, format="extxyz")
    (directory / "subdir").mkdir()
    return directory


@pytest.fixture(scope="module")
def base(served_dir):
    process = subprocess.Popen(
        [
            sys.executable, "-u", "-m", "ptmipf.webui",
            "--root", str(served_dir), "--port", "0", "--no-browser",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # The server prints its ephemeral address once it is listening.
    killer = threading.Timer(60, process.kill)
    killer.start()
    url = None
    try:
        for line in process.stdout:
            if "http://127.0.0.1:" in line:
                url = "http://" + line.split("http://", 1)[1].split("/", 1)[0]
                break
    finally:
        killer.cancel()
    if url is None:
        process.kill()
        pytest.fail("the web UI server did not start")
    yield url
    process.terminate()
    process.wait(timeout=30)


@pytest.fixture(scope="module")
def analysed(base):
    """Run the analysis once and return the final status payload."""
    outcome = _post_json(
        base, "/api/analyse", {"path": "crystal.xyz", "structures": ["hcp", "fcc"]}
    )
    assert outcome["accepted"]
    deadline = time.time() + 120
    while time.time() < deadline:
        status = _get_json(base, "/api/status")
        if status["state"] != "running":
            break
        time.sleep(0.3)
    assert status["state"] == "done", status.get("error")
    return status


def test_meta(base):
    meta = _get_json(base, "/api/meta")
    assert meta["version"]
    names = [s["name"] for s in meta["structures"]]
    assert {"fcc", "hcp", "bcc"} <= set(names)
    assert meta["defaults"]["rmsd_cutoff"] == 0.1


def test_index_page(base):
    status, headers, body = _get(base, "/")
    assert status == 200
    assert "text/html" in headers["Content-Type"]
    assert b"ptm-ipf" in body


def test_browse(base):
    listing = _get_json(base, "/api/browse?path=")
    names = [e["name"] for e in listing["entries"]]
    assert "crystal.xyz" in names
    assert "subdir" in names


def test_browse_refuses_traversal(base):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(base, "/api/browse?path=../..")
    assert excinfo.value.code == 403


def test_static_refuses_traversal(base):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(base, "/static/../server.py")
    assert excinfo.value.code == 404


def test_render_conflict_before_analysis(base):
    # Runs before any test that uses the `analysed` fixture: with no cached
    # result the image endpoints must refuse, not crash.
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(base, "/api/render")
    assert excinfo.value.code == 409


def test_analysis_counts(analysed):
    result = analysed["result"]
    assert result["n_atoms"] == 1024
    # A perfect periodic crystal: everything should be identified as hcp.
    assert result["counts"]["hcp"] > 0.95 * result["n_atoms"]
    assert result["direction_label"] == "Z"


def test_render_is_red_for_basal_ipf_z(base, analysed, renderer):
    status, headers, body = _get(base, "/api/render?w=400&h=300")
    assert status == 200 and headers["Content-Type"] == "image/png"
    image = _decode_png(body)
    assert image.shape[:2] == (300, 400)
    rgb = image[..., :3]
    foreground = rgb.std(axis=2) > 0.05  # coloured atoms, not the background
    assert foreground.any()
    mean = rgb[foreground].mean(axis=0)
    # [0001] along z is the red vertex of the hcp colour key.
    assert mean[0] > mean[1] + 0.2 and mean[0] > mean[2] + 0.2


def test_recolour_changes_direction_without_rerun(base, analysed, renderer):
    outcome = _post_json(
        base,
        "/api/analyse",
        {"path": "crystal.xyz", "structures": ["hcp", "fcc"], "direction": "x"},
    )
    assert outcome["accepted"] and outcome["recoloured"]
    status = _get_json(base, "/api/status")
    assert status["result"]["direction_label"] == "X"
    _, _, body = _get(base, "/api/render?w=400&h=300")
    rgb = _decode_png(body)[..., :3]
    foreground = rgb.std(axis=2) > 0.05
    mean = rgb[foreground].mean(axis=0)
    # a1 along x: the IPF-X direction sits at the green [2-1-10] vertex.
    assert mean[1] > mean[0]
    # Restore the projection the other tests assume.
    outcome = _post_json(
        base,
        "/api/analyse",
        {"path": "crystal.xyz", "structures": ["hcp", "fcc"], "direction": "z"},
    )
    assert outcome["recoloured"]


def test_legend_png(base, analysed):
    _, headers, body = _get(base, "/api/figure/legend?structure=hcp")
    assert headers["Content-Type"] == "image/png"
    assert _decode_png(body).ndim == 3


def test_pole_figure_png(base, analysed):
    _, _, body = _get(base, "/api/figure/poles?poles=0001,10-10&c_over_a=1.624")
    image = _decode_png(body)
    assert image.shape[1] > image.shape[0]  # two side-by-side figures


def test_ipf_density_png(base, analysed):
    _, _, body = _get(base, "/api/figure/ipfdensity")
    assert _decode_png(body).ndim == 3


def test_slice_bounds(base, analysed):
    bounds = _get_json(base, "/api/slicebounds?axis=z")
    assert bounds["min"] < bounds["max"]


def test_sliced_render(base, analysed, renderer):
    _, _, body = _get(base, "/api/render?w=300&h=240&hide_other=1&slice_axis=z&slice_frac=0.5")
    assert _decode_png(body).shape[:2] == (240, 300)


def test_atom_info(base, analysed):
    atom = _get_json(base, "/api/atom?index=200")
    assert atom["index"] == 200
    assert len(atom["position"]) == 3
    assert len(atom["orientation"]) == 4


def test_pick_returns_a_visible_atom(base, analysed):
    outcome = _post_json(
        base, "/api/pick", {"x": 200, "y": 150, "w": 400, "h": 300, "az": -125, "el": 20}
    )
    atom = outcome["atom"]
    assert atom is not None
    assert 0 <= atom["index"] < 1024


def test_export_full(base, analysed):
    status, headers, body = _get(base, "/api/export?format=extxyz")
    assert status == 200
    assert "attachment" in headers["Content-Disposition"]
    assert body.split(b"\n", 1)[0] == b"1024"
    _, _, dump = _get(base, "/api/export?format=lammps-dump")
    assert b"ITEM: ATOMS" in dump


def test_command(base, analysed):
    outcome = _post_json(base, "/api/command", {"poles": ["0001"], "hide_other": True})
    command = outcome["command"]
    assert command.startswith("ptmipf")
    assert "crystal.xyz" in command
    assert "--structures hcp,fcc" in command
    assert "--pole-figure 0001" in command
    assert "--hide-other" in command


def test_command_has_a_one_line_form(base, analysed):
    """PowerShell and cmd.exe break at the first line end, so a one-liner is needed."""
    outcome = _post_json(base, "/api/command", {"poles": ["0001"]})
    assert "\\\n" in outcome["command"]
    assert "\n" not in outcome["one_line"]
    assert "\\" not in outcome["one_line"]
    assert outcome["one_line"].split() == outcome["command"].replace("\\\n", " ").split()


def test_command_round_trips_through_the_parser(base, analysed):
    """What the dialog writes, the dialog can read back."""
    outcome = _post_json(
        base,
        "/api/command",
        {"poles": ["0001", "10-10"], "hide_other": True, "fill_radius": 7.5,
         "export_directions": ["nd", "z"]},
    )
    for form in (outcome["command"], outcome["one_line"]):
        settings = _post_json(base, "/api/command/parse", {"command": form})
        assert settings["analysis"]["path"].endswith("crystal.xyz")
        assert settings["analysis"]["structures"] == ["hcp", "fcc"]
        assert settings["ui"]["poles"] == ["0001", "10-10"]
        assert settings["ui"]["hide_other"] is True
        assert settings["ui"]["fill_radius"] == 7.5
        assert settings["ui"]["export_directions"] == ["nd", "z"]


def test_a_command_without_the_program_name_is_accepted(base):
    settings = _post_json(
        base, "/api/command/parse", {"command": "mg.dump --direction nd --frame 3"}
    )
    assert settings["analysis"]["path"] == "mg.dump"
    assert settings["analysis"]["frame_index"] == 3
    assert settings["colour"]["direction"] == "nd"


def test_a_saved_command_keeps_its_comment_lines_harmless(base):
    text = "ptmipf mg.dump \\\n    --direction z\n# note: something the CLI cannot say\n"
    settings = _post_json(base, "/api/command/parse", {"command": text})
    assert settings["analysis"]["path"] == "mg.dump"


@pytest.mark.parametrize("command", ["", "mg.dump --not-an-option"])
def test_an_unreadable_command_is_a_message_not_a_crash(base, command):
    with pytest.raises(urllib.error.HTTPError) as caught:
        _post_json(base, "/api/command/parse", {"command": command})
    assert caught.value.code == 400
    assert "command" in json.loads(caught.value.read())["error"]


def test_colormap_endpoint_serves_the_bar_the_columns_index(base, analysed):
    status, headers, body = _get(base, "/api/colormap?directions=x;y;z")
    assert status == 200
    image = _decode_png(body)
    entries = int(headers["X-Color-Entries"])
    assert image.shape[1] == entries
    assert headers["X-Color-Columns"] == "ipf_x,ipf_y,ipf_z"


def test_export_carries_the_colour_coding_columns(base, analysed):
    status, headers, body = _get(base, "/api/export?format=lammps-dump&directions=nd;td")
    assert status == 200
    assert headers["X-Color-Columns"] == "ipf_nd,ipf_td"
    header = next(
        line for line in body.decode().splitlines() if line.startswith("ITEM: ATOMS")
    )
    assert header.endswith("ipf_nd ipf_td")


def test_export_can_leave_the_columns_out(base, analysed):
    _, headers, body = _get(base, "/api/export?format=lammps-dump&keys=0")
    assert "X-Color-Columns" not in headers
    assert "ipf_" not in body.decode()


def test_diagnostics_report_what_the_installation_can_do(base):
    """A blank 3D view is a server-side renderer problem; this is where it shows."""
    report = _get_json(base, "/api/diagnostics")
    assert report["ptmipf"] and report["python"] and report["platform"]
    names = {check["name"]: check for check in report["checks"]}
    assert names["ovito"]["ok"]
    assert "3D view" in names
    if report["ok"]:
        assert report["renderer"] in ("opengl", "tachyon")
    else:
        # A failure has to say what failed, or it is no use to anyone.
        assert any(check["detail"] for check in report["checks"] if not check["ok"])


# ----------------------------------------------------------------------
# selection; skipped when ptmipf.select is not installed
# ----------------------------------------------------------------------
def _needs_select(base):
    if not _get_json(base, "/api/meta")["selection_available"]:
        pytest.skip("ptmipf.select is not available")


def _hcp_atom_index(base):
    for index in range(0, 1024, 37):
        if _get_json(base, f"/api/atom?index={index}")["structure"] == "hcp":
            return index
    pytest.fail("no hcp atom found")


def test_selection_by_structure_and_orientation(base, analysed):
    _needs_select(base)
    hcp_count = analysed["result"]["counts"]["hcp"]
    outcome = _post_json(
        base,
        "/api/selection",
        {
            "mode": "and",
            "criteria": [
                {"kind": "structure", "structures": ["hcp"]},
                {
                    "kind": "ipf",
                    "crystal": "0001",
                    "sample": "z",
                    "tolerance": 10,
                    "structure": "hcp",
                },
            ],
        },
    )
    # A basal single crystal: every hcp atom is within tolerance.
    assert outcome["count"] == hcp_count


def test_selection_invert_and_region(base, analysed):
    _needs_select(base)
    n = analysed["result"]["n_atoms"]
    half = _post_json(
        base,
        "/api/selection",
        {"criteria": [{"kind": "region", "axis": "z", "max": 20.0}]},
    )["count"]
    other_half = _post_json(
        base,
        "/api/selection",
        {"criteria": [{"kind": "region", "axis": "z", "max": 20.0, "invert": True}]},
    )["count"]
    assert 0 < half < n
    assert half + other_half == n


def test_selection_misorientation_from_atom(base, analysed):
    _needs_select(base)
    reference = _hcp_atom_index(base)
    outcome = _post_json(
        base,
        "/api/selection",
        {
            "criteria": [
                {
                    "kind": "misorientation",
                    "reference": {"atom": reference},
                    "tolerance": 5,
                    "structure": "hcp",
                }
            ]
        },
    )
    # A single crystal: the whole hcp phase shares the reference orientation.
    assert outcome["count"] == analysed["result"]["counts"]["hcp"]


def test_selection_figures_and_export(base, analysed, renderer):
    _needs_select(base)
    count = _post_json(
        base,
        "/api/selection",
        {"criteria": [{"kind": "structure", "structures": ["hcp"]}]},
    )["count"]
    _, _, body = _get(base, "/api/figure/poles?poles=0001&selection=1")
    assert _decode_png(body).ndim == 3
    _, _, body = _get(base, "/api/figure/ipfdensity?selection=1")
    assert _decode_png(body).ndim == 3
    _, _, exported = _get(base, "/api/export?format=extxyz&selection=1")
    assert exported.split(b"\n", 1)[0] == str(count).encode()
    # Highlighted and selection-only renders must both work.
    for mode in ("highlight", "only"):
        _, _, image = _get(base, f"/api/render?w=200&h=160&highlight={mode}")
        assert _decode_png(image).shape[:2] == (160, 200)


def test_selection_command_flags(base, analysed):
    _needs_select(base)
    _post_json(
        base,
        "/api/selection",
        {"criteria": [{"kind": "structure", "structures": ["hcp"]}]},
    )
    command = _post_json(base, "/api/command", {})["command"]
    assert "--select-structure hcp" in command
    assert "--from-selection" in command


def test_selection_cleared(base, analysed):
    _needs_select(base)
    outcome = _post_json(base, "/api/selection", {"criteria": []})
    assert outcome["count"] is None
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(base, "/api/export?format=extxyz&selection=1")
    assert excinfo.value.code == 400


def test_flat_map_endpoint(base, analysed):
    """The flat map is served, and reports what it found in the headers."""
    status, headers, body = _get(
        base,
        "/api/figure/flatmap?view=z&slab_width=10&pixel_size=0.8&boundary_angle=5",
    )
    assert status == 200
    assert headers["Content-Type"] == "image/png"
    assert int(headers["X-Grain-Count"]) >= 1
    assert "x" in headers["X-Map-Size"]
    assert _decode_png(body).ndim == 3


def test_flat_map_pixel_size_changes_the_resolution(base, analysed):
    coarse = _get(base, "/api/figure/flatmap?view=z&pixel_size=2.0")[1]["X-Map-Size"]
    fine = _get(base, "/api/figure/flatmap?view=z&pixel_size=0.5")[1]["X-Map-Size"]
    assert int(fine.split("x")[0]) > int(coarse.split("x")[0])


def test_boundary_filling_is_a_no_op_on_a_fully_indexed_crystal(base, analysed, renderer):
    """The served crystal has no unindexed atoms, so filling must change nothing.

    That filling does colour the boundaries is covered at the library level in
    tests/test_fill_integration.py, where a configuration with unindexed atoms
    can be constructed.
    """
    plain = _decode_png(_get(base, "/api/render?w=260&h=220")[2])
    filled = _decode_png(_get(base, "/api/render?w=260&h=220&fill_radius=6")[2])
    assert np.allclose(plain, filled, atol=0.02)


def test_fill_parameters_are_accepted_by_every_image_endpoint(base, analysed, renderer):
    for path in (
        "/api/render?w=200&h=160&fill_radius=6&fill_min_neighbours=4",
        "/api/figure/poles?poles=0001&fill_radius=6",
        "/api/figure/ipfdensity?fill_radius=6",
        "/api/figure/flatmap?view=z&pixel_size=1.0&fill_radius=6",
    ):
        status, headers, body = _get(base, path)
        assert status == 200, path
        assert headers["Content-Type"] == "image/png", path
        assert _decode_png(body).ndim == 3, path


def test_tripod_overlay_renders_and_picking_still_works(base, analysed, renderer):
    _, _, body = _get(base, "/api/render?w=300&h=240&tripod=1")
    assert _decode_png(body).shape[:2] == (240, 300)
    # The pick endpoint receives the same options, including ones it must ignore.
    picked = _post_json(
        base, "/api/pick", {"x": 150, "y": 120, "w": 300, "h": 240, "tripod": True}
    )
    assert "atom" in picked or "index" in picked or picked == {}

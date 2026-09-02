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


def test_progress_advances_through_the_stages(base, served_dir):
    """OVITO reports nothing while it works, so the bar is an interpolation.

    What can be checked is that it is monotonic, stays inside [0, 1], names the
    stage it is in, and ends at one.
    """
    outcome = _post_json(base, "/api/analyse", {"path": "crystal.xyz", "structures": ["hcp"]})
    assert outcome["accepted"]
    seen, stages, deadline = [], set(), time.time() + 120
    while time.time() < deadline:
        status = _get_json(base, "/api/status")
        if status["state"] != "running":
            break
        assert 0.0 <= status["progress"] <= 1.0
        assert status["stage"]
        seen.append(status["progress"])
        stages.add(status["stage"])
        time.sleep(0.05)
    assert status["state"] == "done", status.get("error")
    assert status["progress"] == 1.0
    assert seen == sorted(seen), f"the bar went backwards: {seen}"
    assert stages, "no stage was ever reported"


def test_columns_are_offered_for_a_file_with_orientations(base, served_dir):
    """The mapping is chosen from what the file has, not from what it might have."""
    info = _get_json(base, "/api/columns?path=crystal.xyz")
    names = {c["name"]: c["components"] for c in info["columns"]}
    assert names["Position"] == 3
    assert info["n_atoms"] > 0
    assert "guess" in info


def test_columns_refuse_a_path_outside_the_root(base):
    with pytest.raises(urllib.error.HTTPError) as caught:
        _get(base, "/api/columns?path=../../etc/passwd")
    assert caught.value.code == 403


@pytest.fixture(scope="module")
def foreign_file(served_dir):
    """The served crystal, written out the way another session would write it.

    Orientations in four separate columns with names nothing recognises and the
    scalar part first, which is the case this feature exists for.
    """
    from ovito.io import import_file
    from ovito.modifiers import PolyhedralTemplateMatchingModifier as Ptm

    pipeline = import_file(str(served_dir / "crystal.xyz"))
    pipeline.modifiers.append(Ptm(output_orientation=True, rmsd_cutoff=0.1))
    data = pipeline.compute(0)
    positions = np.asarray(data.particles.positions[...])
    quaternions = np.asarray(data.particles["Orientation"][...])
    types = np.asarray(data.particles["Structure Type"][...])

    path = served_dir / "foreign.dump"
    with open(path, "w") as handle:
        handle.write(f"ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n{len(positions)}\n")
        handle.write("ITEM: BOX BOUNDS pp pp pp\n")
        low, high = positions.min(axis=0), positions.max(axis=0)
        for axis in range(3):
            handle.write(f"{low[axis] - 1:.6f} {high[axis] + 1:.6f}\n")
        handle.write("ITEM: ATOMS id type x y z phase qw qx qy qz\n")
        for i, (x, y, z) in enumerate(positions, start=1):
            q = quaternions[i - 1]
            handle.write(
                f"{i} 1 {x:.5f} {y:.5f} {z:.5f} {int(types[i - 1])} "
                f"{q[3]:.8f} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f}\n"
            )
    return "foreign.dump"


def _analyse_and_wait(base, payload):
    outcome = _post_json(base, "/api/analyse", payload)
    assert outcome["accepted"], outcome
    deadline = time.time() + 120
    while time.time() < deadline:
        status = _get_json(base, "/api/status")
        if status["state"] != "running":
            return status
        time.sleep(0.1)
    pytest.fail("the analysis did not finish")


def test_analysing_from_columns_matches_a_real_ptm_run(base, foreign_file):
    """The whole point: the imported map is the same map PTM produced."""
    reference = _analyse_and_wait(
        base, {"path": "crystal.xyz", "structures": ["hcp"], "direction": "z"}
    )["result"]

    info = _get_json(base, "/api/columns?path=" + foreign_file)
    names = {c["name"] for c in info["columns"]}
    assert {"qw", "qx", "qy", "qz", "phase"} <= names
    assert info["guess"]["structure_type"] == "phase"

    imported = _analyse_and_wait(base, {
        "path": foreign_file,
        "structures": ["hcp"],
        "direction": "z",
        "columns": {
            "quaternion": ["qw", "qx", "qy", "qz"],
            "order": "wxyz",
            "structure_type": "phase",
        },
    })["result"]
    assert imported["counts"] == reference["counts"]
    assert imported["n_atoms"] == reference["n_atoms"]


def test_a_mapping_naming_a_column_that_is_not_there_is_reported(base, foreign_file):
    status = _analyse_and_wait(base, {
        "path": foreign_file,
        "structures": ["hcp"],
        "columns": {"quaternion": "Orientation"},
    })
    assert status["state"] == "error"
    assert "Orientation" in status["error"]


def test_a_column_mapping_without_a_quaternion_is_refused(base, served_dir):
    with pytest.raises(urllib.error.HTTPError) as caught:
        _post_json(base, "/api/analyse", {
            "path": "crystal.xyz", "columns": {"order": "xyzw"}
        })
    assert caught.value.code == 400
    assert "quaternion" in json.loads(caught.value.read())["error"]


def test_the_examples_catalogue_quotes_a_cost(base):
    catalogue = _get_json(base, "/api/examples")
    assert catalogue["examples"]
    for entry in catalogue["examples"]:
        assert entry["citation"] and entry["url"].startswith("https://")
        assert entry["estimate"]["n_atoms"] > 0
        assert entry["estimate"]["minutes_one_core"] > entry["estimate"]["minutes_four_cores"]
    # Whether atomsk is here decides which builder the page offers.
    assert "atomsk" in catalogue and "atomsk_help" in catalogue


def test_the_examples_page_is_served(base):
    status, headers, body = _get(base, "/examples")
    assert status == 200
    assert b"<title>ptm-ipf examples</title>" in body
    assert "text/html" in headers["Content-Type"]


def test_building_an_example_refuses_settings_that_would_not_run(base):
    with pytest.raises(urllib.error.HTTPError) as caught:
        _post_json(base, "/api/examples/build", {"element": "Cu", "box": 5.0})
    assert caught.value.code == 400
    assert "box" in json.loads(caught.value.read())["error"]


def test_the_colour_maps_on_offer_are_listed(base):
    meta = _get_json(base, "/api/meta")
    assert "jet" in meta["colormaps"] and "viridis" in meta["colormaps"]


@pytest.mark.parametrize(
    "query", ["cmap=jet", "cmap=rainbow", "smoothing=8", "cmap=turbo&smoothing=4"]
)
def test_pole_figures_take_a_colour_map_and_a_smoothing_width(base, analysed, query):
    status, _, body = _get(base, f"/api/figure/poles?poles=0001&{query}")
    assert status == 200
    assert _decode_png(body).size > 0


def test_smoothing_changes_the_figure(base, analysed):
    """If the option did nothing the two images would be identical."""
    _, _, plain = _get(base, "/api/figure/poles?poles=0001")
    _, _, smoothed = _get(base, "/api/figure/poles?poles=0001&smoothing=10")
    assert plain != smoothed


def test_the_ipf_density_takes_them_too(base, analysed):
    status, _, body = _get(base, "/api/figure/ipfdensity?cmap=jet&smoothing=5")
    assert status == 200
    assert _decode_png(body).size > 0


def test_an_unknown_colour_map_is_a_message_not_a_traceback(base, analysed):
    with pytest.raises(urllib.error.HTTPError) as caught:
        _get(base, "/api/figure/poles?poles=0001&cmap=not-a-colour-map")
    assert caught.value.code == 400
    assert "colour map" in json.loads(caught.value.read())["error"]


def _upload_colormap(base, name, raw):
    import base64

    return _post_json(
        base, "/api/colormap/upload", {"name": name, "data": base64.b64encode(raw).decode()}
    )


def test_a_colour_map_can_be_uploaded_as_a_table(base, analysed):
    table = b"# copied from a paper\n255 255 255\n255 200 0\n200 0 0\n0 0 0\n"
    assert _upload_colormap(base, "paper.txt", table) == {"name": "paper.txt", "entries": 4}
    status, _, body = _get(base, "/api/figure/poles?poles=0001&cmap=custom")
    assert status == 200
    assert _decode_png(body).size > 0


def test_a_colour_map_can_be_uploaded_as_an_image(base, analysed):
    """A screenshot of a colour bar, or the strip this tool writes for OVITO."""
    import io

    Image = pytest.importorskip("PIL.Image")
    ramp = np.linspace(0, 255, 32).astype(np.uint8)
    strip = np.stack([ramp, np.zeros_like(ramp), 255 - ramp], axis=1)[None].repeat(4, axis=0)
    buffer = io.BytesIO()
    Image.fromarray(strip).save(buffer, format="PNG")
    outcome = _upload_colormap(base, "bar.png", buffer.getvalue())
    assert outcome["entries"] == 32
    assert _get(base, "/api/figure/ipfdensity?cmap=custom")[0] == 200


@pytest.mark.parametrize("raw", [b"", b"\x00\x01\x02 not a colour map at all"])
def test_an_upload_that_is_not_a_colour_map_is_refused(base, raw):
    with pytest.raises(urllib.error.HTTPError) as caught:
        _upload_colormap(base, "junk.bin", raw)
    assert caught.value.code == 400


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


# ---------------------------------------------------------------------------
# slices on every figure, vector formats, tripod options, meta additions
# ---------------------------------------------------------------------------
def test_meta_lists_laue_groups_and_pole_presets(base):
    meta = _get_json(base, "/api/meta")
    laue = {s["name"]: s["laue"] for s in meta["structures"]}
    assert laue["fcc"] == laue["bcc"] and laue["hcp"] != laue["fcc"]
    presets = meta["defaults"]["pole_presets"]
    assert "0001" in presets["hexagonal"] and "111" in presets["cubic"]
    assert meta["defaults"]["poles"] == presets["hexagonal"][:3]
    assert set(meta["defaults"]["tripod"]) == {"size", "x", "y"}


@pytest.mark.parametrize(
    "path",
    [
        "/api/figure/poles?poles=0001&slice_axis=z&slice_frac=0.5",
        "/api/figure/ipfdensity?slice_axis=z&slice_distance=8&slice_width=6",
        "/api/figure/legend?slice_axis=z&slice_frac=0.5",
    ],
)
def test_figures_take_the_slice_of_the_view(base, analysed, path):
    status, headers, body = _get(base, path)
    assert status == 200
    assert headers["Content-Type"] == "image/png"
    assert _decode_png(body).ndim == 3


def test_a_slice_with_no_atoms_left_is_a_message_not_a_traceback(base, analysed):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(base, "/api/figure/poles?poles=0001&slice_axis=z&slice_distance=-500")
    assert excinfo.value.code == 400
    assert "no atoms" in json.loads(excinfo.value.read())["error"]


def test_slice_bounds_take_a_vector_axis(base, analysed):
    bounds = _get_json(base, "/api/slicebounds?axis=1,1,0")
    assert bounds["min"] < bounds["max"]
    _, _, body = _get(
        base, "/api/render?w=200&h=160&slice_axis=1,1,0&slice_distance=0&slice_width=8"
    )
    assert _decode_png(body).shape[:2] == (160, 200)


def test_the_flat_map_follows_the_slice(base, analysed):
    """With a slice, the map is a section of that slab, seen along its normal."""
    _, headers, _ = _get(
        base,
        "/api/figure/flatmap?view=x&pixel_size=1.0&slice_axis=z&slice_distance=8&slice_width=4",
    )
    assert abs(float(headers["X-Slab-Center"]) - 8.0) < 1e-6
    _, headers, _ = _get(
        base, "/api/figure/flatmap?view=z&pixel_size=1.0&slab_width=6&slice_axis=z&slice_distance=9"
    )
    # A zero-width slice ends at the plane, so the slab sits just below it.
    assert abs(float(headers["X-Slab-Center"]) - 6.0) < 1e-6


def test_the_ipfmap_alias_and_svg_output(base, analysed):
    status, headers, body = _get(base, "/api/figure/ipfmap?view=z&pixel_size=1.0&format=svg")
    assert status == 200
    assert headers["Content-Type"].startswith("image/svg")
    assert b"<svg" in body
    for figure in ("poles?poles=0001", "density?", "legend?"):
        _, headers, body = _get(base, f"/api/figure/{figure}&format=svg")
        assert headers["Content-Type"].startswith("image/svg"), figure
        assert b"<svg" in body, figure


def test_pole_figures_take_the_projection_axes(base, analysed):
    one = _decode_png(_get(base, "/api/figure/poles?poles=0001&up=rd&right=td")[2])
    other = _decode_png(_get(base, "/api/figure/poles?poles=0001&up=1,1,0&right=nd")[2])
    assert one.shape == other.shape
    assert not np.allclose(one, other)


def test_tripod_options_and_label_are_drawn(base, analysed, renderer):
    plain = _decode_png(_get(base, "/api/render?w=300&h=240&tripod=1")[2])
    custom = _decode_png(
        _get(
            base,
            "/api/render?w=300&h=240&tripod=1&tripod_axes=x;y;1,1,0"
            "&tripod_labels=;;loading&tripod_size=0.3&tripod_x=0.6&tripod_y=0.5"
            "&label=frame%2042",
        )[2]
    )
    assert plain.shape == custom.shape
    assert not np.allclose(plain, custom)


# ---------------------------------------------------------------------------
# trajectory series
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def series_files(served_dir):
    """Three numbered frames of the served crystal, named so that 100 sorts after 20."""
    atoms = ase_build.bulk("Mg", "hcp", a=3.2094, c=5.2108).repeat((4, 4, 4))
    names = []
    for step in (0, 20, 100):
        name = f"dump_{step}.xyz"
        ase_io.write(str(served_dir / name), atoms, format="extxyz")
        names.append(name)
    return names


def test_a_numbered_file_series_is_detected_in_numeric_order(base, series_files):
    series = _get_json(base, "/api/series?path=dump_20.xyz")
    assert series["kind"] == "files"
    assert [item["path"] for item in series["items"]] == series_files
    assert [item["step"] for item in series["items"]] == [0, 20, 100]
    assert series["current"] == 1


def test_a_lone_file_is_no_series(base, analysed):
    series = _get_json(base, "/api/series?path=crystal.xyz")
    assert series["kind"] == "none"
    assert series["items"] == []
    assert series["stem"] == "crystal"


def test_series_needs_a_path_inside_the_root(base):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(base, "/api/series?path=../outside.xyz")
    assert excinfo.value.code in (400, 403, 404)


def _wait_for_series(base):
    deadline = time.time() + 300
    while time.time() < deadline:
        status = _get_json(base, "/api/series/status")
        if status["state"] not in ("running", "idle"):
            return status
        time.sleep(0.2)
    pytest.fail("the series render did not finish")


def test_a_series_renders_stills_and_a_movie(base, analysed, series_files, renderer, served_dir):
    outcome = _post_json(
        base,
        "/api/series/render",
        {
            "path": "dump_0.xyz",
            "start": 0,
            "stop": 1,
            "step": 1,
            "outputs": ["view:png", "poles:gif", "legend:svg"],
            "seconds_per_frame": 0.2,
            "label": True,
            "view_query": {"w": 160, "h": 120, "tripod": 1},
            "poles_query": {"poles": "0001", "up": "rd", "right": "td"},
        },
    )
    assert outcome["accepted"] and outcome["n_items"] == 2
    analysed_path = _get_json(base, "/api/status")["result"]["path"]
    status = _wait_for_series(base)
    assert status["state"] == "done", status.get("error")
    files = set(status["files"])
    assert {"dump_0_view.png", "dump_20_view.png", "dump_0_legend.svg"} <= files
    movie = [f for f in files if f.endswith(".gif")]
    assert movie and movie[0].endswith("_poles.gif")
    out_dir = served_dir / status["out_dir"]
    assert (out_dir / "dump_0_view.png").stat().st_size > 0
    # The files are served back one at a time and as a zip.
    _, headers, body = _get(base, "/api/series/file?path=dump_0_view.png")
    assert headers["Content-Type"] == "image/png"
    assert _decode_png(body).shape[:2] == (120, 160)
    _, headers, archive = _get(base, "/api/series/zip")
    assert headers["Content-Type"] == "application/zip"
    import zipfile

    names = zipfile.ZipFile(io.BytesIO(archive)).namelist()
    assert any(name.endswith("dump_0_view.png") for name in names)
    # The analysis in the browser is untouched by the batch.
    assert _get_json(base, "/api/status")["result"]["path"] == analysed_path


def test_series_output_refuses_paths_outside_its_folder(base, analysed):
    for name in ("../crystal.xyz", "/etc/passwd", "nope.png"):
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _get(base, f"/api/series/file?path={name}")
        assert excinfo.value.code in (400, 404), name


def test_cancelling_when_nothing_runs_is_harmless(base):
    outcome = _post_json(base, "/api/series/cancel", {})
    assert isinstance(outcome, dict)


# ---------------------------------------------------------------------------
# rotations and slab analysis: these change the server's state, so they run
# last and put the full analysis back when they are done
# ---------------------------------------------------------------------------
def test_rotations_recolour_and_reach_the_command_line(base, analysed):
    plain = _decode_png(_get(base, "/api/figure/ipfdensity")[2])
    status = _analyse_and_wait(
        base,
        {
            "path": "crystal.xyz",
            "structures": ["hcp", "fcc"],
            "rotations": [{"axis": "x", "angle": 90}, ["z", 0]],
        },
    )
    assert status["state"] == "done", status.get("error")
    assert status["result"]["rotations"] == [["x", 90.0]]
    assert status["result"]["n_atoms"] == analysed["result"]["n_atoms"]
    turned = _decode_png(_get(base, "/api/figure/ipfdensity")[2])
    assert not np.allclose(plain, turned)
    outcome = _post_json(base, "/api/command", {"poles": ["0001"]})
    assert "--rotate x:90" in outcome["command"]
    parsed = _post_json(base, "/api/command/parse", {"command": outcome["command"]})
    assert parsed["colour"]["rotations"] == [["x", 90.0]]


def test_a_slab_analysis_is_a_subset_of_the_full_one(base, analysed):
    full = analysed["result"]["n_atoms"]
    status = _analyse_and_wait(
        base,
        {
            "path": "crystal.xyz",
            "structures": ["hcp", "fcc"],
            "slab": {"axis": "z", "distance": 10, "width": 6},
        },
    )
    assert status["state"] == "done", status.get("error")
    assert 0 < status["result"]["n_atoms"] < full
    assert status["result"]["slab"] == {"axis": "z", "distance": 10.0, "width": 6.0}
    assert status["result"]["full_n_atoms"] == full
    export = _get(base, "/api/export?format=extxyz")[2]
    assert export.split(b"\n", 1)[0] == str(status["result"]["n_atoms"]).encode()
    outcome = _post_json(base, "/api/command", {"poles": ["0001"], "slice_axis": "z"})
    command = outcome["command"]
    assert "--ptm-slice" in command
    assert "--slice z" in command and "--slice-distance 10" in command
    assert "--slice-width 6" in command
    parsed = _post_json(base, "/api/command/parse", {"command": command})
    assert parsed["ui"]["ptm_slice"] is True
    assert parsed["ui"]["slice_distance"] == 10.0


def test_the_tripod_axes_round_trip_through_the_command(base, analysed):
    outcome = _post_json(
        base, "/api/command", {"tripod": True, "tripod_axes": ["x", "y", "1,1,0=loading"]}
    )
    assert "--tripod-axes 'x;y;1,1,0=loading'" in outcome["command"]
    parsed = _post_json(base, "/api/command/parse", {"command": outcome["command"]})
    assert parsed["ui"]["tripod"] is True
    assert parsed["ui"]["tripod_axes"] == ["x", "y", "1,1,0=loading"]


def test_the_full_analysis_comes_back_from_the_cache(base, analysed):
    started = time.time()
    status = _analyse_and_wait(base, {"path": "crystal.xyz", "structures": ["hcp", "fcc"]})
    assert status["state"] == "done", status.get("error")
    assert status["result"]["n_atoms"] == analysed["result"]["n_atoms"]
    assert status["result"]["slab"] is None
    assert status["result"]["full_n_atoms"] == analysed["result"]["n_atoms"]
    # No PTM run was needed: the whole cell was already matched.
    assert time.time() - started < 10

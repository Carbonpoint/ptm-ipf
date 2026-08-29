"""The full ptm-ipf figure set for one showcase deformation run.

Given a run directory with ``<name>.*.dump`` frames and ``<name>.stress``, this
produces, in an output directory:

  stress_strain.png        the stress-strain curve with the frame strains marked
  final_ipf_{x,y,z}.png    3D renders of the final frame, triad on, one per projection
  grid_flat.png            the 3 by 3 section-and-projection grid, wireframes on
  grid_render.png          the same as rendered atoms, boundaries filled
  boundaries.png           final z section, boundaries coloured by misorientation
  hagb_lagb.png            final z section, high angle black and low angle grey
  pole_figures.png         pole figures of the final frame in the sample frame
  ipf_density.png          IPF density of the final frame
  evolution.mp4            flat map of the z section through every frame
  evolution_render.gif     rendered z section through every frame
  stats.json               structure counts, grain count, twin fraction per frame

Every figure is produced by the library the showcase is demonstrating.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ptmipf import boundaries  # noqa: E402
from ptmipf.analysis import analyse  # noqa: E402
from ptmipf.animate import animate_flat_map, animate_render, frame_files  # noqa: E402
from ptmipf.fill import fill_boundary_orientations  # noqa: E402
from ptmipf.flatmap import flat_ipf_map, save_flat_map  # noqa: E402
from ptmipf.frames import SampleFrame  # noqa: E402
from ptmipf.legend import ipf_legend  # noqa: E402
from ptmipf.polefigure import ipf_density, pole_figure  # noqa: E402
from ptmipf.render import render_result  # noqa: E402
from ptmipf.select import select_by_region  # noqa: E402
from ptmipf.wireframe import grain_wireframes  # noqa: E402

FRAME = SampleFrame({"rd": "1,0,0", "td": "0,1,0", "nd": "0,0,1"})
AXES = ("x", "y", "z")
ELEMENT = {
    "cu": ("fcc", "m-3m", ["111", "100", "110"], 1.0),
    "fe": ("bcc", "m-3m", ["110", "100", "111"], 1.0),
    "ti": ("hcp", "6/mmm", ["0001", "10-10", "10-12"], 1.587),
}


def main(run_dir, out_dir, rate, workers=8):
    run_dir, out = Path(run_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = run_dir.name
    element = name.split("_")[0]
    structure, laue, poles, c_over_a = ELEMENT[element]
    structures = ("fcc", "hcp", "bcc")
    files = frame_files(run_dir / f"{name}.*.dump")
    print(f"[{name}] {len(files)} frames, {structure}")
    t_all = time.time()

    # Stress-strain, with the frame strains marked.
    ss = np.loadtxt(run_dir / f"{name}.stress", skiprows=1)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(100 * np.abs(ss[:, 0]), np.abs(ss[:, 1]), color="0.2", lw=1.2)
    frame_strains = [int(f.stem.split(".")[-1]) * 0.002 * 1e-3 * abs(rate) for f in files]
    for s in frame_strains:
        ax.axvline(100 * s, color="tab:red", lw=0.5, alpha=0.5)
    ax.set_xlabel("engineering strain (%)")
    ax.set_ylabel("|stress| (GPa)")
    ax.set_title(name.replace("_", " "))
    fig.tight_layout()
    fig.savefig(out / "stress_strain.png", dpi=160)
    plt.close(fig)

    # Final frame, analysed once.
    final = analyse(files[-1], direction="x", structures=structures, frame=FRAME)
    stats = {"name": name, "frames": len(files), "final_counts": final.counts,
             "n_atoms": final.n_atoms}
    print(f"[{name}] final: {final.counts}")

    for axis in AXES:
        coloured = final.recolor(axis)
        render_result(coloured, out / f"final_ipf_{axis}.png", hide_other=True, tripod=True,
                      size=(1200, 1000))
        ipf_legend(laue, direction_label=axis.upper(), structure_label=structure,
                   filename=out / f"key_{axis}.png")

    rot = final.rotations(structure)
    pole_figure(rot, poles, laue, sample_frame=FRAME, up="rd", right="td", center="nd",
                c_over_a=c_over_a, filename=out / "pole_figures.png", dpi=170)
    ipf_density(rot, "rd", laue, sample_frame=FRAME, filename=out / "ipf_density.png", dpi=170)

    # Sections through the middle, flat and rendered, for the grids.
    grid_dir = out / "panels"
    grid_dir.mkdir(exist_ok=True)
    grain_counts = {}
    for cut in AXES:
        axis_index = "xyz".index(cut)
        centre = float(np.median(final.positions[:, axis_index]))
        section = final.subset(
            select_by_region(final, cut, minimum=centre - 5, maximum=centre + 5)
        )
        section = fill_boundary_orientations(section, radius=6.0, min_neighbours=3,
                                             structure=structure)
        for projection in AXES:
            col = section.recolor(projection)
            flat = flat_ipf_map(col, view=cut, slab_width=10.0, pixel_size=0.5,
                                boundary_angle=5.0, structure=structure)
            frames = grain_wireframes(flat, min_area_pixels=600, color="invert",
                                      c_over_a=c_over_a)
            save_flat_map(flat, grid_dir / f"{name}_cut{cut}_ipf{projection}.png",
                          scale_bar=(projection == "x"), axes_labels=False, title=None,
                          dpi=150, wireframes=frames, wireframe_linewidth=1.2)
            render_result(col, grid_dir / f"{name}_render_cut{cut}_ipf{projection}.png",
                          size=(900, 900), camera_dir={"x": (-1, 0, 0), "y": (0, -1, 0),
                          "z": (0, 0, -1)}[cut], perspective=False)
            if projection == "x":
                grain_counts[cut] = flat.n_grains
        if cut == "z":
            # Boundary colourings of the z section, IPF-X.
            col = section.recolor("x")
            flat = flat_ipf_map(col, view="z", slab_width=10.0, pixel_size=0.5,
                                boundary_angle=5.0, structure=structure)
            # The scale runs to the largest disorientation the symmetry allows.
            top = 93.9 if structure == "hcp" else 62.8
            rgb = boundaries.color_boundaries_by_angle(flat, 0, top, "plasma", width=2)
            save_flat_map(flat, out / "boundaries.png", rgb=rgb,
                          colorbar=(0, top, "plasma", "misorientation (degrees)"),
                          title=f"{name}: boundaries by misorientation", dpi=170)
            rgb = boundaries.color_boundaries_by_threshold(flat, 15.0, "black", "0.6", width=2)
            save_flat_map(flat, out / "hagb_lagb.png", rgb=rgb,
                          title=f"{name}: HAGB black, LAGB grey (15 degrees)", dpi=170)
    stats["grains_in_sections"] = grain_counts

    # Animations through the trajectory: the z section, IPF-X.
    animate_flat_map(files, out / "evolution.mp4", direction="x", view="z",
                     structures=structures, frame=FRAME, pixel_size=0.6, fill=6.0,
                     boundary_scale=(0, 60, "plasma"), wireframes=False, rate=rate,
                     title=name.replace("_", " "), fps=3, workers=1)
    animate_render(files, out / "evolution_render.gif", direction="x", view="z",
                   structures=structures, frame=FRAME, slab_width=10.0, fill=6.0,
                   rate=rate, fps=3, size=(800, 700))

    stats["seconds"] = round(time.time() - t_all, 1)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[{name}] done in {stats['seconds']} s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 8)

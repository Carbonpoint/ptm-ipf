"""Command line interface for ptm-ipf."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __version__
from .frames import SampleFrame
from .structures import DEFAULT_STRUCTURES, STRUCTURES, get_structure
from .symmetry import get_laue_group

_EPILOG = """\
examples:
  # IPF-Z map of an hcp magnesium configuration, with the colour key
  ptmipf mg.dump -o mg_ipf.xyz --direction z --legend mg_key.png

  # Extruded Mg: define the sample frame, then colour along the extrusion axis
  ptmipf mg.dump -o mg_ipf.xyz --ed 1,0,0 --nd 0,0,1 --direction ed

  # An arbitrary loading axis, plus basal and prismatic pole figures
  ptmipf mg.dump -o mg_ipf.xyz --direction 1,1,0 \\
      --pole-figure 0001 --pole-figure 10-10 --pole-figure-file mg_pf.png

  # A dual-phase Mg/Al configuration: identify both, colour only the hcp phase
  ptmipf alloy.dump -o out.xyz --structures hcp,fcc --color-only hcp

  # Only the basal-oriented grains: select them, then plot and render the subset
  ptmipf mg.dump --structures hcp --direction nd --from-selection \\
      --select-orientation '0001|nd|15' --selection-output basal.xyz \\
      --pole-figure 0001 --render basal.png --hide-other

  # A 10 angstrom section through the middle, seen face on like an EBSD map
  ptmipf mg.dump --structures hcp --direction z \\
      --slice z --slice-width 10 --view z --render section.png --hide-other

  # A flat EBSD-style orientation map of a section, boundaries filled in
  ptmipf mg.dump --structures hcp --direction nd --fill-boundaries 6 \\
      --view nd --slice-width 10 --flat-map map.png

  # One grain, picked from an atom in it, in the top half of the cell
  ptmipf mg.dump --structures hcp --from-selection --select-grain 12345 \\
      --select-region 'z|60|' --ipf-density grain.png
"""


def _parse_structure_list(text: str) -> tuple[str, ...]:
    names = [t.strip() for t in text.split(",") if t.strip()]
    return tuple(get_structure(n).name for n in names)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ptmipf",
        description=(
            "Colour atoms by crystal orientation using OVITO's polyhedral template "
            "matching and the EBSD inverse pole figure colour key."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", nargs="?", help="atomistic configuration or trajectory file")
    parser.add_argument(
        "-o", "--output", help="write the coloured configuration here (.xyz or .dump)"
    )
    parser.add_argument("--format", help="force the output format (extxyz or lammps-dump)")
    parser.add_argument(
        "--export-direction",
        metavar="DIR",
        action="append",
        help="also write a scalar colour-coding column for this direction, so OVITO's "
        "own Color coding modifier can repaint the atoms along it.  Repeatable; "
        "accepts anything --direction does.  Default: x, y and z",
    )
    parser.add_argument(
        "--no-export-directions",
        action="store_true",
        help="write no colour-coding columns, only Color.R/G/B for --direction",
    )
    parser.add_argument(
        "--color-map",
        metavar="PNG",
        help="where to write the colour bar the colour-coding columns index "
        "(default: <output>_colormap.png)",
    )
    parser.add_argument(
        "--color-map-gradient",
        metavar="NAME",
        help="index one of OVITO's built-in colour bars (jet, rainbow, viridis, ...) "
        "instead of writing an exact one.  A built-in bar is a curve through colour "
        "space and the IPF colours are a surface in it, so this only approximates "
        "them; the error is reported",
    )
    parser.add_argument("--version", action="version", version=f"ptm-ipf {__version__}")
    parser.add_argument(
        "--list-structures", action="store_true", help="list the supported structures and exit"
    )

    group = parser.add_argument_group("orientation reference")
    group.add_argument(
        "-d",
        "--direction",
        default="z",
        help="sample direction projected in the IPF: an axis (+z, -x), a named sample "
        "axis (rd, td, nd, ed) or a vector (1,1,0).  Default: z",
    )
    for axis in ("rd", "td", "nd", "ed"):
        group.add_argument(
            f"--{axis}",
            help=f"define the {axis.upper()} sample axis in cell coordinates, "
            f"e.g. --{axis} 1,1,0",
        )

    group = parser.add_argument_group("structure identification")
    group.add_argument(
        "-s",
        "--structures",
        default=",".join(DEFAULT_STRUCTURES),
        help="comma-separated structures for PTM to identify.  Default: %(default)s",
    )
    group.add_argument(
        "--color-only",
        "--colour-only",
        dest="color_only",
        help="colour only these structures; the others keep the 'other' colour",
    )
    group.add_argument(
        "--rmsd-cutoff",
        type=float,
        default=0.1,
        help="PTM RMSD cutoff, 0 disables it (default: %(default)s)",
    )
    group.add_argument(
        "--other-color",
        default="0.35,0.35,0.35",
        help="RGB colour for unidentified atoms (default: %(default)s)",
    )
    group.add_argument(
        "--frame", type=int, default=0, help="trajectory frame to analyse (default: 0)"
    )

    group = parser.add_argument_group("boundary filling")
    group.add_argument(
        "--fill-boundaries",
        nargs="?",
        type=float,
        const=6.0,
        metavar="RADIUS",
        help="give atoms PTM left unindexed the average orientation of the "
        "indexed atoms within RADIUS angstrom (default 6.0), so grain "
        "boundaries are coloured instead of blank",
    )
    group.add_argument(
        "--fill-min-neighbours",
        type=int,
        default=3,
        metavar="N",
        help="leave an atom unindexed if it has fewer than N indexed neighbours "
        "(default: %(default)s)",
    )

    group = parser.add_argument_group(
        "flat orientation map",
        "A section rasterised into a flat EBSD-style map: colours and grain "
        "boundaries, no atoms.",
    )
    group.add_argument("--flat-map", metavar="PNG", help="write a flat orientation map here")
    group.add_argument(
        "--pixel-size",
        type=float,
        default=0.5,
        metavar="ANGSTROM",
        help="pixel size of the flat map (default: %(default)s)",
    )
    group.add_argument(
        "--boundary-angle",
        type=float,
        default=5.0,
        metavar="DEG",
        help="misorientation above which neighbouring pixels are a grain "
        "boundary; 0 draws none (default: %(default)s)",
    )
    group.add_argument(
        "--flat-map-raw",
        action="store_true",
        help="paint unindexed atoms black instead of filling the map from the "
        "indexed ones, showing how wide the boundaries really are",
    )
    group.add_argument(
        "--no-scale-bar", action="store_true", help="omit the scale bar on the flat map"
    )
    group.add_argument(
        "--boundary-scale",
        nargs="?",
        const="0,90",
        metavar="MIN,MAX",
        help="colour the boundaries by misorientation on a colour scale over "
        "this range in degrees (default 0,90)",
    )
    group.add_argument(
        "--boundary-cmap", default="viridis", help="colormap for --boundary-scale"
    )
    group.add_argument(
        "--boundary-axis",
        metavar="HKIL",
        help="measure the boundary angle as the tilt of this crystal axis, e.g. "
        "0001 for the hcp c axis, instead of the full disorientation",
    )
    group.add_argument(
        "--boundary-threshold",
        type=float,
        metavar="DEG",
        help="draw boundaries at or above this angle in --boundary-high-color "
        "and the rest in --boundary-low-color (or hide them)",
    )
    group.add_argument("--boundary-high-color", default="black")
    group.add_argument(
        "--boundary-low-color",
        help="colour for boundaries below the threshold; omit to hide them",
    )
    group.add_argument(
        "--boundary-width", type=int, default=1, help="boundary line width in pixels"
    )
    group.add_argument(
        "--wireframes",
        action="store_true",
        help="draw a unit cell wireframe on each distinct orientation of the flat map",
    )
    group.add_argument(
        "--wireframe-tolerance",
        type=float,
        default=5.0,
        metavar="DEG",
        help="grains within this misorientation count as one orientation (default: 5)",
    )
    group.add_argument(
        "--wireframe-size",
        type=float,
        metavar="ANGSTROM",
        help="a fixed cell edge length; by default cells scale with the grain area",
    )
    group.add_argument(
        "--wireframe-scale",
        type=float,
        default=1.0,
        help="multiplier on the wireframe size, fixed or proportional (default: 1)",
    )
    group.add_argument(
        "--wireframe-color",
        default="invert",
        help="'invert' for the inverse of the grain colour, or any matplotlib "
        "colour such as black or '#ffffff' (default: invert)",
    )
    group.add_argument(
        "--wireframe-min-area",
        type=int,
        default=200,
        metavar="PIXELS",
        help="grains smaller than this get no wireframe (default: 200)",
    )
    group.add_argument(
        "--wireframe-one-per-orientation",
        action="store_true",
        help="one wireframe per orientation class, on its largest grain, rather "
        "than one on every grain",
    )

    group = parser.add_argument_group(
        "selection",
        "Build a subset of atoms from one or more criteria.  Fields of the "
        "composite options are separated by '|', so a direction may still "
        "contain commas.",
    )
    group.add_argument(
        "--select-structure", metavar="NAMES", help="select atoms identified as these structures"
    )
    group.add_argument(
        "--select-type", metavar="NAMES", help="select these particle types, by name or id"
    )
    group.add_argument(
        "--select-rmsd-below", type=float, metavar="F", help="select a better fit than F"
    )
    group.add_argument(
        "--select-rmsd-above", type=float, metavar="F", help="select a worse fit than F"
    )
    group.add_argument(
        "--select-region",
        action="append",
        default=[],
        metavar="AXIS|MIN|MAX",
        help="select a slab, e.g. 'z|10|60' or 'nd||60'.  Repeatable",
    )
    group.add_argument(
        "--select-orientation",
        action="append",
        default=[],
        metavar="CRYSTAL|SAMPLE|TOL",
        help="select atoms whose crystal direction lies within TOL degrees of a "
        "sample direction, e.g. '0001|nd|15' for the basal-oriented grains.  Repeatable",
    )
    group.add_argument(
        "--select-grain",
        type=int,
        metavar="INDEX",
        help="select atoms whose full orientation matches that of atom INDEX, "
        "which picks out one grain",
    )
    group.add_argument(
        "--select-grain-tolerance",
        type=float,
        default=10.0,
        metavar="DEG",
        help="misorientation tolerance for --select-grain (default: %(default)s)",
    )
    group.add_argument(
        "--select-mode",
        choices=("and", "or"),
        default="and",
        help="how to combine several criteria (default: %(default)s)",
    )
    group.add_argument("--invert-selection", action="store_true", help="select everything else")
    group.add_argument(
        "--orientation-structure",
        help="structure the orientation queries apply to (default: the first coloured structure)",
    )
    group.add_argument("--selection-output", metavar="FILE", help="write only the selection here")
    group.add_argument(
        "--from-selection",
        action="store_true",
        help="restrict the output, plots and rendering to the selection",
    )

    group = parser.add_argument_group("plots")
    group.add_argument(
        "--legend", nargs="?", const="", help="write the IPF colour key to this PNG"
    )
    group.add_argument(
        "--pole-figure",
        action="append",
        default=[],
        dest="pole_figures",
        metavar="HKIL",
        help="add a pole figure for this pole family, e.g. 0001 or 10-10 (repeatable)",
    )
    group.add_argument("--pole-figure-file", help="write the pole figures to this PNG")
    group.add_argument(
        "--pole-figure-mode",
        choices=("density", "scatter"),
        default="density",
        help="pole figure style (default: %(default)s)",
    )
    group.add_argument(
        "--pole-figure-structure",
        help="structure whose orientations are plotted (default: the first coloured structure)",
    )
    group.add_argument("--ipf-density", help="write the IPF orientation density plot to this PNG")
    group.add_argument(
        "--c-over-a",
        type=float,
        default=None,
        help="axial ratio for hexagonal poles such as {10-11} (default: ideal 1.633)",
    )
    group.add_argument(
        "--max-orientations",
        type=int,
        default=200_000,
        help="subsample this many orientations for the plots (default: %(default)s)",
    )
    group.add_argument("--dpi", type=int, default=200, help="resolution of the written figures")

    group = parser.add_argument_group(
        "animation",
        "Animate one section through every frame of a trajectory.  The input is "
        "then a glob such as 'run.*.dump', quoted so the shell leaves it alone.",
    )
    group.add_argument("--animate", metavar="FILE", help="write an .mp4 or .gif here")
    group.add_argument(
        "--animate-render",
        action="store_true",
        help="animate rendered atoms instead of a flat map",
    )
    group.add_argument(
        "--strain-rate",
        type=float,
        metavar="PER_PS",
        help="signed strain rate of the run, to stamp each frame with its strain",
    )
    group.add_argument("--fps", type=int, default=4, help="frames per second (default: 4)")
    group.add_argument(
        "--workers", type=int, default=1, help="frames rendered in parallel (flat maps only)"
    )

    group = parser.add_argument_group("rendering")
    group.add_argument("--render", metavar="PNG", help="render an IPF-coloured image with OVITO")
    group.add_argument(
        "--render-size", default="800x600", help="image size, e.g. 1600x1200 (default: %(default)s)"
    )
    group.add_argument(
        "--hide-other", action="store_true", help="hide atoms without a recognised orientation"
    )
    group.add_argument(
        "--slice",
        dest="slice_normal",
        help="cut the cell open with a plane of this normal (axis, sample axis or vector)",
    )
    group.add_argument(
        "--slice-distance", type=float, help="offset of the slice plane along its normal"
    )
    group.add_argument(
        "--slice-width",
        type=float,
        default=0.0,
        metavar="ANGSTROM",
        help="keep a slab of this thickness centred on the slice plane instead of "
        "cutting the cell in half; 0 cuts in half (default: %(default)s)",
    )
    group.add_argument(
        "--view",
        metavar="AXIS",
        help="view along this direction (axis, named sample axis or vector), "
        "orthographically.  '--slice z --slice-width 10 --view z' gives a 10 A "
        "section seen face on, like an EBSD map",
    )
    group.add_argument(
        "--perspective",
        action="store_true",
        help="use a perspective camera with --view instead of an orthographic one",
    )
    group.add_argument(
        "--transparent", action="store_true", help="render on a transparent background"
    )
    group.add_argument(
        "--tripod",
        action="store_true",
        help="draw a coordinate tripod labelled with the sample axes (RD, TD, ND)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress the summary")
    return parser


def _split_fields(text: str, n: int, option: str):
    fields = [f.strip() for f in text.split("|")]
    if not 1 <= len(fields) <= n:
        raise SystemExit(f"{option} expects up to {n} fields separated by '|', got {text!r}")
    return fields + [""] * (n - len(fields))


def _optional_float(text: str, option: str):
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        raise SystemExit(f"{option}: {text!r} is not a number") from None


def _color_keys(args, result):
    """Scalar colour-coding columns for the requested export directions.

    Returns ``(keys, palette, info)``; ``keys`` is empty when the columns are
    switched off, which keeps the writers on their original output.
    """
    if args.no_export_directions:
        return {}, None, None
    directions = args.export_direction or ["x", "y", "z"]
    from .colormap import color_keys

    return color_keys(result, directions, gradient=args.color_map_gradient)


def _report_color_keys(args, keys, palette, info, output, quiet: bool) -> None:
    """Write the colour bar beside the output and say how to use it."""
    if not keys or info is None:
        return
    columns = ", ".join(f"{name} ({label})" for name, label in info["directions"].items())
    if palette is None:
        if not quiet:
            print(
                f"colour-coding columns: {columns}\n"
                f"  load them with Color coding and OVITO's built-in "
                f"{info['gradient']} colour bar, range 0 to 1\n"
                f"  approximation error: {info['mean_error']:.3f} mean, "
                f"{info['max_error']:.3f} worst (RGB units of 1)"
            )
        return

    from pathlib import Path

    from .colormap import write_color_map

    target = args.color_map or str(Path(output).with_suffix("")) + "_colormap.png"
    write_color_map(palette, target)
    if not quiet:
        print(
            f"colour-coding columns: {columns}\n"
            f"wrote {target} ({info['entries']} colours, "
            f"worst colour error {info['max_error']:.4f})\n"
            "  in OVITO: Color coding, input property one of the columns above, "
            "range 0 to 1,\n"
            "  colour gradient 'Load custom color map' and pick that PNG"
        )


def _build_selection(args, result, structures, coloured):
    """Turn the --select-* options into one boolean mask, or None."""
    from . import select as selection

    masks = []
    if args.select_structure:
        wanted = _parse_structure_list(args.select_structure)
        masks.append(selection.select_by_structure(result, wanted))
    if args.select_type:
        names = [t.strip() for t in args.select_type.split(",") if t.strip()]
        masks.append(selection.select_by_type(result, names))
    if args.select_rmsd_below is not None or args.select_rmsd_above is not None:
        masks.append(
            selection.select_by_rmsd(
                result, maximum=args.select_rmsd_below, minimum=args.select_rmsd_above
            )
        )
    for spec in args.select_region:
        axis, low, high = _split_fields(spec, 3, "--select-region")
        masks.append(
            selection.select_by_region(
                result,
                axis or "z",
                minimum=_optional_float(low, "--select-region"),
                maximum=_optional_float(high, "--select-region"),
            )
        )

    orientation_structure = args.orientation_structure or next(
        (s for s in coloured if get_structure(s).colorable), None
    )
    needs_structure = args.select_orientation or args.select_grain is not None
    if needs_structure and orientation_structure is None:
        raise SystemExit("an orientation query needs a colourable structure")

    for spec in args.select_orientation:
        crystal, sample, tolerance = _split_fields(spec, 3, "--select-orientation")
        if not crystal or not sample:
            raise SystemExit(f"--select-orientation needs CRYSTAL|SAMPLE, got {spec!r}")
        masks.append(
            selection.select_by_ipf_direction(
                result,
                crystal,
                sample,
                _optional_float(tolerance, "--select-orientation") or 15.0,
                structure=orientation_structure,
                c_over_a=args.c_over_a if args.c_over_a is not None else float(np.sqrt(8.0 / 3.0)),
            )
        )

    if args.select_grain is not None:
        if not 0 <= args.select_grain < result.n_atoms:
            raise SystemExit(
                f"--select-grain {args.select_grain} is outside the configuration "
                f"(0 to {result.n_atoms - 1})"
            )
        masks.append(
            selection.select_by_misorientation(
                result,
                args.select_grain,
                args.select_grain_tolerance,
                structure=orientation_structure,
            )
        )

    if not masks:
        return None
    mask = selection.combine(*masks, mode=args.select_mode)
    return selection.invert(mask) if args.invert_selection else mask


def _animate(args) -> int:
    """The --animate path: many frames, one section, one video."""
    from .animate import animate_flat_map, animate_render, frame_files

    files = frame_files(args.input)
    if not files:
        raise SystemExit(f"no frames match {args.input!r}")
    axes = {name: getattr(args, name) for name in ("rd", "td", "nd", "ed") if getattr(args, name)}
    frame = SampleFrame(axes)
    structures = _parse_structure_list(args.structures)
    view = args.view or args.slice_normal or "z"
    slab = args.slice_width or 10.0
    fill = args.fill_boundaries
    if args.animate_render:
        pngs = animate_render(
            files, args.animate, direction=args.direction, view=view, structures=structures,
            frame=frame, slab_width=slab, hide_other=args.hide_other, fill=fill,
            tripod=args.tripod, rate=args.strain_rate, fps=args.fps,
            rmsd_cutoff=args.rmsd_cutoff,
        )
    else:
        scale = None
        if args.boundary_scale:
            lo, hi = (float(v) for v in args.boundary_scale.split(","))
            scale = (lo, hi, args.boundary_cmap)
        pngs = animate_flat_map(
            files, args.animate, direction=args.direction, view=view, structures=structures,
            frame=frame, slab_width=slab, pixel_size=args.pixel_size, fill=fill,
            boundary_angle=args.boundary_angle, boundary_scale=scale,
            wireframes=args.wireframes, rate=args.strain_rate, fps=args.fps,
            workers=args.workers, rmsd_cutoff=args.rmsd_cutoff,
        )
    if not args.quiet:
        print(f"wrote {args.animate} ({len(pngs)} frames)")
    return 0


def _color(text: str):
    parts = [float(t) for t in text.replace(",", " ").split()]
    if len(parts) != 3:
        raise SystemExit(f"--other-color needs three components, got {text!r}")
    return tuple(parts)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_structures:
        print(f"{'name':<16}{'Laue group':<12}description")
        for name, structure in STRUCTURES.items():
            laue = get_laue_group(structure.laue).name if structure.colorable else "-"
            print(f"{name:<16}{laue:<12}{structure.description}")
        return 0

    if not args.input:
        parser.error("an input file is required (or use --list-structures)")

    if args.animate:
        return _animate(args)

    from .analysis import analyse

    axes = {name: getattr(args, name) for name in ("rd", "td", "nd", "ed") if getattr(args, name)}
    frame = SampleFrame(axes)
    for warning in frame.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    structures = _parse_structure_list(args.structures)
    only = _parse_structure_list(args.color_only) if args.color_only else None

    result = analyse(
        args.input,
        direction=args.direction,
        structures=structures,
        frame=frame,
        frame_index=args.frame,
        rmsd_cutoff=args.rmsd_cutoff,
        other_color=_color(args.other_color),
        only=only,
    )

    if args.fill_boundaries is not None:
        from .fill import fill_boundary_orientations

        result = fill_boundary_orientations(
            result,
            radius=args.fill_boundaries,
            min_neighbours=args.fill_min_neighbours,
            structure=args.orientation_structure,
        )
        if not args.quiet:
            filled = int(result.interpolated.sum())
            print(
                f"filled {filled} unindexed atoms from neighbours within "
                f"{args.fill_boundaries:g} A"
            )

    if not args.quiet:
        print(result.summary())

    coloured = only or structures
    mask = _build_selection(args, result, structures, coloured)
    if mask is not None:
        selected = result.subset(mask)
        if not args.quiet:
            percent = 100 * selected.n_atoms / max(result.n_atoms, 1)
            print(f"selection: {selected.n_atoms} atoms ({percent:.1f} %)")
        if args.selection_output:
            from .io import write_result

            selection_keys, _, _ = _color_keys(args, selected)
            fmt = write_result(
                selected, args.selection_output, args.format, keys=selection_keys
            )
            if not args.quiet:
                print(f"wrote {args.selection_output} ({fmt}, selection only)")
        if args.from_selection:
            if selected.n_atoms == 0:
                raise SystemExit("the selection is empty; nothing to output or plot")
            result = selected
    elif args.selection_output or args.from_selection:
        raise SystemExit("no selection criteria were given; see the 'selection' options")

    if args.output:
        from .io import write_result

        keys, palette, key_info = _color_keys(args, result)
        fmt = write_result(result, args.output, args.format, keys=keys)
        if not args.quiet:
            print(f"wrote {args.output} ({fmt})")
        _report_color_keys(args, keys, palette, key_info, args.output, args.quiet)

    plot_structure = args.pole_figure_structure or next(
        (s for s in coloured if get_structure(s).colorable), None
    )

    if args.legend is not None:
        from .legend import ipf_legend

        for name in coloured:
            structure = get_structure(name)
            if not structure.colorable:
                continue
            filename = args.legend or f"ipf_key_{name}_{result.direction_label.lower()}.png"
            ipf_legend(
                get_laue_group(structure.laue),
                direction_label=result.direction_label,
                structure_label=name,
                filename=filename,
                dpi=args.dpi,
            )
            if not args.quiet:
                print(f"wrote {filename}")
            if args.legend:
                break  # an explicit filename means a single legend

    if args.pole_figures or args.ipf_density:
        if plot_structure is None:
            raise SystemExit("no colourable structure selected for the requested plots")
        structure = get_structure(plot_structure)
        laue = get_laue_group(structure.laue)
        rotations = result.rotations(structure.name)
        if len(rotations) == 0:
            raise SystemExit(f"no atoms were identified as {structure.name}; nothing to plot")
        c_over_a = args.c_over_a
        if c_over_a is None:
            c_over_a = float(np.sqrt(8.0 / 3.0))

        if args.pole_figures:
            from .polefigure import pole_figure

            filename = args.pole_figure_file or f"pole_figures_{structure.name}.png"
            pole_figure(
                rotations,
                args.pole_figures,
                laue,
                sample_frame=frame,
                c_over_a=c_over_a,
                mode=args.pole_figure_mode,
                max_orientations=args.max_orientations,
                filename=filename,
                dpi=args.dpi,
            )
            if not args.quiet:
                print(f"wrote {filename}")

        if args.ipf_density:
            from .polefigure import ipf_density

            ipf_density(
                rotations,
                args.direction,
                laue,
                sample_frame=frame,
                max_orientations=args.max_orientations,
                filename=args.ipf_density,
                dpi=args.dpi,
            )
            if not args.quiet:
                print(f"wrote {args.ipf_density}")

    if args.flat_map:
        from .flatmap import flat_ipf_map, save_flat_map

        view = args.view or args.slice_normal or "z"
        flat = flat_ipf_map(
            result,
            view=view,
            slab_width=args.slice_width or 10.0,
            slab_center=args.slice_distance,
            pixel_size=args.pixel_size,
            boundary_angle=args.boundary_angle,
            structure=args.orientation_structure,
            fill_unindexed=not args.flat_map_raw,
        )
        wireframes = None
        if args.wireframes:
            from .wireframe import grain_wireframes

            wireframes = grain_wireframes(
                flat,
                tolerance_deg=args.wireframe_tolerance,
                min_area_pixels=args.wireframe_min_area,
                size=args.wireframe_size,
                scale=args.wireframe_scale,
                color=args.wireframe_color,
                c_over_a=args.c_over_a if args.c_over_a is not None else float(np.sqrt(8.0 / 3.0)),
                one_per_orientation=args.wireframe_one_per_orientation,
            )
        rgb_override = None
        colorbar = None
        if args.boundary_scale or args.boundary_threshold is not None:
            from . import boundaries

            angles = None
            if args.boundary_axis:
                angles = boundaries.boundary_axis_angles(
                    flat,
                    args.boundary_axis,
                    c_over_a=(
                        args.c_over_a if args.c_over_a is not None else float(np.sqrt(8.0 / 3.0))
                    ),
                )
            if args.boundary_threshold is not None:
                rgb_override = boundaries.color_boundaries_by_threshold(
                    flat,
                    threshold=args.boundary_threshold,
                    high_color=args.boundary_high_color,
                    low_color=args.boundary_low_color,
                    angles=angles,
                    width=args.boundary_width,
                )
            else:
                lo, hi = (float(v) for v in args.boundary_scale.split(","))
                rgb_override = boundaries.color_boundaries_by_angle(
                    flat, vmin=lo, vmax=hi, cmap=args.boundary_cmap, angles=angles,
                    width=args.boundary_width,
                )
                label = (
                    f"{args.boundary_axis} tilt (degrees)" if args.boundary_axis
                    else "misorientation (degrees)"
                )
                colorbar = (lo, hi, args.boundary_cmap, label)
        save_flat_map(
            flat,
            args.flat_map,
            scale_bar=not args.no_scale_bar,
            title=f"IPF {result.direction_label}",
            dpi=args.dpi,
            wireframes=wireframes,
            rgb=rgb_override,
            colorbar=colorbar,
        )
        if not args.quiet:
            print(
                f"wrote {args.flat_map} ({flat.shape[1]} x {flat.shape[0]} px, "
                f"{flat.slab_width:g} A section along {flat.view_label})"
            )

    if args.render:
        from .render import render_result

        try:
            width, height = (int(v) for v in args.render_size.lower().split("x"))
        except ValueError:
            raise SystemExit(
                f"--render-size must look like 800x600, got {args.render_size!r}"
            ) from None
        camera_dir = (-1.0, -1.0, -0.5)
        perspective = True
        if args.view:
            # The camera looks along the given direction, so it sits on the far
            # side of the cell from it.
            camera_dir = tuple(-c for c in frame.direction(args.view))
            perspective = args.perspective
        render_result(
            result,
            args.render,
            hide_other=args.hide_other,
            slice_normal=args.slice_normal,
            slice_distance=args.slice_distance,
            slice_width=args.slice_width,
            size=(width, height),
            camera_dir=camera_dir,
            perspective=perspective,
            transparent=args.transparent,
            tripod=args.tripod,
        )
        if not args.quiet:
            print(f"wrote {args.render}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

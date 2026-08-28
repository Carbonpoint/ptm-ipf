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
        "--transparent", action="store_true", help="render on a transparent background"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress the summary")
    return parser


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

    if not args.quiet:
        print(result.summary())

    if args.output:
        from .io import write_result

        fmt = write_result(result, args.output, args.format)
        if not args.quiet:
            print(f"wrote {args.output} ({fmt})")

    coloured = only or structures
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

    if args.render:
        from .render import render_ipf

        try:
            width, height = (int(v) for v in args.render_size.lower().split("x"))
        except ValueError:
            raise SystemExit(
                f"--render-size must look like 800x600, got {args.render_size!r}"
            ) from None
        render_ipf(
            args.input,
            args.render,
            direction=args.direction,
            structures=structures,
            frame=frame,
            frame_index=args.frame,
            rmsd_cutoff=args.rmsd_cutoff,
            other_color=_color(args.other_color),
            hide_other=args.hide_other,
            slice_normal=args.slice_normal,
            slice_distance=args.slice_distance,
            size=(width, height),
            transparent=args.transparent,
        )
        if not args.quiet:
            print(f"wrote {args.render}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

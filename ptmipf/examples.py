"""Ready-made starter examples: build a polycrystal, fetch the potential, write the run.

One call produces a directory that contains everything needed to deform a small
polycrystal and then look at the result with ptm-ipf: the structure in both
LAMMPS and extended XYZ form, the EAM potential from NIST, the LAMMPS input,
the grain orientations that were built, and a README saying what to type.

The defaults are sized for a laptop.  Nothing here runs LAMMPS; it writes the
run and says what it will cost, because a tool that quietly starts a multi
minute job on someone's machine is a worse tool.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = ["DEFAULTS", "ExampleSpec", "build_example", "main"]


@dataclass
class ExampleSpec:
    """Everything that decides what an example contains."""

    element: str = "Cu"
    box: float = 48.0  #: cube edge in angstrom
    n_grains: int = 6
    temperature: float = 300.0
    strain: float = 0.10
    strain_rate: float = 2.5e-3  #: engineering strain per picosecond
    seed: int = 1
    builder: str = "atomsk"

    def validate(self) -> None:
        from .potentials import potential_for

        potential_for(self.element)
        if not 12.0 <= self.box <= 200.0:
            raise ValueError("box must be between 12 and 200 angstrom")
        if not 1 <= self.n_grains <= 64:
            raise ValueError("n_grains must be between 1 and 64")
        if not 0.005 <= self.strain <= 0.5:
            raise ValueError("strain must be between 0.005 and 0.5")
        if not 1e-5 <= self.strain_rate <= 0.1:
            raise ValueError("strain rate must be between 1e-5 and 0.1 per picosecond")
        if not 1.0 <= self.temperature <= 2000.0:
            raise ValueError("temperature must be between 1 and 2000 K")


#: The catalogue the interface offers, chosen to run in a couple of minutes.
DEFAULTS = {
    "cu_compression": ExampleSpec(element="Cu", box=48.0, n_grains=6),
    "al_compression": ExampleSpec(element="Al", box=54.0, n_grains=6),
    "ni_compression": ExampleSpec(element="Ni", box=47.0, n_grains=6),
    "fe_compression": ExampleSpec(element="Fe", box=46.0, n_grains=6),
}


def example_directory(root, spec: ExampleSpec) -> Path:
    """Where an example lands: one directory per element under ``examples``."""
    name = f"{spec.element.lower()}_{spec.n_grains}grain_compression"
    return Path(root) / "examples" / name


def build_example(root, spec: ExampleSpec | None = None, **overrides) -> dict:
    """Build one example under ``<root>/examples/`` and report what was made.

    Returns a dictionary the web interface renders directly: the directory, the
    files, the structure statistics, the run cost and the commands to type.
    """
    from .lammps import compression_script
    from .polycrystal import build_polycrystal
    from .potentials import download_potential, potential_for

    spec = spec or ExampleSpec()
    if overrides:
        spec = ExampleSpec(**{**asdict(spec), **overrides})
    spec.validate()

    potential = potential_for(spec.element)
    directory = example_directory(root, spec)
    directory.mkdir(parents=True, exist_ok=True)

    # The download first: it is the only step that can fail for reasons outside
    # this machine, and failing before building saves the wait.
    potential_path = download_potential(spec.element, directory)

    crystal = build_polycrystal(
        spec.element,
        spec.box,
        spec.n_grains,
        directory,
        structure=potential.structure,
        a0=potential.a0,
        seed=spec.seed,
        builder=spec.builder,
    )

    script, settings = compression_script(
        potential,
        crystal.files["data"],
        crystal.n_atoms,
        temperature=spec.temperature,
        strain=spec.strain,
        strain_rate=spec.strain_rate,
        seed=spec.seed * 977 + 13,
    )
    (directory / "in.compression").write_text(script, encoding="utf-8", newline="\n")

    # The orientations that were built, so a map can be checked against them.
    (directory / "grains.json").write_text(
        json.dumps(
            {
                "element": crystal.element,
                "structure": crystal.structure,
                "a0": crystal.a0,
                "box": crystal.box,
                "builder": crystal.builder,
                "seeds": crystal.seeds.tolist(),
                "rotations": crystal.rotations.tolist(),
            },
            indent=1,
        ),
        encoding="utf-8",
        newline="\n",
    )

    report = {
        "directory": str(directory),
        "relative": str(Path("examples") / directory.name),
        "element": spec.element,
        "structure": crystal.structure,
        "builder": crystal.builder,
        "n_atoms": crystal.n_atoms,
        "n_grains": crystal.n_grains,
        "box": round(crystal.box, 3),
        "grain_size": round(crystal.mean_grain_size, 1),
        "density": round(crystal.density, 4),
        "min_separation": round(crystal.min_separation, 3),
        "potential": {
            "file": potential_path.name,
            "citation": potential.citation,
            "url": potential.entry_url,
            "pair_style": potential.pair_style,
        },
        "run": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in settings.items()},
        "files": sorted(p.name for p in directory.iterdir() if p.is_file()),
        "spec": asdict(spec),
    }
    (directory / "README.md").write_text(_readme(report, crystal), encoding="utf-8", newline="\n")
    report["files"] = sorted(p.name for p in directory.iterdir() if p.is_file())
    return report


def _readme(report: dict, crystal) -> str:
    run = report["run"]
    low = run["minutes_four_cores"]
    high = run["minutes_one_core"]
    builder_note = (
        ""
        if report["builder"] == "atomsk"
        else "\nThis structure was built by the fallback NumPy builder, not atomsk, "
        "because atomsk was not found on this machine. It is a few percent less "
        "dense at the grain boundaries, which softens the elastic response a "
        "little. Install atomsk from https://atomsk.univ-lille.fr and rebuild for "
        "the reference structure.\n"
    )
    return f"""# {report["element"]} polycrystal under compression

{crystal.summary()}.
Closest pair as built: {report["min_separation"]} A, which the minimisation at the
top of the input script relaxes before any dynamics runs.
{builder_note}
## Run it

```bash
lmp -in in.compression            # or: mpirun -np 4 lmp -in in.compression
```

About {run["atom_steps"]:,} atom-steps: roughly {low:.0f} to {high:.0f} minutes
depending on how many cores it gets.

It writes `stress_strain.txt` (engineering strain against stress in GPa),
`compression.dump` ({run["load_steps"] // run["dump_every"]} frames) and
`compressed.lmp`.

## Look at it

```bash
ptmipf compression.dump --structures {report["structure"]} --direction z \\
    --legend key.png --render map.png --hide-other
ptmipf-ui compression.dump        # or open it in the web interface
```

The last frame is the deformed structure. `--frame 0` is the starting one, which
should be a clean set of randomly oriented grains: that is the control that says
the build and the colouring are both working.

## What is here

| file | what it is |
| --- | --- |
| `{crystal.files.get("data", "structure.lmp")}` | the structure, LAMMPS data format |
| `{crystal.files.get("xyz", "structure.xyz")}` | the same structure for ptm-ipf |
| `in.compression` | the LAMMPS input |
| `{report["potential"]["file"]}` | the EAM potential, from NIST |
| `grains.json` | the seed points and rotation matrices that were built |

The potential is {report["potential"]["citation"]}, taken from the NIST
Interatomic Potentials Repository, {report["potential"]["url"]}.
Cite it if you publish anything from this.
"""


def main(argv=None) -> int:
    """``ptmipf-example``: build a starter example from the terminal."""
    import argparse

    from .potentials import POTENTIALS

    parser = argparse.ArgumentParser(
        prog="ptmipf-example",
        description="Build a small polycrystal, fetch its EAM potential from NIST "
        "and write a LAMMPS compression run for it.",
    )
    parser.add_argument(
        "element", nargs="?", default="Cu", choices=sorted(POTENTIALS), help="default: Cu"
    )
    parser.add_argument("--root", default=".", help="examples land under <root>/examples")
    parser.add_argument("--box", type=float, help="cube edge in angstrom")
    parser.add_argument("--grains", type=int, help="number of grains")
    parser.add_argument("--strain", type=float, help="engineering strain to reach")
    parser.add_argument("--strain-rate", type=float, help="engineering strain per picosecond")
    parser.add_argument("--temperature", type=float, help="kelvin")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument(
        "--builder",
        choices=("atomsk", "voronoi", "auto"),
        default="atomsk",
        help="atomsk builds at full density and is the default; voronoi is the "
        "built-in fallback; auto uses atomsk when it is installed",
    )
    args = parser.parse_args(argv)

    spec = DEFAULTS.get(f"{args.element.lower()}_compression", ExampleSpec(element=args.element))
    overrides = {"builder": args.builder}
    for name, value in (
        ("box", args.box),
        ("n_grains", args.grains),
        ("strain", args.strain),
        ("strain_rate", args.strain_rate),
        ("temperature", args.temperature),
        ("seed", args.seed),
    ):
        if value is not None:
            overrides[name] = value

    try:
        report = build_example(args.root, spec, **overrides)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    run = report["run"]
    print(f"wrote {report['directory']}")
    print(
        f"  {report['n_atoms']} atoms, {report['n_grains']} grains averaging "
        f"{report['grain_size']} A, {100 * report['density']:.1f} % dense "
        f"({report['builder']} builder)"
    )
    print(
        f"  {run['atom_steps']:,} atom-steps: roughly "
        f"{run['minutes_four_cores']:.0f} to {run['minutes_one_core']:.0f} minutes"
    )
    print(f"  cd {report['directory']} && lmp -in in.compression")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

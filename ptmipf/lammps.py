"""LAMMPS input for the built-in compression examples.

The generated script is meant to be read, not just run: it is the shortest
uniaxial compression that produces something worth colouring, with every choice
that matters written where it can be changed.

Two details are easy to get wrong and are pinned down here.  The barostat must
not control the loading axis, because ``fix deform`` already does; it relaxes
the two transverse axes to zero pressure so the cell can contract the way a
real specimen does.  And the potential's pair style comes from the catalogue
entry rather than from habit: ``Fe_2.eam.fs`` read as ``eam/alloy`` is not an
error, it is a different and wrong potential.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["compression_script", "estimate_cost", "write_data_file"]

#: LAMMPS metal units report pressure in bar.
_BAR_PER_GPA = 10000.0


def write_data_file(crystal, path, mass: float | None = None) -> str:
    """Write a polycrystal as a LAMMPS data file, ``atom_style atomic``."""
    from .potentials import POTENTIALS

    if mass is None:
        entry = POTENTIALS.get(crystal.element)
        mass = entry.mass if entry else 1.0
    path = Path(path)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            f"# {crystal.n_grains}-grain {crystal.element} {crystal.structure} "
            f"polycrystal from ptm-ipf ({crystal.builder} builder)\n\n"
        )
        handle.write(f"{crystal.n_atoms} atoms\n1 atom types\n\n")
        for axis in "xyz":
            handle.write(f"0.0 {crystal.box:.8f} {axis}lo {axis}hi\n")
        handle.write(f"\nMasses\n\n1 {mass:.4f}\n\nAtoms # atomic\n\n")
        for index, (x, y, z) in enumerate(crystal.positions, start=1):
            handle.write(f"{index} 1 {x:.6f} {y:.6f} {z:.6f}\n")
    return str(path)


#: Atom-steps per second, measured by timing a generated run of this shape on a
#: workstation (8559 Cu atoms, 22000 steps, 92 s on four cores).  A laptop will
#: be slower, which is why the estimate is quoted as a range.
_RATE_ONE_CORE = 6.0e5
_RATE_FOUR_CORES = 2.0e6


def estimate_cost(n_atoms: int, steps: int) -> dict:
    """Rough wall clock for the generated run.

    Quoted as a range, and with the raw atom-steps behind it, because the rate
    varies by a factor of two or so between machines and there is no honest
    single number.
    """
    atom_steps = n_atoms * steps
    return {
        "atom_steps": atom_steps,
        "minutes_one_core": atom_steps / _RATE_ONE_CORE / 60.0,
        "minutes_four_cores": atom_steps / _RATE_FOUR_CORES / 60.0,
    }


def compression_script(
    potential,
    data_file: str,
    n_atoms: int,
    temperature: float = 300.0,
    strain: float = 0.10,
    strain_rate: float = 2.5e-3,
    timestep: float = 0.002,
    equilibration_ps: float = 4.0,
    dump_frames: int = 20,
    seed: int = 8765,
) -> tuple[str, dict]:
    """The compression input script, and the settings it was written with.

    *strain_rate* is engineering strain per picosecond, which is what
    ``fix deform ... erate`` takes in metal units.
    """
    load_ps = strain / strain_rate
    load_steps = int(round(load_ps / timestep))
    equilibration_steps = int(round(equilibration_ps / timestep))
    dump_every = max(1, load_steps // max(dump_frames, 1))
    # ave/time needs sample_every * (samples - 1) <= dump_every.
    samples = min(10, dump_every)
    sample_every = max(1, dump_every // max(samples, 1))
    settings = {
        "temperature": temperature,
        "strain": strain,
        "strain_rate": strain_rate,
        "timestep": timestep,
        "load_steps": load_steps,
        "equilibration_steps": equilibration_steps,
        "dump_every": dump_every,
        "pair_style": potential.pair_style,
        **estimate_cost(n_atoms, load_steps + equilibration_steps),
    }

    script = f"""# Uniaxial compression of a {potential.element} polycrystal along z.
# Written by ptm-ipf.  Run it with:  lmp -in in.compression
#
# Potential: {potential.citation}
#   from the NIST Interatomic Potentials Repository,
#   {potential.entry_url}

units           metal
dimension       3
boundary        p p p
atom_style      atomic

read_data       {data_file}

pair_style      {potential.pair_style}
pair_coeff      * * {potential.filename} {potential.element}

neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes

# The as-built grain boundaries hold atoms closer than any relaxed structure
# would.  Minimisation pushes them apart before any dynamics is attempted.
thermo          200
thermo_style    custom step temp pe press lx ly lz
minimize        1.0e-10 1.0e-10 20000 100000

reset_timestep  0
timestep        {timestep}
velocity        all create {2 * temperature:g} {seed} mom yes rot yes dist gaussian

# Equilibrate at zero pressure so the cell finds its own volume.
fix             settle all npt temp {temperature:g} {temperature:g} 0.1 iso 0.0 0.0 1.0
run             {equilibration_steps}
unfix           settle

reset_timestep  0
# $(lz) evaluates once, here.  Writing "variable lz0 equal lz" would re-read the
# current length every time it was used, and the strain would always be zero.
variable        lz0 equal $(lz)
# Both reported as positive under compression, which is how a compression curve
# is usually read.  The underlying sign convention is the usual one: the true
# stress component is -pzz and the true engineering strain is (lz-lz0)/lz0.
variable        strain equal (v_lz0-lz)/v_lz0
variable        stress equal pzz/{_BAR_PER_GPA:g}

# fix deform drives z; the barostat must therefore leave z alone and relax only
# the transverse axes, which is what lets the cell bulge as it is squashed.
fix             load all npt temp {temperature:g} {temperature:g} 0.1 &
                x 0.0 0.0 1.0 y 0.0 0.0 1.0
fix             squash all deform 1 z erate -{strain_rate:g} units box remap x

thermo          {dump_every}
thermo_style    custom step temp pe press pzz lz v_strain v_stress
# ave/time takes the variables by name and evaluates them when it runs, which
# fix print does not reliably do: the input parser can substitute ${{strain}}
# once, at parse time, and write the same number on every line.  Ten samples per
# line rather than one instant: a cell this small is noisy, and averaging over
# the interval costs nothing.
fix             curve all ave/time {sample_every} {samples} {dump_every} v_strain v_stress &
                file stress_strain.txt &
                title1 "# uniaxial compression of {potential.element}, written by ptm-ipf" &
                title2 "# step compressive_strain compressive_stress_GPa"

dump            traj all custom {dump_every} compression.dump id type x y z
dump_modify     traj sort id

run             {load_steps}

write_data      compressed.lmp
print           "compression finished: {strain:g} engineering strain reached"
"""
    return script, settings

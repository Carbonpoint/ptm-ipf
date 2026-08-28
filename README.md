# ptm-ipf

**EBSD-style inverse pole figure colouring for atomistic simulations.**

`ptm-ipf` runs [OVITO](https://www.ovito.org)'s polyhedral template matching (PTM) on an
atomistic configuration, turns the per-atom orientation quaternions into inverse pole
figure (IPF) colours using the standard EDAX/TSL colour key, and draws the matching
colour key, pole figures and orientation densities — so a molecular dynamics
microstructure can be put side by side with an EBSD orientation map and read the same way.

<p align="center">
  <img src="docs/mg_ipf_nd.png" width="45%" alt="IPF-ND coloured magnesium polycrystal">
  <img src="docs/ipf_key_hcp_nd.png" width="34%" alt="hexagonal IPF colour key">
</p>

<p align="center"><em>A textured hcp magnesium polycrystal, cut open and coloured by
IPF-ND, with its colour key. Reds mean c axes near ND — the basal texture of rolled or
extruded magnesium.</em></p>

## Why

OVITO computes the orientations but does not colour them the way EBSD software does: its
[example M3](https://docs.ovito.org/python/introduction/examples/modifiers/visualize_local_lattice_orientation.html)
maps quaternions through Rodrigues space, which is a valid colouring but not an inverse
pole figure and not comparable with an EBSD map. Conversely
[MTEX](https://mtex-toolbox.github.io) and [orix](https://orix.readthedocs.io) implement
the proper IPF colour keys but work on EBSD maps, not on atoms.

`ptm-ipf` closes that gap:

* **Colours that match EBSD software.** The key is the EDAX/TSL scheme of Nolze &
  Hielscher, verified against `orix` to better than `1e-3` in RGB for every supported
  Laue group (`tests/test_orix_agreement.py`).
* **Any reference direction.** Project along a cell axis, along a named sample axis
  (`RD`, `TD`, `ND`, `ED`) that you define for your cell, or along an arbitrary vector.
* **More than cubic.** hcp, fcc, bcc, simple cubic, cubic and hexagonal diamond, and
  graphene, each with the right Laue group.
* **Pole figures too**, in the same sample frame, as equal-area MRD contour plots.
* **Verified conventions.** The PTM quaternion convention and the hcp template frame the
  colours depend on are pinned down by tests that run real PTM
  (`tests/test_ptm_convention.py`), so an OVITO change cannot silently rotate your maps.

## Install

```bash
pip install git+https://github.com/Carbonpoint/ptm-ipf.git
```

On a bare Linux machine OVITO also needs the OpenGL runtime:
`sudo apt install libopengl0 libegl1`.

## Quick start

```bash
# IPF-Z map of an hcp magnesium configuration, plus the colour key
ptmipf mg.dump -o mg_ipf.xyz --structures hcp --direction z --legend key.png
```

The output file carries the colour as three extra columns, so it can be dropped straight
into OVITO, VMD or a plotting script. To render the picture directly:

```bash
ptmipf mg.dump --structures hcp --direction nd \
    --render mg_map.png --hide-other --slice nd --render-size 1600x1200
```

### Choosing the reference direction

The direction is whatever you would project an EBSD map along. Give it as an axis, a
named sample axis, or a vector:

```bash
ptmipf mg.dump -o out.xyz --direction z          # a cell axis
ptmipf mg.dump -o out.xyz --direction 1,1,0      # an arbitrary vector
ptmipf mg.dump -o out.xyz --direction -x         # a negative axis
```

Named axes are defined once and then used by name, which is the natural way to describe a
rolled or extruded sample whose axes are not the cell axes:

```bash
# Extrusion along the cell's [110], sheet normal along z
ptmipf mg.dump -o out.xyz --ed 1,1,0 --nd 0,0,1 --direction ed
```

`--rd`, `--td` and `--nd` are completed to a right-handed orthonormal frame: give two and
the third follows, give one and the others are chosen perpendicular to it. Non-orthogonal
input is orthogonalised with a warning rather than silently accepted. `--ed` is stored as
an extra named direction, so `ED` and `RD` can differ.

### Pole figures

```bash
ptmipf mg.dump --structures hcp --direction nd --c-over-a 1.6236 \
    --pole-figure 0001 --pole-figure 10-10 --pole-figure-file pf.png \
    --ipf-density ipf_density.png
```

<p align="center">
  <img src="docs/mg_pole_figures.png" width="58%" alt="basal and prismatic pole figures">
  <img src="docs/mg_ipf_density.png" width="35%" alt="IPF orientation density">
</p>

Pole figures are Lambert equal-area projections on the upper hemisphere, contoured in
multiples of a random distribution (MRD); equal area means equal solid angle, which is
what makes MRD meaningful. Poles are given in Miller or Miller–Bravais notation
(`0001`, `10-10`, `11-20`, `111`) and are expanded over all symmetry equivalents.
Non-basal, non-prismatic hexagonal poles such as `{10-11}` depend on the axial ratio, so
pass `--c-over-a` for your material (1.6236 for magnesium; the ideal 1.633 is the default).

### Supported structures

| structure | Laue group | red / green / blue corners |
|---|---|---|
| `hcp`, `hex_diamond`, `graphene` | `6/mmm` | `[0001]`, `[2-1-10]`, `[10-10]` |
| `fcc`, `bcc`, `sc`, `cubic_diamond` | `m-3m` | `[001]`, `[101]`, `[111]` |

`ptmipf --list-structures` prints the full list. Tetragonal (`4/mmm`), trigonal (`-3m`)
and orthorhombic (`mmm`) keys are implemented too and are available through the Python API
for orientation data from other sources.

In a multiphase configuration, identify several structures but colour only the phase of
interest:

```bash
ptmipf alloy.dump -o out.xyz --structures hcp,fcc --color-only hcp
```

## Python API

```python
from ptmipf import analyse, ipf_legend, pole_figure, get_laue_group
from ptmipf.frames import SampleFrame

frame = SampleFrame({"ed": "1,1,0", "nd": "0,0,1"})
result = analyse("mg.dump", direction="ed", structures=("hcp",), frame=frame)

print(result.summary())
colors = result.colors                  # (n, 3) RGB per atom
rotations = result.rotations("hcp")     # crystal-to-sample rotation matrices

ipf_legend("6/mmm", direction_label="ED", filename="key.png")
pole_figure(rotations, ["0001", "10-10"], "6/mmm", sample_frame=frame,
            c_over_a=1.6236, filename="pf.png")
```

The colour key is independent of OVITO, so orientations from any source can be coloured:

```python
from ptmipf import IPFColorKey, get_laue_group

key = IPFColorKey(get_laue_group("hexagonal"))
rgb = key.orientation2color(rotation_matrices, sample_direction=[0, 0, 1])
```

### Inside an OVITO pipeline

`ipf_color_modifier` returns a plain OVITO modifier function, so the colours can be part
of an existing pipeline and rendered by OVITO itself:

```python
from ovito.io import import_file
from ovito.modifiers import PolyhedralTemplateMatchingModifier
from ptmipf import ipf_color_modifier

pipeline = import_file("mg.dump")
pipeline.modifiers.append(PolyhedralTemplateMatchingModifier(output_orientation=True))
pipeline.modifiers.append(ipf_color_modifier(direction="z", structures=("hcp",)))
```

## Conventions

These were determined empirically from OVITO and are asserted by the test suite:

* OVITO stores the PTM orientation as a quaternion in `(x, y, z, w)` order, and the
  rotation it represents maps the **crystal (template) frame onto the sample frame**.
  The direction shown in an inverse pole figure is therefore `Rᵀ d`, not `R d`.
* The hexagonal templates have **c along z and a₁ along x**; the cubic templates have
  their crystal axes along the Cartesian axes.
* Orientations are reduced into the fundamental zone by PTM, and the colour key is
  invariant under the Laue group, so the reduction does not affect the colours.

## Limitations

* PTM assigns no orientation to disordered atoms, so grain boundary and surface atoms are
  drawn in the `--other-color` grey (or hidden with `--hide-other`). In the example above
  that is about a third of the atoms, which is normal for a nanocrystal.
* Icosahedral environments have no crystallographic orientation and are never coloured.
* The pole figure density is a Gaussian kernel on the equal-area plane, which is fine for
  inspection but is not a spherical harmonic ODF; use MTEX for quantitative texture
  analysis.
* PTM's hexagonal templates assume a near-ideal axial ratio. `--c-over-a` affects pole
  positions, not the identification itself.

## Citing

If this tool is useful in published work, please cite PTM (Larsen, Schmidt & Schiøtz,
*Modelling Simul. Mater. Sci. Eng.* **24** (2016) 055007), OVITO (Stukowski,
*Modelling Simul. Mater. Sci. Eng.* **18** (2010) 015012) and the colour key
(Nolze & Hielscher, *J. Appl. Cryst.* **49** (2016) 1786). See `CITATION.cff`.

## Licence and acknowledgements

GPL-3.0-or-later. The inverse pole figure colour key follows the scheme described by
Nolze & Hielscher and implemented in [MTEX](https://mtex-toolbox.github.io) and
[orix](https://orix.readthedocs.io); it is reimplemented here in NumPy so that millions of
atoms can be coloured quickly, and the GPL is inherited from that lineage. `orix` is used
in the test suite as the reference implementation.

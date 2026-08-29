# ptm-ipf showcase: deformation of Cu, Fe and Ti polycrystals

Molecular dynamics runs made to show what ptm-ipf does, on materials the reader can check
against the literature. Every starting structure was built from explicit rotation
matrices recorded beside it, so the orientations the analysis reports can be compared with
the ones that went in; ptm-ipf recovers every built grain to within 0.001 degrees.

## Simulations

Twenty-grain Voronoi polycrystals in 200 A periodic boxes, 410k to 644k atoms, minimised,
equilibrated 20 ps at 300 K and zero pressure, then strained along x at a constant
engineering rate with the transverse directions at zero pressure. Frames every 5000 steps
(10 ps).

| element | potential | structure | textures | loading | rate | strain |
|---|---|---|---|---|---|---|
| Cu | Mishin et al. 2001, eam/alloy | fcc | random, rolled (brass + copper), extruded (<111> + <100>) | tension and compression | 0.001/ps | 15% |
| Fe | Mendelev et al. 2003, eam/fs | bcc | random, rolled (alpha fibre) | tension and compression | 0.001/ps | 15% |
| Ti | Mendelev et al. 2016, eam/fs | hcp | random, rolled (basal), extruded (<10-10> fibre) | tension and compression | 0.0005/ps | 12% |

The rolled and extruded textures are the ideal components with an 8 to 15 degree spread,
built by `scripts/build_poly.py`; the JSON beside each `.data` file lists every grain's
rotation matrix. Runs are named `<element>_<texture>_<T|C>`.

LAMMPS 29 Aug 2024 on the lab workstations, CPU only: 32 cores give 12 million atom-steps
per second, about 1.5 hours per run.

## Figures per run (`figures/<run>/`)

| file | what | ptm-ipf feature |
|---|---|---|
| `stress_strain.png` | stress-strain curve, frame strains marked | |
| `final_ipf_{x,y,z}.png` | 3D renders of the final frame | IPF colouring, coordinate triad |
| `key_{x,y,z}.png` | the colour keys | |
| `pole_figures.png` | equal-area pole figures in the sample frame | pole figures |
| `ipf_density.png` | orientation density in the fundamental sector | IPF density |
| `panels/` | 10 A sections, flat and rendered, for the 3 by 3 grids | flat maps, sections, wireframes |
| `boundaries.png` | z section, boundaries coloured by misorientation | boundary colouring |
| `hagb_lagb.png` | z section, high angle black, low angle grey | boundary threshold |
| `evolution.mp4` | flat map of the z section through every frame | animation |
| `evolution_render.gif` | rendered z section through every frame | animation |
| `stats.json` | structure counts, grains per section, timing | |

## What the first runs show

**cu_random_T** (632k atoms, 15% tension): yield at 2.2 GPa near 6 percent strain. By the
final frame 12 percent of the atoms are hcp, which in fcc copper means stacking faults and
deformation twins. The boundary map colours the twin boundaries yellow at 60 degrees, the
Sigma 3 relation, as parallel lamellae through five of the grains, against the purple
low angle boundaries the grains started with. The animation shows them appear one at a
time from 4 percent strain on.

**ti_random_T** (413k atoms, 12% tension): the section shows no twin lamellae; the
deformation is in small low angle cells inside the grains, the purple loops in the boundary
map, consistent with prismatic slip carrying most of the strain in random-texture titanium
at this grain size. The 321 grains the segmentation counts in the z section are those
cells, not twins. The rolled and extruded Ti runs, loaded along and across the c axes, are
the ones expected to twin.

**cu_rolled_C** (638k atoms, 15% compression): the built texture survives deformation. The
{111} and {100} pole figures show the brass and copper components as sharp spots of 40 to
50 MRD, against the scattered 15 to 18 MRD of the random run, and the IPF-X map is two
colour families, one per component, meeting along a single boundary. Twinning is lighter
than in random tension (8 percent hcp against 12), and shows as thin lamellae inside the
blue grain and one wide deformation band through the pink one; the sections count 266 and
274 grains where the random run counts 150 to 321. `compare_cu_texture_ipf{x,z}.png` puts
the two side by side, same sections, same projection, same colour key.

**ti_rolled_C** (410k atoms, 12% compression along RD, basal texture): the built texture
is a single 180 MRD {0001} spot at ND, and it does not twin. Compression perpendicular to
the c axis cannot drive {10-12} extension twinning, which needs the c axis to lengthen, so
the strain goes to prismatic slip and the section stays a handful of large grains in one
colour family, with only a few small twin nuclei (the pink islands at 50 degrees in the
boundary map). The grain counts fall to 30 to 88 per section from the 150 to 321 cells of
the random run: a single crystallographic texture, loaded the wrong way for twinning, is
the least fragmented microstructure of the set. The extruded Ti run, pulled along a
<10-10> fibre, is the one loaded to twin.

**cu_rolled_T** (638k atoms, 15% tension): the same starting structure as cu_rolled_C,
pulled instead of pushed. It twins far more: 12.4 percent hcp against 8.3, and 514 grains
in the z section against 274, most of them twin lamellae. That is the tension-compression
asymmetry of a textured fcc metal, which comes from the Schmid factors of the partial
dislocations reversing sign with the load; the same grains, the same boundaries, and a
different microstructure at the end. `compare_cu_rolled_loading_ipf{x,z}.png` shows the
pair, and `compare_cu_all_ipfx.png` puts all three Cu runs in one figure.

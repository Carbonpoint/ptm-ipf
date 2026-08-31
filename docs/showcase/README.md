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

The rolled and extruded textures are the ideal components with an 8 to 15 degree spread;
the JSON beside each `.data` file lists every grain's rotation matrix. The campaign ran on
`build_poly.py`, which placed rotated lattice blocks by hand and trimmed overlaps, leaving
the as-built cells at 82 to 93 percent of ideal density; that is the origin of the
unindexable zone in the random iron cell. `build_poly2.py` replaces it: atomsk builds the
Voronoi geometry at full density (99.9 to 100.0 percent, verified), and the same explicit
rotation matrices are converted to atomsk's node angles, which are not Euler angles but
extrinsic rotations about x, y and z composing as Rz Ry Rx, established by building single
grains at known angles and measuring them with PTM (exact to 0.01 degrees). PTM recovers
every grain of a v2 build to 0.000 degrees, and a rolled Ti test cell reproduces its basal
fibre. Use v2 for any future run. Runs are named `<element>_<texture>_<T|C>`.

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

## The ten runs at a glance

| run | atoms | texture | load | peak stress | at strain | final stress | atoms PTM labels close-packed | grains per section (x, y, z) |
|---|---|---|---|---|---|---|---|---|
| cu_random_T | 632,412 | random | tension | 2.23 GPa | 7.0% | 1.94 GPa | hcp 12.0% | 279, 355, 525 |
| cu_rolled_T | 637,665 | rolled | tension | 2.31 GPa | 4.6% | 1.85 GPa | hcp 12.4% | 209, 260, 514 |
| cu_rolled_C | 637,665 | rolled | compression | 2.64 GPa | 5.2% | 2.29 GPa | hcp 8.3% | 266, 124, 274 |
| cu_extruded_T | 631,371 | extruded | tension | 2.52 GPa | 13.9% | 2.40 GPa | hcp 11.2% | 166, 232, 261 |
| cu_extruded_C | 631,371 | extruded | compression | 2.52 GPa | 5.9% | 2.16 GPa | hcp 13.7% | 690, 426, 529 |
| fe_random_T | 610,364 | random | tension | 4.99 GPa | 7.2% | 4.15 GPa | hcp 3.8% | 279, 434, 422 |
| fe_rolled_C | 643,992 | rolled | compression | 5.80 GPa | 10.7% | 5.34 GPa | hcp 3.2% | 303, 157, 366 |
| ti_random_T | 413,296 | random | tension | 2.30 GPa | 11.1% | 2.27 GPa | fcc 0.6% | 150, 200, 321 |
| ti_rolled_C | 410,297 | rolled | compression | 2.16 GPa | 8.1% | 1.97 GPa | fcc 1.0% | 60, 30, 88 |
| ti_extruded_T | 410,253 | extruded | tension | 2.22 GPa | 11.3% | 2.15 GPa | fcc 0.7% | 136, 91, 220 |

Peak and final stress are the absolute value of the tensile stress along x. "Atoms PTM
labels close-packed" is the fraction of atoms PTM assigns to hcp or fcc at the final
frame: in copper that is stacking faults and twins, a real quantity; in iron at the 0.15
cutoff it is thermal misidentification of distorted bcc, not a phase, and in titanium the
sub-percent fcc is the same. Grains per section are what the flat map segmentation counts
in the three 10 A mid-cell sections, and include the cells that slip produces inside the
built grains.

## What the campaign shows

Three crystal structures, three deformation modes, and the tool separates them from the
orientation field alone. Copper twins in every texture, and the sign of its
tension-compression asymmetry flips between the rolled cell (three times more twinning in
tension) and the extruded cell (more in compression), both as the Schmid factors of the
partial dislocations predict. Titanium on this potential twins in none of its three
textures at this grain size and rate; it fragments by slip, and the texture sets how much,
from 30 cells per section for the basal cell under compression to 321 for random. Iron
deforms by slip in both textures and is the strongest of the three by a factor of two.

The unit cell wireframes carry the texture story without the colour key: cubes in every
attitude for random copper, in aligned pairs for the rolled components, standing on a body
diagonal for the <111> fibre; hexagonal prisms face-on for basal titanium and edge-on with
a common axis for the <10-10> fibre. The boundary colouring carries the deformation story:
yellow 60 degree Sigma 3 lamellae through the copper grains, purple low angle cells inside
the titanium and iron grains, and nothing in between.

Every run took about 1.5 hours on 32 cores for the molecular dynamics and 6 to 7 minutes on
ghost for the full figure set, with the flat map animations at one frame per 5 to 10
seconds.

## What the runs show, one by one

**cu_random_T** (632k atoms, 15% tension): yield at 2.2 GPa near 7 percent strain. By the
final frame 12 percent of the atoms are hcp, which in fcc copper means stacking faults and
deformation twins. The boundary map colours the twin boundaries yellow at 60 degrees, the
Sigma 3 relation, as parallel lamellae through most of the grains, against the purple low
angle boundaries the grains started with; the sections count 279 to 525 grains from the
20 built. The animation shows the twins appear one at a time from 4 percent strain on.

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

**cu_extruded_T and cu_extruded_C** (631k atoms, 15% tension and compression along the
<111> + <100> double fibre): the asymmetry runs the other way from the rolled cell. In
tension 11.2 percent of the atoms end up hcp and the sections count 166 to 261 grains; in
compression it is 13.6 percent and 426 to 690 grains, the most fragmented microstructure of
the campaign. For a <111> fibre loaded along its axis, the leading partial has the higher
Schmid factor in compression, so the twins form more readily when the fibre is pushed; for
the rolled cell the same argument favours tension. Same tool, same sections, opposite
answers, both the ones crystal plasticity predicts. `compare_cu_extruded_loading_ipf{x,z}.png`
is the pair, and `compare_cu_tension_textures_ipfx.png` puts the three textures in tension
side by side.

**ti_rolled_C** (410k atoms, 12% compression along RD, basal texture): the built texture
is a single 180 MRD {0001} spot at ND, and it does not twin. Compression perpendicular to
the c axis cannot drive {10-12} extension twinning, which needs the c axis to lengthen, so
the strain goes to prismatic slip and the section stays a handful of large grains in one
colour family, with only a few small twin nuclei (the pink islands at 50 degrees in the
boundary map). The grain counts fall to 30 to 88 per section from the 150 to 321 cells of
the random run: a single crystallographic texture, loaded the wrong way for twinning, is
the least fragmented microstructure of the set. The animation makes the
distinction a still map cannot: the twin nuclei appear early and never grow, so this is a
cell that did not twin, rather than one that twinned and detwinned. The extruded Ti run,
pulled along a <10-10> fibre, is the one loaded to twin.

**ti_extruded_T** (410k atoms, 12% tension along the <10-10> fibre): built so that every
c axis is perpendicular to the tensile axis, which is the loading that {10-12} extension
twinning needs, and it still does not twin in the section: the boundary map is a single
green colour family with ordinary grain boundaries and low angle cells, no lamellae, and
the 220 grains counted in the z section are cells. Two things are consistent with that. The
Mendelev 2016 Ti potential has a high twin nucleation stress, and at 20 grains in a 200 A
box with a 0.0005/ps rate there is little room for a twin to nucleate before slip relieves
the stress. So across the three Ti runs the microstructure fragments by slip in every
case, and the difference the texture makes is how much: 30 to 88 cells per section for the
basal cell under compression, 91 to 220 for the <10-10> fibre under tension, 150 to 321 for
random. `compare_ti_all_ipf{x,z}.png` puts the three side by side.

**fe_random_T and fe_rolled_C** (610k and 644k atoms, 15% tension and compression): bcc
iron on the Mendelev 2003 potential deforms by slip in both textures. Neither section
shows a twin lamella; the boundary maps are ordinary 40 to 50 degree grain boundaries with
low angle cells inside the grains, and the grain counts (279 to 434 per section for random,
157 to 366 for the rolled alpha fibre) are those cells. Iron is the strongest material of
the set, peaking near 5 GPa in tension and 5.6 in compression, against 2.2 to 2.6 for
copper and 2.2 to 2.3 for titanium. The rolled cell's {110} pole figure shows the alpha
fibre as a single 24 MRD spot at RD, and its IPF-X map is one green colour family: every
grain has <110> within a few degrees of the compression axis, as built.

Two things this pair taught about the pipeline rather than the metal. The PTM RMSD cutoff
of 0.1 that suits fcc copper is too tight for bcc iron at 300 K: its larger thermal
displacements push 3 percent of the grain interiors over it, and at 0.1 they go unindexed.
Both iron figure sets use 0.15. And the large white region in the random cell's z section
is not a phase, a twin, or a cutoff effect: it is there in the first frame, before any
strain, and no fill radius closes it. It is a poorly crystallised zone left by the builder,
whose overlap trimming at the Voronoi boundaries thins the as-built cell to 82 percent of
the ideal bcc density, and 20 ps at 300 K is far too short for iron to recrystallise 30 A
of disorder. The flat map reports it correctly as unindexable, which is the right answer
from the tool and a lesson for the next build: relax the boundaries longer, or trim less.

`compare_fe_texture_ipf{x,z}.png` puts the two side by side.

**cu_rolled_T** (638k atoms, 15% tension): the same starting structure as cu_rolled_C,
pulled instead of pushed. It twins far more: 12.4 percent hcp against 8.3, and 514 grains
in the z section against 274, most of them twin lamellae. That is the tension-compression
asymmetry of a textured fcc metal, which comes from the Schmid factors of the partial
dislocations reversing sign with the load; the same grains, the same boundaries, and a
different microstructure at the end. `compare_cu_rolled_loading_ipf{x,z}.png` shows the
pair, and `compare_cu_all_ipfx.png` puts all three Cu runs in one figure.

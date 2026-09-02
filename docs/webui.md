# Web interface

`ptm-ipf` ships a local web interface that wraps the whole toolchain (PTM
structure identification, IPF colouring, the colour key, pole figures, the IPF
density plot, atom selections and file export) behind an interactive page:

```bash
ptmipf-ui mg.dump                    # analyse a file and open the browser
python -m ptmipf.webui mg.dump       # the same, without the console script
python -m ptmipf.webui --root ~/simulations --port 8465
ptmipf-ui --check                    # can this installation draw the 3D view?
```

On Windows, call the executable in the environment rather than activating it:

```powershell
.venv\Scripts\ptmipf-ui.exe --check
.venv\Scripts\ptmipf-ui.exe mg.dump
```

*(Packaging note: the console script is a one-line entry point,
`ptmipf-ui = "ptmipf.webui:main"` under `[project.scripts]`.)*

![The web interface](webui_light.png)

The interface follows the browser's light or dark theme (a toggle is in the
header):

![Dark theme](webui_dark.png)

## What it does

* **Load a configuration** with the path box or the built-in file browser,
  pick a trajectory frame, and press *Analyse*. Atom count, cell dimensions
  and the per-structure PTM counts appear once the analysis finishes; long
  runs show a live status instead of a frozen page. Beside *Analyse* is
  *Analyse slice*, which runs PTM on the slab the 3D view is showing (plus a
  margin so its faces are matched correctly) and keeps only that slab, so
  every following figure and recolouring works on fewer atoms. When the whole
  cell has already been matched the slice is cut from that cached result and
  comes back at once, and *Analyse* restores the whole cell the same way.
* **File series**: a file whose name ends in a number (`dump_0.cfg`,
  `dump_100.cfg`, `dump_120.cfg`) is recognised as one frame of a series,
  sorted numerically, and the previous/next arrows and the menu under the
  path box step through it, keeping the current settings (and an analysed
  slice) from frame to frame. A multi-frame file steps through its frames
  the same way. A selection belongs to one frame and is not carried over.
* **Busy indicators**: anything that takes longer than a moment shows a
  spinner on the card it will replace, and the header lists the jobs in
  flight with the progress bar that the PTM run uses.
* **Every CLI option is a control**: the structures PTM identifies, the RMSD
  cutoff, colour-only, the "other" colour, the RD/TD/ND/ED sample frame
  definitions, and the IPF projection direction (axis buttons or any vector).
  Wherever a direction is asked for (the projection direction, the sample
  axes, the slice normal, the rotation axis, the pole figure axes and the
  triad) an *xyz* button opens three component boxes for a specific vector.
  Changing only the projection direction or the colouring re-uses the cached
  PTM result and updates in a fraction of a second; PTM re-runs only when the
  file, the structure set, the cutoff or the frame change.
* **3D view**: the IPF-coloured atoms, rendered server-side by OVITO from the
  cached result. Drag to orbit, scroll to zoom, cut the cell open with the
  slice control, and hide the unidentified grain-boundary atoms. Click an
  atom to see its position, structure, RMSD and orientation, and to use it
  as a misorientation reference. The slice slider has a number box beside
  it (in angstroms along the normal) and applies, when *figures too* is
  ticked, to the pole figures, the IPF density and the IPF map as well as to
  the view, so all four show the same atoms.
* **Triad**: the triad button labels the sample axes; its *advanced* gear
  chooses between the sample axes (RD, TD, ND), the cell axes (x, y, z) and
  three custom directions with their own labels, and has sliders for the
  triad's size and position in the viewport. Long custom labels can run past
  the edge of the tripod box (an OVITO limit); a larger triad gives them
  room.
* **System rotation**: turn the whole system (atoms, orientations and cell
  together) by any angle about any axis, in order, with the sample frame
  staying put. Rotations only recolour the cached result, so they are
  immediate, and the *Sample frame* panel assigns specific directions to RD,
  TD and ND for the projection.
* **Boundary filling**: the same interpolation as `--fill-boundaries`, with a
  radius and a minimum neighbour count. It applies to the 3D view and to every
  figure, and is cached, so turning it on costs one pass and nothing after that.
* **IPF map**: a section rasterised into an EBSD-style map of colours and
  grain boundaries, with controls for the view axis, the slab thickness, the
  pixel size and the boundary angle, and a raw mode that paints the unindexed
  atoms black. The grain count is reported under the map. With the slice
  applied to the figures, the map is a section of that slab seen along its
  normal.
* **Sections**: the slice control cuts the cell open; giving it a thickness keeps a slab
  of that many angstroms instead, and the X/Y/Z buttons look straight down an axis, which
  turns the slab into an EBSD-style orientation map of a section.
* **Figures**: the IPF colour key, pole figures for any pole families and
  the IPF orientation density, regenerated whenever the settings change. The
  pole list is editable and has a menu of the common families for the
  dominant structure's Laue group; it initialises with that structure's
  first three families, as the colour key does, so an fcc system starts with
  `100,110,111` and an hcp one with `0001,10-10,11-20`. The *up* and *right*
  boxes set the projection axes of the pole figures, and there is a c/a
  input for non-ideal lattices and a density or scatter mode. Every figure
  has PNG and SVG download buttons, the rendered view a PNG one, and the
  coloured configuration exports to `.xyz` or `.dump`.
* **Selections**: build up a selection from structure, particle type, RMSD
  range, spatial slab, orientation within a tolerance of a sample direction
  ("the basal-oriented grains"), or misorientation from a reference
  orientation or a clicked atom. Criteria combine with *and*/*or* and each
  can be inverted; the selected atom count updates live. The selection can be
  highlighted (or shown alone) in the 3D view, the pole figures and IPF
  density can be restricted to it, and it exports to its own `.xyz`/`.dump`.
* **Colour scales and smoothing**: both density plots carry a colour map menu
  (`viridis`, `magma`, `jet`, `rainbow`, `turbo` and the rest) with an *upload
  your own* entry that takes an image strip or a text table of RGB triples, so a
  colour bar screenshotted from a paper can be used directly. An uploaded scale
  is held in memory for the session and never written to disk. Beside it is a
  smoothing box in degrees, off by default: a simulated cell has few, nearly
  perfect grains, so its poles are far sharper than any measured texture and the
  peak MRD comes out correspondingly high. A few degrees puts it on a
  comparable scale, and the figure is annotated with the width used.
* **Export**: the coloured configuration goes out as `.xyz` or `.dump`, with
  `Color.R/G/B` for the direction on screen so the file opens already coloured,
  plus one scalar column per direction listed in the *Colour-coding columns*
  box. Those columns drive OVITO's own Color coding modifier: load the
  *colour map .png* from the same panel as a custom colour map, set the range
  to 0 and 1, and switching the input property between `ipf_x`, `ipf_y` and
  `ipf_z` switches between the three IPF maps inside OVITO, with no re-export.
  The selection exports the same way.
* **Progress while PTM runs**: the status line carries an approximate
  percentage and a bar. OVITO reports nothing while it is working, so the
  percentage is interpolated inside each of the three stages (reading the file,
  matching, colouring) from a throughput the server calibrates on the runs it
  has already done. It is an estimate, and the wording says so; what it is good
  for is telling a long run apart from a stuck one.
* **Orientations already in the file**: for a configuration another OVITO
  session has run PTM on, name the columns that hold the quaternions and they
  are coloured directly, with no second PTM run. The panel reads the file's
  actual columns and offers a guess, and asks for the two things no file states:
  the component order (`x, y, z, w` as OVITO writes it, or `w, x, y, z` as most
  other tools do) and whether the rotation is crystal to sample or the other way
  round. Those two differ by a transpose, which turns an IPF map into a
  plausible looking but wrong one, so they are asked rather than guessed. A
  structure type column is read with OVITO's own PTM codes; without one, every
  atom carrying an orientation is taken as a single named phase.
* **Reproducibility**: the *CLI command* button prints the `ptmipf` command
  line, selection flags included, that reproduces the current session, so
  interactive exploration turns directly into a scriptable analysis. It can be
  copied wrapped over several lines for a script, copied as a single line for a
  shell that does not honour backslash continuations (PowerShell and `cmd.exe`
  break at the first line end), or saved to a file. The same dialog reads a
  command back: load or paste one under *Resume from a saved command* and the
  whole form is set from it, which is how a previous session is picked up.
  Parsing goes through the real `ptmipf` argument parser, so the two can never
  drift apart. Rotations, an analysed slice and a custom triad appear as
  `--rotate`, `--ptm-slice` and `--tripod-axes`.

## Rendering a series

The *Render series* card at the bottom of the page turns a file series (or
the frames of a multi-frame file) into a set of images or a movie, with the
settings of the page as they stand: structures, sample frame, projection
direction, rotations, an analysed slice, boundary filling, the camera of the
3D view and the options of each figure.

Choose the range (`from`, `to`, `every`), the seconds per frame for movies,
and tick the outputs wanted: the 3D view, the IPF map, the pole figures, the
IPF density and the colour key, each as PNG or SVG stills or as a GIF or MP4
movie (MP4 needs `imageio[ffmpeg]`, as `--animate` does). Several
outputs can be produced in one run. Stills are written one per frame as
`<frame>_<kind>.<ext>`, movies as `<series>_<kind>.<ext>`, into the output
folder (`<series>_series` beside the files by default). *Label frames* stamps
each frame with its name. The card shows the progress, an estimate of the
time remaining, and the finished files, which can be opened one by one or
downloaded together as a zip. Rendering runs on the server one frame at a
time, so a long series can be left to run while the page is used for other
things; *Cancel* stops it after the current frame. A selection is not carried
across frames, so the batch renders whole frames.

## Design notes

The server is standard library only (`http.server`), so the web interface
adds no dependencies beyond what `ptm-ipf` already needs; the front end is a
single static page with no JavaScript frameworks. It binds to `127.0.0.1` and
serves files only from below the `--root` directory (by default the input
file's directory). It is a single-user local tool, not something to expose on
a network.

Rendering is done by OVITO on the server from the cached analysis result, so
orbiting the view never re-runs PTM, and what the browser shows is what
`ptmipf --render` writes. All plots are produced by the same functions as the
CLI, which keeps the two front ends pixel-identical.

## The examples page

The *Examples* button in the header opens a page that builds something to look
at. Pick copper, aluminium, nickel or iron and it will:

1. build a periodic cube of randomly oriented grains with
   [atomsk](https://atomsk.univ-lille.fr),
2. download the matching EAM potential from the
   [NIST Interatomic Potentials Repository](https://www.ctcms.nist.gov/potentials/),
   checking it against a recorded checksum,
3. write a LAMMPS input that compresses the cube along z, plus the grain
   orientations it built and a README,

all into `<served root>/examples/<element>_<n>grain_compression/`. Nothing is
run for you. The page quotes the cost before you ask for it and again
afterwards: the defaults are about 8,600 atoms and 22,000 steps, which took
1 minute 34 seconds on four cores of a workstation and will take a few minutes
on a laptop.

Then `lmp -in in.compression`, and open `compression.dump` back in the
interface. On the reference copper run the hcp fraction goes from 3.7 per cent
at zero strain to 15.4 per cent at ten per cent compression: stacking faults and
twin boundaries appearing between the grains, which is exactly what an IPF map
of a deformed fcc metal is for.

The same thing from the terminal:

```bash
ptmipf-example Cu --root . --grains 6 --box 48
ptmipf-example Fe --strain 0.15 --builder auto
```

atomsk is not a Python package and will not arrive with `pip install`. Install
it from <https://atomsk.univ-lille.fr/dl.php>, which has binaries for Linux,
macOS and Windows, and put it on the PATH or point `PTMIPF_ATOMSK` at it. Without
it the page falls back to a built-in NumPy Voronoi builder and says so, on the
page and in the README it writes. The fallback reaches about 97 per cent of
single-crystal density against atomsk's 97.4 on the same cell, which is close,
but "close" is not something to quote in a paper without saying which one ran.

## When the 3D view stays empty

The 3D view is drawn by OVITO in the server process and reaches the browser as
a PNG, so a blank viewer is nearly always a server-side renderer problem rather
than a browser one. The page says so in the empty frame instead of showing a
broken image, and `ptmipf-ui --check` reports the same probe on the terminal:
which of OVITO, matplotlib, Pillow and the selection module imported, and
whether a test scene rendered and with which renderer. The plots, the flat
orientation map and the exports need no renderer and keep working without one.

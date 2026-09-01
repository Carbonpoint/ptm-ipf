"""Colour-coding keys and colour bars, so OVITO can paint the IPF map itself.

An exported file already carries ``Color.R/G/B`` for the direction that was
analysed, which is why it opens already coloured.  That is also its limit: the
direction is fixed at export time, and OVITO's own Color coding modifier has
nothing to work with, because it maps *one scalar per atom* through a colour
bar rather than reading a colour triple.

This module writes the other half of the file.  Each requested direction gets
one scalar column, and the distinct colours of the run are written out as a
colour bar image.  OVITO's image colour bar samples the nearest pixel instead
of interpolating between them (pinned down in ``tests/test_colormap.py``), so
a key of ``(index + 0.5) / n_entries`` comes back as exactly the palette
colour.  Loading that image into Color coding and switching the input property
then swaps between IPF-X, IPF-Y and IPF-Z inside OVITO, with no re-export and
no loss of colour.

A built-in bar such as Jet or Rainbow can be used instead, through
:func:`gradient_keys`, but only as an approximation: a built-in bar is a curve
through colour space while the IPF colours cover a two-dimensional patch of
it, so many IPF colours simply do not lie on the curve.  That function reports
the error it incurs rather than hiding it.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

__all__ = [
    "BUILTIN_GRADIENTS",
    "PLOT_COLORMAPS",
    "MAX_PALETTE_ENTRIES",
    "color_keys",
    "direction_column",
    "gradient_keys",
    "keys_from_indices",
    "load_colormap",
    "read_colormap_table",
    "quantise_colors",
    "sample_gradient",
    "write_color_map",
]

#: OVITO colour bars usable with :func:`gradient_keys`, lower case.
BUILTIN_GRADIENTS = (
    "jet", "rainbow", "cyclicrainbow", "viridis", "magma", "hot",
    "grayscale", "bluewhitered",
)

#: Palette ceiling.  The keys are written with eight decimals, so even a
#: palette this large addresses its entries with a wide margin.
MAX_PALETTE_ENTRIES = 65536


def direction_column(label: str, taken=()) -> str:
    """A column name for the IPF key along the direction named *label*.

    Kept to letters, digits and underscores because both readers that matter
    (OVITO's extended XYZ and LAMMPS dump parsers) treat the column name as an
    identifier.  A leading minus becomes ``m`` so that ``-Z`` does not collide
    with ``+Z``.
    """
    text = label.strip().lower().replace("-", "m").replace("+", "").replace(".", "p")
    text = re.sub(r"[^0-9a-z]+", "_", text).strip("_") or "dir"
    name = f"ipf_{text}"
    if name not in taken:
        return name
    for suffix in range(2, 100):
        if f"{name}_{suffix}" not in taken:
            return f"{name}_{suffix}"
    raise ValueError(f"cannot find a free column name for {label!r}")


def quantise_colors(color_sets, max_entries: int = MAX_PALETTE_ENTRIES):
    """Reduce several colour arrays to one shared 8-bit palette.

    Parameters
    ----------
    color_sets
        Sequence of ``(n, 3)`` float arrays in [0, 1], one per direction.  They
        share a palette so that a single colour bar serves every direction.
    max_entries
        Ceiling on the palette size.  IPF colours vary continuously, so a large
        run can hold more distinct 8-bit colours than that; the channels are
        then snapped to a coarser lattice until the palette fits.

    Returns
    -------
    tuple
        ``(palette, indices, error)``: the ``(m, 3)`` uint8 palette, a list of
        integer index arrays matching *color_sets*, and the largest colour
        error introduced, in RGB units of 1.
    """
    arrays = [np.asarray(c, dtype=float) for c in color_sets]
    if not arrays:
        raise ValueError("at least one colour array is required")
    lengths = [len(a) for a in arrays]
    stacked = np.concatenate(arrays, axis=0)
    exact = np.clip(np.rint(stacked * 255.0), 0, 255).astype(np.int32)

    step = 1
    while True:
        snapped = exact if step == 1 else np.clip(np.rint(exact / step) * step, 0, 255).astype(
            np.int32
        )
        packed = (snapped[:, 0] << 16) | (snapped[:, 1] << 8) | snapped[:, 2]
        unique, inverse = np.unique(packed, return_inverse=True)
        if len(unique) <= max_entries or step >= 128:
            break
        step *= 2

    palette = np.stack(
        [(unique >> 16) & 0xFF, (unique >> 8) & 0xFF, unique & 0xFF], axis=1
    ).astype(np.uint8)
    order = _readable_order(palette)
    palette = palette[order]
    # ``order`` lists old indices in new order; invert it to remap the atoms.
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    inverse = rank[np.asarray(inverse).reshape(-1)]

    error = float(np.abs(palette[inverse].astype(float) / 255.0 - stacked).max()) if len(
        stacked
    ) else 0.0

    indices, start = [], 0
    for length in lengths:
        indices.append(inverse[start : start + length])
        start += length
    return palette, indices, error


def _readable_order(palette: np.ndarray) -> np.ndarray:
    """Sort a palette so the colour bar reads as a sweep rather than as noise.

    The order is irrelevant to the colours the atoms end up with, but a bar
    that runs through the hues is far easier to recognise in OVITO's panel as
    the IPF palette and not a leftover from another modifier.
    """
    from matplotlib.colors import rgb_to_hsv

    hsv = rgb_to_hsv(palette.astype(float) / 255.0)
    return np.lexsort((hsv[:, 1], hsv[:, 2], np.rint(hsv[:, 0] * 60.0)))


def keys_from_indices(indices: np.ndarray, n_entries: int) -> np.ndarray:
    """Colour-coding values addressing the centre of each palette entry."""
    return (np.asarray(indices, dtype=float) + 0.5) / float(n_entries)


def write_color_map(palette: np.ndarray, path, height: int = 1) -> str:
    """Write *palette* as a PNG for OVITO's "Load custom color map".

    One pixel per entry, left to right.  *height* only affects how the file
    looks when opened in an image viewer; OVITO reads the top row.
    """
    from PIL import Image

    array = np.asarray(palette, dtype=np.uint8).reshape(1, -1, 3)
    if height > 1:
        array = np.repeat(array, height, axis=0)
    Image.fromarray(array, mode="RGB").save(str(path))
    return str(path)


def sample_gradient(name: str, samples: int = 2048) -> np.ndarray:
    """Sample one of OVITO's built-in colour bars into an ``(samples, 3)`` array."""
    from ovito.modifiers import ColorCodingModifier

    lookup = {n.lower(): n for n in dir(ColorCodingModifier) if not n.startswith("_")}
    key = name.strip().lower().replace("_", "").replace("-", "")
    if key not in lookup:
        raise ValueError(
            f"unknown colour bar {name!r}; OVITO offers {', '.join(BUILTIN_GRADIENTS)}"
        )
    gradient = getattr(ColorCodingModifier, lookup[key])()
    values = (np.arange(samples) + 0.5) / samples
    return np.array([gradient.value_to_color(float(v)) for v in values], dtype=float)


def gradient_keys(color_sets, gradient: str = "jet", samples: int = 2048):
    """Project colours onto a built-in OVITO colour bar.

    Each colour is replaced by the position along *gradient* whose colour is
    nearest to it.  This is what lets a stock Jet or Rainbow bar stand in for
    the IPF palette, and it is inherently lossy: the bar is a one-dimensional
    curve and the IPF colours are not on it.

    Returns ``(keys, max_error, mean_error)`` with the errors in RGB units of
    1, so the caller can say how far the picture has moved.
    """
    ramp = sample_gradient(gradient, samples)
    keys, worst, total, count = [], 0.0, 0.0, 0
    for colors in color_sets:
        colors = np.asarray(colors, dtype=float)
        # Chunked so a multi-million atom run does not build an (n, samples)
        # distance matrix all at once.
        index = np.empty(len(colors), dtype=np.int64)
        errors = np.empty(len(colors), dtype=float)
        for start in range(0, len(colors), 65536):
            block = colors[start : start + 65536]
            distance = np.linalg.norm(block[:, None, :] - ramp[None, :, :], axis=2)
            nearest = distance.argmin(axis=1)
            index[start : start + len(block)] = nearest
            errors[start : start + len(block)] = distance[np.arange(len(block)), nearest]
        keys.append((index + 0.5) / samples)
        if len(colors):
            worst = max(worst, float(errors.max()))
            total += float(errors.sum())
            count += len(colors)
    return keys, worst, (total / count if count else 0.0)


def color_keys(
    result,
    directions=("x", "y", "z"),
    max_entries: int = MAX_PALETTE_ENTRIES,
    gradient: str | None = None,
):
    """Colour-coding keys for *result* along each of *directions*.

    Recolouring reuses the cached PTM orientations, so asking for three
    directions costs three colour-key evaluations, not three PTM runs.

    Returns
    -------
    tuple
        ``(keys, palette, info)``.  *keys* maps column name to a float array in
        [0, 1]; *palette* is the ``(m, 3)`` uint8 colour bar, or ``None`` when
        a built-in *gradient* was used instead; *info* records the direction
        labels, the palette size or gradient name, and the colour error.
    """
    labels, colors = [], []
    for spec in directions:
        recoloured = result.recolor(spec)
        labels.append(recoloured.direction_label)
        colors.append(recoloured.colors)

    names, taken = [], set()
    for label in labels:
        name = direction_column(label, taken)
        taken.add(name)
        names.append(name)

    if gradient:
        values, worst, mean = gradient_keys(colors, gradient)
        palette = None
        info = {
            "directions": dict(zip(names, labels)),
            "gradient": gradient,
            "entries": 0,
            "max_error": worst,
            "mean_error": mean,
        }
    else:
        palette, indices, worst = quantise_colors(colors, max_entries)
        values = [keys_from_indices(i, len(palette)) for i in indices]
        info = {
            "directions": dict(zip(names, labels)),
            "gradient": None,
            "entries": int(len(palette)),
            "max_error": worst,
            "mean_error": worst,
        }
    return dict(zip(names, values)), palette, info


# ----------------------------------------------------------------------
# colour maps for the plots
# ----------------------------------------------------------------------
#: Colour maps offered for the density plots.  Sequential ones first, because
#: a density is a sequential quantity and jet reads a smooth gradient as a set
#: of false bands; jet and rainbow are here because a great deal of the
#: published texture literature uses them and a figure has to be comparable
#: with the one it sits next to.
PLOT_COLORMAPS = (
    "viridis", "magma", "inferno", "plasma", "cividis",
    "jet", "rainbow", "turbo", "coolwarm", "hot", "Greys",
)


def read_colormap_table(data) -> np.ndarray:
    """Read a colour map given as text: one RGB triple per line.

    Accepts values in 0 to 1 or 0 to 255, separated by whitespace, commas or
    semicolons, with ``#`` comments and an optional leading index column.
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="replace")
    rows = []
    for line in str(data).splitlines():
        line = line.split("#")[0].replace(",", " ").replace(";", " ").strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) == 4:
            fields = fields[1:]  # an index column, which some tools write
        if len(fields) != 3:
            raise ValueError(
                f"a colour map table needs three numbers a line, got {len(fields)}"
            )
        try:
            rows.append([float(f) for f in fields])
        except ValueError:
            raise ValueError(f"cannot read {line!r} as a colour") from None
    if len(rows) < 2:
        raise ValueError("a colour map needs at least two colours")
    table = np.asarray(rows, dtype=float)
    if table.max() > 1.0:
        table = table / 255.0
    if table.min() < 0.0 or table.max() > 1.0:
        raise ValueError("colour components must lie in 0 to 1 or 0 to 255")
    return table


def read_colormap_image(data) -> np.ndarray:
    """Read a colour map from an image strip, left to right.

    Any height is accepted and the rows are averaged, so a bar exported for
    OVITO, a screenshot of a colour scale or a one pixel strip all work.
    """
    import io

    from PIL import Image

    with Image.open(io.BytesIO(data) if isinstance(data, bytes) else data) as image:
        array = np.asarray(image.convert("RGB"), dtype=float) / 255.0
    if array.ndim != 3:
        raise ValueError("that image could not be read as a colour map")
    return array.mean(axis=0)


def load_colormap(spec):
    """Resolve *spec* to a matplotlib colour map.

    Accepts a matplotlib colour map name, an already built colour map, an
    ``(n, 3)`` array of colours, or the path to an image strip or a text table
    of RGB triples.  Paths are how a colour scale from a paper, or the bar this
    package writes for OVITO, is reused for the plots.
    """
    from matplotlib.colors import Colormap, ListedColormap

    if spec is None:
        return None
    if isinstance(spec, Colormap):
        return spec
    if isinstance(spec, np.ndarray) or isinstance(spec, (list, tuple)):
        table = np.asarray(spec, dtype=float)
        if table.ndim != 2 or table.shape[1] != 3:
            raise ValueError("a colour map array must have shape (n, 3)")
        return ListedColormap(np.clip(table, 0.0, 1.0), name="custom")

    text = str(spec)
    path = Path(text)
    if path.exists() and path.is_file():
        raw = path.read_bytes()
        if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"):
            return ListedColormap(read_colormap_image(raw), name=path.stem)
        return ListedColormap(read_colormap_table(raw), name=path.stem)

    import matplotlib

    try:
        return matplotlib.colormaps[text]
    except KeyError:
        raise ValueError(
            f"unknown colour map {text!r}: give a matplotlib name "
            f"({', '.join(PLOT_COLORMAPS)}, ...) or a path to an image or a table"
        ) from None

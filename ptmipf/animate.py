"""Animate a flat orientation map, or a rendered section, over a trajectory.

A deformation simulation writes a frame every few thousand steps; an animation
of the same section through each frame shows twins nucleate and grow, grains
rotate and boundaries move, which no single map can.  The frames share one
section, one projection and one colour key, so the only thing changing between
them is the microstructure.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

__all__ = ["frame_files", "animate_flat_map", "animate_render", "write_video"]


def frame_files(pattern) -> list[Path]:
    """Trajectory frames matching a glob, sorted by the integer in their name.

    ``run.*.dump`` sorts as 0, 5000, 10000 rather than 0, 10000, 5000.
    """
    pattern = str(pattern)
    directory, _, name = pattern.rpartition("/")
    files = list(Path(directory or ".").glob(name))

    def step(path: Path) -> int:
        numbers = re.findall(r"\d+", path.stem)
        return int(numbers[-1]) if numbers else 0

    return sorted(files, key=step)


def _strain_of(path: Path, rate: float | None, timestep_fs: float = 2.0) -> float | None:
    """Engineering strain of a frame from its step number, if the rate is known."""
    if rate is None:
        return None
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        return None
    return int(numbers[-1]) * timestep_fs * 1e-3 * abs(rate)


def animate_flat_map(
    files,
    out,
    direction="x",
    view="z",
    structures=("fcc", "hcp", "bcc"),
    frame=None,
    slab_width: float = 10.0,
    pixel_size: float = 0.6,
    fill: float | None = 6.0,
    boundary_angle: float = 5.0,
    boundary_scale: tuple | None = None,
    wireframes: bool = False,
    rate: float | None = None,
    title: str = "",
    fps: int = 4,
    dpi: int = 120,
    workers: int = 1,
    rmsd_cutoff: float = 0.1,
):
    """Render every frame as a flat map and join them into a video.

    Parameters
    ----------
    files
        Trajectory frames, in order; see :func:`frame_files`.
    out
        Output path, ``.mp4`` or ``.gif``.
    rate
        Signed strain rate in 1/ps, used to label each frame with its strain.
    boundary_scale
        ``(vmin, vmax, cmap)`` to colour the boundaries by misorientation.
    rmsd_cutoff
        PTM RMSD cutoff.  The 0.1 default suits fcc at room temperature; bcc
        metals need about 0.15 or their distorted grain interiors go unindexed.
    workers
        Frames are independent, so they can be rendered in parallel.  Keep
        this at 1 in a process that has already used OVITO: its thread pool
        does not survive a fork, and the workers hang rather than fail.

    Returns
    -------
    list[str]
        The per-frame PNG paths, kept beside the video.
    """
    out = Path(out)
    stills = out.with_suffix("")
    stills.mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(path), str(stills / f"{i:04d}.png"), _strain_of(Path(path), rate))
        for i, path in enumerate(files)
    ]
    settings = dict(
        direction=direction, view=view, structures=tuple(structures), frame=frame,
        slab_width=slab_width, pixel_size=pixel_size, fill=fill,
        boundary_angle=boundary_angle, boundary_scale=boundary_scale,
        wireframes=wireframes, title=title, dpi=dpi, rmsd_cutoff=rmsd_cutoff,
    )
    if workers > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        # Fork rather than the 3.14 forkserver default: the workers need the
        # loaded OVITO and the caller's module state, and fork keeps both
        # without re-importing a __main__ that may not exist.
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(workers, mp_context=context) as pool:
            list(pool.map(_flat_frame, jobs, [settings] * len(jobs)))
    else:
        for job in jobs:
            _flat_frame(job, settings)
    write_video([j[1] for j in jobs], out, fps=fps)
    return [j[1] for j in jobs]


def _flat_frame(job, settings):
    """One frame; a top-level function so a process pool can pickle it."""
    import matplotlib

    matplotlib.use("Agg")
    from .analysis import analyse
    from .fill import fill_boundary_orientations
    from .flatmap import flat_ipf_map, save_flat_map
    from .select import select_by_region

    path, png, strain = job
    result = analyse(
        path, direction=settings["direction"], structures=settings["structures"],
        frame=settings["frame"], rmsd_cutoff=settings.get("rmsd_cutoff", 0.1),
    )
    axis = "xyz".index(settings["view"]) if settings["view"] in "xyz" else 2
    centre = float(np.median(result.positions[:, axis]))
    half = settings["slab_width"] / 2.0
    section = result.subset(
        select_by_region(result, settings["view"], minimum=centre - half, maximum=centre + half)
    )
    if settings["fill"]:
        section = fill_boundary_orientations(section, radius=settings["fill"], min_neighbours=3)
    flat = flat_ipf_map(
        section, view=settings["view"], slab_width=settings["slab_width"],
        pixel_size=settings["pixel_size"], boundary_angle=settings["boundary_angle"],
    )
    rgb = None
    colorbar = None
    if settings["boundary_scale"]:
        from .boundaries import color_boundaries_by_angle

        lo, hi, cmap = settings["boundary_scale"]
        rgb = color_boundaries_by_angle(flat, lo, hi, cmap, width=2)
        colorbar = (lo, hi, cmap, "misorientation (degrees)")
    frames = None
    if settings["wireframes"] and flat.labels is not None:
        from .wireframe import grain_wireframes

        frames = grain_wireframes(flat, min_area_pixels=600, color="invert")
    label = settings["title"]
    if strain is not None:
        label = f"{label}   strain {100 * strain:.1f}%".strip()
    save_flat_map(
        flat, png, title=label, dpi=settings["dpi"], rgb=rgb, colorbar=colorbar,
        wireframes=frames,
    )
    return png


def animate_render(
    files,
    out,
    direction="x",
    view="z",
    structures=("fcc", "hcp", "bcc"),
    frame=None,
    slab_width: float | None = 10.0,
    hide_other: bool = True,
    fill: float | None = None,
    tripod: bool = True,
    size=(900, 800),
    rate: float | None = None,
    fps: int = 4,
    rmsd_cutoff: float = 0.1,
):
    """Render every frame with OVITO, as a section or the whole cell, to a video."""
    from .analysis import analyse
    from .render import render_result
    from .select import select_by_region

    out = Path(out)
    stills = out.with_suffix("")
    stills.mkdir(parents=True, exist_ok=True)
    pngs = []
    camera = {"x": (-1.0, 0, 0), "y": (0, -1.0, 0), "z": (0, 0, -1.0)}.get(view, (-1, -1, -0.5))
    for i, path in enumerate(files):
        result = analyse(
            path, direction=direction, structures=structures, frame=frame,
            rmsd_cutoff=rmsd_cutoff,
        )
        if slab_width:
            axis = "xyz".index(view) if view in "xyz" else 2
            centre = float(np.median(result.positions[:, axis]))
            result = result.subset(
                select_by_region(result, view, minimum=centre - slab_width / 2,
                                 maximum=centre + slab_width / 2)
            )
        if fill:
            from .fill import fill_boundary_orientations

            result = fill_boundary_orientations(result, radius=fill, min_neighbours=3)
        png = stills / f"{i:04d}.png"
        render_result(
            result, png, hide_other=hide_other, size=size, camera_dir=camera,
            perspective=view not in "xyz", tripod=tripod,
        )
        strain = _strain_of(Path(path), rate)
        if strain is not None:
            _stamp(png, f"strain {100 * strain:.1f}%")
        pngs.append(str(png))
    write_video(pngs, out, fps=fps)
    return pngs


def _stamp(png, text):
    """Write a label into the corner of a rendered PNG."""
    from PIL import Image, ImageDraw

    with Image.open(png) as image:
        draw = ImageDraw.Draw(image)
        draw.text((14, 10), text, fill=(0, 0, 0))
        image.save(png)


def _pad_to_common_size(pngs):
    """Pad every still to the largest even width and height among them, on white.

    A deforming cell changes the map's aspect from frame to frame, and a video
    needs every frame the same size.  Padding, rather than resizing, keeps the
    scale bar honest.
    """
    from PIL import Image

    sizes = [Image.open(p).size for p in pngs]
    width = max(w for w, _ in sizes)
    height = max(h for _, h in sizes)
    # H.264 with yuv420p chroma subsampling cannot encode an odd width or
    # height, and macro_block_size=1 below turns off imageio's own rounding,
    # so an odd-sized frame reaches ffmpeg and kills it with a broken pipe.
    # One extra white pixel costs nothing and keeps the scale bar honest.
    width += width % 2
    height += height % 2
    for p, (w, h) in zip(pngs, sizes):
        with Image.open(p) as image:
            # Every frame becomes RGB, whatever matplotlib or OVITO wrote, so
            # the encoder sees one channel count throughout.
            canvas = Image.new("RGB", (width, height), "white")
            rgb = image.convert("RGBA")
            canvas.paste(rgb, ((width - w) // 2, (height - h) // 2), mask=rgb.split()[3])
            canvas.save(p)


def write_video(pngs, out, fps: float = 4):
    """Join PNGs into an MP4 (via imageio/ffmpeg) or a GIF (via Pillow).

    *fps* may be fractional: a frame every two seconds is ``fps=0.5``.
    """
    out = Path(out)
    _pad_to_common_size(pngs)
    if out.suffix.lower() == ".gif":
        from PIL import Image

        frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in pngs]
        duration = max(1, int(round(1000 / fps)))
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=duration, loop=0)
        return str(out)
    try:
        import imageio.v2 as imageio
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "MP4 output needs imageio and imageio-ffmpeg: pip install 'imageio[ffmpeg]'"
        ) from error
    with imageio.get_writer(
        str(out), fps=fps, codec="libx264", quality=8, macro_block_size=1
    ) as writer:
        for p in pngs:
            writer.append_data(imageio.imread(p))
    return str(out)

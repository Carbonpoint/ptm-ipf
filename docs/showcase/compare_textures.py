"""One projection, several runs side by side: rows are the three sections."""
import sys
sys.path.insert(0, "/home/user/replotting")
sys.path.insert(0, "/home/user/mg_ipf_analysis")
import figcheck
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from assemble_grid import AXES, _draw_parent_axes
from ptmipf.legend import ipf_legend

PLANES = {"x": "Y-Z", "y": "X-Z", "z": "X-Y"}


def compare(runs, labels, projection, out_path, title, structure, laue):
    n = len(runs)
    fig = plt.figure(figsize=(4.4 * n + 1.0, 13.6), facecolor="white")
    left, bottom, height = 0.075, 0.03, 0.235
    width = 0.80 / n
    gap_x, gap_y = 0.84 / n, 0.257
    for row, cut in enumerate(AXES):
        for col, (run, label) in enumerate(zip(runs, labels)):
            ax = fig.add_axes([left + col * gap_x, bottom + (2 - row) * gap_y, width, height])
            ax.imshow(mpimg.imread(f"/home/user/showcase/figures/{run}/panels/{run}_cut{cut}_ipf{projection}.png"))
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
            if row == 0:
                ax.set_title(label, fontsize=14, pad=8)
            if col == 0:
                ax.set_ylabel(f"section $\\perp$ {cut.upper()}\n({PLANES[cut]} plane)",
                              fontsize=13, labelpad=8, y=0.62)
    # With fewer columns the panels reach further right, so the key sits
    # above them rather than beside the last one, and gets its own room.
    box = [0.70, 0.815, 0.16, 0.125] if laue != "6/mmm" else [0.70, 0.860, 0.14, 0.072]
    key = fig.add_axes(box)
    ipf_legend(laue, direction_label=projection.upper(), structure_label=structure, ax=key)
    _draw_parent_axes(fig, left, bottom, gap_y, structure, laue)
    fig.suptitle(title, fontsize=15, y=0.992)
    fig.text(0.5, 0.012, f"Every panel is the same IPF {projection.upper()} projection of a "
             "10 A section through the middle of the cell, boundaries filled, unit cells drawn.",
             ha="center", fontsize=10, color="0.3")
    figcheck.save_checked(fig, out_path, strict=False, dpi=140)


if __name__ == "__main__":
    # Labels are separated by "|" so they may contain commas.
    runs = sys.argv[1].split(","); labels = sys.argv[2].split("|")
    compare(runs, labels, sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])

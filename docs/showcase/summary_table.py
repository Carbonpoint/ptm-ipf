"""The campaign summary table for the README, from stats.json and peaks.txt."""
import json, re
from pathlib import Path

F = Path("/home/user/showcase/figures")
peaks = {}
for line in Path("/home/user/showcase/peaks.txt").read_text().splitlines():
    m = re.match(r"(\S+) peak ([\d.]+) GPa at strain ([\d.]+), final ([\d.]+)", line)
    if m:
        peaks[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))

ORDER = ["cu_random_T", "cu_rolled_T", "cu_rolled_C", "cu_extruded_T", "cu_extruded_C",
         "fe_random_v2_T", "fe_rolled_v2_C", "ti_random_T", "ti_rolled_C", "ti_extruded_T"]
LABEL = {"random": "random", "rolled": "rolled", "extruded": "extruded"}
rows = ["| run | atoms | texture | load | peak stress | at strain | final stress | atoms PTM labels close-packed | grains per section (x, y, z) |",
        "|---|---|---|---|---|---|---|---|---|"]
for name in ORDER:
    p = F / name / "stats.json"
    parts = name.split("_")
    el, tex, mode = parts[0], parts[1], parts[-1]
    load = "tension" if mode == "T" else "compression"
    if not p.exists():
        rows.append(f"| {name} | | {tex} | {load} | | | | | in progress |")
        continue
    d = json.loads(p.read_text())
    n = d["n_atoms"]; c = d["final_counts"]
    parent = {"cu": "fcc", "fe": "bcc", "ti": "hcp"}[el]
    other = {"cu": "hcp", "fe": "hcp", "ti": "fcc"}[el]
    frac = 100 * c.get(other, 0) / n
    g = d["grains_in_sections"]
    pk = peaks.get(name)
    stress = f"{pk[0]:.2f} GPa | {100*pk[1]:.1f}% | {pk[2]:.2f} GPa" if pk else " | | "
    rows.append(f"| {name} | {n:,} | {tex} | {load} | {stress} | {other} {frac:.1f}% | {g['x']}, {g['y']}, {g['z']} |")
print("\n".join(rows))

"""Build a textured Voronoi polycrystal with atomsk, from explicit rotation matrices.

Version 2 of the builder. The first version placed rotated lattice blocks by
hand and trimmed overlaps at 0.75 of the nearest-neighbour distance, which left
the as-built cell at 82 to 93 percent of the ideal density and produced
unindexable zones that 20 ps of equilibration could not heal. atomsk builds its
Voronoi polycrystals at full density, so it does the geometry here.

atomsk's node file takes three angles per grain. They are not Euler angles:
they are extrinsic rotations about the Cartesian x, y and z axes, composing as
R = Rz(theta_z) @ Ry(theta_y) @ Rx(theta_x), crystal to sample. This was
established by building single grains at known angles and measuring the result
with polyhedral template matching (exact to 0.01 degrees); the same script
records every grain's rotation matrix in JSON so the analysis can always be
checked against what was built.

  python build_poly2.py <Cu|Fe|Ti> <random|rolled|extruded> <box A> <n grains> <seed> <out>

Textures as in version 1 (crystal->sample, RD = x, TD = y, ND = z).
"""
import json, subprocess, sys
import numpy as np

element, texture, box, n_grains, seed, out = (
    sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
LATTICE = {"Cu": ("fcc", "3.615"), "Fe": ("bcc", "2.8553"), "Ti": ("hcp", "2.951 4.684")}
structure, params = LATTICE[element]
rng = np.random.default_rng(seed)
RD, TD, ND = np.eye(3)


def rot(axis, deg):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis); t = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def random_rotation():
    q = rng.normal(size=4); q /= np.linalg.norm(q); x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def align(crystal_dir, sample_axis, spin=None):
    d = np.asarray(crystal_dir, float); d /= np.linalg.norm(d)
    s = np.asarray(sample_axis, float); s /= np.linalg.norm(s)
    v = np.cross(d, s); c = float(d @ s)
    if np.linalg.norm(v) < 1e-9:
        R = np.eye(3) if c > 0 else rot([1, 0, 0] if abs(d[0]) < 0.9 else [0, 1, 0], 180)
    else:
        R = rot(v, np.degrees(np.arctan2(np.linalg.norm(v), c)))
    return rot(s, rng.uniform(0, 360) if spin is None else spin) @ R


def plane_and_direction(plane, direction):
    R = align(plane, ND, spin=0.0)
    d = R @ (np.asarray(direction, float) / np.linalg.norm(direction))
    turn = np.degrees(np.arctan2(np.cross(d, RD) @ ND, d @ RD))
    return rot(ND, turn) @ R


def spread(R, deg):
    return rot(rng.normal(size=3), rng.normal(0, deg)) @ R


def orientation():
    if texture == "random":
        return random_rotation()
    if texture == "rolled":
        if structure == "fcc":
            if rng.random() < 0.5:
                return spread(plane_and_direction([1, 1, 0], [1, -1, 2]), 8.0)
            return spread(plane_and_direction([1, 1, 2], [1, 1, -1]), 8.0)
        if structure == "hcp":
            return spread(align([0, 0, 1], ND), 15.0)
        return spread(align([1, 1, 0], RD), 10.0)
    if texture == "extruded":
        if structure == "fcc":
            return spread(align([1, 1, 1] if rng.random() < 0.65 else [1, 0, 0], RD), 8.0)
        if structure == "hcp":
            return spread(align([1, 0, 0], RD), 10.0)
        return spread(align([1, 1, 0], RD), 10.0)
    raise SystemExit(f"unknown texture {texture}")


def xyz_angles(R):
    """Decompose crystal->sample R as Rz(tz) @ Ry(ty) @ Rx(tx), atomsk's convention."""
    ty = np.degrees(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    if abs(R[2, 0]) < 1.0 - 1e-10:
        tx = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        tz = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:  # gimbal: fold everything into tx
        tx = np.degrees(np.arctan2(-R[0, 1], R[1, 1]))
        tz = 0.0
    return tx, ty, tz


rotations = [orientation() for _ in range(n_grains)]
seeds = rng.uniform(0, box, size=(n_grains, 3))

# Verify the decomposition reproduces every matrix before anything is built.
for R in rotations:
    tx, ty, tz = xyz_angles(R)
    C = rot([0, 0, 1], tz) @ rot([0, 1, 0], ty) @ rot([1, 0, 0], tx)
    assert np.abs(C - R).max() < 1e-9, "angle decomposition failed"

with open(out + ".nodes", "w") as fh:
    fh.write(f"box {box} {box} {box}\n")
    for s, R in zip(seeds, rotations):
        tx, ty, tz = xyz_angles(R)
        fh.write(f"node {s[0]:.4f} {s[1]:.4f} {s[2]:.4f} {tx:.6f} {ty:.6f} {tz:.6f}\n")

subprocess.run(f"atomsk --create {structure} {params} {element} {out}_unit.xsf -overwrite >/dev/null 2>&1", shell=True, check=True)
# atomsk appends .lmp to the output name for the lammps format.
subprocess.run(
    f"atomsk --polycrystal {out}_unit.xsf {out}.nodes {out} lammps -wrap -overwrite "
    f">/dev/null 2>&1",
    shell=True, check=True,
)
subprocess.run(f"mv {out}.lmp {out}.data", shell=True, check=True)

n_atoms = int(next(line.split()[0] for line in open(out + ".data") if "atoms" in line))
json.dump({"element": element, "structure": structure, "texture": texture, "box": box,
           "builder": "atomsk-v2", "n_atoms": n_atoms, "seeds": seeds.tolist(),
           "rotations": [R.tolist() for R in rotations]}, open(out + ".json", "w"), indent=1)
ideal = {"fcc": 4/3.615**3, "bcc": 2/2.8553**3, "hcp": 4/(2.951**2*np.sin(np.pi/3)*4.684*2)}[structure]
print(f"{out}: {n_atoms} {element} atoms, {n_grains} grains, {texture}, "
      f"density {n_atoms/box**3:.4f}/A^3 = {100*n_atoms/box**3/ideal:.1f}% of ideal")

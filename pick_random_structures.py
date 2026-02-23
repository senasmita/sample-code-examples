#!/usr/bin/env python3
"""
Randomly pick N structures (frames) from a LAMMPS dump trajectory (*.lammpstrj)
and write each as a separate geometry .txt file (XYZ format).

Usage:
  python pick_random_structures.py traj_prod.lammpstrj -n 100 -o geoms --seed 42

Output format (XYZ-in-.txt):
  line 1: number of atoms
  line 2: comment (timestep + box bounds)
  next lines: "T<type> x y z" (sorted by atom id if present)
"""

import argparse
import os
import random
from typing import List, Tuple


def collect_frame_offsets(path: str) -> List[int]:
    """Return file offsets (byte positions) where each frame starts."""
    offsets: List[int] = []
    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.startswith(b"ITEM: TIMESTEP"):
                offsets.append(pos)
    return offsets


def read_frame_at(path: str, offset: int):
    """Read one frame starting at offset and return (timestep, natoms, bounds, cols, atoms)."""
    with open(path, "rb") as f:
        f.seek(offset)

        def rline() -> str:
            return f.readline().decode("utf-8").strip()

        header = rline()
        if not header.startswith("ITEM: TIMESTEP"):
            raise ValueError("Offset does not point to a TIMESTEP header")

        timestep = int(rline())

        if not rline().startswith("ITEM: NUMBER OF ATOMS"):
            raise ValueError("Unexpected dump format (missing NUMBER OF ATOMS)")
        natoms = int(rline())

        box_header = rline()
        if not box_header.startswith("ITEM: BOX BOUNDS"):
            raise ValueError("Unexpected dump format (missing BOX BOUNDS)")
        bounds: List[Tuple[float, float]] = []
        for _ in range(3):
            lo, hi = rline().split()[:2]
            bounds.append((float(lo), float(hi)))

        atoms_header = rline()
        if not atoms_header.startswith("ITEM: ATOMS"):
            raise ValueError("Unexpected dump format (missing ATOMS header)")
        cols = atoms_header.split()[2:]

        atoms: List[List[str]] = []
        for _ in range(natoms):
            atoms.append(rline().split())

        return timestep, natoms, bounds, cols, atoms


def write_xyz_txt(outpath: str, timestep: int, natoms: int, bounds, cols, atoms):
    """Write a single frame as XYZ (but with .txt extension)."""
    for c in ("x", "y", "z"):
        if c not in cols:
            raise ValueError(f"Dump must contain '{c}' column, found: {cols}")

    idx_id = cols.index("id") if "id" in cols else None
    idx_type = cols.index("type") if "type" in cols else None
    idx_x, idx_y, idx_z = cols.index("x"), cols.index("y"), cols.index("z")

    if idx_id is not None:
        atoms = sorted(atoms, key=lambda a: int(a[idx_id]))

    bstr = " ".join([f"{lo:g} {hi:g}" for lo, hi in bounds])

    with open(outpath, "w") as w:
        w.write(f"{natoms}\n")
        w.write(f"timestep {timestep} | box {bstr}\n")
        for a in atoms:
            typ = a[idx_type] if idx_type is not None else "X"
            w.write(f"T{typ} {a[idx_x]} {a[idx_y]} {a[idx_z]}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj", help="LAMMPS trajectory file (*.lammpstrj / dump)")
    ap.add_argument("-n", "--nframes", type=int, default=100, help="How many frames to pick")
    ap.add_argument("-o", "--outdir", default="random_geoms", help="Output directory")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (for reproducibility)")
    args = ap.parse_args()

    offsets = collect_frame_offsets(args.traj)
    total = len(offsets)
    if total == 0:
        raise SystemExit("No frames found (did you pass a valid LAMMPS dump?)")

    n = min(args.nframes, total)
    if args.seed is not None:
        random.seed(args.seed)

    picks = sorted(random.sample(range(total), k=n))

    os.makedirs(args.outdir, exist_ok=True)

    for j, frame_idx in enumerate(picks, start=1):
        timestep, natoms, bounds, cols, atoms = read_frame_at(args.traj, offsets[frame_idx])
        fname = f"geom_{j:03d}_ts{timestep}.txt"
        outpath = os.path.join(args.outdir, fname)
        write_xyz_txt(outpath, timestep, natoms, bounds, cols, atoms)

    print(f"Wrote {n} geometries to: {args.outdir} (picked from {total} frames)")


if __name__ == "__main__":
    main()


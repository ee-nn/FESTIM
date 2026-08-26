#!/usr/bin/env python3
"""Overlay the grain boundaries of a `neper -M map.tesr` mesh on the raster.

    python mesh_overlay.py map.tesr poly.msh4 -o check-mesh.png

The raster is drawn with its cell ids shuffled onto a categorical colormap
(consecutive ids are neighbours, so a continuous map hides the boundaries), and
every 1D element of the mesh on top of it: black where the edge lies between
two grains, grey where it lies on the specimen surface. These 1D elements are
the `edge#` element sets the transport driver selects its network from, so
what is drawn is exactly the set of segments FESTIM can put a boundary on --
before the disorientation filter, which the driver applies and draws itself
(check-network.png).

The msh4 is parsed directly (no gmsh/meshio dependency), and the two functions
`read_tesr` / `read_msh4` are reused by ebsd_gb_diffusion.py.
"""

import argparse
import sys

import numpy as np


def read_tesr(path):
    """(cell ids as (ny, nx) array, (voxel size x, y)) from an ascii tesr."""
    with open(path) as fh:
        tok = fh.read().split()
    i = tok.index("**general")
    if int(tok[i + 1]) != 2:
        sys.exit(f"{path}: not a 2D tesr")
    nx, ny = int(tok[i + 2]), int(tok[i + 3])
    vox = float(tok[i + 4]), float(tok[i + 5])
    i = tok.index("**data")
    if tok[i + 1] != "ascii":
        sys.exit(f"{path}: **data is {tok[i + 1]}, write it as ascii")
    return np.array(tok[i + 2 : i + 2 + nx * ny], dtype=int).reshape(ny, nx), vox


def read_msh4(path):
    """Nodes and tagged 1D / 2D elements of a Gmsh 4.1 ascii mesh.

    Returns ``(xyz, seg, tri)`` where ``xyz`` maps node tag -> coordinates,
    ``seg`` is a list of (edge id, [n1, n2]) and ``tri`` a list of
    (face id, [n1, n2, n3]); the ids are the entity tags Neper wrote, which for
    its own meshes equal the tessellation entity ids (edge#, face#).
    """
    with open(path) as fh:
        lines = fh.read().split("\n")
    i = lines.index("$Nodes")
    nblocks = int(lines[i + 1].split()[0])
    p = i + 2
    xyz = {}
    for _ in range(nblocks):
        _dim, _tag, _par, n = map(int, lines[p].split())
        p += 1
        tags = [int(lines[p + k]) for k in range(n)]
        p += n
        for k in range(n):
            xyz[tags[k]] = np.array(lines[p + k].split(), dtype=float)
        p += n
    i = lines.index("$Elements")
    nblocks = int(lines[i + 1].split()[0])
    p = i + 2
    seg, tri = [], []
    for _ in range(nblocks):
        dim, tag, _typ, n = map(int, lines[p].split())
        p += 1
        for k in range(n):
            nodes = [int(v) for v in lines[p + k].split()[1:]]
            if dim == 1:
                seg.append((tag, nodes))
            elif dim == 2:
                tri.append((tag, nodes))
        p += n
    return xyz, seg, tri


def edge_sides(seg, tri):
    """edge id -> set of face ids its segments belong to (1: surface, 2: interior)."""
    tri_of = {}
    for face, v in tri:
        for a, b in ((v[0], v[1]), (v[1], v[2]), (v[2], v[0])):
            tri_of.setdefault(frozenset((a, b)), set()).add(face)
    sides = {}
    for edge, v in seg:
        sides.setdefault(edge, set()).update(tri_of.get(frozenset(v), ()))
    return sides


def draw_raster(ax, cells, vox, seed=0):
    """Raster with shuffled categorical colours; returns the permutation."""
    ncell = int(cells.max())
    perm = np.random.default_rng(seed).permutation(ncell) + 1
    shown = np.where(cells > 0, perm[np.maximum(cells - 1, 0)], 0).astype(float)
    shown[cells == 0] = np.nan
    ny, nx = cells.shape
    ax.imshow(
        shown,
        cmap="tab20",
        interpolation="nearest",
        origin="lower",
        extent=(0, nx * vox[0], 0, ny * vox[1]),
    )
    ax.set_xlim(0, nx * vox[0])
    ax.set_ylim(0, ny * vox[1])
    return perm


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("tesr")
    p.add_argument("msh4")
    p.add_argument("-o", "--output", default="check-mesh.png")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells, vox = read_tesr(args.tesr)
    xyz, seg, tri = read_msh4(args.msh4)
    sides = edge_sides(seg, tri)
    faces = {t for t, _ in tri}
    if len(faces) != int(cells.max()):
        print(f"note: raster has {int(cells.max())} cells, mesh has {len(faces)} faces")

    ny, nx = cells.shape
    fig, ax = plt.subplots(figsize=(7, 7 * ny * vox[1] / (nx * vox[0])))
    draw_raster(ax, cells, vox)
    n_int = n_surf = 0
    for edge, v in seg:
        interior = len(sides[edge]) == 2
        n_int += interior
        n_surf += not interior
        a, b = xyz[v[0]], xyz[v[1]]
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color="black" if interior else "0.55",
            lw=1.0 if interior else 0.7,
        )
    ax.set_title(
        f"{len(sides)} edges ({sum(len(s) == 2 for s in sides.values())} interior), "
        f"{n_int + n_surf} segments over {int(cells.max())} raster cells"
    )
    ax.set_xlabel("x (tesr units)")
    ax.set_ylabel("y (tesr units)")
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

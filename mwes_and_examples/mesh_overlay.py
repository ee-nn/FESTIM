"""Overlay the grain boundaries of a `neper -M map.tesr` mesh on the raster.

    from mesh_overlay import overlay
    overlay("map.tesr", "poly.msh4", output="check-mesh.png", unit="um")

The raster is drawn with its cell ids shuffled onto a categorical colormap
(consecutive ids are neighbours, so a continuous map hides the boundaries), and
every 1D element of the mesh on top of it: black where the edge lies between
two grains, grey where it lies on the specimen surface. These 1D elements are
the `edge#` element sets the transport driver selects its network from, so
what is drawn is exactly the set of segments FESTIM can put a boundary on --
before the disorientation filter, which the driver applies and draws itself
(check-network.png).

The msh4 is parsed directly (no gmsh/meshio dependency), and the readers
`read_tesr` / `read_msh4` are reused by ebsd_gb_diffusion.py and
grain_area_change.py.

Importing this module has no side effects: the Agg backend and the font sizes
are set by `overlay`, which is the function that writes a file, and never by
the drawing primitive `draw_raster`, which takes an axes the caller owns.
"""

import numpy as np
from micrograph import scale_bar_ax

# Every figure the pipeline writes uses these.
RCPARAMS = {
    "font.size": 16,  # Baseline text, labels, ticks, and legends
    "axes.titlesize": 16,  # Subplot titles
    "figure.titlesize": 16,  # Global figure super title
}


def use_agg():
    """Select the non-interactive backend and the pipeline's font sizes.

    Called by every function here that writes a PNG. Kept out of module scope
    so that importing this file does not reach into a caller's matplotlib.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(RCPARAMS)
    return plt


def read_tesr(path):
    """(cell ids as (ny, nx) array, (voxel size x, y)) from an ascii tesr.

    Raises ValueError on a file this reader cannot handle, rather than exiting:
    the caller is another module, not a shell.
    """
    with open(path) as fh:
        tok = fh.read().split()
    i = tok.index("**general")
    if int(tok[i + 1]) != 2:
        raise ValueError(f"{path}: not a 2D tesr")
    nx, ny = int(tok[i + 2]), int(tok[i + 3])
    vox = float(tok[i + 4]), float(tok[i + 5])
    i = tok.index("**data")
    if tok[i + 1] != "ascii":
        raise ValueError(f"{path}: **data is {tok[i + 1]}, write it as ascii")
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


# Neper's colour palette for integer values, which is what `neper -V map.tesr`
# uses to colour cells by id. It is the colour list in
# https://neper.info/doc/exprskeys.html with entries of brightness (mean channel
# value / 255) below 0.2 or above 0.8 removed, kept in order; deriving it from
# that table reproduces the 92-entry palette the same page prints. Cell id k
# takes entry k: the Neper tutorial colours values 1, 2, 3 and 4 red, green,
# blue and yellow, which are entries 1-4.
NEPER_PALETTE = np.array(
    [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (127, 255, 0),
        (0, 255, 127),
        (128, 128, 0),
        (128, 0, 128),
        (0, 128, 128),
        (128, 128, 128),
        (0, 191, 255),
        (124, 252, 0),
        (64, 64, 64),
        (255, 69, 0),
        (192, 192, 192),
        (255, 140, 0),
        (0, 0, 205),
        (75, 0, 130),
        (240, 128, 128),
        (255, 127, 80),
        (250, 128, 114),
        (127, 255, 212),
        (255, 215, 0),
        (255, 165, 0),
        (139, 0, 139),
        (0, 139, 139),
        (205, 133, 63),
        (70, 130, 180),
        (0, 250, 154),
        (72, 61, 139),
        (184, 134, 11),
        (255, 160, 122),
        (135, 206, 250),
        (255, 99, 71),
        (112, 128, 144),
        (255, 105, 180),
        (189, 183, 107),
        (0, 206, 209),
        (60, 179, 113),
        (199, 21, 133),
        (238, 130, 238),
        (173, 255, 47),
        (143, 188, 143),
        (188, 143, 143),
        (255, 20, 147),
        (139, 69, 19),
        (148, 0, 211),
        (30, 144, 255),
        (119, 136, 153),
        (222, 184, 135),
        (123, 104, 238),
        (64, 224, 208),
        (135, 206, 235),
        (72, 209, 204),
        (210, 180, 140),
        (50, 205, 50),
        (233, 150, 122),
        (176, 196, 222),
        (65, 105, 225),
        (152, 251, 152),
        (220, 20, 60),
        (186, 85, 211),
        (240, 230, 140),
        (144, 238, 144),
        (47, 79, 79),
        (153, 50, 204),
        (46, 139, 87),
        (154, 205, 50),
        (138, 43, 226),
        (219, 112, 147),
        (107, 142, 35),
        (147, 112, 219),
        (244, 164, 96),
        (85, 107, 47),
        (102, 205, 170),
        (106, 90, 205),
        (34, 139, 34),
        (25, 25, 112),
        (32, 178, 170),
        (218, 112, 214),
        (100, 149, 237),
        (160, 82, 45),
        (178, 34, 34),
        (205, 92, 92),
        (105, 105, 105),
        (210, 105, 30),
        (165, 42, 42),
        (218, 165, 32),
        (221, 160, 221),
        (95, 158, 160),
    ],
    dtype=np.uint8,
)


def draw_raster(ax, cells, vox, alpha=1.0):
    """Raster coloured by cell id with Neper's own palette; returns the RGBA.

    Cell id k gets ``NEPER_PALETTE[(k - 1) % 92]``, so this background is the
    same picture as check-grains.png and the two can be compared grain by
    grain. Empty voxels (id 0) are left transparent.

    Ids more than 92 apart repeat a colour, exactly as they do in
    check-grains.png, so a map with more than 92 grains has repeats in both
    images and in the same places.

    Pass ``alpha`` below 1 to fade the background. Worth doing on
    check-network.png, where the inferno theta lines have to stay readable
    over saturated primaries; the hues still match check-grains.png.
    """
    ny, nx = cells.shape
    ncell = int(cells.max())
    lut = np.zeros((ncell + 1, 4), dtype=np.uint8)  # row 0 = empty = transparent
    lut[1:, :3] = NEPER_PALETTE[np.arange(ncell) % len(NEPER_PALETTE)]
    lut[1:, 3] = round(255 * alpha)
    rgba = lut[cells]
    ax.imshow(
        rgba,
        interpolation="nearest",
        origin="lower",
        extent=(0, nx * vox[0], 0, ny * vox[1]),
    )
    ax.set_xlim(0, nx * vox[0])
    ax.set_ylim(0, ny * vox[1])
    return rgba


def overlay(tesr, msh4, output="check-mesh.png", dpi=150, unit="um", log=print):
    """Write the overlay PNG. Returns the output path.

    `tesr` and `msh4` are paths; everything else is cosmetic. The counts are
    printed through `log`, which can be set to None to silence them.
    """
    plt = use_agg()

    cells, vox = read_tesr(tesr)
    xyz, seg, tri = read_msh4(msh4)
    sides = edge_sides(seg, tri)
    faces = {t for t, _ in tri}
    if log and len(faces) != int(cells.max()):
        log(f"note: raster has {int(cells.max())} cells, mesh has {len(faces)} faces")

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
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")
    scale_bar_ax(ax, nx * vox[0], unit)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    if log:
        log(f"  wrote {output}")
    return output

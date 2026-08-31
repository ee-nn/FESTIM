"""Convert an Oxford/Channel .ctf EBSD map into a Neper raster tessellation
(.tesr), including the grain segmentation a .ctf does not carry.

    from ctf_to_tesr import convert
    res = convert("map.ctf", "ebsd.tesr", min_pixels=20, diagnostics=True)
    res["segmentation_error"]["indexed"]["rms"]   # degrees

then set TESR = "ebsd.tesr" in ebsd_gb_diffusion.py.

There is no official converter; the transcription is straightforward but the
segmentation is not. A .ctf holds an orientation per pixel and nothing else,
while a .tesr needs a `**cell` section and a `**data` section putting every
pixel in a grain. Grains are found here by flood-filling across neighbouring
pixels whose disorientation is below `threshold`, under cubic symmetry.

Every choice lives in the `Settings` dataclass, which `convert` builds from
keyword arguments or accepts ready-made. It is dumped to
<output>-provenance.json beside the .tesr and read back by
`settings_from_provenance`, because the crop window, the y mirror and the
orientation convention cannot be recovered from the .tesr itself.

With diagnostics=True the conversion also writes, beside the .tesr:
<output>-quality.png (why each rejected pixel was rejected), -segerror.png/.csv
(the per-voxel cost of segmenting), and, when the `neper` binary resolves,
-ori.png and -grains.png, the rendered maps. Everything that depends only on
this stage is produced here; the mesh diagnostics live with the mesh.

What is written
---------------
**general   dimension, XCells/YCells, XStep/YStep (microns unless `scale`)
**cell      grain count, ids, crysym, one mean orientation per grain
**data      grain id of every pixel, contiguous from 1, 0 where unindexed
**oridata   per-pixel orientation (optional; large, needed for -V and GOS)
**oridef    per-pixel indexing flag, 0 where the point was rejected

The driver rescales to metres after meshing (TESR_UNIT), so leave `scale` at 1.

Orientation convention
----------------------
Bunge (phi1, Phi, phi2) in degrees go to Rodrigues vectors under Neper's
default `passive` convention. Rodrigues rather than passing the angles through
because it removes any degrees-vs-radians ambiguity in the tesr reader; reduced
to the cubic fundamental zone first, which also keeps the vector finite (a
180 deg rotation has none). `orientation.self_test`, which `convert` runs
first, pins this against Neper's own convention table.

The file declares tesr format 2.2 deliberately: Neper 4.10.0 swapped the
meaning of `active` and `passive` and bumped the version, and a file claiming
2.1 has its `**cell/*ori` descriptor silently flipped on read
(neut_tesr_fscanf2.c, "Fixing orientation convention") while `**oridata` is
taken literally -- leaving the two sections in opposite conventions.

Any symmetry-equivalent representative is equally correct, since the crysym is
declared and Neper applies it. The active/passive choice is not free, so check
check-ori.png against AZtec or MTEX; if the colours look inverted, re-run with
active=True.

Limitations: cubic only (the disorientation is a closed form specific to that
group -- for hex or lower, segment in MTEX and import the grain ids), and
square grids only (a hexagonal acquisition needs MTEX's `gridify` first).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path

import numpy as np
from mesh_overlay import use_agg
from micrograph import annotate_png, append_key, scale_bar_ax
from orientation import (
    crystal_equivalents,
    cubic_disorientation_angle,
    cubic_symmetry_quaternions,
    euler_bunge_to_quat,
    qconj,
    qmul,
    quat_to_rodrigues,
    rodrigues_to_quat,
    self_test,
    to_fundamental_zone,
)
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from segmentation_error import format_report, read_tesr_full, segmentation_error
from segmentation_error import write_png as write_segerr_png

# --- Channel conventions -----------------------------------------------------
# Field 4 of a .ctf phase line is the Channel Laue group index. Mapped onto the
# crystal symmetry keys Neper accepts (https://neper.info/doc/exprskeys.html).
LAUE_TO_CRYSYM = {
    1: "-1",
    2: "2/m",
    3: "mmm",
    4: "4/m",
    5: "4/mmm",
    6: "-3",
    7: "-3m",
    8: "6/m",
    9: "6/mmm",
    10: "m-3",
    11: "cubic",  # m-3m; Neper's `cubic` and `m-3m` both carry 24 operators
}
CUBIC_LAUE = (10, 11)


# --- conversion settings -----------------------------------------------------
@dataclass
class Settings:
    """Every choice `convert` makes, in one object.

    Passed whole to the writers, so a caller building one by hand gets exactly
    the same behaviour and the fields reaching the provenance json have a
    single definition. The four with a non-obvious rationale are commented; the
    rest say what they do.
    """

    ctf: str
    #: xmin,xmax,ymin,ymax in the .ctf's units, applied before segmentation.
    #: Prefer this to Neper's -transform crop, which clips grains into 1-2 px
    #: slivers that no `min_pixels` prune has seen and that abort the fit.
    crop: str | None = None
    phase: int = 1  #: phase to keep
    threshold: float = 10.0  #: grain boundary misorientation, degrees
    max_mad: float = 1.0  #: MAD cutoff
    min_bands: int = 0  #: minimum Bands; 0 disables the test
    allow_error: bool = False  #: keep points whose Error column is non-zero
    min_pixels: int = 5  #: discard grains smaller than this
    #: step size multiplier giving the tesr length unit. Do not write metres:
    #: Neper's fit objective and its val/eps criteria are in absolute lengths,
    #: so a metre-scale map stops the fit at iteration 1.
    scale: float = 1.0
    #: mirror in y. EBSD coordinates usually run downwards and Neper's y runs
    #: up, so set this to put the specimen's top surface at y = Ly.
    flip_y: bool = False
    active: bool = False  #: write orientations active rather than passive
    fill: bool = True  #: grow cells into unassigned voxels; see fill_holes
    #: clean up enclosed grains and corner-only self-contacts, because
    #: `neper -M` aborts on a face whose boundary is not a single loop.
    topology_fix: bool = True
    voxel_ori: bool = True  #: write **oridata/**oridef (large; needed by -V)
    #: also write <output>-quality.png, -segerror.png/.csv, and, if `neper`
    #: resolves, the two rendered maps -ori.png and -grains.png
    diagnostics: bool = False
    #: neper binary for the rendered check images, or None to skip them. They
    #: are the only part of this module that shells out, and a missing binary
    #: is a warning, not an error.
    neper: str | None = "neper"
    povray: str = "povray"  #: neper -V renders through this

    @property
    def unit(self):
        return "um" if np.isclose(self.scale, 1.0) else f"x{self.scale:g} um"


# --- .ctf parsing ------------------------------------------------------------
class CtfMap:
    """A parsed Channel Text File: header fields plus the pixel table."""

    def __init__(self, path):
        self.path = Path(path)
        self.header = {}
        self.phases = []
        self._parse()

    def _parse(self):
        with open(self.path, errors="replace") as fh:
            lines = fh.read().splitlines()

        col_row = None
        for i, line in enumerate(lines):
            fields = line.split("\t")
            key = fields[0].strip()
            if key == "Phase" and len(fields) > 5 and "Euler1" in fields:
                col_row = i
                break
            if key in ("XCells", "YCells"):
                self.header[key] = int(float(fields[1]))
            elif key in ("XStep", "YStep"):
                self.header[key] = float(fields[1])
            elif key == "Phases":
                self.header["Phases"] = int(fields[1])
            elif ";" in key and len(fields) >= 5:
                # a phase line: "a;b;c <tab> al;be;ga <tab> name <tab> laue <tab> sg"
                try:
                    self.phases.append(
                        {"name": fields[2].strip(), "laue": int(fields[3])}
                    )
                except (ValueError, IndexError):
                    pass

        if col_row is None:
            raise ValueError(
                f"{self.path}: no column header row found. Expected a line "
                "starting with 'Phase' and containing 'Euler1'."
            )
        for key in ("XCells", "YCells", "XStep", "YStep"):
            if key not in self.header:
                raise ValueError(f"{self.path}: header is missing {key}")

        self.columns = [c.strip() for c in lines[col_row].split("\t") if c.strip()]
        for required in ("Phase", "X", "Y", "Euler1", "Euler2", "Euler3"):
            if required not in self.columns:
                raise ValueError(f"{self.path}: no '{required}' column")

        data = np.genfromtxt(
            self.path, skip_header=col_row + 1, usecols=range(len(self.columns))
        )
        if data.ndim == 1:
            data = data[None, :]
        self.table = {c: data[:, i] for i, c in enumerate(self.columns)}
        self.npoints = data.shape[0]

    def __getitem__(self, key):
        return self.table[key]

    def has(self, key):
        return key in self.table

    @property
    def shape(self):
        return self.header["YCells"], self.header["XCells"]

    def crysym(self, phase_index=1):
        if not self.phases:
            return None, None
        ph = self.phases[phase_index - 1]
        return LAUE_TO_CRYSYM.get(ph["laue"]), ph


# --- pipeline ----------------------------------------------------------------
def build_grid(ctf, phase, max_mad, require_zero_error, min_bands):
    """Place the pixel table on the (ny, nx) grid and build the quality mask.

    Points are indexed from their X/Y coordinates rather than from row order,
    so a file that is not written in strict raster order still lands correctly
    and a truncated file leaves holes rather than shearing the map.
    """
    ny, nx = ctf.shape
    ix = np.rint(ctf["X"] / ctf.header["XStep"]).astype(int)
    iy = np.rint(ctf["Y"] / ctf.header["YStep"]).astype(int)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not inside.all():
        print(f"  warning: {int((~inside).sum())} points fall outside XCells x YCells")

    good = inside & (ctf["Phase"] == phase)
    if require_zero_error and ctf.has("Error"):
        good &= ctf["Error"] == 0
    if ctf.has("MAD"):
        good &= ctf["MAD"] <= max_mad
    if min_bands and ctf.has("Bands"):
        good &= ctf["Bands"] >= min_bands

    euler = np.stack((ctf["Euler1"], ctf["Euler2"], ctf["Euler3"]), axis=-1)
    quat = euler_bunge_to_quat(euler[:, 0], euler[:, 1], euler[:, 2])

    qgrid = np.zeros((ny, nx, 4))
    qgrid[..., 0] = 1.0
    ok = np.zeros((ny, nx), dtype=bool)
    qgrid[iy[inside], ix[inside]] = quat[inside]
    ok[iy[good], ix[good]] = True

    # Per-pixel provenance for diagnostics=True: what the .ctf itself says about
    # each point, so a rejected pixel can be traced to the column that
    # rejected it rather than blamed on the conversion.
    diag = {}
    for col, fill in (("Error", -1), ("MAD", np.nan), ("Bands", -1), ("Phase", -1)):
        if ctf.has(col):
            g = np.full((ny, nx), fill, dtype=float)
            g[iy[inside], ix[inside]] = ctf[col][inside]
            diag[col] = g
    return qgrid, ok, diag


def crop_grid(qgrid, ok, spec, xstep, ystep):
    """Cut a rectangular window out of the map, before segmentation.

    Cropping here rather than in Neper matters for two reasons:
     1. The segmentation, the prune and the cell ids all describe
        the same region, and a clipped grain is either big enough
        to keep or dropped like any other.

     2. Per-voxel orientations. Possible bug is that Neper 5.0.0 cannot read
        back a raster when the file carries a `**oridata` section and has
        been through (auto)crop. Cropping upstream keeps the file small enough
        to keep orientations, so -V colouring & -S intragranular measures still work.

    Bounds are in the .ctf's own 'as- acquired' length units,
    i.e. before any flip_y.
    """
    try:
        x0, x1, y0, y1 = (float(v) for v in spec.split(","))
    except ValueError:
        raise SystemExit(
            f"crop={spec!r}: expected four comma-separated numbers, "
            "xmin,xmax,ymin,ymax, in the same units as XStep"
        )
    ny, nx = ok.shape
    ix0, ix1 = max(round(x0 / xstep), 0), min(round(x1 / xstep), nx)
    iy0, iy1 = max(round(y0 / ystep), 0), min(round(y1 / ystep), ny)
    if ix1 - ix0 < 2 or iy1 - iy0 < 2:
        raise SystemExit(
            f"crop={spec} keeps {max(ix1 - ix0, 0)} x {max(iy1 - iy0, 0)} "
            f"pixels. The map is {nx} x {ny} pixels of {xstep} x {ystep}, "
            f"i.e. {nx * xstep:g} x {ny * ystep:g} in those units."
        )
    window = (slice(iy0, iy1), slice(ix0, ix1))
    return qgrid[window], ok[window], window


def segment_grains(qgrid, ok, threshold, sym):
    """Flood-fill across neighbours whose disorientation is below `threshold`.

    Four-connected: a pixel is joined to the one on its right and the one below
    when the two orientations are close enough. The resulting graph's connected
    components are the grains. Rejected pixels join nothing and end up as id 0.
    """
    ny, nx = ok.shape
    idx = np.arange(ny * nx).reshape(ny, nx)
    rows, cols = [], []

    for shift_axis in (1, 0):  # right neighbour, then lower neighbour
        if shift_axis == 1:
            a, b = ok[:, :-1] & ok[:, 1:], None
            qa, qb = qgrid[:, :-1], qgrid[:, 1:]
            ia, ib = idx[:, :-1], idx[:, 1:]
        else:
            a, b = ok[:-1, :] & ok[1:, :], None
            qa, qb = qgrid[:-1, :], qgrid[1:, :]
            ia, ib = idx[:-1, :], idx[1:, :]
        del b
        pa, pb = qa[a], qb[a]
        if pa.size == 0:
            continue
        ang = cubic_disorientation_angle(qmul(qconj(pa), pb))
        same = ang < threshold
        rows.append(ia[a][same])
        cols.append(ib[a][same])

    if rows:
        r = np.concatenate(rows)
        c = np.concatenate(cols)
    else:
        r = c = np.zeros(0, dtype=int)

    graph = coo_matrix(
        (np.ones(len(r)), (r, c)), shape=(ny * nx, ny * nx), dtype=np.int8
    )
    _n, labels = connected_components(graph, directed=False)
    labels = labels.reshape(ny, nx)
    labels[~ok] = -1
    return labels


def fill_holes(cellids):
    """Assign every empty voxel to its nearest cell.

    Rejected points and pruned grains leave holes in `**data`, and a hole is an
    interior surface as far as the fit's `pts(region=surf)` control points are
    concerned -- so a map that is half holes has the objective function chasing
    the boundaries of the noise rather than the boundaries of the grains.

    Neper's own `grow` transform does this, but reaching it means putting the
    file back through `neper -T -transform`, which is the write path that
    produces an unreadable raster when the file carries `**oridata` (Neper
    5.0.0). Doing it here keeps the orientations and avoids that entirely.

    `**oridef` is deliberately left alone: it still records which points were
    actually indexed, so the provenance of a filled voxel is not lost.
    """
    from scipy.ndimage import distance_transform_edt

    empty = cellids == 0
    n = int(empty.sum())
    if n == 0 or n == empty.size:
        return cellids, n
    # distance_transform_edt measures distance to the nearest zero element, so
    # feeding it the empty mask returns, for each empty voxel, the index of the
    # nearest non-empty one
    _, idx = distance_transform_edt(empty, return_indices=True)
    return cellids[tuple(idx)], n


STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def _absorb_enclosed(cellids):
    """Absorb every group of cells fully enclosed by one grain into that grain.

    A grain containing an island has a boundary made of two loops. Neper's
    walk (see `make_meshable`) finishes the outer one and then has edges left
    over. The island itself is degenerate too: the ring around it carries no
    triple junction, so no vertex is created on it and its edge ends up with
    none.
    """
    from scipy.ndimage import label

    absorbed = []
    for cell in range(1, int(cellids.max()) + 1):
        # the complement of one grain, in the 4-connectivity of the voxel faces
        # the reconstruction works with; a component of it that does not reach
        # the map border is enclosed by that grain
        comp, n = label(cellids != cell, structure=STRUCT4)
        if n <= 1:
            continue
        border = set(comp[0]) | set(comp[-1]) | set(comp[:, 0]) | set(comp[:, -1])
        inside = [k for k in range(1, n + 1) if k not in border]
        if inside:
            mask = np.isin(comp, inside)
            absorbed += sorted(set(cellids[mask].tolist()))
            cellids[mask] = cell
    return absorbed


def _pinch_nodes(cellids):
    """Nodes where one grain occupies both pixels of a diagonal and neither of
    the other two, i.e. it touches itself at a point.

    Its boundary is then a figure-eight through that node rather than a loop,
    which breaks the same walk. The grain need not be in two pieces for this:
    an arm folding back on itself pinches while staying connected elsewhere.
    Pairs of pixels are returned, one pair per node.
    """
    a, b = cellids[:-1, :-1], cellids[:-1, 1:]
    c, d = cellids[1:, :-1], cellids[1:, 1:]
    out = []
    for mask, off in (
        ((a == d) & (a != b) & (a != c), ((0, 0), (1, 1))),
        ((b == c) & (b != a) & (b != d), ((0, 1), (1, 0))),
    ):
        ys, xs = np.nonzero(mask)
        out += [
            ((y + off[0][0], x + off[0][1]), (y + off[1][0], x + off[1][1]))
            for y, x in zip(ys.tolist(), xs.tolist())
        ]
    return out


def _unpinch(cellids):
    """Hand one pixel of each corner-only self-contact to a neighbouring grain.

    The pixel taken is whichever of the two is least attached to its own grain
    (a one-pixel spur left by the back-fill, usually), and it goes to whichever
    grain holds most of its four neighbours, so the change is one pixel per
    node and always in favour of an already-adjacent grain.
    """
    ny, nx = cellids.shape
    fixed = 0
    for p, q in _pinch_nodes(cellids):
        cell = cellids[p]
        if cellids[q] != cell:  # already resolved by an earlier fix
            continue

        def neighbours(pixel):
            y, x = pixel
            return [
                (y + dy, x + dx)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= y + dy < ny and 0 <= x + dx < nx
            ]

        victim = min(
            (p, q), key=lambda t: sum(cellids[n] == cell for n in neighbours(t))
        )
        others = [cellids[n] for n in neighbours(victim) if cellids[n] != cell]
        if not others:
            continue
        vals, counts = np.unique(others, return_counts=True)
        cellids[victim] = vals[np.argmax(counts)]
        fixed += 1
    return fixed


def make_meshable(cellids):
    """Remove the two raster configurations `neper -M` cannot reconstruct.

    Neper reconstructs the raster's interfaces into a vertex/edge/face topology
    and then orders each face's edges by walking them as a single closed loop
    (neut_tess_init_facetopo_fromver, in neut_tess_op1.c). Anything whose
    boundary is not one loop makes the walk run out of edges, and Neper aborts
    with `ut_print_neperbug()` right after "Reconstructing 0D mesh... 100%".
    Two things do that: a grain enclosing another (two loops) and a grain
    touching itself at a corner (a figure-eight).

    Both are fixed in one loop because each fix can create the other: absorbing
    an island can leave the host touching itself where the island had separated
    two arms, and handing away a pinched pixel can close a ring around a
    neighbour.

    What is lost is small and inert. An enclosed grain's boundary is a closed
    ring that meets nothing else, i.e. an isolated component of the grain
    boundary network (an enclosed pair gives a ring with one chord, likewise
    isolated), so removing it changes no connected path and leaves percolation
    untouched; a pinch is a one-pixel spur. The ids and counts are returned so
    the loss stays on the record.
    """
    cellids = cellids.copy()
    absorbed, pinches = [], 0
    for _ in range(20):
        n = _unpinch(cellids)
        new = _absorb_enclosed(cellids)
        pinches += n
        absorbed += new
        if not n and not new:
            break
    else:
        print(
            "  WARNING: enclosure/pinch cleanup did not converge; `neper -M` "
            "may still abort in neut_tess_init_facetopo_fromver"
        )

    ids = np.unique(cellids[cellids > 0])
    remap = np.zeros(int(cellids.max()) + 1, dtype=np.int64)
    remap[ids] = np.arange(1, len(ids) + 1)
    return remap[cellids], sorted(absorbed), pinches


def relabel_and_prune(labels, ok, min_pixels):
    """Drop tiny grains, then renumber what survives contiguously from 1.

    Neper requires the `**data` section to be numbered contiguously from 1, with
    0 for empty voxels. Pruned pixels become 0 and are treated exactly like
    unindexed ones -- the tessellation fit only uses cell boundaries, and the
    `grow` transform can fill the holes later if they matter.
    """
    flat = labels.ravel()
    valid = flat >= 0
    uniq, inv, counts = np.unique(flat[valid], return_inverse=True, return_counts=True)
    keep = counts >= min_pixels
    newid = np.zeros(len(uniq), dtype=np.int64)
    newid[keep] = np.arange(1, int(keep.sum()) + 1)

    out = np.zeros_like(flat, dtype=np.int64)
    out[valid] = newid[inv]
    out = out.reshape(labels.shape)
    dropped = int((~keep).sum())
    lost = int(counts[~keep].sum())
    return out, int(keep.sum()), dropped, lost


def grain_mean_orientations(qgrid, cellids, ncells, sym, chunk=50_000):
    """One orientation per grain: the symmetry-aligned quaternion mean.

    Each pixel is first mapped to the symmetry equivalent closest to its grain's
    reference orientation, otherwise the average of two equivalent descriptions
    of the same orientation is not that orientation.
    """
    flat_q = qgrid.reshape(-1, 4)
    flat_id = cellids.ravel()
    order = np.argsort(flat_id, kind="stable")
    sorted_id = flat_id[order]
    starts = np.searchsorted(sorted_id, np.arange(1, ncells + 1))
    ends = np.searchsorted(sorted_id, np.arange(1, ncells + 1), side="right")

    means = np.zeros((ncells, 4))
    for k in range(ncells):
        members = order[starts[k] : ends[k]]
        qs = flat_q[members]
        ref = qs[0]
        acc = np.zeros(4)
        for lo in range(0, len(qs), chunk):
            blk = qs[lo : lo + chunk]
            cand = crystal_equivalents(blk, sym)
            dots = cand @ ref
            best = np.argmax(np.abs(dots), axis=1)
            picked = cand[np.arange(len(blk)), best]
            sign = np.sign(dots[np.arange(len(blk)), best])
            sign[sign == 0] = 1.0
            acc += (picked * sign[:, None]).sum(axis=0)
        means[k] = acc / np.linalg.norm(acc)
    return to_fundamental_zone(means, sym)


# --- writing -----------------------------------------------------------------
def write_tesr(path, cellids, ori_cell, ori_vox, oridef, voxsize, crysym, precision=12):
    """Write the .tesr.

    Section layout follows the EBSD tutorial and the file-format reference:
    https://neper.info/doc/tutorials/ebsd_process.html
    https://neper.info/doc/fileformat.html

    Voxels run with x varying fastest, matching the 4 x 3 example in the
    tutorial where twelve `**data` values describe a map four wide.
    """
    ny, nx = cellids.shape
    fmt = f"%.{precision}f"

    with open(path, "w") as fh:
        fh.write("***tesr\n")
        fh.write(" **format\n   2.2\n")
        fh.write(" **general\n   2\n")
        fh.write(f"   {nx} {ny}\n")
        fh.write(f"   {voxsize[0]:.12g} {voxsize[1]:.12g}\n")

        ncells = int(cellids.max())
        fh.write(" **cell\n")
        fh.write(f"   {ncells}\n")
        fh.write("  *id\n")
        ids = np.arange(1, ncells + 1)
        for lo in range(0, ncells, 20):
            fh.write("   " + " ".join(str(i) for i in ids[lo : lo + 20]) + "\n")
        fh.write("  *crysym\n")
        fh.write(f"   {crysym}\n")
        fh.write("  *ori\n")
        fh.write("   rodrigues:passive\n")
        for r in ori_cell:
            fh.write("   " + " ".join(fmt % v for v in r) + "\n")

        fh.write(" **data\n   ascii\n")
        flat = cellids.ravel()
        for lo in range(0, flat.size, 40):
            fh.write(" ".join(str(int(v)) for v in flat[lo : lo + 40]) + "\n")

        if ori_vox is not None:
            fh.write(" **oridata\n   rodrigues:passive\n   ascii\n")
            for r in ori_vox.reshape(-1, 3):
                fh.write("   " + " ".join(fmt % v for v in r) + "\n")
            fh.write(" **oridef\n   ascii\n")
            flags = oridef.ravel().astype(int)
            for lo in range(0, flags.size, 60):
                fh.write(" ".join(str(v) for v in flags[lo : lo + 60]) + "\n")

        fh.write("***end\n")


def write_provenance(path, opt, seg, vox, log=print):
    """Everything needed to line the .tesr back up with the .ctf, plus the
    segmentation error, as json.

    The window, the mirror and the orientation convention are choices made in
    `Settings` and are not recoverable from the .tesr itself, so
    `measure_tesr_against_ctf` cannot re-measure the conversion without them.
    The statistics are copied in so that a later stage can quote stage 1's
    error without re-reading the .ctf. Dumping the whole dataclass means a
    field added to `Settings` reaches the file without a second edit here.
    """
    rec = asdict(opt)
    rec.update(
        {
            "ctf": str(opt.ctf),
            "unit": opt.unit,
            "voxsize": list(vox),
            "ncells": int(seg["ncells"]),
            "segmentation_error_deg": {
                k: seg[k] for k in ("all", "indexed", "backfilled") if k in seg
            },
        }
    )
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1)
    if log:
        log(f"  wrote {path}")
    return path


def settings_from_provenance(path):
    """Rebuild the `Settings` of a conversion from its provenance json."""
    rec = json.loads(Path(path).read_text())
    known = {f.name for f in dataclass_fields(Settings)}
    return Settings(**{k: v for k, v in rec.items() if k in known})


# --- rendered check images ---------------------------------------------------
def _run(cmd, cwd, log):
    """Run a Neper command, returning True on success and reporting on failure."""
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if out.returncode:
        tail = (out.stderr or out.stdout).strip().splitlines()[-3:]
        log(f"  WARNING: {Path(cmd[0]).name} {cmd[1]} failed: {' / '.join(tail)}")
    return out.returncode == 0


def render_checks(tesr, width, unit="um", neper="neper", povray="povray", log=print):
    """Render the written raster with neper -V. Returns the paths written.

    Two images, both of the .tesr this module just wrote and of nothing else,
    which is why they belong here rather than in the meshing stage:

      <stem>-ori.png     per-voxel orientation, IPF-Z, with the colour key
      <stem>-grains.png  cell ids in Neper's integer palette

    Look at these before trusting anything downstream: an inverted orientation
    convention shows up as IPF colours that disagree with AZtec or MTEX, and a
    bad segmentation as speckle or as obviously back-filled grains.

    -V colours by orientation but does not print the key
    (neper.info/tutorials/orientation_color_key.html), so the key is built the
    way that page documents -- tessellate the standard stereographic triangle,
    mesh it, read the node colours out with `-statnode col_stdtriangle` -- then
    pasted beside the map and deleted.

    Needs the neper binary and POV-Ray. A missing one is reported and skipped,
    since everything else the conversion produces is pure Python.
    """
    tesr = Path(tesr)
    work, stem = tesr.parent, tesr.stem
    if shutil.which(neper) is None and not Path(neper).is_file():
        log(f"  note: {neper!r} not found, skipping the rendered check images")
        return []

    written = []
    for name, opts in (
        ("ori", ["-datavoxcol", "ori", "-datavoxcolscheme", "ipf"]),
        ("grains", []),
    ):
        png = work / f"{stem}-{name}.png"
        cmd = [neper, "-V", tesr.name, "-povray", povray, *opts, "-print", png.stem]
        if not _run(cmd, work, log):
            continue
        # neper -V frames the flat map in the middle of a 3D canvas, so the
        # border comes off first; after that the image width *is* `width`
        annotate_png(png, width, unit, trim_border=True, log=log)
        written.append(png)

    ori = work / f"{stem}-ori.png"
    if ori in written:
        tri, key = f"{stem}-stdtriangle", work / f"{stem}-ipfkey.png"
        if (
            _run(
                [
                    neper,
                    "-T",
                    "-n",
                    "1",
                    "-domain",
                    "stdtriangle(20)",
                    "-dim",
                    "2",
                    "-o",
                    tri,
                ],
                work,
                log,
            )
            and _run(
                [
                    neper,
                    "-M",
                    f"{tri}.tess",
                    "-cl",
                    "0.02",
                    "-statnode",
                    "col_stdtriangle",
                ],
                work,
                log,
            )
            and _run(
                [
                    neper,
                    "-V",
                    f"{tri}.msh",
                    "-povray",
                    povray,
                    "-datanodecol",
                    f"col:file({tri}.stnode)",
                    "-dataeltcol",
                    "from_nodes",
                    "-dataelt2dedgerad",
                    "0",
                    "-dataelt1drad",
                    "0.001",
                    "-showelt1d",
                    "all",
                    "-imagesize",
                    "800:400",
                    "-print",
                    key.stem,
                ],
                work,
                log,
            )
        ):
            append_key(ori, key, log=log)
            key.unlink()
        for ext in (".tess", ".msh", ".stnode"):
            (work / (tri + ext)).unlink(missing_ok=True)
    return written


# --- diagnostics -------------------------------------------------------------
def verify_readback(path, qgrid, ok, cellids, flip_y, sym):
    """Re-read the written tesr, compare with what was meant, return a report.

    Three checks: the cell map is identical, `**oridef` is the quality mask,
    and every voxel orientation read back is the same rotation as the raw Euler
    triple. The file holds a fundamental-zone representative, so equality is
    only expected up to the symmetry group -- which is what a disorientation
    measures, hence ~0 rather than exactly 0.
    """
    back = read_tesr_full(path)
    exp_cells = cellids[::-1] if flip_y else cellids
    exp_ok = ok[::-1] if flip_y else ok
    exp_q = qgrid[::-1] if flip_y else qgrid
    same_cells = np.array_equal(back["cells"], exp_cells)
    report = [
        f"read-back: cell ids {'identical' if same_cells else 'DIFFER'} "
        f"({back['nx']} x {back['ny']} voxels)"
    ]
    if "vox_ori" in back:
        same_def = np.array_equal(back["oridef"], exp_ok)
        report.append(
            f"read-back: **oridef {'identical' if same_def else 'DIFFERS'} "
            "to the quality mask"
        )
        dis = cubic_disorientation_angle(
            qmul(qconj(exp_q.reshape(-1, 4)), rodrigues_to_quat(back["vox_ori"]))
        )
        report.append(
            f"read-back: voxel orientations vs raw Euler angles: max "
            f"{dis.max():.2e} deg, mean {dis.mean():.2e} deg over {dis.size} voxels"
            + ("" if dis.max() < 1e-3 else "  <-- NOT a round trip")
        )
    return report


def write_quality_png(
    path, diag, ok, unassigned, cellids, opt, vox, unit, flip_y=False, log=print
):
    """Three panels tracing every grey pixel of `neper -V ... -datavoxcol ori`.

    Neper paints a voxel grey where `**oridef` is 0, and that is written from
    the quality mask, so a grey pixel is one the .ctf's own columns failed. The
    panels are the MAD column (which is what the max_mad cut has to be chosen
    against), which test rejected each pixel, and which pixels the cell map
    back-filled because they were rejected or belonged to a pruned grain.
    """

    plt = use_agg()
    from matplotlib.colors import ListedColormap

    def orient(a):
        return a[::-1] if flip_y else a

    ny, nx = ok.shape
    mad, err, bands, phase = (diag.get(k) for k in ("MAD", "Error", "Bands", "Phase"))

    # rejection reason, first failing test wins
    reason = np.zeros((ny, nx), dtype=int)  # 0 kept
    if phase is not None:
        reason[(reason == 0) & (phase != opt.phase)] = 4
    if err is not None and not opt.allow_error:
        reason[(reason == 0) & (err != 0)] = 1
    if mad is not None:
        reason[(reason == 0) & (mad > opt.max_mad)] = 2
    if bands is not None and opt.min_bands:
        reason[(reason == 0) & (bands < opt.min_bands)] = 3
    reason[ok] = 0
    labels = ["kept", "Error != 0", f"MAD > {opt.max_mad:g}", "Bands < min", "phase"]
    counts = np.bincount(reason.ravel(), minlength=5)

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 2.4 + 5.2 * ny / nx), layout="constrained"
    )
    kw = dict(
        interpolation="nearest", origin="lower", extent=(0, nx * vox[0], 0, ny * vox[1])
    )

    ax = axes[0]
    if mad is not None:
        im = ax.imshow(
            orient(mad), cmap="viridis", vmin=0, vmax=max(opt.max_mad * 1.5, 1.0), **kw
        )
        fig.colorbar(im, ax=ax, fraction=0.046).set_label("MAD (deg)")
        ax.set_title(f"MAD: {int((mad > opt.max_mad).sum())} px over max_mad")
    else:
        ax.set_title("no MAD column")

    ax = axes[1]
    cmap = ListedColormap(["white", "tab:red", "tab:orange", "tab:purple", "tab:brown"])
    im = ax.imshow(orient(reason), cmap=cmap, vmin=-0.5, vmax=4.5, **kw)
    fig.colorbar(im, ax=ax, ticks=range(5), fraction=0.046).set_ticklabels(labels)
    ax.set_title(
        f"rejected (grey in -V): {int((~ok).sum())}/{ok.size} px "
        f"({100 * (~ok).mean():.1f} %)"
    )

    ax = axes[2]
    ncell = int(cellids.max())
    perm = np.random.default_rng(0).permutation(ncell) + 1
    shown = np.where(cellids > 0, perm[np.maximum(cellids - 1, 0)], np.nan)
    ax.imshow(orient(shown), cmap="tab20", **kw)
    filled = unassigned & (cellids > 0)
    ax.imshow(
        orient(np.where(filled, 1.0, np.nan)),
        cmap=ListedColormap(["black"]),
        alpha=0.45,
        **kw,
    )
    ax.set_title(
        f"{ncell} cells, {int(filled.sum())} px back-filled (dark), "
        f"{int((cellids == 0).sum())} empty"
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        scale_bar_ax(ax, nx * vox[0], unit)
    fig.suptitle(
        f"{Path(opt.ctf).name}: "
        + ", ".join(f"{labels[k]} {counts[k]}" for k in range(5) if counts[k])
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if log:
        log(f"  wrote {path}")
    return path


def convert(ctf_path=None, output=None, *, settings=None, log=print, **kwargs):
    """Convert a .ctf into a .tesr, and measure what the segmentation cost.

    Either give `ctf_path` plus any `Settings` field as a keyword argument, or
    build a `Settings` yourself and pass it as `settings`. `output` defaults to
    the .ctf's name with a .tesr suffix. `log` takes every progress line and can
    be set to None to run silently.

    Returns a dict describing the result: the paths written (`tesr`,
    `provenance`), the `settings` actually used, and the arrays a caller is
    likely to want next -- `cellids` and `ok` in the *written* (post flip_y)
    row order, `qcell` and `ori_cell`, the per-grain voxel counts `npx`, and
    the `segmentation_error` dict, whose "indexed" entry holds the headline
    RMS disorientation in degrees.

    Raises ValueError if the map has no cubic phase line or if no grain
    survives the prune, rather than exiting the interpreter.
    """
    opt = settings if settings is not None else Settings(ctf=ctf_path, **kwargs)
    if kwargs and settings is not None:
        raise TypeError("pass either a Settings object or keyword arguments")
    log = log or (lambda *a, **k: None)

    self_test()

    ctf = CtfMap(opt.ctf)
    ny, nx = ctf.shape
    log(
        f"{opt.ctf}: {nx} x {ny} pixels, step {ctf.header['XStep']} x "
        f"{ctf.header['YStep']}, {ctf.npoints} rows"
    )
    if not np.isclose(ctf.header["XStep"], ctf.header["YStep"]):
        log("  note: XStep != YStep -- voxels will not be square")

    crysym, phase = ctf.crysym(opt.phase)
    if crysym is None:
        raise ValueError("no phase line found; cannot determine crystal symmetry")
    if phase["laue"] not in CUBIC_LAUE:
        raise ValueError(
            f"phase '{phase['name']}' has Laue group {phase['laue']} -> {crysym}, "
            "which this script cannot segment. The disorientation used here is "
            "specific to the cubic group. Segment in MTEX and write the grain "
            "ids into the **data section instead."
        )
    log(f"  phase {opt.phase}: {phase['name']}, Laue {phase['laue']} -> {crysym}")

    sym = cubic_symmetry_quaternions()
    qgrid, ok, diag = build_grid(
        ctf, opt.phase, opt.max_mad, not opt.allow_error, opt.min_bands
    )
    window = (slice(0, ny), slice(0, nx))
    if opt.crop:
        qgrid, ok, window = crop_grid(
            qgrid, ok, opt.crop, ctf.header["XStep"], ctf.header["YStep"]
        )
        diag = {k: v[window] for k, v in diag.items()}
        ny, nx = ok.shape
        log(f"  cropped to {nx} x {ny} pixels ({opt.crop})")
    frac = ok.mean()
    log(f"indexed & above quality cutoffs: {ok.sum()} of {ok.size} ({100 * frac:.1f}%)")
    if frac < 0.8:
        log(
            "  WARNING: a large fraction of the map was rejected. Loosen "
            "max_mad or set allow_error, or expect a holed tessellation"
        )

    labels = segment_grains(qgrid, ok, opt.threshold, sym)
    cellids, ncells, dropped, lost = relabel_and_prune(labels, ok, opt.min_pixels)
    log(
        f"  grains at {opt.threshold:g} deg: {ncells} "
        f"({dropped} below {opt.min_pixels} px dropped, {lost} px)"
    )
    if ncells == 0:
        raise ValueError("no grains survived; lower min_pixels or threshold")

    # Degenerate cells are what abort `neper -T -n from_morpho` partway through
    # "Listing cell voxels", and the objective function cannot place
    # pts(res=N) control points on a cell two pixels across either. Report the
    # bottom of the distribution so the failure is visible here, not there.
    unassigned = cellids == 0
    empty_before = int(unassigned.sum())
    log(
        f"  unassigned voxels: {empty_before} of {cellids.size} "
        f"({100 * empty_before / cellids.size:.1f} %)"
    )
    if opt.fill:
        cellids, filled = fill_holes(cellids)
        log(f"  filled {filled} voxels from the nearest cell")
    elif empty_before > 0.05 * cellids.size:
        log(
            "  WARNING: the raster has substantial holes and fill=False was "
            "given. The tessellation fit will treat hole boundaries as grain "
            "boundaries"
        )

    if opt.topology_fix:
        cellids, absorbed, pinches = make_meshable(cellids)
        ncells = int(cellids.max())
        if absorbed:
            log(
                f"  absorbed {len(absorbed)} enclosed grain(s) into their "
                f"surrounding grain: {', '.join(map(str, absorbed))}"
            )
        if pinches:
            log(f"  unpinched {pinches} corner-only self-contact(s), 1 px each")
        if absorbed or pinches:
            log(f"  grains: {ncells}")

    counts = np.bincount(cellids.ravel())[1:]
    log(f"  smallest grains (px): {', '.join(str(c) for c in np.sort(counts)[:8])}")
    if counts.min() < 10:
        log(
            f"  WARNING: {int((counts < 10).sum())} grains under 10 px. Neper's "
            "tessellation fit is liable to abort on these -- raise "
            "min_pixels (20 is a reasonable floor)"
        )

    qfz = to_fundamental_zone(qgrid.reshape(-1, 4), sym).reshape(ny, nx, 4)
    qcell = grain_mean_orientations(qgrid, cellids, ncells, sym)
    ori_cell = quat_to_rodrigues(qcell)
    ori_vox = None if not opt.voxel_ori else quat_to_rodrigues(qfz)
    vox = (ctf.header["XStep"] * opt.scale, ctf.header["YStep"] * opt.scale)

    # What the segmentation cost. Computed here, before --active flips the sign
    # and --flip-y mirrors the arrays, so everything is still in the .ctf's own
    # frame; neither transformation changes a disorientation. This is the
    # headline quality number for stage 1 of the pipeline: the RMS angle
    # between a pixel's measured orientation and the single orientation its
    # grain will carry from here on.
    seg = segmentation_error(
        qgrid,
        cellids,
        qcell,
        vox,
        ok=ok,
        threshold=opt.threshold,
        backfilled=unassigned,
    )
    for line in format_report(seg):
        log("  " + line)

    if opt.active:  # active is the opposite rotation, i.e. -r
        ori_cell = -ori_cell
        if ori_vox is not None:
            ori_vox = -ori_vox

    if opt.flip_y:
        cellids = cellids[::-1]
        ok = ok[::-1]
        if ori_vox is not None:
            ori_vox = ori_vox[::-1]

    out = Path(output or Path(opt.ctf).with_suffix(".tesr"))
    write_tesr(out, cellids, ori_cell, ori_vox, ok, vox, crysym)
    write_provenance(
        out.with_name(out.stem + "-provenance.json"), opt, seg, vox, log=log
    )

    if opt.diagnostics:
        unit = opt.unit
        write_segerr_png(
            out.with_name(out.stem + "-segerror.png"),
            seg,
            # back to the .ctf's row order: `cellids` was mirrored in place
            # above, while seg["theta"] was measured before that
            cellids[::-1] if opt.flip_y else cellids,
            unit=unit,
            threshold=opt.threshold,
            log=log,
        )
        png = out.with_name(out.stem + "-quality.png")
        write_quality_png(
            png,
            diag,
            ok,
            unassigned,
            cellids,
            opt,
            vox,
            unit,
            flip_y=opt.flip_y,
            log=log,
        )
        for line in verify_readback(out, qgrid, ok, cellids, opt.flip_y, sym):
            log(f"  {line}")

    lx, ly = nx * vox[0], ny * vox[1]
    if opt.diagnostics and opt.neper:
        render_checks(
            out, lx, unit=opt.unit, neper=opt.neper, povray=opt.povray, log=log
        )
    mean_px = counts.mean()
    grain_size = np.sqrt(mean_px) * vox[0]
    log(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    log(f"  domain      : {lx:.4g} x {ly:.4g}")
    log(f"  grain size  : ~{grain_size:.4g} (equivalent square)")
    if not opt.diagnostics:
        log(f"  set diagnostics=True for {out.stem}-quality.png and the rest")
    if ncells > 400:
        side = grain_size * np.sqrt(250)
        log(
            f"\n{ncells} grains is a long tessellation fit. Crop to ~250 by "
            f"re-running with a window about {side:.3g} x {side:.3g} in the "
            f'.ctf\'s units, i.e. crop="x0,x0+{side:.3g},y0,y0+{side:.3g}"'
        )

    return {
        "tesr": out,
        "provenance": out.with_name(out.stem + "-provenance.json"),
        "settings": opt,
        "crysym": crysym,
        "ncells": ncells,
        "voxsize": vox,
        "extent": (lx, ly),
        "cellids": cellids,
        "ok": ok,
        "ori_cell": ori_cell,
        "qcell": qcell,
        "npx": counts,
        "window": window,
        "segmentation_error": seg,
    }


def measure_tesr_against_ctf(
    ctf_path,
    tesr_path,
    settings=None,
    provenance=None,
    against="grain",
    png=None,
    csv=None,
    log=print,
):
    """Re-measure the segmentation error of a .ctf/.tesr pair already on disk.

    `convert` reports this as it goes; this is the same measurement made
    afterwards, straight off the two files, so it also verifies that what was
    written is what was meant. Give the conversion's `Settings` (or
    `provenance`, the json path `convert` wrote) so the crop window, the
    mirror and the orientation convention are known -- none of them is
    recoverable from the .tesr alone.

    `against` selects what each pixel is compared with: "grain" (default) uses
    the **cell/*ori orientation of the grain the pixel was put in, which is the
    segmentation error and the only orientation the rest of the pipeline sees;
    "voxel" uses the pixel's own **oridata entry, which measures transcription
    only and should come back at the 1e-6 deg noise floor; "both" reports each.

    Returns the result dict from `segmentation_error`.
    """
    log = log or (lambda *a, **k: None)
    if provenance is not None:
        settings = settings_from_provenance(provenance)
    opt = settings or Settings(ctf=str(ctf_path))

    ctf = CtfMap(ctf_path)
    crysym, phase = ctf.crysym(opt.phase)
    if phase is None or phase["laue"] not in CUBIC_LAUE:
        raise ValueError(
            "cubic phases only: the disorientation used here is cubic-specific"
        )
    qgrid, ok, _diag = build_grid(
        ctf, opt.phase, opt.max_mad, not opt.allow_error, opt.min_bands
    )
    if opt.crop:
        qgrid, ok, _w = crop_grid(
            qgrid, ok, opt.crop, ctf.header["XStep"], ctf.header["YStep"]
        )

    t = read_tesr_full(tesr_path)
    if (t["ny"], t["nx"]) != ok.shape:
        raise ValueError(
            f"{tesr_path} is {t['nx']} x {t['ny']} voxels but the .ctf window "
            f"is {ok.shape[1]} x {ok.shape[0]}. Pass the Settings (or the "
            "provenance json) of the conversion that wrote it."
        )
    cells = t["cells"][::-1] if opt.flip_y else t["cells"]
    sign = -1.0 if opt.active else 1.0
    qcell = rodrigues_to_quat(sign * t["cell_ori"])

    qvox = None
    if against in ("voxel", "both"):
        if "vox_ori" in t:
            r = t["vox_ori"][::-1] if opt.flip_y else t["vox_ori"]
            qvox = rodrigues_to_quat(sign * r).reshape((*ok.shape, 4))
        else:
            log("  note: no **oridata in the tesr (voxel_ori=False); check skipped")

    res = segmentation_error(
        qgrid, cells, qcell, t["vox"], ok=ok, threshold=opt.threshold, qvox=qvox
    )
    for line in format_report(res):
        log("  " + line)

    # `convert` knows which voxels fill_holes back-filled, including the ones
    # whose grain was pruned by min_pixels; working from the files alone only
    # the quality rejections are visible, so the two populations differ by the
    # pruned pixels and the numbers differ with them. Say so rather than leave
    # two unequal RMS values lying about.
    ref = {}
    if provenance is not None:
        ref = json.loads(Path(provenance).read_text())
        ref = ref.get("segmentation_error_deg", {}).get("indexed", {})
    if ref and ref.get("n") != res["indexed"]["n"]:
        log(
            f"  note: convert() measured {ref['rms']:.3f} deg over {ref['n']} "
            f"voxels. The {res['indexed']['n'] - ref['n']} extra voxels here "
            "belonged to grains the prune removed and were then back-filled; "
            "only the conversion can tell them apart from voxels in a grain of "
            "their own."
        )

    if png:
        write_segerr_png(
            png, res, cells, unit=opt.unit, threshold=opt.threshold, log=log
        )
    return res

"""Convert an Oxford/Channel .ctf EBSD map into a Neper Raster Tessellation
File (.tesr), including the grain segmentation that a .ctf does not contain.

There is no official converter, one simply needs to do this by hand.

The part that is not a transcription is the segmentation. A .ctf holds an
orientation per pixel and nothing else; `neper -T -morpho tesr` fits its cells
to the *cells* of the raster, so the file needs a `**cell` section and a
`**data` section assigning every pixel to a grain. Grains are found here by
flood-filling across neighbouring pixels whose disorientation is below a
threshold, with cubic crystal symmetry taken into account.

Example usage:

    python ctf_to_tesr.py map.ctf -o ebsd.tesr

then point the transport pipeline at the result:

    TESR = "ebsd.tesr"   in festim_ebsd_gb_diffusion.py

What is written
---------------
**general   dimension, XCells/YCells, XStep/YStep (in microns by default)
**cell      grain count, ids, crysym, and one mean orientation per grain
**data      grain id of every pixel, contiguous from 1, 0 where unindexed
**oridata   per-pixel orientation (optional; large, but needed for -V and GOS)
**oridef    per-pixel indexing flag, 0 where the point was rejected

Length unit
-----------
The tesr is written in the .ctf's own unit (microns) unless --scale is given.
The transport driver converts to metres after reading
the mesh (TESR_UNIT in ebsd_gb_diffusion.py).

Orientation convention
----------------------
Euler angles in a .ctf are Bunge (phi1, Phi, phi2) in degrees, referred to the
sample frame. They are converted to Rodrigues vectors under Neper's default
`passive` convention -- the rotation of the sample coordinate system into the
crystal one. The file is written as tesr format 2.2: Neper 4.10.0 swapped the
meaning of `active` and `passive` and bumped the tesr version to 2.2, and a
file declaring 2.1 has its `**cell/*ori` descriptor silently flipped on read
(neut_tesr_fscanf2.c, "Fixing orientation convention") while `**oridata` is
taken literally, leaving the two sections in opposite conventions. The
conversion is verified against Neper's own convention table,
which gives Bunge (0, 30, 0) as Rodrigues (0.267949192, 0, 0) and quaternion
(0.965925826, 0.258819045, 0, 0); see the self-test at the bottom.

Rodrigues is used rather than passing the Euler angles through unchanged
because it removes any degrees-versus-radians ambiguity in the tesr reader.
Orientations are reduced to the cubic fundamental zone first, which also keeps
the Rodrigues vector finite (a 180-degree rotation has none).

Because the crystal symmetry is declared in the file, any symmetry-equivalent
representative is equally correct -- Neper applies the symmetry itself. What
is *not* free is the active/passive choice, so verify the output visually:

    neper -V ebsd.tesr -datavoxcol ori -datavoxcolscheme ipf -print check

and compare against the same map plotted in AZtec or MTEX. If the colours are
wrong in a way that looks like an inversion, re-run with --active.

Limitations
-----------
* Cubic symmetry only. The disorientation uses a closed form specific to the
  cubic group. For hex or lower, segment in MTEX and write out grain ids instead.
* Square grids only, which is what JobMode Grid with XStep == YStep gives. A
  hexagonal acquisition must be resampled first (MTEX's `gridify`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

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


# --- quaternion helpers ------------------------------------------------------
def cubic_symmetry_quaternions():
    """The 24 rotations of the cubic group, as unit quaternions.

    Nine 90/180/270-degree rotations about the <100> axes, six 180-degree
    rotations about the <110> axes, and eight 120/240-degree rotations about
    the <111> axes, plus the identity.
    """
    r = np.sqrt(0.5)
    q = [(1.0, 0.0, 0.0, 0.0)]
    for axis in range(3):
        for w, s in ((r, r), (0.0, 1.0), (r, -r)):
            v = [0.0, 0.0, 0.0]
            v[axis] = s
            q.append((w, *v))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        for sign in (1.0, -1.0):
            v = [0.0, 0.0, 0.0]
            v[i], v[j] = r, sign * r
            q.append((0.0, *v))
    for sx in (0.5, -0.5):
        for sy in (0.5, -0.5):
            for sz in (0.5, -0.5):
                q.append((0.5, sx, sy, sz))
    out = np.array(q, dtype=float)
    assert out.shape == (24, 4), out.shape
    return out


def qmul(a, b):
    """Hamilton product, broadcasting over leading axes."""
    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        (
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
        ),
        axis=-1,
    )


def qconj(q):
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def euler_bunge_to_quat(phi1, Phi, phi2, degrees=True):
    """Bunge Euler angles -> unit quaternion, passive convention.

    Passive here means what Neper means by it: the rotation carrying the sample
    coordinate system onto the crystal coordinate system, which is the standard
    reading of Bunge angles and what a .ctf stores.
    """
    if degrees:
        phi1, Phi, phi2 = np.radians(phi1), np.radians(Phi), np.radians(phi2)
    sigma = 0.5 * (phi1 + phi2)
    delta = 0.5 * (phi1 - phi2)
    c, s = np.cos(0.5 * Phi), np.sin(0.5 * Phi)
    q = np.stack(
        (c * np.cos(sigma), s * np.cos(delta), s * np.sin(delta), c * np.sin(sigma)),
        axis=-1,
    )
    # a quaternion and its negative are the same rotation; fix the sign so that
    # averaging and fundamental-zone reduction are well defined
    return np.where(q[..., :1] < 0, -q, q)


def crystal_equivalents(q, sym):
    """All symmetry-equivalent descriptions of the orientations q, (n, 24, 4).

    The crystal symmetry multiplies on the *right* in this quaternion
    convention: q maps sample to crystal (Bunge, passive), so a symmetry
    operator S, which relabels crystal axes, composes as q * S. Multiplying on
    the left, S * q, would instead rotate the sample frame and yield a
    physically different orientation. Checked against Neper: for a cell pair
    (q, q*S) `-statedge theta` under -crysym cubic is 0; for (q, S*q) it is
    17 degrees.
    """
    return qmul(q[:, None, :], sym[None, :, :])


def to_fundamental_zone(q, sym, chunk=50_000):
    """Pick, for each orientation, the symmetry equivalent closest to identity.

    Any equivalent is as correct as any other -- the crysym is declared in the
    tesr and Neper applies the symmetry itself. The point of choosing this one
    is that its rotation angle is at most ~62.8 degrees for cubic, so the
    scalar part never approaches zero and the Rodrigues vector stays finite.
    """
    out = np.empty_like(q)
    for lo in range(0, len(q), chunk):
        blk = q[lo : lo + chunk]
        cand = crystal_equivalents(blk, sym)  # (n, 24, 4)
        best = np.argmax(np.abs(cand[..., 0]), axis=1)
        picked = cand[np.arange(len(blk)), best]
        out[lo : lo + chunk] = np.where(picked[..., :1] < 0, -picked, picked)
    return out


def quat_to_rodrigues(q):
    """Rodrigues vector = (q1, q2, q3) / q0. Requires q already in the FZ."""
    q0 = q[..., :1]
    if np.any(np.abs(q0) < 1e-8):
        raise ValueError("scalar part near zero; reduce to the fundamental zone first")
    return q[..., 1:] / q0


def cubic_disorientation_angle(m):
    """Disorientation angle (degrees) of a cubic misorientation quaternion.

    Closed form rather than a search over 24 x 24 symmetry pairs: with the
    absolute components sorted descending as a >= b >= c >= d, the largest
    attainable cos(omega/2) over the cubic group is

        max( a, (a + b)/sqrt(2), (a + b + c + d)/2 )

    which is the standard result for the cubic misorientation function (Grimmer,
    Acta Cryst. A36 (1980) 382). Checked in the self-test against two cases with
    known answers: a 90-degree rotation about <100> (a symmetry operation, so
    zero) and a 60-degree rotation about <111> (the Sigma-3 twin).
    """
    s = np.sort(np.abs(m), axis=-1)[..., ::-1]
    a, b, c, d = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    best = np.maximum.reduce([a, (a + b) / np.sqrt(2.0), 0.5 * (a + b + c + d)])
    return np.degrees(2.0 * np.arccos(np.clip(best, -1.0, 1.0)))


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

    # Per-pixel provenance for --diagnostics: what the .ctf itself says about
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
    i.e. before any --flip-y.
    """
    try:
        x0, x1, y0, y1 = (float(v) for v in spec.split(","))
    except ValueError:
        raise SystemExit(
            f"--crop {spec!r}: expected four comma-separated numbers, "
            "xmin,xmax,ymin,ymax, in the same units as XStep"
        )
    ny, nx = ok.shape
    ix0, ix1 = max(round(x0 / xstep), 0), min(round(x1 / xstep), nx)
    iy0, iy1 = max(round(y0 / ystep), 0), min(round(y1 / ystep), ny)
    if ix1 - ix0 < 2 or iy1 - iy0 < 2:
        raise SystemExit(
            f"--crop {spec} keeps {max(ix1 - ix0, 0)} x {max(iy1 - iy0, 0)} "
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


# --- diagnostics -------------------------------------------------------------
def _scale_bar(ax, width, unit):
    """Scale bar from micrograph.py if it sits next to this script; else none."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from micrograph import scale_bar_ax
    except ImportError:
        return
    scale_bar_ax(ax, width, unit)


def ipf_z_colours(q):
    """Simplified IPF-Z colouring of quaternions (n, 4), cubic symmetry.

    The crystal-frame direction of the sample z axis is taken to the standard
    triangle by |components| sorted so that k <= h <= l, and coloured
    (l - h, h - k, k) normalised to unit maximum: [001] red, [101] green,
    [111] blue, as in the usual IPF key. It is a simplified scheme -- AZtec and
    neper -V use their own angle-based interpolations, so shades differ from
    check-ori.png -- but the raw and read-back panels use the *same* scheme,
    which is what makes them comparable.
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # The sample Z axis in crystal coordinates is g e_z with g the Bunge
    # sample->crystal matrix. For this quaternion convention the standard
    # quaternion->matrix formula gives R(q) = g^T (self_test checks it), so the
    # vector wanted is the third *row* of R(q). The third column would be the
    # crystal [001] axis in sample coordinates -- a valid colouring, but not
    # an IPF-Z, and not what neper -V's ipf scheme shows.
    d = np.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), 1)
    d = np.sort(np.abs(d), axis=1)  # k <= h <= l
    k, h, l = d[:, 0], d[:, 1], d[:, 2]  # noqa: E741
    rgb = np.stack((l - h, h - k, k), 1)
    rgb /= np.maximum(rgb.max(axis=1, keepdims=True), 1e-12)
    return rgb


def read_tesr_back(path):
    """
    Minimal reader for what this script writes:
    header, **data, **oridata, **oridef.
    """
    with open(path) as fh:
        tok = fh.read().split()
    i = tok.index("**general")
    nx, ny = int(tok[i + 2]), int(tok[i + 3])
    vox = (float(tok[i + 4]), float(tok[i + 5]))
    n = nx * ny
    out = {"nx": nx, "ny": ny, "vox": vox}
    i = tok.index("**data")
    out["cells"] = np.array(tok[i + 2 : i + 2 + n], dtype=int).reshape(ny, nx)
    if "**oridata" in tok:
        i = tok.index("**oridata")
        out["orides"] = tok[i + 1]
        r = np.array(tok[i + 3 : i + 3 + 3 * n], dtype=float).reshape(n, 3)
        qq = np.column_stack((np.ones(n), r))
        out["quat"] = (qq / np.linalg.norm(qq, axis=1)[:, None]).reshape(ny, nx, 4)
        i = tok.index("**oridef")
        out["oridef"] = (
            np.array(tok[i + 2 : i + 2 + n], dtype=int).reshape(ny, nx).astype(bool)
        )
    return out


def verify_readback(path, qgrid, ok, cellids, flip_y, sym):
    """Re-read the written tesr and compare with what was meant to be written.

    Three checks: the cell map is identical, `**oridef` is the quality mask,
    and every voxel orientation read back is the same rotation as the raw
    Euler triple (disorientation under cubic symmetry ~ 0; the file holds a
    symmetry-equivalent, fundamental-zone representative, so identity is only
    expected up to the symmetry group, which is what the disorientation
    measures).
    """
    back = read_tesr_back(path)
    exp_cells = cellids[::-1] if flip_y else cellids
    exp_ok = ok[::-1] if flip_y else ok
    exp_q = qgrid[::-1] if flip_y else qgrid
    report = []
    same_cells = np.array_equal(back["cells"], exp_cells)
    report.append(
        f"read-back: cell ids {'identical' if same_cells else 'DIFFER'} "
        f"({back['nx']} x {back['ny']} voxels)"
    )
    result = {"cells_ok": same_cells, "dis": None, "oridef_ok": None}
    if "quat" in back:
        same_def = np.array_equal(back["oridef"], exp_ok)
        result["oridef_ok"] = same_def
        report.append(
            f"read-back: **oridef \
                {'identical' if same_def else 'DIFFERS'} to the quality mask"
        )
        dis = cubic_disorientation_angle(
            qmul(qconj(exp_q.reshape(-1, 4)), back["quat"].reshape(-1, 4))
        ).reshape(exp_q.shape[:2])
        result["dis"] = dis[::-1] if flip_y else dis
        report.append(
            f"read-back: voxel orientations vs raw Euler angles: max "
            f"{dis.max():.2e} deg, mean {dis.mean():.2e} deg over {dis.size} voxels"
            + ("" if dis.max() < 1e-3 else "  <-- NOT a round trip")
        )
    result["report"] = report
    return result


def write_raw_png(
    path, qgrid_full, ok_full, window, qgrid, ok, check, ctf, args, vox, unit
):
    """<output>-raw.png: the .ctf as read, the window as written, and their difference.

    (a) the whole map, IPF-Z from the raw Euler angles, with the crop window
    drawn; (b) the window from the raw Euler angles; (c) the window as read
    back from the .tesr just written, same colouring, `**oridef` = 0 in grey;
    (d) the per-voxel disorientation between (b) and (c), which is the
    quantitative statement that nothing was lost or altered in the reading
    and writing -- it should be ~1e-12 degrees everywhere.
    """

    NY, NX = ok_full.shape
    ny, nx = ok.shape
    xs, ys = ctf.header["XStep"] * args.scale, ctf.header["YStep"] * args.scale

    full_rgb = ipf_z_colours(qgrid_full.reshape(-1, 4)).reshape(NY, NX, 3)
    full_rgb[~ok_full] = 0.6
    win_rgb = ipf_z_colours(qgrid.reshape(-1, 4)).reshape(ny, nx, 3)
    win_rgb[~ok] = 0.6

    fig, axes = plt.subplots(2, 2, figsize=(12, 5.5 + 5.5 * ny / nx))
    kw_full = dict(
        interpolation="nearest", origin="lower", extent=(0, NX * xs, 0, NY * ys)
    )
    kw = dict(
        interpolation="nearest", origin="lower", extent=(0, nx * vox[0], 0, ny * vox[1])
    )

    ax = axes[0, 0]
    ax.imshow(full_rgb, **kw_full)
    y0, y1 = window[0].start, window[0].stop
    x0, x1 = window[1].start, window[1].stop
    ax.add_patch(
        Rectangle(
            (x0 * xs, y0 * ys),
            (x1 - x0) * xs,
            (y1 - y0) * ys,
            fill=False,
            ec="white",
            lw=1.5,
        )
    )
    ax.set_title(
        f"whole .ctf: {NX} x {NY} px, {ctf.npoints} rows read of {NX * NY}\n"
        f"grey = rejected ({100 * (~ok_full).mean():.1f} %), box = window"
    )
    _scale_bar(ax, NX * xs, unit)

    ax = axes[0, 1]
    ax.imshow(win_rgb, **kw)
    ax.set_title("window, raw Euler angles (simplified IPF-Z)")
    _scale_bar(ax, nx * vox[0], unit)

    ax = axes[1, 0]
    if check["dis"] is not None:
        back = read_tesr_back(args.output or str(Path(args.ctf).with_suffix(".tesr")))
        q_back = back["quat"][::-1] if args.flip_y else back["quat"]
        def_back = back["oridef"][::-1] if args.flip_y else back["oridef"]
        back_rgb = ipf_z_colours(q_back.reshape(-1, 4)).reshape(ny, nx, 3)
        back_rgb[~def_back] = 0.6
        ax.imshow(back_rgb, **kw)
        ax.set_title(
            f"window read back from \
                {Path(args.output).name if args.output else 'tesr'} ({back['orides']})"
        )
    else:
        ax.set_title("no **oridata in the tesr (--no-voxel-ori)")
    _scale_bar(ax, nx * vox[0], unit)

    ax = axes[1, 1]
    if check["dis"] is not None:
        # floor the scale at 1e-3 deg so an exact round trip shows as a flat
        # black panel and any real discrepancy stands out
        im = ax.imshow(
            check["dis"], cmap="magma", vmin=0, vmax=max(check["dis"].max(), 1e-3), **kw
        )
        fig.colorbar(im, ax=ax, fraction=0.046).set_label(
            "disorientation raw vs read back (deg)"
        )
        ax.set_title(
            f"max {check['dis'].max():.1e} deg: \
                {'round trip exact' if check['dis'].max() < 1e-3 else 'MISMATCH'}"
        )
    _scale_bar(ax, nx * vox[0], unit)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def write_quality_png(
    path, diag, ok, unassigned, cellids, args, vox, unit, flip_y=False
):
    """Four panels tracing every grey pixel of `neper -V ... -datavoxcol ori`.

    Neper paints a voxel grey when `**oridef` is 0, and this script writes
    `**oridef` from the quality mask `ok` -- so a grey pixel is a point the
    .ctf's own columns failed: Error != 0 (unless --allow-error), MAD above
    --max-mad, Bands below --min-bands, or the wrong phase. The panels show:
    Error column, MAD column, rejection reason, and which pixels of the
    final cell map were back-filled from the nearest surviving cell
    because they were rejected or belonged to a pruned grain.
    """

    def orient(a):
        return a[::-1] if flip_y else a

    ny, nx = ok.shape
    err = diag.get("Error")
    mad = diag.get("MAD")
    bands = diag.get("Bands")
    phase = diag.get("Phase")

    # rejection reason, first failing test wins
    reason = np.zeros((ny, nx), dtype=int)  # 0 kept
    if phase is not None:
        reason[(reason == 0) & (phase != args.phase)] = 4
    if err is not None and not args.allow_error:
        reason[(reason == 0) & (err != 0)] = 1
    if mad is not None:
        reason[(reason == 0) & (mad > args.max_mad)] = 2
    if bands is not None and args.min_bands:
        reason[(reason == 0) & (bands < args.min_bands)] = 3
    reason[ok] = 0
    labels = [
        "kept",
        "Error != 0",
        f"MAD > {args.max_mad:g}",
        "Bands < min",
        "other phase",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 10 * ny / nx))
    kw = dict(
        interpolation="nearest",
        origin="lower",
        extent=(0, nx * vox[0], 0, ny * vox[1]),
    )

    ax = axes[0, 0]
    if err is not None:
        codes = np.unique(err[err >= 0]).astype(int)
        im = ax.imshow(orient(err), cmap="tab10", vmin=-0.5, vmax=9.5, **kw)
        cb = fig.colorbar(im, ax=ax, ticks=codes, fraction=0.046)
        cb.set_label("Error column (0 = indexed, 3 = no solution)")
        ax.set_title(
            "ctf Error code: "
            + ", ".join(f"{c}: {int((err == c).sum())}" for c in codes)
        )
    else:
        ax.set_title("no Error column")

    ax = axes[0, 1]
    if mad is not None:
        im = ax.imshow(
            orient(mad), cmap="viridis", vmin=0, vmax=max(args.max_mad * 1.5, 1.0), **kw
        )
        fig.colorbar(im, ax=ax, fraction=0.046).set_label("MAD (deg)")
        over = int((mad > args.max_mad).sum())
        ax.set_title(f"ctf MAD: {over} px above --max-mad {args.max_mad:g}")
    else:
        ax.set_title("no MAD column")

    ax = axes[1, 0]
    cmap = ListedColormap(["white", "tab:red", "tab:orange", "tab:purple", "tab:brown"])
    im = ax.imshow(orient(reason), cmap=cmap, vmin=-0.5, vmax=4.5, **kw)
    cb = fig.colorbar(im, ax=ax, ticks=range(5), fraction=0.046)
    cb.set_ticklabels(labels)
    counts = np.bincount(reason.ravel(), minlength=5)
    ax.set_title(
        f"rejected (grey in neper -V): {int((~ok).sum())} of {ok.size} px "
        f"({100 * (~ok).mean():.1f} %)"
    )

    ax = axes[1, 1]
    ncell = int(cellids.max())
    perm = np.random.default_rng(0).permutation(ncell) + 1
    shown = np.where(cellids > 0, perm[np.maximum(cellids - 1, 0)], 0).astype(float)
    shown[cellids == 0] = np.nan
    ax.imshow(orient(shown), cmap="tab20", **kw)
    filled = unassigned & (cellids > 0)
    ax.imshow(
        orient(np.where(filled, 1.0, np.nan)),
        cmap=ListedColormap(["black"]),
        alpha=0.45,
        **kw,
    )
    ax.set_title(
        f"{ncell} cells; {int(filled.sum())} px back-filled (dark), "
        f"{int((cellids == 0).sum())} left empty"
    )

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        _scale_bar(ax, nx * vox[0], unit)
    fig.suptitle(
        f"{Path(args.ctf).name}: \
            {', '.join(f'{labels[k]} {counts[k]}' for k in range(5) if counts[k])}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


# --- self-test ---------------------------------------------------------------
def self_test():
    """Check the conventions against values Neper publishes, on every run.

    Neper's orientation-convention table gives, for a 30-degree rotation about
    the sample x axis under the passive convention, Bunge angles (0, 30, 0),
    quaternion (0.965925826, 0.258819045, 0, 0) and Rodrigues vector
    (0.267949192, 0, 0). If this assertion ever fires, the Euler conversion has
    drifted away from what the tesr reader will assume.
    """
    sym = cubic_symmetry_quaternions()
    q = euler_bunge_to_quat(np.array([0.0]), np.array([30.0]), np.array([0.0]))
    assert np.allclose(q[0], [0.965925826, 0.258819045, 0, 0], atol=1e-8), q
    r = quat_to_rodrigues(to_fundamental_zone(q, sym))
    assert np.allclose(r[0], [0.267949192, 0, 0], atol=1e-8), r

    # 90 degrees about z is a cubic symmetry operation -> disorientation 0.
    # The tolerance is loose because arccos has an infinite derivative at 1, so
    # a rounding error of 1e-16 in the argument surfaces as ~1e-6 degrees. That
    # amplification is harmless at a 10-degree segmentation threshold but it is
    # the reason not to compare disorientations to exact zero anywhere.
    q90 = np.array([[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]])
    assert cubic_disorientation_angle(q90)[0] < 1e-4
    # R(q) = g^T for Bunge (0, 30, 0): the IPF-Z direction is the third row
    w, x, y, z = q[0]
    third_row = np.array(
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    )
    assert np.allclose(third_row, [0.0, np.sin(np.radians(30)), np.cos(np.radians(30))])

    # 60 degrees about <111> is the Sigma-3 twin -> disorientation 60
    v = np.sin(np.pi / 6) / np.sqrt(3)
    q60 = np.array([[np.cos(np.pi / 6), v, v, v]])
    assert abs(cubic_disorientation_angle(q60)[0] - 60.0) < 1e-6

    # symmetry operators are unit quaternions and closed under multiplication
    assert np.allclose(np.linalg.norm(sym, axis=1), 1.0)

    # The crystal symmetry acts on the right (see crystal_equivalents): every
    # equivalent must be at zero disorientation from the original, the
    # fundamental-zone representative must be the same orientation, and the
    # left-multiplied version must NOT be (it is a sample-frame rotation). A
    # regression of the S*q bug that used to corrupt **oridata and *ori.
    qq = euler_bunge_to_quat(np.array([37.0]), np.array([52.0]), np.array([131.0]))
    equiv = crystal_equivalents(qq, sym)[0]
    d = cubic_disorientation_angle(qmul(qconj(np.repeat(qq, 24, 0)), equiv))
    assert d.max() < 1e-4, d.max()
    assert (
        cubic_disorientation_angle(qmul(qconj(qq), to_fundamental_zone(qq, sym)))[0]
        < 1e-4
    )
    wrong = qmul(sym[13][None], qq)
    assert cubic_disorientation_angle(qmul(qconj(qq), wrong))[0] > 10.0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("ctf", help="input Channel Text File")
    p.add_argument("-o", "--output", default=None, help="output .tesr")
    p.add_argument(
        "--crop",
        default=None,
        help="xmin,xmax,ymin,ymax in the .ctf's own units (microns), applied "
        "before segmentation. Prefer this to Neper's -transform crop: cropping "
        "afterwards clips grains into 1-2 pixel slivers that no --min-pixels "
        "prune has seen, and those degenerate cells abort the tessellation fit",
    )
    p.add_argument("--phase", type=int, default=1, help="phase to keep (default 1)")
    p.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="grain boundary misorientation threshold, degrees (default 10)",
    )
    p.add_argument(
        "--max-mad", type=float, default=1.0, help="MAD cutoff (default 1.0)"
    )
    p.add_argument("--min-bands", type=int, default=0, help="minimum Bands (default 0)")
    p.add_argument(
        "--allow-error",
        action="store_true",
        help="keep points whose Error column is non-zero (default: reject them)",
    )
    p.add_argument(
        "--min-pixels",
        type=int,
        default=5,
        help="discard grains smaller than this many pixels (default 5)",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="multiply step sizes by this to get the tesr length unit "
        "(default 1: keep the .ctf's microns). Do not write metres: Neper's "
        "fit objective and its val/eps stopping criteria are in absolute "
        "length units, so a metre-scale map stops the fit at iteration 1. "
        "The driver rescales to metres after meshing (TESR_UNIT)",
    )
    p.add_argument(
        "--flip-y",
        action="store_true",
        help="mirror the map in y. EBSD map coordinates usually run downwards "
        "while Neper's y runs upwards, so use this if the physical top surface "
        "of the specimen must end up at y = Ly",
    )
    p.add_argument(
        "--active",
        action="store_true",
        help="write orientations under the active convention instead of passive",
    )
    p.add_argument(
        "--no-fill",
        action="store_true",
        help="leave unassigned voxels empty instead of growing the cells into "
        "them. Holes act as interior surfaces in the tessellation fit, so this "
        "is only useful for inspecting how much of the map was rejected",
    )
    p.add_argument(
        "--no-voxel-ori",
        action="store_true",
        help="omit **oridata/**oridef. Much smaller file; keeps the grain "
        "orientations but loses per-pixel colouring in -V and GOS in -S",
    )
    p.add_argument(
        "--diagnostics",
        action="store_true",
        help="also write <output>-quality.png (the .ctf's Error codes and MAD "
        "over the window, the pixels this script rejected and why -- these are "
        "the grey pixels in neper -V's orientation map, **oridef = 0 -- and the "
        "pixels the cell map back-filled) and <output>-raw.png (the whole .ctf "
        "with the window marked, the window from the raw Euler angles, the "
        "window read back from the written tesr, and their per-voxel "
        "disorientation, i.e. proof of a lossless round trip). Needs matplotlib",
    )
    args = p.parse_args(argv)

    self_test()

    ctf = CtfMap(args.ctf)
    ny, nx = ctf.shape
    print(
        f"{args.ctf}: {nx} x {ny} pixels, step {ctf.header['XStep']} x "
        f"{ctf.header['YStep']}, {ctf.npoints} rows"
    )
    if not np.isclose(ctf.header["XStep"], ctf.header["YStep"]):
        print("  note: XStep != YStep -- voxels will not be square")

    crysym, phase = ctf.crysym(args.phase)
    if crysym is None:
        raise SystemExit("no phase line found; cannot determine crystal symmetry")
    if phase["laue"] not in CUBIC_LAUE:
        raise SystemExit(
            f"phase '{phase['name']}' has Laue group {phase['laue']} -> {crysym}, "
            "which this script cannot segment. The disorientation used here is "
            "specific to the cubic group. Segment in MTEX and write the grain "
            "ids into the **data section instead."
        )
    print(f"  phase {args.phase}: {phase['name']}, Laue {phase['laue']} -> {crysym}")

    sym = cubic_symmetry_quaternions()
    qgrid, ok, diag = build_grid(
        ctf, args.phase, args.max_mad, not args.allow_error, args.min_bands
    )
    qgrid_full, ok_full = qgrid, ok
    window = (slice(0, ny), slice(0, nx))
    if args.crop:
        qgrid, ok, window = crop_grid(
            qgrid, ok, args.crop, ctf.header["XStep"], ctf.header["YStep"]
        )
        diag = {k: v[window] for k, v in diag.items()}
        ny, nx = ok.shape
        print(f"  cropped to {nx} x {ny} pixels ({args.crop})")
    frac = ok.mean()
    print(
        f"indexed & above quality cutoffs: {ok.sum()} of {ok.size} ({100 * frac:.1f}%)"
    )
    if frac < 0.8:
        print(
            "  WARNING: a large fraction of the map was rejected. Loosen "
            "--max-mad or pass --allow-error, or expect a holed tessellation"
        )

    labels = segment_grains(qgrid, ok, args.threshold, sym)
    cellids, ncells, dropped, lost = relabel_and_prune(labels, ok, args.min_pixels)
    print(
        f"  grains at {args.threshold:g} deg: {ncells} "
        f"({dropped} below {args.min_pixels} px dropped, {lost} px)"
    )
    if ncells == 0:
        raise SystemExit("no grains survived; lower --min-pixels or --threshold")

    # Degenerate cells are what abort `neper -T -n from_morpho` partway through
    # "Listing cell voxels", and the objective function cannot place
    # pts(res=N) control points on a cell two pixels across either. Report the
    # bottom of the distribution so the failure is visible here, not there.
    unassigned = cellids == 0
    empty_before = int(unassigned.sum())
    print(
        f"  unassigned voxels: {empty_before} of {cellids.size} "
        f"({100 * empty_before / cellids.size:.1f} %)"
    )
    if not args.no_fill:
        cellids, filled = fill_holes(cellids)
        print(f"  filled {filled} voxels from the nearest cell")
    elif empty_before > 0.05 * cellids.size:
        print(
            "  WARNING: the raster has substantial holes and --no-fill was "
            "given. The tessellation fit will treat hole boundaries as grain "
            "boundaries"
        )

    counts = np.bincount(cellids.ravel())[1:]
    print(f"  smallest grains (px): {', '.join(str(c) for c in np.sort(counts)[:8])}")
    if counts.min() < 10:
        print(
            f"  WARNING: {int((counts < 10).sum())} grains under 10 px. Neper's "
            "tessellation fit is liable to abort on these -- raise "
            "--min-pixels (20 is a reasonable floor)"
        )

    qfz = to_fundamental_zone(qgrid.reshape(-1, 4), sym).reshape(ny, nx, 4)
    ori_cell = quat_to_rodrigues(grain_mean_orientations(qgrid, cellids, ncells, sym))
    ori_vox = None if args.no_voxel_ori else quat_to_rodrigues(qfz)

    if args.active:  # active is the opposite rotation, i.e. -r
        ori_cell = -ori_cell
        if ori_vox is not None:
            ori_vox = -ori_vox

    if args.flip_y:
        cellids = cellids[::-1]
        ok = ok[::-1]
        if ori_vox is not None:
            ori_vox = ori_vox[::-1]

    vox = (ctf.header["XStep"] * args.scale, ctf.header["YStep"] * args.scale)
    out = Path(args.output or Path(args.ctf).with_suffix(".tesr"))
    write_tesr(out, cellids, ori_cell, ori_vox, ok, vox, crysym)

    if args.diagnostics:
        unit = "um" if np.isclose(args.scale, 1.0) else f"x{args.scale:g} um"
        png = out.with_name(out.stem + "-quality.png")
        write_quality_png(
            png, diag, ok, unassigned, cellids, args, vox, unit, flip_y=args.flip_y
        )
        check = verify_readback(out, qgrid, ok, cellids, args.flip_y, sym)
        for line in check["report"]:
            print(f"  {line}")
        write_raw_png(
            out.with_name(out.stem + "-raw.png"),
            qgrid_full,
            ok_full,
            window,
            qgrid,
            ok,
            check,
            ctf,
            args,
            vox,
            unit,
        )

    lx, ly = nx * vox[0], ny * vox[1]
    mean_px = counts.mean()
    grain_size = np.sqrt(mean_px) * vox[0]
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  domain      : {lx:.4g} x {ly:.4g}")
    print(f"  grain size  : ~{grain_size:.4g} (equivalent square)")
    print("\ncheck it before meshing (ebsd_to_mesh.sh does the first two itself):")
    print(f"  neper -V {out} -datavoxcol ori -datavoxcolscheme ipf -print check-ori")
    print(f"  neper -V {out} -print check-grains")
    if not args.diagnostics:
        print("  re-run with --diagnostics to see why pixels are grey in check-ori")
    if ncells > 400:
        side = grain_size * np.sqrt(250)
        print(
            f"\n{ncells} grains is a long tessellation fit. Crop to ~250 by "
            f"re-running with a window about {side:.3g} x {side:.3g} in the "
            ".ctf's units, e.g. --crop xmin,xmax,ymin,ymax"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

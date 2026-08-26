#!/usr/bin/env python3
"""Convert an Oxford/Channel .ctf EBSD map into a Neper Raster Tessellation
File (.tesr), including the grain segmentation that a .ctf does not contain.

Neper cannot read .ctf, .ang or .h5. The entry point is a .tesr, and there is no
official converter -- the documented workflow is simply "write the EBSD data as
a tesr file", after which -V, -T and -M all work on it. This script is that
step, for the Channel Text File format.

The part that is not a transcription is the segmentation. A .ctf holds an
orientation per pixel and nothing else; `neper -T -morpho tesr` fits its cells
to the *cells* of the raster, so the file needs a `**cell` section and a
`**data` section assigning every pixel to a grain. Grains are found here by
flood-filling across neighbouring pixels whose disorientation is below a
threshold, with cubic crystal symmetry taken into account.

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
This is deliberate: Neper's tesr-fitting objective is `avdiameq * rms(dist)` in
the tesr's absolute length unit (net_tess_opt_comp_objective_fval_tesr2.c), so
its `val` and `eps` stopping criteria are not dimensionless. In metres the
initial objective of a ~100 um map is ~1e-10 and any `val<1e-6` or `eps<1e-6`
criterion is met before the first iteration, i.e. the "fit" returned is the
initial Laguerre guess. Gmsh's absolute geometric tolerances are also marginal
for 1e-6-sized elements. The transport driver converts to metres after reading
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
taken literally, leaving the two sections in opposite conventions. The conversion is verified against Neper's own convention table,
which gives Bunge (0, 30, 0) as Rodrigues (0.267949192, 0, 0) and quaternion
(0.965925826, 0.258819045, 0, 0); see the self-test at the bottom, which runs
on every invocation.

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
  cubic group; for hexagonal or lower, segment in MTEX and write the grain ids
  out instead.
* Square grids only, which is what JobMode Grid with XStep == YStep gives. A
  hexagonal acquisition must be resampled first (MTEX's `gridify`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
except ImportError:  # pragma: no cover
    sys.exit("this script needs scipy (conda install scipy)")


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
        cand = qmul(sym[None, :, :], blk[:, None, :])  # (n, 24, 4)
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
    return qgrid, ok


def crop_grid(qgrid, ok, spec, xstep, ystep):
    """Cut a rectangular window out of the map, before segmentation.

    Cropping here rather than in Neper matters for two reasons.

    First, sliver cells. `neper -T -transform crop(...)` clips whatever grains
    straddle the window, leaving cells one or two voxels wide that no
    `--min-pixels` prune ever saw, because the prune ran on the uncropped map.
    Those degenerate cells are what make `-n from_morpho` abort partway through
    "Listing cell voxels". Cropping first means the segmentation, the prune and
    the cell ids all describe the same region, and a clipped grain is either big
    enough to keep or dropped like any other.

    Second, per-voxel orientations. Neper 5.0.0 writes a raster it cannot read
    back when the file carries a `**oridata` section and has been through
    crop/autocrop. Cropping upstream keeps the file small enough to leave the
    orientations in, so -V colouring and -S intragranular measures still work.

    Bounds are in the .ctf's own length units (microns for a Channel file) and
    refer to the map as acquired, i.e. before any --flip-y.
    """
    try:
        x0, x1, y0, y1 = (float(v) for v in spec.split(","))
    except ValueError:
        raise SystemExit(
            f"--crop {spec!r}: expected four comma-separated numbers, "
            "xmin,xmax,ymin,ymax, in the same units as XStep"
        )
    ny, nx = ok.shape
    ix0, ix1 = max(int(round(x0 / xstep)), 0), min(int(round(x1 / xstep)), nx)
    iy0, iy1 = max(int(round(y0 / ystep)), 0), min(int(round(y1 / ystep)), ny)
    if ix1 - ix0 < 2 or iy1 - iy0 < 2:
        raise SystemExit(
            f"--crop {spec} keeps {max(ix1 - ix0, 0)} x {max(iy1 - iy0, 0)} "
            f"pixels. The map is {nx} x {ny} pixels of {xstep} x {ystep}, "
            f"i.e. {nx * xstep:g} x {ny * ystep:g} in those units."
        )
    return qgrid[iy0:iy1, ix0:ix1], ok[iy0:iy1, ix0:ix1]


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
            cand = qmul(sym[None, :, :], blk[:, None, :])
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
    # 60 degrees about <111> is the Sigma-3 twin -> disorientation 60
    v = np.sin(np.pi / 6) / np.sqrt(3)
    q60 = np.array([[np.cos(np.pi / 6), v, v, v]])
    assert abs(cubic_disorientation_angle(q60)[0] - 60.0) < 1e-6

    # symmetry operators are unit quaternions and closed under multiplication
    assert np.allclose(np.linalg.norm(sym, axis=1), 1.0)


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
    qgrid, ok = build_grid(
        ctf, args.phase, args.max_mad, not args.allow_error, args.min_bands
    )
    if args.crop:
        qgrid, ok = crop_grid(
            qgrid, ok, args.crop, ctf.header["XStep"], ctf.header["YStep"]
        )
        ny, nx = ok.shape
        print(f"  cropped to {nx} x {ny} pixels ({args.crop})")
    frac = ok.mean()
    print(
        f"  indexed and above quality cutoffs: {ok.sum()} of {ok.size} ({100 * frac:.1f} %)"
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
    empty_before = int((cellids == 0).sum())
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

    lx, ly = nx * vox[0], ny * vox[1]
    mean_px = counts.mean()
    grain_size = np.sqrt(mean_px) * vox[0]
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  domain      : {lx:.4g} x {ly:.4g}")
    print(f"  grain size  : ~{grain_size:.4g} (equivalent square)")
    print("\ncheck it before fitting:")
    print(f"  neper -V {out} -datavoxcol ori -datavoxcolscheme ipf -print check-ori")
    print(f"  neper -V {out} -print check-grains")
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

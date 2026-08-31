#!/usr/bin/env python3
"""How much orientation information the .ctf -> .tesr segmentation throws away.

    python segmentation_error.py map.ctf map.tesr [-o seg-error.png]
    python segmentation_error.py map.ctf map.tesr --provenance map-provenance.json

A .ctf carries one orientation per pixel; a .tesr carries one orientation per
*grain* (``**cell/*ori``), and that per-grain value is the only orientation the
rest of the pipeline ever sees -- ``ebsd_to_mesh.sh`` reads it out with
``-statcell rodrigues`` into ``${STEM}-grainori.txt`` and the transport driver
builds every boundary theta from it. The difference between the two is the
error introduced by segmenting, and it is what this script measures.

Definition
----------
For voxel i with area a_i = XStep x YStep, assigned to cell c(i),

    theta_i = disorientation( q_ctf(i), q_cell(c(i)) )        [degrees]

under the crystal symmetry declared in the tesr (cubic here), i.e. the minimum
rotation angle over all symmetry-equivalent descriptions of the two
orientations -- the same quantity the segmentation threshold is applied to.
The reported norms are the discrete L2 norm of that field,

    ||theta||_L2      = sqrt( sum_i theta_i^2 a_i )            [deg * length]
    ||theta||_L2 / sqrt(|Omega|)
                      = sqrt( sum_i theta_i^2 a_i / sum_i a_i )
                      = sqrt( mean_i theta_i^2 )               [deg]

The second is the headline number: it is the RMS disorientation of a pixel
from the grain it was put in, in degrees, independent of how big the map is,
and it is directly comparable with the segmentation threshold (--threshold,
10 deg by default) and with the theta cutoff the transport script filters the
boundary network on (THETA_MIN). Voxel areas are all equal on a square grid,
so the area-weighted and unweighted forms coincide; the weighted form is
computed anyway so that the number stays meaningful if XStep != YStep.

This is the grain-orientation-spread (GOS) field in RMS form
(Wright, Nowell & Field, Microsc. Microanal. 17 (2011) 316, sec. "Grain
Orientation Spread"), reported per grain as well as over the map.

Two errors, not one
-------------------
``--against grain`` (default) compares each pixel with its *cell* orientation:
the segmentation error, of order the intragranular spread, and irreducible --
one orientation per grain is what a tessellation is.

``--against voxel`` compares each pixel with the ``**oridata`` entry written
for that same pixel: the transcription error of the Euler -> Rodrigues
conversion and the ascii write. It should be ~1e-6 deg, which is the
arccos-near-1 noise floor of the disorientation formula, not a real difference
(ctf_to_tesr.self_test explains the amplification). Anything larger means the
convention flipped or the file is misaligned. ``--against both`` prints both.

Alignment
---------
The tesr covers the --crop window of the .ctf and may have been mirrored with
--flip-y, so the window and the flip have to be known to line the two files up.
ctf_to_tesr.py writes them into ``<output>-provenance.json``; pass that with
--provenance, or repeat the conversion flags by hand. A mismatch in shape is a
hard error, and a suspiciously large error with the right shape usually means
the window is right but the flip is not (try toggling --flip-y: a wrong flip
shows a mirror-symmetric theta map).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _ctf():
    """The ctf_to_tesr module, whether it is imported or is the running script.

    ctf_to_tesr.py imports this file to print the error at conversion time, and
    this file needs ctf_to_tesr's quaternion algebra, so the import has to be
    resolved late. Looking in sys.modules first also avoids loading a second
    copy of ctf_to_tesr when that script is the one being run (in which case it
    is registered as __main__, not as ctf_to_tesr).
    """
    for name in ("ctf_to_tesr", "__main__"):
        mod = sys.modules.get(name)
        if getattr(mod, "cubic_disorientation_angle", None) is not None:
            return mod
    import ctf_to_tesr

    return ctf_to_tesr


# --- the measurement ---------------------------------------------------------
def rodrigues_to_quat(r):
    """Rodrigues vectors -> unit quaternions, q0 > 0."""
    r = np.asarray(r, dtype=float).reshape(-1, 3)
    q = np.column_stack((np.ones(len(r)), r))
    return q / np.linalg.norm(q, axis=1)[:, None]


def theta_field(qgrid, qref):
    """Per-voxel disorientation (degrees) between two (ny, nx, 4) quat fields."""
    c = _ctf()
    a = qgrid.reshape(-1, 4)
    b = qref.reshape(-1, 4)
    return c.cubic_disorientation_angle(c.qmul(c.qconj(a), b)).reshape(qgrid.shape[:2])


def l2_stats(theta, mask, vox, threshold=None):
    """L2 norms and the order statistics of `theta` over the voxels in `mask`.

    `vox` is (XStep, YStep) in the raster's length unit; it only sets the units
    of the unnormalised norm and, if the voxels are not square, the weights.
    """
    t = np.asarray(theta)[mask]
    a = vox[0] * vox[1]
    n = t.size
    if n == 0:
        return {"n": 0}
    sq = float(np.sum(t**2))
    out = {
        "n": int(n),
        "area": n * a,
        "l2": float(np.sqrt(sq * a)),  # deg * length
        "rms": float(np.sqrt(sq / n)),  # deg, = l2 / sqrt(area)
        "mean": float(t.mean()),
        "median": float(np.median(t)),
        "p90": float(np.percentile(t, 90)),
        "p99": float(np.percentile(t, 99)),
        "max": float(t.max()),
    }
    if threshold is not None:
        out["threshold"] = float(threshold)
        out["frac_over"] = float((t > threshold).mean())
    return out


def per_grain_rms(theta, cellids, ncells):
    """(rms, count) per cell id 1..ncells, over the voxels assigned to it."""
    flat_id = cellids.ravel()
    flat_t = np.asarray(theta).ravel()
    keep = flat_id > 0
    cnt = np.bincount(flat_id[keep], minlength=ncells + 1)[1:]
    ssq = np.bincount(
        flat_id[keep], weights=flat_t[keep] ** 2, minlength=ncells + 1
    )[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        rms = np.sqrt(ssq / np.where(cnt > 0, cnt, np.nan))
    return rms, cnt


def segmentation_error(
    qgrid, cellids, qcell, vox, ok=None, threshold=None, qvox=None, backfilled=None
):
    """The full diagnostic. Returns a dict of arrays and statistics.

    qgrid   (ny, nx, 4) per-pixel quaternions straight from the .ctf
    cellids (ny, nx)    grain id per pixel, 0 = unassigned, as in the tesr
    qcell   (ncell, 4)  one quaternion per grain, as in the tesr's **cell/*ori
    ok      (ny, nx)    the quality mask, i.e. the tesr's **oridef. Voxels that
                        failed it carry a meaningless orientation, so they are
                        excluded from the headline number and reported apart.
    qvox    (ny, nx, 4) the tesr's **oridata, if it was written, for the
                        transcription check.
    backfilled (ny, nx) voxels that fill_holes assigned to their nearest cell
                        rather than to a grain of their own -- quality
                        rejections *and* pixels whose grain was pruned by
                        --min-pixels. Their theta is not a segmentation error
                        but the price of filling, so they are counted apart.
                        Defaults to ~ok, which catches only the first kind.
    """
    ncells = int(cellids.max())
    assigned = cellids > 0
    ok = np.ones_like(assigned) if ok is None else np.asarray(ok, dtype=bool)
    filled_in = ~ok if backfilled is None else np.asarray(backfilled, dtype=bool)

    qref = qcell[np.maximum(cellids - 1, 0)]  # id 0 -> cell 1, masked out below
    theta = theta_field(qgrid, qref)

    good = assigned & ~filled_in & ok  # in a grain of its own: the honest set
    filled = assigned & filled_in
    res = {
        "theta": np.where(assigned, theta, np.nan),
        "good": good,
        "filled": filled,
        "ncells": ncells,
        "vox": tuple(vox),
        "all": l2_stats(theta, assigned, vox, threshold),
        "indexed": l2_stats(theta, good, vox, threshold),
        "backfilled": l2_stats(theta, filled, vox, threshold),
    }
    rms, cnt = per_grain_rms(
        np.where(good, theta, 0.0), np.where(good, cellids, 0), ncells
    )
    res["grain_rms"], res["grain_npx"] = rms, cnt
    if qvox is not None:
        tv = theta_field(qgrid, qvox)
        res["theta_voxel"] = np.where(ok, tv, np.nan)
        res["voxel"] = l2_stats(tv, ok, vox)
    return res


def format_report(res, label="segmentation"):
    lines = []
    a = res["indexed"]
    if a["n"] == 0:
        return [f"{label}: no indexed voxel is assigned to a grain"]
    lines.append(
        f"{label} L2: RMS {a['rms']:.3f} deg over {a['n']} indexed voxels "
        f"(||theta||_2 = {a['l2']:.4g} deg*len over {a['area']:.4g} len^2)"
    )
    lines.append(
        f"{label}   : mean {a['mean']:.3f}, median {a['median']:.3f}, "
        f"p90 {a['p90']:.3f}, p99 {a['p99']:.3f}, max {a['max']:.3f} deg"
    )
    if "frac_over" in a:
        lines.append(
            f"{label}   : {100 * a['frac_over']:.2f} % of voxels sit further "
            f"than the {a['threshold']:g} deg segmentation threshold from "
            "their own grain's orientation"
        )
    b = res["backfilled"]
    if b["n"]:
        lines.append(
            f"{label}   : back-filled voxels ({b['n']}, excluded above): "
            f"RMS {b['rms']:.3f} deg, max {b['max']:.3f} deg"
        )
    rms, cnt = res["grain_rms"], res["grain_npx"]
    fin = np.flatnonzero(np.isfinite(rms))
    if fin.size:
        worst = fin[np.argsort(rms[fin])[::-1][:5]]
        lines.append(
            f"{label}   : worst grains (id: RMS deg, px) "
            + ", ".join(f"{i + 1}: {rms[i]:.2f}, {cnt[i]}" for i in worst)
        )
    if "voxel" in res:
        v = res["voxel"]
        verdict = "" if v["max"] < 1e-3 else "  <-- NOT a round trip"
        lines.append(
            f"transcription L2: RMS {v['rms']:.2e} deg, max {v['max']:.2e} deg "
            f"(**oridata vs the raw Euler angles){verdict}"
        )
    return lines


# --- picture -----------------------------------------------------------------
def write_png(path, res, cellids, unit="um", dpi=150, threshold=None):
    """theta map, its distribution, and the per-grain RMS on the same map."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15})
    try:
        from micrograph import scale_bar_ax
    except ImportError:  # pragma: no cover - scale bar is cosmetic
        scale_bar_ax = None

    theta = res["theta"]
    ny, nx = theta.shape
    vx, vy = res["vox"]
    kw = dict(
        interpolation="nearest", origin="lower", extent=(0, nx * vx, 0, ny * vy)
    )
    a = res["indexed"]
    vmax = max(a["p99"], 1e-3)
    aspect = ny * vy / (nx * vx)

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 3.0 + 5.5 * aspect), layout="constrained"
    )

    ax = axes[0]
    im = ax.imshow(theta, cmap="inferno", vmin=0, vmax=vmax, **kw)
    fig.colorbar(im, ax=ax, fraction=0.046).set_label("theta (deg)")
    ax.set_title(f"per-voxel segmentation error: RMS {a['rms']:.2f} deg")
    ax.set_xlabel(f"clipped at p99 = {vmax:.2f} deg")
    ax.set_xticks([])
    ax.set_yticks([])
    if scale_bar_ax:
        scale_bar_ax(ax, nx * vx, unit)

    ax = axes[1]
    good = theta[res["good"]]
    filled = theta[res["filled"]]
    hi = max(np.percentile(good, 99.9) * 1.5, threshold or 0, 1e-3)
    bins = np.linspace(0, hi, 80)
    ax.hist(good, bins=bins, color="0.35", label=f"indexed ({good.size})")
    if filled.size:
        ax.hist(
            np.clip(filled, 0, hi),
            bins=bins,
            color="tab:orange",
            alpha=0.8,
            label=f"back-filled ({filled.size}), clipped",
        )
    ax.set_yscale("log")
    ax.axvline(a["rms"], color="tab:red", lw=2, label=f"RMS {a['rms']:.2f} deg")
    ax.axvline(
        a["median"], color="tab:blue", lw=2, ls="--", label=f"median {a['median']:.2f}"
    )
    if threshold and threshold <= hi:
        ax.axvline(
            threshold, color="k", lw=2, ls=":", label=f"threshold {threshold:g} deg"
        )
    ax.set_xlabel("theta to the grain orientation (deg)")
    ax.set_ylabel("voxels")
    ax.set_title("distribution")
    ax.legend(fontsize=12)

    ax = axes[2]
    rms = res["grain_rms"]
    painted = np.full(theta.shape, np.nan)
    m = cellids > 0
    painted[m] = rms[cellids[m] - 1]
    im = ax.imshow(painted, cmap="viridis", **kw)
    fig.colorbar(im, ax=ax, fraction=0.046).set_label("RMS theta (deg)")
    ax.set_title(f"{res['ncells']} grains, worst {np.nanmax(rms):.2f} deg")
    ax.set_xticks([])
    ax.set_yticks([])
    if scale_bar_ax:
        scale_bar_ax(ax, nx * vx, unit)

    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {path}")


def write_csv(path, res):
    rms, cnt = res["grain_rms"], res["grain_npx"]
    with open(path, "w") as fh:
        fh.write("cell_id,n_voxels,rms_theta_deg\n")
        for k, (r, c) in enumerate(zip(rms, cnt), start=1):
            fh.write(f"{k},{int(c)},{'' if not np.isfinite(r) else f'{r:.6g}'}\n")
    print(f"  wrote {path}")


# --- standalone --------------------------------------------------------------
def read_tesr_full(path):
    """header, **cell/*ori, **data, **oridata, **oridef of an ascii tesr."""
    with open(path) as fh:
        tok = fh.read().split()
    i = tok.index("**general")
    if int(tok[i + 1]) != 2:
        sys.exit(f"{path}: not a 2D tesr")
    nx, ny = int(tok[i + 2]), int(tok[i + 3])
    n = nx * ny
    out = {"nx": nx, "ny": ny, "vox": (float(tok[i + 4]), float(tok[i + 5]))}

    i = tok.index("**cell")
    ncell = int(tok[i + 1])
    out["ncell"] = ncell
    j = tok.index("*crysym", i)
    out["crysym"] = tok[j + 1]
    j = tok.index("*ori", i)
    out["orides"] = tok[j + 1]
    if not out["orides"].startswith("rodrigues"):
        sys.exit(
            f"{path}: **cell/*ori is {out['orides']!r}; this reader only "
            "understands the rodrigues descriptor ctf_to_tesr.py writes"
        )
    out["cell_ori"] = np.array(tok[j + 2 : j + 2 + 3 * ncell], dtype=float).reshape(
        ncell, 3
    )

    i = tok.index("**data")
    if tok[i + 1] != "ascii":
        sys.exit(f"{path}: **data is {tok[i + 1]}, write it as ascii")
    out["cells"] = np.array(tok[i + 2 : i + 2 + n], dtype=int).reshape(ny, nx)

    if "**oridata" in tok:
        i = tok.index("**oridata")
        r = np.array(tok[i + 3 : i + 3 + 3 * n], dtype=float).reshape(n, 3)
        out["vox_ori"] = r.reshape(ny, nx, 3)
        i = tok.index("**oridef")
        out["oridef"] = (
            np.array(tok[i + 2 : i + 2 + n], dtype=int).reshape(ny, nx).astype(bool)
        )
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("ctf")
    p.add_argument("tesr")
    p.add_argument("-o", "--output", default=None, help="PNG (default: none)")
    p.add_argument("--csv", default=None, help="per-grain RMS table")
    p.add_argument(
        "--against",
        choices=("grain", "voxel", "both"),
        default="grain",
        help="compare each pixel with its grain's orientation (default), with "
        "the per-pixel **oridata, or both",
    )
    p.add_argument(
        "--provenance",
        default=None,
        help="the <output>-provenance.json ctf_to_tesr.py wrote; supplies "
        "--crop / --flip-y / --phase / the quality cutoffs",
    )
    p.add_argument("--crop", default=None)
    p.add_argument("--flip-y", action="store_true")
    p.add_argument("--active", action="store_true")
    p.add_argument("--phase", type=int, default=1)
    p.add_argument("--threshold", type=float, default=10.0)
    p.add_argument("--max-mad", type=float, default=1.0)
    p.add_argument("--min-bands", type=int, default=0)
    p.add_argument("--allow-error", action="store_true")
    p.add_argument("--unit", default="um")
    args = p.parse_args(argv)
    c = _ctf()

    prov = {}
    if args.provenance:
        prov = json.loads(Path(args.provenance).read_text())
        for k in (
            "crop",
            "flip_y",
            "active",
            "phase",
            "threshold",
            "max_mad",
            "min_bands",
            "allow_error",
        ):
            if k in prov:
                setattr(args, k, prov[k])
        args.unit = prov.get("unit", args.unit)

    ctf = c.CtfMap(args.ctf)
    _crysym, phase = ctf.crysym(args.phase)
    if phase is None or phase["laue"] not in c.CUBIC_LAUE:
        sys.exit("cubic phases only: the disorientation used here is cubic-specific")
    qgrid, ok, _diag = c.build_grid(
        ctf, args.phase, args.max_mad, not args.allow_error, args.min_bands
    )
    if args.crop:
        qgrid, ok, _w = c.crop_grid(
            qgrid, ok, args.crop, ctf.header["XStep"], ctf.header["YStep"]
        )

    t = read_tesr_full(args.tesr)
    if (t["ny"], t["nx"]) != ok.shape:
        sys.exit(
            f"{args.tesr} is {t['nx']} x {t['ny']} voxels but the .ctf window is "
            f"{ok.shape[1]} x {ok.shape[0]}. Pass the --crop used for the "
            "conversion, or --provenance."
        )
    cells = t["cells"][::-1] if args.flip_y else t["cells"]
    sign = -1.0 if args.active else 1.0
    qcell = rodrigues_to_quat(sign * t["cell_ori"])
    qvox = None
    if args.against in ("voxel", "both") and "vox_ori" in t:
        r = t["vox_ori"][::-1] if args.flip_y else t["vox_ori"]
        qvox = rodrigues_to_quat(sign * r).reshape(ok.shape + (4,))
    elif args.against in ("voxel", "both"):
        print("  note: no **oridata in the tesr (--no-voxel-ori); voxel check skipped")

    res = segmentation_error(
        qgrid, cells, qcell, t["vox"], ok=ok, threshold=args.threshold, qvox=qvox
    )
    for line in format_report(res):
        print("  " + line)

    # ctf_to_tesr.py knows which voxels fill_holes back-filled, including the
    # ones whose grain was pruned by --min-pixels; this script can only see the
    # quality rejections, so the two populations differ by the pruned pixels
    # and the numbers differ slightly with them. Say so rather than leave two
    # unequal RMS values lying about.
    ref = prov.get("segmentation_error_deg", {}).get("indexed", {})
    if ref and ref.get("n") != res["indexed"]["n"]:
        print(
            f"  note: ctf_to_tesr.py measured {ref['rms']:.3f} deg over "
            f"{ref['n']} voxels. The {res['indexed']['n'] - ref['n']} extra "
            "voxels here belonged to grains the prune removed and were then "
            "back-filled; only the converter can tell them apart from voxels "
            "in a grain of their own."
        )

    if args.output:
        write_png(args.output, res, cells, unit=args.unit, threshold=args.threshold)
    if args.csv:
        write_csv(args.csv, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())

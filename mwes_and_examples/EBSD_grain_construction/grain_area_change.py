"""How much the grains change size between the raster and the mesh.

    from grain_area_change import area_change, format_report, write_png
    res = area_change("map.tesr", "poly.msh4")
    for line in format_report(res):
        print(line)

Stage 2 of the pipeline replaces each grain's pixel staircase by a polygon:
`neper -M map.tesr` reconstructs the raster interfaces into a vertex/edge/face
topology, runs -tesrsmooth (Laplacian by default) over it, and meshes the
result. Both halves move the boundary, so the meshed grain is not the same set
of points as the rastered one. This script measures the difference in the one
number the transport model is sensitive to through its bulk terms -- the area
each grain ends up with -- and in the boundary position itself.

    A_raster(k) = (voxels with cell id k) x XStep x YStep
    A_mesh(k)   = sum of the triangle areas of the mesh's face k
    delta(k)    = 100 (A_mesh(k) - A_raster(k)) / A_raster(k)      [%]

reported as min / mean / median / max over the grains, which is the combined
effect of smoothing and meshing; there is no intermediate file between the two
to separate them from. To separate them anyway, re-run the mesh stage with
TESR_SMOOTH=none (`-tesrsmooth none`) into a different STEM and difference the
two tables: what is left is the discretisation alone.

Because a linear-triangle mesh represents a face by a polygon whose vertices
lie on the smoothed boundary, delta is not a bias-free quantity -- it is
dominated by the smoothing, which cuts pixel corners and therefore shrinks a
convex grain and grows a concave one. The signed mean over all grains is
correspondingly near zero while individual grains move by several percent, so
both are printed.

Two further numbers are printed because the area alone can be preserved while
the boundary moves:

  size (ECD)     the same change expressed as an equivalent circular diameter,
                 sqrt(A). A percent in diameter is what compares with a grain
                 size, and it is about half the percent in area.
  displaced area the fraction of the meshed area that sits over a *different*
                 grain of the raster, found by locating each triangle's
                 centroid in the raster. This counts boundary motion whatever
                 its sign, and it is the quantity that is directly comparable
                 with the ~17 % of misplaced voxels that made the convex-cell
                 Laguerre fit (route B) unusable; see the header of
                 ebsd_to_mesh.sh.

The same centroid lookup checks the assumption the rest of the pipeline rests
on, that mesh face k is raster cell k. If it does not hold the script says so
and refuses to report areas, because every number here (and every theta in the
transport driver) would then be attached to the wrong grain.
"""

from __future__ import annotations

import numpy as np
from mesh_overlay import read_msh4, read_tesr, use_agg
from micrograph import scale_bar_ax


def tesr_origin(path):
    """(ox, oy) of the raster, 0 unless the file carries an **origin section."""
    with open(path) as fh:
        tok = fh.read().split()
    if "**origin" not in tok:
        return (0.0, 0.0)
    i = tok.index("**origin")
    return (float(tok[i + 1]), float(tok[i + 2]))


def raster_areas(cells, vox, ncell=None):
    """Area of every cell id 1..ncell, as (voxel count) x (voxel area)."""
    ncell = int(cells.max()) if ncell is None else ncell
    counts = np.bincount(cells.ravel(), minlength=ncell + 1)[1 : ncell + 1]
    return counts * vox[0] * vox[1], counts


def triangle_areas(xyz, tri):
    """(face tag, area, centroid) for every 2D element, by the shoelace rule.

    The node tags are gathered through a flat lookup array rather than by
    indexing the dict per triangle, which is what keeps this linear in the
    element count on a real mesh.
    """
    tags = np.fromiter((t for t, _ in tri), dtype=np.int64, count=len(tri))
    nodes = np.array([v[:3] for _t, v in tri], dtype=np.int64)
    coord = np.zeros((max(xyz) + 1, 2))
    for t, c in xyz.items():
        coord[t] = c[:2]
    p = coord[nodes]  # (ntri, 3, 2)
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    area = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1])
        + b[:, 0] * (c[:, 1] - a[:, 1])
        + c[:, 0] * (a[:, 1] - b[:, 1])
    )
    return tags, area, p.mean(axis=1)


def face_to_cell(cells, vox, origin, tags, area, centroid, ncell):
    """Which raster cell each mesh face sits on, and how much area is displaced.

    Every triangle's centroid is dropped into the raster and its area added to
    the (face, cell) pair it lands on; the face is attributed to whichever cell
    holds the most of its area. Only the pairs that actually occur are
    accumulated -- a dense (nface, ncell) table would be 100 MB at 3500 grains
    and is almost all zeros, since a face touches one cell and its neighbours.

    Returns the mapping face tag -> cell id, the meshed area of each face, the
    area of each face lying on its mapped cell and on the cell of its own id,
    and the fraction of the whole meshed area that does not lie over the cell
    of its own id -- boundary motion, counted without regard to sign.
    """
    ny, nx = cells.shape
    ix = np.clip(((centroid[:, 0] - origin[0]) / vox[0]).astype(int), 0, nx - 1)
    iy = np.clip(((centroid[:, 1] - origin[1]) / vox[1]).astype(int), 0, ny - 1)
    under = cells[iy, ix].astype(np.int64)  # raster cell each triangle sits on
    ntag = int(tags.max())

    key = tags * (ncell + 1) + under
    uk, inv = np.unique(key, return_inverse=True)
    w = np.bincount(inv, weights=area)
    utag, ucell = uk // (ncell + 1), uk % (ncell + 1)

    # uk is sorted, so entries are already grouped by face; sorting each group
    # by area puts the winner last
    order = np.lexsort((w, utag))
    ut, uc, uw = utag[order], ucell[order], w[order]
    last = np.flatnonzero(np.r_[ut[1:] != ut[:-1], True])

    mapping = np.zeros(ntag + 1, dtype=np.int64)
    best = np.zeros(ntag + 1)
    mapping[ut[last]] = uc[last]
    best[ut[last]] = uw[last]

    total = np.bincount(tags, weights=area, minlength=ntag + 1)
    self_area = np.zeros(ntag + 1)
    same = utag == ucell
    self_area[utag[same]] = w[same]
    displaced = 1.0 - self_area.sum() / total.sum() if total.sum() else 0.0
    return mapping, total, best, self_area, displaced


def summarise(delta, weights=None):
    """min / mean / median / max, plus the shape-blind and weighted versions."""
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {}
    out = {
        "n": int(d.size),
        "min": float(d.min()),
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "max": float(d.max()),
        "abs_mean": float(np.abs(d).mean()),
        "abs_median": float(np.median(np.abs(d))),
        "rms": float(np.sqrt(np.mean(d**2))),
        "p05": float(np.percentile(d, 5)),
        "p95": float(np.percentile(d, 95)),
    }
    if weights is not None:
        w = np.asarray(weights, dtype=float)[np.isfinite(delta)]
        out["area_weighted_mean"] = float(np.sum(d * w) / np.sum(w))
    return out


def area_change(tesr_path, msh4_path, allow_mismatch=False):
    """The whole diagnostic. Returns a dict of per-grain arrays and summaries."""
    cells, vox = read_tesr(tesr_path)
    origin = tesr_origin(tesr_path)
    xyz, _seg, tri = read_msh4(msh4_path)
    if not tri:
        raise ValueError(f"{msh4_path}: no 2D elements. Mesh with -dim all.")

    ncell = int(cells.max())
    a_ras, npx = raster_areas(cells, vox, ncell)
    tags, tarea, cent = triangle_areas(xyz, tri)
    mapping, f_total, f_best, f_self, displaced = face_to_cell(
        cells, vox, origin, tags, tarea, cent, ncell
    )

    ntag = int(tags.max())
    faces = np.unique(tags)
    identity = bool(np.array_equal(mapping[faces], faces))

    a_mesh = (
        np.bincount(tags, weights=tarea, minlength=ncell + 1)[1 : ncell + 1]
        if ntag <= ncell
        else None
    )

    res = {
        "ncell": ncell,
        "nface": int(faces.size),
        "vox": vox,
        "cells": cells,
        "npx": npx,
        "identity": bool(identity),
        "allow_mismatch": bool(allow_mismatch),
        "mapping": mapping,
        "face_area": f_total,
        "face_matched": f_best,
        "face_self": f_self,
        "displaced": float(displaced),
        "a_raster": a_ras,
        "a_mesh": a_mesh,
        "domain": (cells.shape[1] * vox[0], cells.shape[0] * vox[1]),
        "mesh_total": float(tarea.sum()),
        "raster_total": float(a_ras.sum()),
    }
    if a_mesh is None or not (identity or allow_mismatch):
        return res

    with np.errstate(invalid="ignore", divide="ignore"):
        delta = 100.0 * (a_mesh - a_ras) / np.where(a_ras > 0, a_ras, np.nan)
        # equivalent circular diameter: d ~ sqrt(A), so this is the same change
        # expressed as a length rather than an area
        d_ecd = 100.0 * (np.sqrt(a_mesh / np.where(a_ras > 0, a_ras, np.nan)) - 1.0)
    res["delta"] = delta
    res["delta_ecd"] = d_ecd
    res["area"] = summarise(delta, weights=a_ras)
    res["ecd"] = summarise(d_ecd, weights=a_ras)
    return res


def format_report(res):
    lines = []
    if res["nface"] != res["ncell"]:
        lines.append(
            f"WARNING: the raster has {res['ncell']} cells but the mesh has "
            f"{res['nface']} faces. Grains were lost or split in the "
            "reconstruction; the area comparison below is not one-to-one."
        )
    if not res["identity"]:
        tag = "WARNING" if res.get("allow_mismatch") else "ERROR"
        m, tot, matched = res["mapping"], res["face_area"], res["face_matched"]
        bad = [
            (f, int(m[f]), matched[f] / tot[f] if tot[f] else np.nan)
            for f in range(1, len(m))
            if int(m[f]) not in (0, f) and tot[f] > 0
        ]
        lines.append(
            f"{tag}: mesh face k is not raster cell k. Faces sitting mostly on "
            "another cell (face -> cell, fraction of the face's area there): "
            + ", ".join(f"{f} -> {k} ({100 * q:.0f} %)" for f, k, q in bad[:10])
            + (" ..." if len(bad) > 10 else "")
        )
        lines.append(
            f"  {len(bad)} of {res['nface']} faces. Every per-grain quantity "
            "downstream, theta included, is indexed by face id, so this has to "
            "be resolved before the mesh is used. A face whose fraction is "
            "barely over half is an ambiguous small grain rather than a "
            "mis-tagged one; --allow-mismatch carries on with the identity "
            "mapping and reports the areas anyway."
        )
        if not res.get("allow_mismatch"):
            return lines
    if "area" not in res:
        return lines

    a, e = res["area"], res["ecd"]
    lines.append(
        f"grain area change (smoothing + meshing, {a['n']} grains): "
        f"min {a['min']:+.2f} %, mean {a['mean']:+.2f} %, "
        f"median {a['median']:+.2f} %, max {a['max']:+.2f} %"
    )
    lines.append(
        f"  |change|: mean {a['abs_mean']:.2f} %, median {a['abs_median']:.2f} %, "
        f"rms {a['rms']:.2f} %; area-weighted mean {a['area_weighted_mean']:+.2f} %; "
        f"5-95 % of grains within [{a['p05']:+.2f}, {a['p95']:+.2f}] %"
    )
    lines.append(
        f"  as a size (ECD ~ sqrt(area)): min {e['min']:+.2f} %, "
        f"mean {e['mean']:+.2f} %, median {e['median']:+.2f} %, max {e['max']:+.2f} %"
    )
    tot = 100.0 * (res["mesh_total"] - res["raster_total"]) / res["raster_total"]
    lines.append(
        f"  total meshed area {res['mesh_total']:.6g} vs rastered "
        f"{res['raster_total']:.6g} ({tot:+.3f} %); "
        f"displaced area {100 * res['displaced']:.2f} % "
        "(meshed area lying over a different raster cell)"
    )
    return lines


def write_csv(path, res, log=print):
    with open(path, "w") as fh:
        fh.write(
            "cell_id,n_voxels,area_raster,area_mesh,delta_area_pct,delta_ecd_pct\n"
        )
        for k in range(len(res["a_raster"])):
            fh.write(
                f"{k + 1},{int(res['npx'][k])},{res['a_raster'][k]:.8g},"
                f"{res['a_mesh'][k]:.8g},{res['delta'][k]:.6g},"
                f"{res['delta_ecd'][k]:.6g}\n"
            )
    if log:
        log(f"  wrote {path}")
    return path


def write_png(path, res, unit="um", dpi=150, log=print):
    """The grains coloured by their area change, next to its distribution."""
    plt = use_agg()
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15})

    cells, (vx, vy) = res["cells"], res["vox"]
    ny, nx = cells.shape
    delta = res["delta"]
    painted = np.full(cells.shape, np.nan)
    m = cells > 0
    painted[m] = delta[cells[m] - 1]
    lim = max(np.nanpercentile(np.abs(delta), 98), 1e-3)

    aspect = ny * vy / (nx * vx)
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 3.0 + 5.5 * aspect), layout="constrained"
    )

    ax = axes[0]
    im = ax.imshow(
        painted,
        cmap="coolwarm",
        vmin=-lim,
        vmax=lim,
        interpolation="nearest",
        origin="lower",
        extent=(0, nx * vx, 0, ny * vy),
    )
    fig.colorbar(im, ax=ax, fraction=0.046).set_label("area change (%)")
    ax.set_title("mesh vs raster, per grain")
    ax.set_xlabel(f"blue = shrunk, red = grew; colour limit {lim:.2g} %")
    ax.set_xticks([])
    ax.set_yticks([])
    scale_bar_ax(ax, nx * vx, unit)

    ax = axes[1]
    a = res["area"]
    ax.hist(delta[np.isfinite(delta)], bins=40, color="0.35")
    ax.axvline(0, color="k", lw=1)
    ax.axvline(
        a["median"],
        color="tab:blue",
        ls="--",
        lw=2,
        label=f"median {a['median']:+.2f} %",
    )
    ax.axvline(a["mean"], color="tab:red", lw=2, label=f"mean {a['mean']:+.2f} %")
    ax.set_xlabel("area change (%)")
    ax.set_ylabel("grains")
    ax.set_title(f"min {a['min']:+.2f} %, max {a['max']:+.2f} % over {a['n']} grains")
    ax.legend(fontsize=12)

    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    if log:
        log(f"  wrote {path}")
    return path


def measure(
    tesr, msh4, csv=None, png=None, unit="um", dpi=150, allow_mismatch=False, log=print
):
    """Measure, report, and optionally write the table and the picture.

    Returns the result dict. Raises ValueError when a mesh face does not sit on
    the raster cell of the same id, unless `allow_mismatch`: every per-grain
    quantity downstream is indexed by face id, so continuing past that would
    attach the areas -- and the transport driver's theta -- to the wrong
    grains.
    """
    res = area_change(tesr, msh4, allow_mismatch=allow_mismatch)
    lines = format_report(res)
    if log:
        for line in lines:
            log("  " + line)
    if "delta" not in res:
        raise ValueError("\n".join(lines))
    if csv:
        write_csv(csv, res, log=log)
    if png:
        write_png(png, res, unit=unit, dpi=dpi, log=log)
    return res

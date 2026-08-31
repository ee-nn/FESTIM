"""Short-circuit diffusion through the GB network of a polycrystal
measured by EBSD, in 2D.

Whole network is a single codim-1 subdomain carrying 1 species, so
the submesh built from all the GB facets is topologically connected
and hydrogen crosses from one boundary to another with no junction
condition.

Nothing about a disorientation requires three dimensions, it is computed
from the two grain orientations either way.

The mesh is Neper's direct meshing of the raster (`neper -M map.tesr`, 2D
only), so the grain boundaries are the measured ones. Neper writes
no .tess for that route (`-format tess` segfaults in 5.0.0), so the
tessellation-level bookkeeping -- which grains an edge separates, whether it
lies on the specimen surface, theta, length, junctions -- is rebuilt here from
the mesh: the msh4 carries the reconstructed boundary topology as 1D element
sets `edge#` and the grains as 2D element sets `face#`, with face k being
raster cell k, and theta follows from the two grain orientations under cubic
symmetry using the same disorientation function that segmented the map.

The one thing 2D genuinely costs is connectivity, which is what this script
measures. See the note at the end of ebsd_to_mesh.sh: percolation thresholds
for GB networks are far lower in 2D than in 3D, so a THETA_MIN that
fragments this network may leave the corresponding 3D one connected. Treat the
enhancement factor as a lower bound unless the microstructure is columnar, in
which case 2D is exact.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from mpi4py import MPI

import dolfinx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx.io import gmsh as gmshio

matplotlib.use("Agg")
from matplotlib.collections import LineCollection
from mesh_overlay import draw_raster, read_tesr
from micrograph import scale_bar_ax

import festim as F

plt.rcParams.update(
    {
        "font.size": 16,  # Baseline text, labels, ticks, and legends
        "axes.titlesize": 16,  # Subplot titles
        "figure.titlesize": 16,  # Global figure super title
    }
)

# --- input map ---------------------------------------------------------------
# The .tesr is the EBSD map written as a raster tessellation. Neper does not
# read .ang/.ctf/.h5
TESR = "d7.tesr"
CRYSYM = "cubic"
ORIDES = "rodrigues:passive"  # must match the descriptor in the tesr

# Interface smoothing applied by neper -M before meshing. The reconstructed
# boundaries are pixel staircases; Laplacian smoothing with factor A for N
# iterations rounds them off. "none" keeps the staircase.
TESR_SMOOTH = "laplacian"
TESR_SMOOTH_FACT = 0.5
TESR_SMOOTH_ITER = 5

# Metres per tesr length unit. ctf_to_tesr.py keeps the .ctf's microns so that
# Gmsh's absolute geometric tolerances are exercised at O(1-100) rather than
# O(1e-6). The mesh is converted here, once, right after it is read, so the
# transport parameters below and everything derived from the mesh stay in SI.
TESR_UNIT = 1e-6
UNIT_NAME = {1e-9: "nm", 1e-6: "um", 1e-3: "mm", 1.0: "m"}.get(TESR_UNIT, "tesr units")

# --- transport ---------------------------------------------------------------
D_B = 1e-14  # lattice diffusivity              [m^2/s]
D_GB = 1e-8  # GB diffusivity       [m^2/s]
DELTA = 5e-9  # GB width             [m]
K_EX = 1e-4  # bulk <-> GB exchange  [m/s]
C0 = 1.0  # surface concentration

T_END, DT = 36000.0, 600.0

# Keep only boundaries above this disorientation. 15 deg is the usual high-angle
# threshold. Expect it to fragment the network more readily in 2D than 3D
THETA_MIN = 10.0
THETA_DEPENDENT_D = False  # see gb_diffusivity_field, CHECK before enabling

# --- meshing -----------------------------------------------------------------
# RCL means "relative characteristic length" and is passed to Neper as -rcl.
# Neper uses it relative to the average raster cell size to choose a target
# element edge length. Smaller values request a finer mesh (more triangles
# and more 1D boundary segments); larger values request a coarser mesh. It
# doesn't change raster geometry or smoothed interfaces.
RCL = 0.25

# Multimeshing retries each face with several algorithms until MESH_QUAL_MIN is
# reached, so quality target and meshing time trade off directly; 0.9 is Neper's
# default and 0.7 is reasonable while iterating.
MESH_QUAL_MIN = 0.7
MESH_MAX_TIME = None  # seconds per face; try 30 when diagnosing (default = 1000)

try:
    _HERE = Path(__file__).resolve().parent
except NameError:  # interactive session: no __file__
    _HERE = Path.cwd()
WORKDIR = _HERE / "results"
MESH_SCRIPT = _HERE / "ebsd_to_mesh.sh"

# Neper is a command-line program, so it does not have to live in the same conda
# environment as FESTIM. Keeping a separate environment avoids depdendency issues.
#
# GMSH_BIN is the *executable*, which Neper calls for 2D meshing. The
# conda-forge package providing it is `gmsh`; `python-gmsh` is only the
# bindings, so a dolfinx environment may have the API without the command.
NEPER_ENV = "/home/fenna/anaconda3/envs/neper-env/bin"
NEPER_BIN = os.path.join(NEPER_ENV, "neper")
GMSH_BIN = os.path.join(NEPER_ENV, "gmsh")

# neper -V renders PNGs through a separate `povray` process
POVRAY_BIN = os.path.join(NEPER_ENV, "povray")
if not Path(POVRAY_BIN).is_file():
    POVRAY_BIN = "povray"

# The pipeline modules sit next to this file and are imported, not run:
# orientation.py owns the cubic disorientation function used to segment the map,
# so a boundary's theta here means the same thing as the threshold that created
# it, and micrograph / mesh_overlay / grain_area_change supply the diagnostics
# that ebsd_to_mesh.sh used to shell out for.
sys.path.insert(0, str(_HERE))


def run_interruptible(cmd, cwd=None, env=None):
    """Run the meshing pipeline, surviving a Ctrl+C long enough for it to finish.

    Neper treats SIGINT as "stop optimizing, keep the current solution and write
    the output" -- which matters here, because the tessellation fit is the long
    stage and a partially converged fit is a perfectly usable microstructure.
    Catching it here and waiting lets the pipeline
    land its output; a second Ctrl+C still gets you out.
    """
    proc = subprocess.Popen(cmd, cwd=cwd, env=env)
    try:
        code = proc.wait()
    except KeyboardInterrupt:
        print(
            "\ninterrupted: waiting for neper to write its output "
            "(Ctrl+C again to abandon it)"
        )
        try:
            code = proc.wait()
        except KeyboardInterrupt:
            proc.kill()
            raise
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def run_ebsd_pipeline(tesr=TESR, workdir=WORKDIR, force=True):
    """Mesh the EBSD map. Returns the base path (no extension).

    All Neper invocations live in ebsd_to_mesh.sh; this only marshals the
    parameters and checks the binaries. Caching is per stage inside the script.
    """
    base = (Path(workdir) / "poly").resolve()
    base.parent.mkdir(parents=True, exist_ok=True)
    print(f"neper outputs -> {base.parent}")

    tesr_path = Path(tesr)
    if not tesr_path.is_absolute():
        tesr_path = (_HERE / tesr_path).resolve()
    if not tesr_path.is_file():
        raise FileNotFoundError(
            f"no EBSD map .tesr file at {tesr_path}. Map must be written as a"
            + "raster tessellation first."
        )

    if shutil.which(NEPER_BIN) is None and not Path(NEPER_BIN).is_file():
        raise FileNotFoundError(
            f"neper not found at {NEPER_BIN!r}. Install it with "
            "conda and set NEPER_BIN to the binary's path."
        )
    if shutil.which(GMSH_BIN) is None and not Path(GMSH_BIN).is_file():
        raise FileNotFoundError(
            f"no gmsh executable at {GMSH_BIN!r}. Neper calls it for 2D meshing."
        )
    for p in (GMSH_BIN, NEPER_BIN, str(tesr_path)):
        if any(c.isspace() for c in str(p)):
            raise ValueError(
                f"the path {str(p)!r} contains whitespace. Neper's input-file"
                "argument is a structured field supporting .csv files and "
                "colon-separated transformations, so a path with whitespace"
                " arrives as several unusable fragments."
            )

    env = dict(os.environ)
    # the equivalent of activating neper-env for the child processes only:
    # neper, gmsh and povray are passed as explicit paths, and PATH is
    # prepended so that helper programs Neper spawns by name resolve there too
    env["PATH"] = NEPER_ENV + os.pathsep + env.get("PATH", "")
    env.update(
        {
            "TESR": str(tesr_path),
            "NEPER_BIN": NEPER_BIN,
            "GMSH_BIN": GMSH_BIN,
            "POVRAY_BIN": POVRAY_BIN,
            "NEPER_ENV": NEPER_ENV,
            "STEM": "poly",
            "WORKDIR": str(base.parent),
            "FORCE": "1" if force else "0",
            "CRYSYM": CRYSYM,
            "ORIDES": ORIDES,
            "RCL": str(RCL),
            "TESR_SMOOTH": TESR_SMOOTH,
            "TESR_SMOOTH_FACT": str(TESR_SMOOTH_FACT),
            "TESR_SMOOTH_ITER": str(TESR_SMOOTH_ITER),
            "CHECK_IMAGES": "1",
            "UNIT_NAME": UNIT_NAME,
        }
    )
    if MESH_QUAL_MIN:
        env["MESH_QUAL_MIN"] = str(MESH_QUAL_MIN)
    if MESH_MAX_TIME:
        env["MESH_MAX_TIME"] = str(MESH_MAX_TIME)

    run_interruptible(["bash", str(MESH_SCRIPT)], cwd=str(base.parent), env=env)
    finish_diagnostics(base)
    return base


def raster_extent(base):
    """(Lx, Ly, voxel x, voxel y) out of the .sttesr the mesh script wrote."""
    dim, lx, ly, vsx, vsy = Path(f"{base}.sttesr").read_text().split()
    if int(dim) != 2:
        raise ValueError(f"{base}.sttesr: expected a 2D raster, got {dim}D")
    return float(lx), float(ly), float(vsx), float(vsy)


def finish_diagnostics(base, unit=UNIT_NAME, check_images=True):
    """The Python half of the meshing stage, run once the shell script is done.

    ebsd_to_mesh.sh drives Neper and Gmsh only; these four outputs come from the
    pipeline's library modules and are done here so that those modules stay
    importable rather than needing a command line:

      check-ori.png, check-grains.png  the neper -V renders, border trimmed and
                                       a scale bar added
      check-mesh.png                   the reconstructed boundary edges over the
                                       raster cells
      <stem>-areachange.csv, check-area.png
                                       how much each grain changed size between
                                       the raster and the mesh

    A missing check image is reported and skipped, since neper -V needs POV-Ray
    and may have failed upstream. The area table is not optional: it also checks
    that mesh face k is raster cell k, which every theta downstream depends on,
    and grain_area_change.measure raises if it is not.
    """
    from grain_area_change import measure
    from mesh_overlay import overlay
    from micrograph import annotate_png

    base = Path(base)
    work = base.parent
    lx, _ly, _vx, _vy = raster_extent(base)
    tesr, msh4 = f"{base}-raw.tesr", f"{base}.msh4"

    if check_images:
        for name in ("check-ori", "check-grains"):
            png = work / f"{name}.png"
            if not png.is_file():
                print(f"  note: {png.name} was not rendered, skipping")
                continue
            try:
                annotate_png(png, lx, unit, trim_border=True)
            except Exception as exc:  # Pillow missing, or a zero-size render
                print(f"  WARNING: {png.name} left untrimmed ({exc})")
        overlay(tesr, msh4, output=str(work / "check-mesh.png"), unit=unit)

    measure(
        tesr,
        msh4,
        csv=f"{base}-areachange.csv",
        png=str(work / "check-area.png") if check_images else None,
        unit=unit,
    )


def _ids(mask):
    """Boolean mask over entities in id order -> 1-based ids."""
    return np.flatnonzero(mask).astype(np.int32) + 1


def _allreduce(comm, arr, op):
    out = np.empty_like(arr)
    comm.Allreduce(np.ascontiguousarray(arr), out, op=op)
    return out


def _rodrigues_to_quat(r):
    """Rodrigues vector r = (q1, q2, q3)/q0 -> unit quaternion (q0 > 0)."""
    q = np.column_stack((np.ones(len(r)), r))
    return q / np.linalg.norm(q, axis=1)[:, None]


class EdgeTable:
    """Per-edge scalars in id order: ``values[k]`` belongs to edge ``k + 1``."""

    def __init__(self, values):
        self.values = values
        self.n = len(next(iter(values.values())))

    def __getitem__(self, key):
        return self.values[key]


class Microstructure:
    """The GB topology, rebuilt from the mesh Neper wrote.

    With route A there is no tessellation file to take statistics from, but the
    msh4 carries the reconstructed topology: every 1D element set ``edge#`` is
    one boundary edge of the raster (after interface smoothing) and every 2D
    element set ``face#`` is raster cell k, so the grains on either side of an
    edge are the tags of the cells its facets belong to. From that:

    - ``domtype``: -1 for an edge between two grains, 1 for an edge with a
      single grain, i.e. on the specimen surface (in 2D the domain boundary is
      made of edges, so this is the whole story);
    - ``theta``: cubic disorientation between the two grains' mean
      orientations, read from ``-grainori.txt``. It is invariant under a
      global inversion of all orientations (the misorientations are inverted
      and conjugated, neither of which changes an angle under two-sided cubic
      symmetry), so the active/passive question does not enter here;
    - ``length``, ``ymin``, ``ymax``: summed / extremised over the edge's
      facets, in metres because the mesh was scaled on reading;
    - triple junctions: vertices where 3+ distinct edge ids meet and that do
      not lie on the bounding box. On a raster four grains can meet at a pixel
      corner, and those are counted too.

    The adjacency is gathered across ranks because a facet on a partition
    boundary sees only one of its two cells locally; lengths and extrema are
    reduced. The junction count is exact in serial and a lower bound in parallel
    """

    def __init__(self, base, mesh, cell_tags, facet_tags, extent):
        from orientation import cubic_disorientation_angle, qconj, qmul

        if CRYSYM != "cubic":
            raise NotImplementedError(
                "theta is computed with the closed-form cubic disorientation "
                f"from orientation.py; CRYSYM = {CRYSYM!r} needs a general "
                "symmetry-operator search"
            )
        comm = mesh.comm
        top = mesh.topology
        tdim = top.dim
        fdim = tdim - 1
        top.create_connectivity(fdim, tdim)
        top.create_connectivity(0, fdim)
        f2c = top.connectivity(fdim, tdim)
        v2f = top.connectivity(0, fdim)
        fmap, cmap, vmap = top.index_map(fdim), top.index_map(tdim), top.index_map(0)

        facet_edge = np.zeros(fmap.size_local + fmap.num_ghosts, dtype=np.int32)
        facet_edge[facet_tags.indices] = facet_tags.values
        cell_grain = np.zeros(cmap.size_local + cmap.num_ghosts, dtype=np.int32)
        cell_grain[cell_tags.indices] = cell_tags.values
        n_edge = comm.allreduce(int(facet_edge.max(initial=0)), op=MPI.MAX)
        n_grain = comm.allreduce(int(cell_grain.max(initial=0)), op=MPI.MAX)
        if n_edge == 0 or n_grain == 0:
            raise RuntimeError("the mesh carries no edge# / face# element sets")
        self.facet_edge = facet_edge
        self.n_grains = n_grain

        # grains on either side of each edge
        local = {}
        for f in np.flatnonzero(facet_edge):
            local.setdefault(int(facet_edge[f]), set()).update(
                int(cell_grain[c]) for c in f2c.links(f)
            )
        grains = [set() for _ in range(n_edge + 1)]
        for part in comm.allgather({k: sorted(v) for k, v in local.items()}):
            for k, v in part.items():
                grains[k].update(v)
        sides = np.array([len(g) for g in grains[1:]])
        if sides.min() < 1 or sides.max() > 2 or 0 in set().union(*grains[1:]):
            bad = _ids((sides < 1) | (sides > 2))
            raise RuntimeError(
                f"edges {bad[:10]} touch {sides[bad[:10] - 1]} grains; every "
                "edge# set must lie between two face# sets or between one "
                "face# set and the domain. The msh4 is not a neper -M raster "
                "mesh, or the cell tags did not survive the read."
            )
        pair = np.array(
            [sorted(g) + [0] * (2 - len(g)) for g in grains[1:]], dtype=np.int32
        )
        domtype = np.where(sides == 2, -1.0, 1.0)

        # length, ymin, ymax over owned facets, then reduced
        owned = np.arange(fmap.size_local, dtype=np.int32)
        owned = owned[facet_edge[owned] > 0]
        nodes = dolfinx.mesh.entities_to_geometry(mesh, fdim, owned, False)
        x = mesh.geometry.x[nodes]  # (nf, 2, 3)
        e = facet_edge[owned]
        length = np.bincount(
            e, weights=np.linalg.norm(x[:, 1] - x[:, 0], axis=1), minlength=n_edge + 1
        )
        ymin = np.full(n_edge + 1, np.inf)
        ymax = np.full(n_edge + 1, -np.inf)
        np.minimum.at(ymin, e, x[:, :, 1].min(axis=1))
        np.maximum.at(ymax, e, x[:, :, 1].max(axis=1))
        length = _allreduce(comm, length[1:], MPI.SUM)
        ymin = _allreduce(comm, ymin[1:], MPI.MIN)
        ymax = _allreduce(comm, ymax[1:], MPI.MAX)

        # theta from the grain orientations, line k of the file = face k
        self.ori = np.loadtxt(str(base) + "-grainori.txt", ndmin=2)
        if self.ori.shape[0] != n_grain:
            raise RuntimeError(
                f"{base}-grainori.txt has {self.ori.shape[0]} lines but the mesh "
                f"has {n_grain} face# sets; they must be the same raster"
            )
        q = _rodrigues_to_quat(self.ori)
        theta = np.zeros(n_edge)
        inner = sides == 2
        a, b = pair[inner, 0] - 1, pair[inner, 1] - 1
        theta[inner] = cubic_disorientation_angle(qmul(qconj(q[a]), q[b]))

        self.edges = EdgeTable(
            {
                "domtype": domtype,
                "theta": theta,
                "length": length,
                "ymin": ymin,
                "ymax": ymax,
                "grain_a": pair[:, 0].astype(float),
                "grain_b": pair[:, 1].astype(float),
            }
        )

        # junctions: owned vertices where 3+ distinct edge ids meet, off the box
        n_v = vmap.size_local
        verts = np.arange(n_v, dtype=np.int32)
        xv = dolfinx.mesh.compute_midpoints(mesh, 0, verts)
        lx, ly = extent
        tol = 1e-9 * max(lx, ly)
        on_box = (
            (np.abs(xv[:, 0]) < tol)
            | (np.abs(xv[:, 0] - lx) < tol)
            | (np.abs(xv[:, 1]) < tol)
            | (np.abs(xv[:, 1] - ly) < tol)
        )
        edgenb = np.fromiter(
            (len(set(facet_edge[v2f.links(v)].tolist()) - {0}) for v in verts),
            dtype=int,
            count=n_v,
        )
        self.triple_junctions = comm.allreduce(
            int(((edgenb >= 3) & ~on_box).sum()), op=MPI.SUM
        )
        self.surface_edges = int((sides == 1).sum())

    @property
    def interior_mask(self):
        return self.edges["domtype"] < 0

    @property
    def network_mask(self):
        return self.interior_mask & (self.edges["theta"] > THETA_MIN)

    @property
    def network_edge_ids(self):
        return _ids(self.network_mask)

    @property
    def network_length(self):
        return float(self.edges["length"][self.network_mask].sum())

    def junction_only_below(self, y_top, tol=1e-12):
        """Deepest point reached by a boundary that touches the charged edge.

        Below this depth no boundary is fed directly, so whatever the network
        holds there has crossed at least one triple junction.
        """
        touching = self.network_mask & (self.edges["ymax"] > y_top - tol)
        if not touching.any():
            return y_top
        return float(self.edges["ymin"][touching].min())

    def check_orientations(self):
        """Fail loudly if theta is not a real disorientation distribution.

        With theta computed here rather than by Neper the failure modes move:
        an all-zero orientation file, or one that does not belong to this
        raster, are what would make every boundary look alike.
        """
        theta = self.edges["theta"][self.interior_mask]
        problems = []
        if np.allclose(self.ori, 0.0):
            problems.append("every grain orientation in the tesr readout is zero")
        if theta.size and np.allclose(theta, 0.0):
            problems.append("every interior edge has theta = 0")
        if theta.size and theta.max() > 63.0:
            # the maximum disorientation is ~62.8 deg for cubic symmetry
            problems.append(
                f"max theta = {theta.max():.1f} deg exceeds the cubic bound"
            )
        if problems:
            raise RuntimeError(
                "the orientations are not usable: "
                + "; ".join(problems)
                + ". Check that -grainori.txt was written from the same tesr "
                "that was meshed."
            )
        return self.n_grains

    def report(self, extent, n_grains):
        kept, total = int(self.network_mask.sum()), int(self.interior_mask.sum())
        theta = self.edges["theta"][self.network_mask]
        lx, ly = extent
        print(f"microstructure: {n_grains} grains from {TESR} (2D, raster mesh)")
        print(f"  domain                          : {lx:g} x {ly:g}")
        print(f"  edges                           : {self.edges.n}")
        print(f"  on the specimen surface         : {self.surface_edges}")
        print(f"  grain boundaries (interior)     : {total}")
        print(f"  kept above {THETA_MIN:g} deg    : {kept}")
        if kept:
            print(
                f"  disorientation                  : "
                f"{theta.min():.1f} - {theta.max():.1f} deg"
                f" (mean {theta.mean():.1f})"
            )
        print(f"  triple junctions                : {self.triple_junctions}")
        print(f"  boundary length                 : {self.network_length:.4g}")


def write_network_png(base, mesh, micro, tesr_path):
    """check-network.png: the raster with the boundaries as FESTIM will use them.

    Same background as check-mesh.png (mesh_overlay.py), but the edges are the
    driver's: those above THETA_MIN coloured by theta, interior edges below it
    dashed white, specimen-surface edges grey. Compare with check-mesh.png to
    see what the disorientation filter removed, and with check-grains.png to
    see how far interface smoothing moved the boundaries off the pixels.
    """
    if mesh.comm.size > 1:
        return

    cells, vox = read_tesr(tesr_path)
    fdim = mesh.topology.dim - 1
    owned = np.arange(mesh.topology.index_map(fdim).size_local, dtype=np.int32)
    owned = owned[micro.facet_edge[owned] > 0]
    nodes = dolfinx.mesh.entities_to_geometry(mesh, fdim, owned, False)
    segs = mesh.geometry.x[nodes][:, :, :2] / TESR_UNIT  # back to tesr units
    e = micro.facet_edge[owned] - 1
    surface = micro.edges["domtype"][e] > 0
    kept = micro.network_mask[e]
    dropped = ~surface & ~kept

    ny, nx = cells.shape
    fig, ax = plt.subplots(figsize=(8, 8 * ny * vox[1] / (nx * vox[0])))
    draw_raster(ax, cells, vox)
    ax.add_collection(LineCollection(segs[surface], colors="0.6", lw=0.7))
    ax.add_collection(
        LineCollection(segs[dropped], colors="white", lw=1.3, linestyles="--")
    )
    net = LineCollection(segs[kept], cmap="inferno", lw=1.8)
    net.set_array(micro.edges["theta"][e[kept]])
    net.set_clim(THETA_MIN, 62.8)
    ax.add_collection(net)
    fig.colorbar(net, ax=ax, fraction=0.046).set_label("theta (deg)")
    n_kept, n_int = int(micro.network_mask.sum()), int(micro.interior_mask.sum())
    ax.set_title(
        f"network used by FESTIM: {n_kept} of {n_int} boundaries above "
        f"{THETA_MIN:g} deg\n(dashed white = dropped, grey = specimen surface)"
    )
    ax.set_xlabel(f"x ({UNIT_NAME})")
    ax.set_ylabel(f"y ({UNIT_NAME})")
    scale_bar_ax(ax, nx * vox[0], UNIT_NAME)
    fig.tight_layout()
    out = base.parent / "check-network.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# mesh
def read_mesh(base, comm=MPI.COMM_WORLD, rank=0):
    """Read the Neper mesh into dolfinx.

    ``cell_tags`` carry the grain id (2D element set ``face#`` = raster cell)
    and ``facet_tags`` the reconstructed boundary edge id (1D element set
    ``edge#``); neper -M writes every entity of the topology it reconstructed
    from the raster as an element set. In a 2D mesh the facets are line
    segments, so the tags that matter come from the 1D element sets.
    """
    result = gmshio.read_from_msh(str(base) + ".msh4", comm, rank, gdim=2)
    if hasattr(result, "mesh"):
        mesh, cell_tags, facet_tags = result.mesh, result.cell_tags, result.facet_tags
    else:
        mesh, cell_tags, facet_tags = result[0], result[1], result[2]
    if facet_tags is None or facet_tags.values.size == 0:
        raise RuntimeError(
            "no facet tags were read: the 1D element sets did not survive the "
            "msh4 round trip. Check that -dim all reached neper -M."
        )
    # Neper meshed in the tesr's unit; put the geometry in metres before any
    # locator, submesh or dof coordinate is derived from it
    mesh.geometry.x[:] *= TESR_UNIT
    return mesh, cell_tags, facet_tags


class GrainBoundaryNetwork(F.VolumeSubdomain):
    """The whole network as one codim-1 subdomain: facets of a 2D mesh, ``dim=1``.

    The facets come straight from the tags, so the delicate part of the
    in-situ-tessellation version -- ``locate_entities`` marks an entity only
    when *all* its vertices satisfy the locator, which near a triple junction
    also catches segments lying on no grain boundary in particular -- does not
    arise. There is nothing geometric left to get wrong.
    """

    def __init__(self, id, material, facet_tags, edge_ids):
        super().__init__(id=id, material=material, dim=1)
        self.facet_tags = facet_tags
        self.edge_ids = np.asarray(edge_ids, dtype=np.int32)
        self.entity_edge_ids = None  # tess edge id of each located facet, in order

    def locate_subdomain_entities(self, mesh):
        keep = np.isin(self.facet_tags.values, self.edge_ids)
        self.entity_edge_ids = self.facet_tags.values[keep].astype(np.int32)
        return self.facet_tags.indices[keep].astype(np.int32)


def gb_diffusivity_field(network, micro, d_low, d_high, theta_c=15.0):
    """A per-boundary diffusivity as a DG0 field on the network submesh.

    With measured orientations this stops being a thought experiment: the
    disorientation is a property of the specimen, so a diffusivity that varies
    from boundary to boundary is the natural refinement. It has to enter as a
    coefficient on the *one* submesh -- splitting the network into high- and
    low-angle subdomains would give two disconnected submeshes and put the
    junction conditions straight back.
    """
    submesh = network.submesh
    parent = None
    for name in ("submesh_to_mesh", "submesh_to_parent", "parent_to_submesh"):
        parent = getattr(network, name, None)
        if parent is not None:
            break
    if parent is None or network.entity_edge_ids is None:
        raise NotImplementedError(
            "cannot map submesh cells back to tessellation edges: the subdomain "
            "does not expose its parent entity map under any of the expected "
            "names. Read it off dolfinx.mesh.create_submesh directly."
        )

    # entity_edge_ids is aligned with the facet list returned by
    # locate_subdomain_entities, and create_submesh keeps that order, so the
    # parent map indexes straight into it: parent[c] is the position of submesh
    # cell c in the located facet list, hence its tessellation edge id.
    edge_of_cell = network.entity_edge_ids
    theta = micro.edges["theta"]
    V = dolfinx.fem.functionspace(submesh, ("DG", 0))
    d = dolfinx.fem.Function(V, name="D_gb")
    tdim = submesh.topology.dim
    n_local = submesh.topology.index_map(tdim).size_local
    ids = edge_of_cell[np.asarray(parent)[:n_local]]
    d.x.array[:n_local] = np.where(theta[ids - 1] >= theta_c, d_high, d_low)
    d.x.scatter_forward()
    return d


# build
base = run_ebsd_pipeline()
cols = np.loadtxt(str(base) + ".sttesr", ndmin=2)[0]
LX, LY = float(cols[1]) * TESR_UNIT, float(cols[2]) * TESR_UNIT

mesh, cell_tags, facet_tags = read_mesh(base)
micro = Microstructure(base, mesh, cell_tags, facet_tags, (LX, LY))
N_GRAINS = micro.check_orientations()

write_network_png(base, mesh, micro, base.parent / "poly-raw.tesr")

grains = F.VolumeSubdomain(
    id=1,
    material=F.Material(D_0=D_B, E_D=0.0),
    locator=lambda x: np.full_like(x[0], True, dtype=bool),
)
network = GrainBoundaryNetwork(
    id=2,
    material=F.Material(D_0=D_GB, E_D=0.0),
    facet_tags=facet_tags,
    edge_ids=micro.network_edge_ids,
)
# the charged surface is the top edge of the map, wherever that now is
top = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[1], LY))
# every point where a grain boundary meets the charged edge, in one object
mouths = F.SurfaceSubdomain(id=4, dim=0, locator=lambda x: np.isclose(x[1], LY))

c_b = F.Species("c_b", subdomains=[grains])
c_gb = F.Species("c_gb", subdomains=[network])


def solve(d_gb):
    """
    Run the problem, returning the model and the solution.
    """
    network.material = F.Material(D_0=d_gb, E_D=0.0)
    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[c_b, c_gb],
        subdomains=[grains, network, top, mouths],
        sources=[
            F.ParticleSource(
                value=lambda cb, cg: (2.0 / DELTA) * K_EX * (cb - cg),
                species=c_gb,
                volume=network,
                species_dependent_value={"cb": c_b, "cg": c_gb},
            )
        ],
        boundary_conditions=[
            F.ParticleFluxBC(
                subdomain=network,
                species=c_b,
                value=lambda cb, cg: K_EX * (cg - cb),
                species_dependent_value={"cb": c_b, "cg": c_gb},
            ),
            F.FixedConcentrationBC(subdomain=top, value=C0, species=c_b),
            F.FixedConcentrationBC(subdomain=mouths, value=C0, species=c_gb),
        ],
        temperature=500,
        settings=F.Settings(
            atol=1e-14,
            rtol=1e-12,
            transient=True,
            final_time=T_END,
            stepsize=F.Stepsize(initial_value=DT),
        ),
        exports=[
            F.VTXSpeciesExport(
                str(base.parent / "ebsd_grains.bp"), field=c_b, subdomain=grains
            ),
            F.VTXSpeciesExport(
                str(base.parent / "ebsd_network.bp"), field=c_gb, subdomain=network
            ),
        ],
    )
    model.initialise()
    model.run()
    return (
        model,
        c_b.subdomain_to_post_processing_solution[grains],
        c_gb.subdomain_to_post_processing_solution[network],
    )


model, cb, cgb = solve(D_GB)


# what we built
def submesh_length(subdomain):
    """Total length of the network submesh -- the 2D analogue of its area."""
    sub = subdomain.submesh
    geom = sub.geometry
    # dolfinx >= 0.9 exposes one dofmap per coordinate element; the scalar
    # attribute is deprecated there and absent in later releases
    dofmap = geom.dofmaps[0] if hasattr(geom, "dofmaps") else geom.dofmap
    seg = dofmap.reshape(-1, 2)[: sub.topology.index_map(1).size_local]
    x = sub.geometry.x
    local = float(np.sum(np.linalg.norm(x[seg[:, 1]] - x[seg[:, 0]], axis=1)))
    return sub.comm.allreduce(local, op=MPI.SUM)


def component_count(subdomain):
    """Connected components of the network submesh, through shared vertices.

    In 2D the network is a graph of segments meeting at points, so connectivity
    runs through vertices rather than through edges as it did in 3D. More than
    one component is expected once THETA_MIN starts removing boundaries, and it
    is worth knowing: a component that does not touch the charged edge is never
    fed, and a fragmented network is no longer the single connected object the
    codim-1 formulation was chosen for. Serial only -- in parallel the adjacency
    is partitioned and this would count per-rank pieces.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    sub = subdomain.submesh
    if sub.comm.size > 1:
        return None
    sub.topology.create_connectivity(1, 0)
    sub.topology.create_connectivity(0, 1)
    v2c = sub.topology.connectivity(0, 1)
    n = sub.topology.index_map(1).size_local
    rows, cols = [], []
    for v in range(sub.topology.index_map(0).size_local):
        cells = v2c.links(v)
        for a in cells:
            for b in cells:
                rows.append(a)
                cols.append(b)
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return connected_components(adj, directed=False)[0]


micro.report((LX, LY), N_GRAINS)
len_mesh, len_tess = submesh_length(network), micro.network_length
n_comp = component_count(network)
print(f"mesh : {mesh.topology.index_map(2).size_global} triangles")
print(f"Fraction of network captured by the submesh: {100 * len_mesh / len_tess:.2f}%")


# effect of the network
def inventory(cb, cgb):
    """
    Total hydrogen per unit out-of-plane thickness.
    """
    dx_bulk = ufl.Measure("dx", domain=cb.function_space.mesh)
    dx_gb = ufl.Measure("dx", domain=cgb.function_space.mesh)
    total = dolfinx.fem.assemble_scalar(dolfinx.fem.form(cb * dx_bulk))
    total += DELTA * dolfinx.fem.assemble_scalar(dolfinx.fem.form(cgb * dx_gb))
    return mesh.comm.allreduce(total, op=MPI.SUM)


# Below this depth no boundary is fed directly from the charged edge, so
# everything the network holds there has crossed at least one triple junction.
# This is a property of the tessellation, not of the partitioned mesh, so it is
# the same on every rank with no reduction needed.
junction_only_below = micro.junction_only_below(LY)

gb_y = cgb.function_space.tabulate_dof_coordinates()[:, 1]
deep = gb_y < junction_only_below
c_deep = cgb.x.array[deep].max() if deep.any() else 0.0

# For scale only: Hart's effective diffusivity is the upper bound you would get
# if every boundary ran straight along the gradient. In 2D f is a length fraction
# times delta rather than an area fraction times delta.
beta = DELTA * (D_GB / D_B - 1) / (2 * np.sqrt(D_B * T_END))
print(f"  type-B parameter beta          : {beta:.0f}  (short circuit needs beta >> 1)")

bulk_y = cb.function_space.tabulate_dof_coordinates()[:, 1]
c_grain_deep = cb.x.array[bulk_y < junction_only_below].mean()

print("\njunction transport: no boundary touching the charged edge reaches below")
print(f"y = {junction_only_below:.4g}, so everything the network holds there has")
print("crossed at least one triple junction.")
print(f"  max c on the network there     : {c_deep:.4e}")
print(f"  mean c in the grains there     : {c_grain_deep:.4e}")
# Both are solver noise around zero whenever the Fisher tail is shorter than
# the junction-only depth, and noise has a sign; the ratio is then 0/0
if c_grain_deep > 1e-12 * C0:
    print(f"  ratio                          : x {c_deep / c_grain_deep:.0f}")
else:
    print("ratio: n/a - nothing has reached this depth on either path")

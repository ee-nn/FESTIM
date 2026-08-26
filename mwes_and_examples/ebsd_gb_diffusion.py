"""Short-circuit diffusion through the grain-boundary network of a polycrystal
measured by EBSD, in 2D.

Same formulation as the 3D version: the whole network is **one** codim-1
subdomain carrying **one** species, so the submesh built from all the
grain-boundary facets is topologically connected and hydrogen crosses from one
boundary to another with no junction condition to write. In 2D "codim-1" means
a 1D manifold inside a 2D mesh -- line segments inside triangles -- which is
just as much codimension 1, and the formulation is unchanged.

What moves down a dimension is the bookkeeping. In a 2D tessellation the cells
*are* the faces, so:

    3D                                  2D
    grain boundary  = face              = edge
    triple line     = interior edge     = interior vertex, 3+ edges
    quadruple node  = interior vertex   = does not exist
    theta lives on  faces               edges
    mesh tags from  2D element sets     1D element sets

`theta` is a face key in 3D and an edge key in 2D
(https://neper.info/doc/exprskeys.html), and correspondingly a 2D-elset key in
3D meshes and a 1D-elset key in 2D meshes. Nothing about a disorientation
requires three dimensions -- it is computed from the two grain orientations
either way.

Route B is still the only option: meshing the raster directly (`neper -M
map.tesr`) is *available* in 2D, unlike in 3D, but it produces no tessellation
and therefore no `-statedge`, hence no theta and no way to separate grain
boundaries from specimen surface.

The one thing 2D genuinely costs is connectivity, which is what this script
measures. See the note at the end of ebsd_to_mesh.sh: percolation thresholds
for grain-boundary networks are far lower in 2D than in 3D, so a THETA_MIN that
fragments this network may leave the corresponding 3D one connected. Treat the
enhancement factor as a lower bound unless the microstructure is columnar, in
which case 2D is exact.

All extensive quantities below are per unit thickness out of plane.
"""

import os
import shutil
import subprocess
from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np
import ufl
from dolfinx.io import gmsh as gmshio

import festim as F

# --- input map ---------------------------------------------------------------
# The .tesr is the EBSD map written as a raster tessellation. Neper does not
# read .ang/.ctf/.h5 -- see https://neper.info/doc/tutorials/ebsd_process.html
# for the sections a tesr needs (**general, **cell, **data, **oridata). A single
# map is exactly the right input here; no serial sectioning required.
TESR = "ebsd-centre.tesr"

CRYSYM = "cubic"
ORIDES = "rodrigues:passive"  # must match the descriptor in the tesr

# Crop the map before fitting, in the tesr's own length units. The 2D fit is
# cheap by 3D standards -- three optimization degrees of freedom per grain
# rather than four, and polygons rather than polyhedra -- but the cost is still
# superlinear, so a few hundred grains is a comfortable working size.
# Optional Neper transformation chain for the raster. Leave as None: cropping
# and hole-filling are done by ctf_to_tesr.py, and skipping the Neper pass
# avoids its tesr write path, which produces an unreadable file when the input
# carries **oridata (Neper 5.0.0).
TESR_TRANSFORM = None
OBJ_RES = 8  # control points per grain per direction in the fit objective
MORPHO_STOP = "val<1e-6||iter>=20000||time>=3600"

# --- transport ---------------------------------------------------------------
# CHECK these against the domain size printed at startup. They are written for a
# specimen tens of microns across with metres as the length unit; if the tesr
# pixel size is in microns, all four need rescaling together.
D_B = 1e-16  # lattice diffusivity              [m^2/s]
D_GB = 1e-12  # grain-boundary diffusivity       [m^2/s]
DELTA = 5e-10  # grain-boundary width             [m]
K_EX = 1e-6  # bulk <-> grain-boundary exchange  [m/s]
C0 = 1.0  # surface concentration

T_END, DT = 3600.0, 60.0

# Hydrogen is charged on the top edge of the map, y = LY, and diffuses in -y.
# The in-plane x direction is periodic in nobody's sense here: the left and
# right edges are simply free surfaces of the section.
CHARGED_EDGE = "y1"

# Keep only boundaries above this disorientation. 15 deg is the usual high-angle
# threshold and it is the reason to have gone to EBSD at all -- a synthetic
# tessellation has no meaningful theta distribution to filter on. Expect it to
# fragment the network more readily than the same threshold would in 3D;
# component_count() reports that, and in 2D it is a result rather than a bug.
THETA_MIN = 15.0
THETA_DEPENDENT_D = False  # see gb_diffusivity_field, CHECK before enabling

# --- meshing -----------------------------------------------------------------
# In 2D the cells are faces, so RCL sets the element size inside the grains and
# RCL_EDGE the size along the grain boundaries. RCL_VER refines the triple
# junctions, which is where the interesting transport happens and where the
# elements are worst; None leaves them at the edge value.
RCL = 0.8
RCL_EDGE = 0.2
RCL_VER = None
PL = 2.5  # progression factor: max length ratio between adjacent 1D elements

# -rsel, the small-edge length used by regularization. Neper's default is 1,
# picked to suit the *default* -rcl; the docs say it should track whatever -rcl
# you actually use, and a value of 1 corresponds to a length of 0.125 for a
# unit-area cell in 2D. Leaving it low relative to RCL lets short edges survive
# into the mesh, where they force pinch fixing and degenerate triangles. It
# matters more here than for a Voronoi tessellation, because a fitted Laguerre
# tessellation inherits the awkward near-degenerate edges of whatever grain
# arrangement was actually measured.
REG_RSEL = RCL

# Multimeshing retries each face with several algorithms until MESH_QUAL_MIN is
# reached, so quality target and meshing time trade off directly; 0.9 is Neper's
# default and 0.7 is reasonable while iterating. MESH_MAX_TIME caps the
# per-entity budget: the default is 1000 s, long enough for one pathological
# cell to stall a run without saying so.
MESH_QUAL_MIN = 0.7
MESH_MAX_TIME = None  # seconds per face; try 30 when diagnosing

STEM = "poly"
try:
    _HERE = Path(__file__).resolve().parent
except NameError:  # interactive session: no __file__
    _HERE = Path.cwd()
WORKDIR = _HERE / "results"
MESH_SCRIPT = _HERE / "ebsd_to_mesh.sh"

# Neper is a command-line program, so it does not have to live in the same conda
# environment as FESTIM -- it only has to be a path. Keeping it in its own
# environment avoids letting the solver rearrange a working dolfinx install over
# a dependency (GSL, scotch) that has nothing to do with FESTIM.
#
# GMSH_BIN is the *executable*, which Neper calls for 2D meshing. The
# conda-forge package providing it is `gmsh`; `python-gmsh` is only the
# bindings, so a dolfinx environment may have the API without the command.
NEPER_ENV = "/home/fenna/anaconda3/envs/neper-env/bin"
NEPER_BIN = os.path.join(NEPER_ENV, "neper")
GMSH_BIN = os.path.join(NEPER_ENV, "gmsh")

# The stat keys below are all scalars, one column each, one line per entity in
# id order. They are written by ebsd_to_mesh.sh from the *regularized*
# tessellation, which is the one that got meshed -- regularization renumbers
# edges, so stats taken before it would not match the .msh4 tags. Keep these
# tuples in step with the -stat* options in the shell script.
EDGE_KEYS = ("domtype", "domedge", "theta", "length", "ymin", "ymax")
VER_KEYS = ("domtype", "edgenb")


def run_interruptible(cmd, cwd=None, env=None):
    """Run the meshing pipeline, surviving a Ctrl+C long enough for it to finish.

    Neper treats SIGINT as "stop optimizing, keep the current solution and write
    the output" -- which matters here, because the tessellation fit is the long
    stage and a partially converged fit is a perfectly usable microstructure.
    But Ctrl+C goes to the whole foreground process group, so the Python parent
    gets it too, and if the parent exits immediately the child is killed
    part-way through writing. Catching it here and waiting lets the pipeline
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


def run_ebsd_pipeline(tesr=TESR, stem=STEM, workdir=WORKDIR, force=True):
    """Fit and mesh the EBSD map. Returns the base path (no extension).

    All the Neper invocations live in ebsd_to_mesh.sh; this only marshals the
    parameters and checks the binaries. Caching is per stage inside the script,
    so a failed -M does not cost the fit again.
    """
    base = (Path(workdir) / stem).resolve()
    base.parent.mkdir(parents=True, exist_ok=True)
    print(f"neper outputs -> {base.parent}")

    tesr_path = Path(tesr)
    if not tesr_path.is_absolute():
        tesr_path = (_HERE / tesr_path).resolve()
    if not tesr_path.is_file():
        raise FileNotFoundError(
            f"no EBSD map at {tesr_path}. Neper cannot read .ang/.ctf/.h5 -- "
            "the map has to be written as a raster tessellation first; see "
            "https://neper.info/doc/tutorials/ebsd_process.html"
        )

    if shutil.which(NEPER_BIN) is None and not Path(NEPER_BIN).is_file():
        raise FileNotFoundError(
            f"neper not found at {NEPER_BIN!r}. Install it with "
            "`conda install conda-forge::neper`, ideally into its own "
            "environment, and set NEPER_BIN to the binary's path."
        )
    if shutil.which(GMSH_BIN) is None and not Path(GMSH_BIN).is_file():
        raise FileNotFoundError(
            f"no gmsh executable at {GMSH_BIN!r}. Neper calls it for 2D "
            "meshing; the conda-forge package that provides the command is "
            "`gmsh` (`python-gmsh` is only the bindings)."
        )
    for p in (GMSH_BIN, NEPER_BIN, str(tesr_path)):
        if any(c.isspace() for c in str(p)):
            raise ValueError(
                f"the path {str(p)!r} contains whitespace. Neper re-tokenizes "
                "its arguments -- the input-file argument is a structured field "
                "supporting comma-separated files and colon-separated "
                "transformations -- so a path with a space in it arrives as "
                "several unusable fragments. Move or symlink it."
            )

    env = dict(os.environ)
    env.update(
        {
            "TESR": str(tesr_path),
            "NEPER_BIN": NEPER_BIN,
            "GMSH_BIN": GMSH_BIN,
            "STEM": stem,
            "WORKDIR": str(base.parent),
            "FORCE": "1" if force else "0",
            "CRYSYM": CRYSYM,
            "ORIDES": ORIDES,
            "OBJ_RES": str(OBJ_RES),
            "MORPHO_STOP": MORPHO_STOP,
            "RSEL": str(REG_RSEL),
            "RCL": str(RCL),
            "RCL_EDGE": str(RCL_EDGE),
            "PL": str(PL),
        }
    )
    if TESR_TRANSFORM:
        env["TESR_TRANSFORM"] = TESR_TRANSFORM
    if RCL_VER is not None:
        env["RCL_VER"] = str(RCL_VER)
    if MESH_QUAL_MIN:
        env["MESH_QUAL_MIN"] = str(MESH_QUAL_MIN)
    if MESH_MAX_TIME:
        env["MESH_MAX_TIME"] = str(MESH_MAX_TIME)

    run_interruptible(["bash", str(MESH_SCRIPT)], cwd=str(base.parent), env=env)
    return base


def domain_extent(base):
    """(Lx, Ly) of the meshed domain, from the raster geometry.

    The specimen is not a unit square, so nothing downstream may assume L = 1:
    the charged edge, the depth scan and the boundary area fraction all read
    their lengths from here.
    """
    cols = np.loadtxt(str(base) + ".sttesr", ndmin=2)[0]
    return float(cols[1]), float(cols[2])


class StatFile:
    """One Neper .st* file: scalar keys in columns, entities in id order.

    Ids are 1-based throughout Neper, so ``self.values[k]`` is entity ``k + 1``
    and :meth:`ids` converts a boolean mask back into ids.
    """

    def __init__(self, path, keys):
        raw = np.loadtxt(path, ndmin=2)
        if raw.shape[1] != len(keys):
            raise ValueError(
                f"{path} has {raw.shape[1]} columns but {len(keys)} keys were "
                f"expected ({', '.join(keys)}); the -stat option in "
                "ebsd_to_mesh.sh and the key tuple here have drifted apart"
            )
        self.values = {k: raw[:, i] for i, k in enumerate(keys)}
        self.n = raw.shape[0]

    def __getitem__(self, key):
        return self.values[key]

    @staticmethod
    def ids(mask):
        return np.flatnonzero(mask).astype(np.int32) + 1


class Microstructure:
    """The tessellation's own description of itself, read back from the stats.

    This replaces ``order_ring`` / ``clip_to_box`` / ``polygon_area`` /
    ``face_edges`` / ``connected_components`` and the Counter arithmetic over
    rounded coordinates: every number below is Neper's, computed on the exact
    topology rather than reconstructed from the geometry. With an EBSD-derived
    tessellation ``theta`` is also a measurement rather than a by-product of
    random orientation assignment, which is what makes THETA_MIN meaningful.
    """

    def __init__(self, base):
        self.edges = StatFile(str(base) + ".stedge", EDGE_KEYS)
        self.vertices = StatFile(str(base) + ".stver", VER_KEYS)
        self._check_interior_conventions()

    # In 2D the domain boundary is made of edges, so a tessellation edge is a
    # real grain boundary exactly when it lies on none of them. `domtype` is
    # 0/1 for an entity on a domain vertex/edge and negative otherwise, and
    # `domedge` is the domain edge id or -1. Two independent columns saying the
    # same thing, which is why both are written: if they ever disagree, the
    # assumption about the sign convention is what broke, and it is better to
    # hear about it here than to silently include the specimen surface in the
    # network and watch hydrogen short circuit around the outside.
    def _check_interior_conventions(self):
        by_domtype = self.edges["domtype"] < 0
        by_domedge = self.edges["domedge"] < 0
        if not np.array_equal(by_domtype, by_domedge):
            n = int((by_domtype != by_domedge).sum())
            raise RuntimeError(
                f"domtype and domedge disagree on {n} of {self.edges.n} edges "
                "about which are interior. Inspect the .stedge file: the "
                "interior sentinel is not the negative value assumed here."
            )

    @property
    def interior_mask(self):
        return self.edges["domtype"] < 0

    @property
    def network_mask(self):
        return self.interior_mask & (self.edges["theta"] > THETA_MIN)

    @property
    def network_edge_ids(self):
        return StatFile.ids(self.network_mask)

    @property
    def network_length(self):
        return float(self.edges["length"][self.network_mask].sum())

    def junction_only_below(self, y_top, tol=1e-12):
        """Deepest point reached by a boundary that touches the charged edge.

        Below this depth no boundary is fed directly, so whatever the network
        holds there has crossed at least one triple junction. ``ymin``/``ymax``
        are edge keys, so Neper reports it rather than it being scanned for.
        """
        touching = self.network_mask & (self.edges["ymax"] > y_top - tol)
        if not touching.any():
            return y_top
        return float(self.edges["ymin"][touching].min())

    # A triple junction is an interior vertex meeting three or more edges. In
    # 3D the same object is an interior *edge* meeting three or more faces, and
    # the 3D script's quadruple-point count has no 2D counterpart at all: four
    # grains meeting at a point is not a generic configuration in the plane.
    @property
    def triple_junctions(self):
        m = (self.vertices["domtype"] < 0) & (self.vertices["edgenb"] >= 3)
        return int(m.sum())

    def check_orientations(self, base):
        """Fail loudly if theta is not a real disorientation distribution.

        The failure mode this guards against is quiet: if the orientation file
        did not reach the tessellation, or the crystal symmetry was never set,
        every cell keeps the identity orientation, `theta` comes out zero or
        uniform-random, THETA_MIN filters the wrong edges, and the geometry
        looks flawless throughout.
        """
        ori = np.loadtxt(str(base) + "-grainori.txt", ndmin=2)
        theta = self.edges["theta"][self.interior_mask]
        problems = []
        if np.allclose(ori, 0.0):
            problems.append("every grain orientation in the tesr readout is zero")
        if theta.size and np.allclose(theta, 0.0):
            problems.append("every interior edge has theta = 0")
        if theta.size and CRYSYM == "cubic" and theta.max() > 63.0:
            # the maximum disorientation is ~62.8 deg for cubic symmetry;
            # exceeding it means the symmetry was not applied
            problems.append(
                f"max theta = {theta.max():.1f} deg exceeds the cubic bound, "
                "so CRYSYM did not reach the tessellation"
            )
        if problems:
            raise RuntimeError(
                "the orientations did not survive the tessellation fit: "
                + "; ".join(problems)
                + ". Check that -ori from_morpho and -morphooptiini ori:file() "
                "are both present in ebsd_to_mesh.sh, that the descriptor "
                f"({ORIDES}) matches the tesr, and that the orientation file "
                f"has one line per grain ({ori.shape[0]} lines read)."
            )
        return ori.shape[0]

    def check_units(self, extent, n_grains, delta, d_b, t_end):
        """Two ways a physical domain breaks parameters written for a unit square.

        The grain-boundary width has to be small compared with a grain and the
        diffusion distance small compared with the specimen, and neither is
        automatic once the domain is 40 microns instead of 1.
        """
        lx, ly = extent
        grain_size = np.sqrt(lx * ly / max(n_grains, 1))
        if delta > 0.05 * grain_size:
            print(
                f"  WARNING: delta = {delta:g} is not small against the grain "
                f"size (~{grain_size:g}); the slab idealisation is being "
                "stretched and the area fraction below is not a small number"
            )
        depth = 2 * np.sqrt(d_b * t_end)
        if depth > 0.5 * ly:
            print(
                f"  WARNING: lattice diffusion alone reaches {depth:g} in a "
                f"specimen only {ly:g} deep; there is no short-circuit regime "
                "to observe -- shorten T_END or lower D_B"
            )

    def report(self, extent, n_grains):
        kept, total = int(self.network_mask.sum()), int(self.interior_mask.sum())
        theta = self.edges["theta"][self.network_mask]
        lx, ly = extent
        print(f"microstructure: {n_grains} grains from {TESR} (2D)")
        print(f"  domain                          : {lx:g} x {ly:g}")
        print(f"  edges                           : {self.edges.n}")
        print(f"  grain boundaries (interior)     : {total}")
        print(f"  kept above {THETA_MIN:g} deg          : {kept}")
        if kept:
            print(
                f"  disorientation                  : "
                f"{theta.min():.1f} - {theta.max():.1f} deg"
                f" (mean {theta.mean():.1f})"
            )
        print(f"  triple junctions                : {self.triple_junctions}")
        print(f"  boundary length                 : {self.network_length:.4g}")


# mesh
def read_mesh(base, comm=MPI.COMM_WORLD, rank=0):
    """Read the Neper mesh into dolfinx.

    ``cell_tags`` carry the face (grain) id and ``facet_tags`` the tessellation
    edge id, because Neper writes every tessellation entity as an element set.
    In a 2D mesh the facets are line segments, so the tags that matter come from
    the 1D element sets -- the 3D script's 2D element sets, one dimension down.
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
micro = Microstructure(base)
LX, LY = domain_extent(base)
N_GRAINS = micro.check_orientations(base)
micro.check_units((LX, LY), N_GRAINS, DELTA, D_B, T_END)
mesh, cell_tags, facet_tags = read_mesh(base)

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
# every point where a grain boundary meets the charged edge, in one object: the
# locator runs on the network itself, so dim = mesh dimension - 2 = 0
mouths = F.SurfaceSubdomain(id=4, dim=0, locator=lambda x: np.isclose(x[1], LY))

c_b = F.Species("c_b", subdomains=[grains])
c_gb = F.Species("c_gb", subdomains=[network])


def solve(d_gb):
    """Run the problem, returning the model and the two solutions.

    ``d_gb == D_B`` is the reference case in which the boundaries are not short
    circuits at all.
    """
    network.material = F.Material(D_0=d_gb, E_D=0.0)
    fast = d_gb != D_B
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
        ]
        if fast
        else [],
    )
    model.initialise()
    if fast and THETA_DEPENDENT_D:
        # after initialise() the submesh exists; see the CHECK in the helper
        network.material = F.Material(
            D_0=gb_diffusivity_field(network, micro, D_B, D_GB, theta_c=THETA_MIN),
            E_D=0.0,
        )
        model.initialise()
    model.run()
    return (
        model,
        c_b.subdomain_to_post_processing_solution[grains],
        c_gb.subdomain_to_post_processing_solution[network],
    )


model, cb_fast, cgb_fast = solve(D_GB)


# what we built
def submesh_length(subdomain):
    """Total length of the network submesh -- the 2D analogue of its area."""
    sub = subdomain.submesh
    seg = sub.geometry.dofmap.reshape(-1, 2)[: sub.topology.index_map(1).size_local]
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
print(
    f"  mesh                            : "
    f"{mesh.topology.index_map(2).size_global} triangles"
)
print(
    f"  network captured by the submesh : {len_mesh:.4g} of {len_tess:.4g}"
    f" ({100 * len_mesh / len_tess:.2f} %)"
)
if n_comp is not None:
    print(f"  connected components            : {n_comp}")
print(f"  interior facets                 : {model.manifold_is_interior(network)}")


# effect of the network
def inventory(cb, cgb):
    """Total hydrogen per unit out-of-plane thickness.

    The grains contribute an area integral and the boundaries a line integral
    weighted by the slab width -- one dimension down from the 3D version, where
    it was a volume integral plus delta times an area integral.
    """
    dx_bulk = ufl.Measure("dx", domain=cb.function_space.mesh)
    dx_gb = ufl.Measure("dx", domain=cgb.function_space.mesh)
    total = dolfinx.fem.assemble_scalar(dolfinx.fem.form(cb * dx_bulk))
    total += DELTA * dolfinx.fem.assemble_scalar(dolfinx.fem.form(cgb * dx_gb))
    return mesh.comm.allreduce(total, op=MPI.SUM)


fast = inventory(cb_fast, cgb_fast)

# Below this depth no boundary is fed directly from the charged edge, so
# everything the network holds there has crossed at least one triple junction.
# This is a property of the tessellation, not of the partitioned mesh, so it is
# the same on every rank with no reduction needed.
junction_only_below = micro.junction_only_below(LY)

gb_y = cgb_fast.function_space.tabulate_dof_coordinates()[:, 1]
deep = gb_y < junction_only_below
c_deep = cgb_fast.x.array[deep].max() if deep.any() else 0.0

_, cb_ref, cgb_ref = solve(D_B)
ref = inventory(cb_ref, cgb_ref)

print(
    f"\nafter t = {T_END} (lattice diffusion alone reaches "
    f"~{2 * np.sqrt(D_B * T_END):.3g})"
)
print(f"  inventory with fast boundaries : {fast:.4e}")
print(f"  inventory with D_gb = D_b      : {ref:.4e}")
print(f"  enhancement                    : x {fast / ref:.1f}")

# For scale only: Hart's effective diffusivity is the upper bound you would get
# if every boundary ran straight along the gradient. A real network is tortuous
# and only partly connected to the source, so the observed enhancement is well
# below it -- and with THETA_MIN filtering, the network carrying the flux is
# smaller than the total boundary length anyway. In 2D f is a length fraction
# times delta rather than an area fraction times delta.
f_gb = DELTA * len_tess / (LX * LY)
print(f"  boundary area fraction f       : {f_gb:.3e}")
print(
    f"  Hart bound f D_gb + (1-f) D_b  : {f_gb * D_GB + (1 - f_gb) * D_B:.3e}"
    f"  (vs D_b = {D_B:.3e})"
)

beta = DELTA * (D_GB / D_B - 1) / (2 * np.sqrt(D_B * T_END))
print(f"  type-B parameter beta          : {beta:.0f}  (short circuit needs beta >> 1)")

bulk_y = cb_fast.function_space.tabulate_dof_coordinates()[:, 1]
c_grain_deep = cb_fast.x.array[bulk_y < junction_only_below].mean()

print("\njunction transport: no boundary touching the charged edge reaches below")
print(f"y = {junction_only_below:.4g}, so everything the network holds there has")
print("crossed at least one triple junction.")
print(f"  max c on the network there     : {c_deep:.4e}")
print(f"  mean c in the grains there     : {c_grain_deep:.4e}")
print(f"  ratio                          : x {c_deep / c_grain_deep:.0f}")
print(
    "\nconnectivity in 2D is not 3D connectivity: percolation thresholds are "
    "far lower in the plane, so read the enhancement as a lower bound unless "
    "the microstructure is columnar."
)

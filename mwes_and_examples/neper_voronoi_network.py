"""Short-circuit diffusion through the grain-boundary network of a Neper
polycrystal.

Same formulation as the in-situ Voronoi version: the whole network is **one**
codim-1 subdomain carrying **one** species, so the submesh built from all the
grain-boundary facets is topologically connected and hydrogen crosses from one
boundary to another with no junction condition to write.

What changes is only where the microstructure comes from. Neper is a
command-line program -- there is no library to link against and no bindings to
write -- so the integration is ``subprocess.run`` plus a mesh reader:

* ``neper -T`` writes a tessellation (``.tess``) and, with the ``-stat*``
  options, one line of data per tessellation vertex / edge / face. Those files
  are the lookup tables that turn mesh tags back into geometry and physics.
* ``neper -M`` meshes it. The mesh conforms to every grain boundary by
  construction, so ``occ.fragment`` and the Sutherland-Hodgman clipping of the
  Voronoi ridges both disappear, and so does the Distance/Threshold background
  field: ``-rclface`` sets the element size on the faces and ``-rcl`` the size
  in the cells.
* The 2D element sets come out of ``gmshio`` as ``facet_tags`` whose value is
  the tessellation face id, so the network is picked up from tags exactly as
  before -- no geometric locator, no ``near_faces``, no midpoint test.

Three things the scipy construction could not give at all:

* ``domface``, which separates true grain boundaries from the tessellation
  faces lying on the walls of the specimen (without it the network includes the
  free surface and hydrogen short circuits around the outside);
* ``theta``, the disorientation angle between the two grains, which is what
  actually decides whether a boundary is a fast path;
* exact topology of the triple lines and quadruple points, instead of the
  Counter/union-find reconstruction from rounded vertex coordinates.
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

# parameters
L = 1.0  # specimen size (Neper's default domain is cube(1,1,1))
N_CELLS = 100  # number of grains; Neper is happy into the 1e5 range
SEED = 1  # -id, the rng seed, so the microstructure is reproducible

D_B = 1e-3  # lattice diffusivity
D_GB = 30.0  # grain-boundary diffusivity
DELTA = 1e-3  # grain-boundary width
K_EX = 1.0  # bulk <-> grain-boundary exchange (see the Fisher example on units)
C0 = 1.0  # surface concentration

RCL = 0.8  # element size in the grain interiors, relative to average cell size
RCL_FACE = 0.2  # element size on the grain boundaries
RCL_EDGE = None  # element size on the triple lines; None = same as the faces
PL = 2.5  # progression factor: max length ratio between adjacent 1D elements

# -rsel, the small-edge length used by regularization. Neper's default is 1,
# picked to suit the *default* -rcl; the docs say it should track whatever -rcl
# you actually use. Leaving it low relative to RCL lets short edges and slivers
# survive into the mesh, where they force 2D-mesh pinch fixing and degenerate
# faces -- a large part of what makes -M slow. It costs nothing: it happens
# during -T.
REG_RSEL = RCL

# Meshing effort. Multimeshing retries each face and polyhedron with several
# algorithms until MESH_QUAL_MIN is reached, so quality target and meshing time
# trade off directly against each other; 0.9 is Neper's default and 0.7 is a
# reasonable setting while iterating on element sizes. MESH_MAX_TIME caps the
# per-entity budget: the default is 1000 s, long enough for one pathological
# cell to stall a run without saying so.
MESH_QUAL_MIN = 0.7
MESH_MAX_TIME = None  # seconds per face/polyhedron; try 30 when diagnosing
T_END, DT = 3.0, 0.05

# Two competing costs. The element *count* is dominated by RCL, since the faces
# are 2D and the interiors 3D. But the meshing *time* is driven by the ratio
# RCL/RCL_FACE: a steep size jump means Netgen grades hard, fails the quality
# target more often, and multimeshing retries. Pushing RCL up without moving
# RCL_FACE therefore buys a smaller mesh that takes longer to build. Neper meshes
# each face at RCL_FACE and then fills each polyhedron
# from that boundary mesh at RCL, grading between the two, so refinement near
# the boundaries is preserved however coarse the interior gets. With D_B and
# T_END as set here, lattice diffusion reaches only ~2*sqrt(D_B*T_END), so
# resolving deep grain interiors finely buys nothing -- the gradients are all
# within a fraction of a grain of the network.

THETA_MIN = 0.0  # keep only boundaries above this disorientation (degrees)
THETA_DEPENDENT_D = False  # see gb_diffusivity_field, CHECK before enabling

STEM = "poly"
# Anchored to this file rather than the process working directory, so the
# outputs land next to the script no matter where it was launched from.
# Path(".") would resolve against the CWD, which an IDE or notebook may set
# somewhere unexpected.
try:
    _HERE = Path(__file__).resolve().parent
except NameError:  # interactive session: no __file__
    _HERE = Path.cwd()
WORKDIR = _HERE / "results"

# Neper is a command-line program, so it does not have to live in the same
# conda environment as FESTIM -- it only has to be a path. Keeping it in its
# own environment avoids letting the solver rearrange a working dolfinx
# install over a dependency (GSL, scotch) that has nothing to do with FESTIM.
#
# NEPER_BIN: `neper`, or e.g. os.path.expanduser("~/miniforge3/envs/neper/bin/neper")
# GMSH_BIN : the *executable*, which Neper calls for 2D/3D meshing. The
#            conda-forge package providing it is `gmsh`; `python-gmsh` is only
#            the bindings, so a dolfinx environment may have the API without
#            the command. Passing it explicitly avoids depending on which
#            environment's PATH happens to be active.
NEPER_ENV = "/home/fenna/anaconda3/envs/neper-env/bin"
NEPER_BIN = os.path.join(NEPER_ENV, "neper")
GMSH_BIN = os.path.join(NEPER_ENV, "gmsh")


# generation
#
# -T and -M are two calls rather than one so that the .tess is a persistent
# object: the face ids the .stface lines refer to are the same ids -M turns
# into 2D element sets. Running them together would still work, but keeping the
# file makes the numbering auditable.
#
#   -reg 1            regularization: removes the small edges and faces that
#                     otherwise produce unusable tets at the triple lines --
#                     which is where this whole problem lives, so it matters
#                     more here than in a typical mechanics run. It makes
#                     internal faces slightly non-planar, which is the reason
#                     to let Neper mesh rather than rebuilding the geometry
#                     from the .tess with OCC addPlaneSurface.
#   -periodicity      MUTUALLY EXCLUSIVE with -reg: Neper rejects
#                     `-reg 1` on a periodic tessellation. The scipy version
#                     needed its 27-image tiling because Qhull returns
#                     unbounded ridges at the hull boundary; Neper clips to the
#                     domain natively, so periodicity here only removes
#                     free-surface artefacts on the x/y walls. That is worth
#                     less than good elements, so the default is off.
#                     To have both, suppress the small edges at the
#                     tessellation stage instead of regularizing afterwards:
#                     set PERIODICITY and leave REGULARIZE off, and RSEL is
#                     added to -morpho. `rsel` is the smallest edge length
#                     relative to the average cell size, and the -reg docs
#                     suggest matching it to the -rcl value.
#   -order 1          the default is 2.
#   -format msh4      Gmsh v4. The default `msh` is Neper's own dialect of Gmsh
#                     2.2, carrying extra sections ($Domain, $NSets, $Groups,
#                     $ElsetOrientations) that generic readers do not expect.
#
# The stat keys below are all scalars, one column each, one line per entity in
# id order. Keep the lists and the *_KEYS tuples in step.

REGULARIZE = True  # -reg 1
PERIODICITY = None  # e.g. "x,y"; incompatible with REGULARIZE
RSEL = None  # small-edge control when periodic; try RCL as a first value

# `voronoi` is a Poisson-Voronoi tessellation: uniform random seeds, no
# optimization stage, effectively instant. It is what scipy's Voronoi on
# uniform seeds produced, so it keeps the comparison with the old results
# honest.
#
# `gg` is the grain-growth morphology, an alias for a lognormal equivalent
# diameter and sphericity. It is more realistic, but fitting those
# distributions means solving for the seed positions and weights -- four
# variables per cell -- and that optimization can run for tens of thousands of
# iterations. Switch to it deliberately, to ask whether grain-size statistics
# change the answer, not as a default.
MORPHO = "voronoi"

# Only used when MORPHO involves an optimization. The Neper default is
# `eps<1e-6||val<1e-12`, whose second clause is unreachable for a distribution
# fit, so the run grinds on a plateau until the first one trips. A value or
# iteration cap makes the cost bounded and the result reproducible; Ctrl+C also
# stops the optimization and keeps the current solution.
MORPHO_STOP = "eps<1e-6||val<1e-3||iter>=20000"

FACE_KEYS = ("domface", "theta", "area", "zmin", "zmax")
EDGE_KEYS = ("domtype", "facenb", "length")
VER_KEYS = ("domtype", "edgenb")


def run_interruptible(cmd, cwd=None):
    """Run a Neper command, surviving a Ctrl+C long enough for it to finish.

    ``cwd`` is used to keep every path Neper sees short and relative -- see the
    note in :func:`run_neper` about whitespace in directory names.

    Neper treats SIGINT as "stop optimizing, keep the current solution and
    write the output". But Ctrl+C goes to the whole foreground process group,
    so the Python parent gets it too -- and if the parent exits immediately,
    the child is killed part-way through writing and you are left with partial
    files, or none. Catching it here and waiting lets Neper land its output;
    a second Ctrl+C still gets you out.
    """
    proc = subprocess.Popen(cmd, cwd=cwd)
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


def run_neper(
    n=N_CELLS,
    seed=SEED,
    rcl=RCL,
    rclface=RCL_FACE,
    stem=STEM,
    workdir=WORKDIR,
    force=False,
):
    """Generate and mesh a polycrystal. Returns the base path (no extension)."""
    base = (Path(workdir) / stem).resolve()
    base.parent.mkdir(parents=True, exist_ok=True)
    print(f"neper outputs -> {base.parent}")
    if base.with_suffix(".msh4").exists() and not force:
        print(f"  reusing {base.name}.msh4")
        return base

    if shutil.which(NEPER_BIN) is None and not Path(NEPER_BIN).is_file():
        raise FileNotFoundError(
            f"neper not found at {NEPER_BIN!r}. Install it with "
            "`conda install conda-forge::neper`, ideally into its own "
            "environment, and set NEPER_BIN to the binary's path."
        )
    if GMSH_BIN is None:
        raise FileNotFoundError(
            "no gmsh executable found. Neper calls it for 2D/3D meshing; the "
            "conda-forge package that provides the command is `gmsh` "
            "(`python-gmsh` is only the bindings). Set GMSH_BIN to its path."
        )
    if any(c.isspace() for c in str(GMSH_BIN)):
        raise ValueError(
            f"the gmsh path {str(GMSH_BIN)!r} contains whitespace. Neper "
            "re-tokenizes its arguments, so a path with a space in it is torn "
            "into fragments. Move or symlink the binary somewhere without one."
        )

    if REGULARIZE and PERIODICITY:
        raise ValueError(
            "Neper does not allow -regularization on a periodic tessellation. "
            "Either drop PERIODICITY (recommended: element quality at the "
            "triple lines matters more here than the x/y wall artefacts), or "
            "set REGULARIZE = False and give RSEL a value so the small edges "
            "are suppressed at the tessellation stage instead."
        )

    morpho = MORPHO
    if RSEL is not None:
        # `rsel` avoids small edges like -reg does, but during tessellation,
        # so it is compatible with periodicity. It combines with the `gg`
        # alias, which is itself just a pair of custom properties; combining
        # it with the `voronoi` keyword may not be accepted, since the special
        # morphologies are documented as mutually exclusive.
        morpho = f"{morpho},rsel:{RSEL}"

    tess = [NEPER_BIN, "-T", "-n", str(n), "-id", str(seed), "-morpho", morpho]
    if MORPHO != "voronoi" and MORPHO_STOP:
        tess += ["-morphooptistop", MORPHO_STOP]
    if REGULARIZE:
        tess += ["-reg", "1"]
        if REG_RSEL is not None:
            tess += ["-rsel", str(REG_RSEL)]
    if PERIODICITY:
        tess += ["-periodicity", PERIODICITY]
    tess += [
        "-statface",
        ",".join(FACE_KEYS),
        "-statedge",
        ",".join(EDGE_KEYS),
        "-statver",
        ",".join(VER_KEYS),
        "-o",
        stem,
    ]
    # -T can be the expensive half when the morphology needs optimization, and
    # it is a pure function of (n, seed, morpho), so cache it on its own rather
    # than only on the final mesh: a failure in -M should not cost it again.
    # Neper re-parses its input-file argument -- it is a structured field
    # supporting comma-separated files and colon-separated transformations --
    # and splits it on whitespace, so an absolute path through a directory like
    # "mwes + examples" arrives as several unusable fragments. Running with cwd
    # set to the output directory and passing bare names sidesteps it entirely,
    # and keeps the command lines readable in the log.
    wd = str(base.parent)
    # Gmsh scratch: Neper writes one .geo per tessellation face and polyhedron,
    # meshes it, reads the .msh back and deletes the pair. The default location
    # is the working directory, so a run that aborts before cleanup leaves
    # dozens of tmp*.geo / tmp*.msh files sitting among the real outputs.
    # Giving them their own directory keeps results/ readable and makes the
    # leftovers safe to delete wholesale -- though they are worth inspecting
    # first if it was -M itself that failed, since running gmsh on the
    # offending .geo by hand is the usual way to find out why.
    tmp = base.parent / "tmp"
    tmp.mkdir(exist_ok=True)
    if not base.with_suffix(".tess").exists() or force:
        run_interruptible(tess, cwd=wd)
    else:
        print(f"  reusing {base.name}.tess")
    run_interruptible(
        [
            NEPER_BIN,
            "-M",
            stem + ".tess",
            "-gmsh",
            GMSH_BIN,  # do not rely on whichever PATH is active
            "-order",
            "1",
            "-elttype",
            "tet",
            "-rcl",
            str(rcl),
            "-rclface",
            str(rclface),
            *(["-rcledge", str(RCL_EDGE)] if RCL_EDGE is not None else []),
            "-pl",
            str(PL),
            *(["-meshqualmin", str(MESH_QUAL_MIN)] if MESH_QUAL_MIN else []),
            *(
                [
                    "-mesh2dmaxtime",
                    str(MESH_MAX_TIME),
                    "-mesh3dmaxtime",
                    str(MESH_MAX_TIME),
                ]
                if MESH_MAX_TIME
                else []
            ),
            "-tmp",
            os.path.abspath(os.path.join(wd, "tmp")),
            "-format",
            "msh4",
            "-o",
            stem,
        ],
        cwd=wd,
    )
    leftovers = list(tmp.glob("*"))
    if leftovers:
        # -M succeeded, so anything still here is from an earlier failed run
        for f in leftovers:
            f.unlink()
        print(f"  cleared {len(leftovers)} stale gmsh scratch files")
    return base


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
                f"expected ({', '.join(keys)}); the -stat option and the key "
                "tuple have drifted apart"
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
    topology rather than reconstructed from the geometry.
    """

    def __init__(self, base):
        self.faces = StatFile(str(base) + ".stface", FACE_KEYS)
        self.edges = StatFile(str(base) + ".stedge", EDGE_KEYS)
        self.vertices = StatFile(str(base) + ".stver", VER_KEYS)

    # `domface` is the id of the domain face a tessellation face lies on, and
    # -1 when it lies on none. A face can only meet the boundary by lying on a
    # domain face, so domface < 0 is exactly "interior", i.e. a real grain
    # boundary rather than a piece of free surface.
    @property
    def interior_mask(self):
        return self.faces["domface"] < 0

    @property
    def network_mask(self):
        return self.interior_mask & (self.faces["theta"] > THETA_MIN)

    @property
    def network_face_ids(self):
        return StatFile.ids(self.network_mask)

    @property
    def network_area(self):
        return float(self.faces["area"][self.network_mask].sum())

    def junction_only_below(self, z_top, tol=1e-9):
        """Deepest point reached by a boundary that touches the charged face.

        Below this depth no boundary is fed directly, so whatever the network
        holds there has crossed at least one triple line. The Voronoi script
        scanned its own clipped polygons for this; ``zmin``/``zmax`` are face
        keys, so Neper reports it.
        """
        touching = self.network_mask & (self.faces["zmax"] > z_top - tol)
        if not touching.any():
            return z_top
        return float(self.faces["zmin"][touching].min())

    # For edges and vertices, `domtype` is 0/1/2 when the entity sits on a
    # domain vertex / edge / face, so a negative value means it is interior.
    # A triple line is an interior edge shared by three or more faces; a
    # quadruple point is an interior vertex meeting four or more edges.
    @property
    def triple_lines(self):
        m = (self.edges["domtype"] < 0) & (self.edges["facenb"] >= 3)
        return int(m.sum()), float(self.edges["length"][m].sum())

    @property
    def quadruple_points(self):
        m = (self.vertices["domtype"] < 0) & (self.vertices["edgenb"] >= 4)
        return int(m.sum())

    def report(self):
        n_tl, len_tl = self.triple_lines
        kept, total = int(self.network_mask.sum()), int(self.interior_mask.sum())
        theta = self.faces["theta"][self.network_mask]
        print(f"microstructure: {N_CELLS} cells, {self.faces.n} faces")
        print(f"  grain boundaries (interior)     : {total}")
        print(f"  kept above {THETA_MIN:g} deg          : {kept}")
        if kept:
            print(
                f"  disorientation                  : "
                f"{theta.min():.1f} - {theta.max():.1f} deg"
                f" (mean {theta.mean():.1f})"
            )
        print(f"  triple lines                    : {n_tl} (total length {len_tl:.3f})")
        print(f"  quadruple points                : {self.quadruple_points}")
        print(f"  boundary area                   : {self.network_area:.4f}")


# mesh
def read_mesh(base, comm=MPI.COMM_WORLD, rank=0):
    """Read the Neper mesh into dolfinx.

    ``cell_tags`` carry the polyhedron (grain) id and ``facet_tags`` the
    tessellation face id, because Neper writes every tessellation entity as an
    element set and the 2D physical ids run 1..facenb independently of the 3D
    ones.
    """
    result = gmshio.read_from_msh(str(base) + ".msh4", comm, rank, gdim=3)
    if hasattr(result, "mesh"):
        mesh, cell_tags, facet_tags = result.mesh, result.cell_tags, result.facet_tags
    else:
        mesh, cell_tags, facet_tags = result[0], result[1], result[2]
    if facet_tags is None or facet_tags.values.size == 0:
        raise RuntimeError(
            "no facet tags were read: the 2D element sets did not survive the "
            "msh4 round trip"
        )
    return mesh, cell_tags, facet_tags


class GrainBoundaryNetwork(F.VolumeSubdomain):
    """The whole network as one codim-1 subdomain: facets of a 3D mesh, ``dim=2``.

    The facets come straight from the tags, so the delicate part of the Voronoi
    version -- ``locate_entities`` marks an entity only when *all* its vertices
    satisfy the locator, which near a triple line also catches triangles that
    lie on no grain boundary in particular -- does not arise. There is nothing
    geometric left to get wrong.
    """

    def __init__(self, id, material, facet_tags, face_ids):
        super().__init__(id=id, material=material, dim=2)
        self.facet_tags = facet_tags
        self.face_ids = np.asarray(face_ids, dtype=np.int32)
        self.entity_face_ids = None  # face id of each located facet, in order

    def locate_subdomain_entities(self, mesh):
        keep = np.isin(self.facet_tags.values, self.face_ids)
        self.entity_face_ids = self.facet_tags.values[keep].astype(np.int32)
        return self.facet_tags.indices[keep].astype(np.int32)


def gb_diffusivity_field(network, micro, d_low, d_high, theta_c=15.0):
    """A per-face diffusivity as a DG0 field on the network submesh.

    Low-angle boundaries are not fast paths, and the disorientation is right
    there in the face table, so the natural refinement of the model is a
    diffusivity that varies from face to face. It has to enter as a coefficient
    on the *one* submesh: splitting the network into a high-angle and a
    low-angle subdomain would give two disconnected submeshes and put the
    junction conditions straight back.
    """
    submesh = network.submesh
    parent = None
    for name in ("submesh_to_mesh", "submesh_to_parent", "parent_to_submesh"):
        parent = getattr(network, name, None)
        if parent is not None:
            break
    if parent is None or network.entity_face_ids is None:
        raise NotImplementedError(
            "cannot map submesh cells back to tessellation faces: the "
            "subdomain does not expose its parent entity map under any of the "
            "expected names. Read it off dolfinx.mesh.create_submesh directly."
        )

    # entity_face_ids is aligned with the facet list returned by
    # locate_subdomain_entities, and create_submesh keeps that order, so the
    # parent map indexes straight into it: parent[c] is the position of
    # submesh cell c in the located facet list, hence its face id.
    face_of_cell = network.entity_face_ids
    theta = micro.faces["theta"]
    V = dolfinx.fem.functionspace(submesh, ("DG", 0))
    d = dolfinx.fem.Function(V, name="D_gb")
    tdim = submesh.topology.dim
    n_local = submesh.topology.index_map(tdim).size_local
    ids = face_of_cell[np.asarray(parent)[:n_local]]
    d.x.array[:n_local] = np.where(theta[ids - 1] >= theta_c, d_high, d_low)
    d.x.scatter_forward()
    return d


# build
base = run_neper()
micro = Microstructure(base)
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
    face_ids=micro.network_face_ids,
)
top = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[2], L))
# every curve where a grain boundary meets the charged face, in one object: the
# locator runs on the network itself, so dim = mesh dimension - 2
mouths = F.SurfaceSubdomain(id=4, dim=1, locator=lambda x: np.isclose(x[2], L))

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
                str(base.parent / "neper_grains.bp"), field=c_b, subdomain=grains
            ),
            F.VTXSpeciesExport(
                str(base.parent / "neper_network.bp"), field=c_gb, subdomain=network
            ),
        ]
        if fast
        else [],
    )
    model.initialise()
    if fast and THETA_DEPENDENT_D:
        # after initialise() the submesh exists; see the CHECK in the helper
        network.material = F.Material(
            D_0=gb_diffusivity_field(network, micro, D_B, D_GB), E_D=0.0
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
def submesh_area(subdomain):
    sub = subdomain.submesh
    tri = sub.geometry.dofmap.reshape(-1, 3)[: sub.topology.index_map(2).size_local]
    x = sub.geometry.x
    local = float(
        np.sum(
            0.5
            * np.linalg.norm(
                np.cross(x[tri[:, 1]] - x[tri[:, 0]], x[tri[:, 2]] - x[tri[:, 0]]),
                axis=1,
            )
        )
    )
    return sub.comm.allreduce(local, op=MPI.SUM)


def component_count(subdomain):
    """Connected components of the network submesh, through shared edges.

    The 2D version of this counted components by union-find over shared polygon
    endpoints; here the submesh's own topology answers it. Serial only -- in
    parallel the adjacency is partitioned and this would count per-rank pieces.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    sub = subdomain.submesh
    if sub.comm.size > 1:
        return None
    sub.topology.create_connectivity(2, 1)
    sub.topology.create_connectivity(1, 2)
    c2e = sub.topology.connectivity(2, 1)
    e2c = sub.topology.connectivity(1, 2)
    n = sub.topology.index_map(2).size_local
    rows, cols = [], []
    for e in range(sub.topology.index_map(1).size_local):
        cells = e2c.links(e)
        for a in cells:
            for b in cells:
                rows.append(a)
                cols.append(b)
    del c2e
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return connected_components(adj, directed=False)[0]


micro.report()
area_mesh, area_tess = submesh_area(network), micro.network_area
n_comp = component_count(network)
print(
    f"  mesh                            : "
    f"{mesh.topology.index_map(3).size_global} cells"
)
print(
    f"  network captured by the submesh : {area_mesh:.4f} of {area_tess:.4f}"
    f" ({100 * area_mesh / area_tess:.2f} %)"
)
if n_comp is not None:
    print(f"  connected components            : {n_comp}")
print(f"  interior facets                 : {model.manifold_is_interior(network)}")


# effect of the network
def inventory(cb, cgb):
    """Total hydrogen: the grains plus the boundary slabs (delta x area in 3D)."""
    dx_bulk = ufl.Measure("dx", domain=cb.function_space.mesh)
    dx_gb = ufl.Measure("dx", domain=cgb.function_space.mesh)
    total = dolfinx.fem.assemble_scalar(dolfinx.fem.form(cb * dx_bulk))
    total += DELTA * dolfinx.fem.assemble_scalar(dolfinx.fem.form(cgb * dx_gb))
    return mesh.comm.allreduce(total, op=MPI.SUM)


fast = inventory(cb_fast, cgb_fast)

# Below this depth no boundary is fed directly from the charged face, so
# everything the network holds there has crossed at least one triple line.
# This is a property of the tessellation, not of the partitioned mesh, so it
# is the same on every rank with no reduction needed.
junction_only_below = micro.junction_only_below(L)

gb_z = cgb_fast.function_space.tabulate_dof_coordinates()[:, 2]
deep = gb_z < junction_only_below
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
# below it.
f_gb = DELTA * area_tess / L**3
print(f"  boundary volume fraction f     : {f_gb:.3e}")
print(
    f"  Hart bound f D_gb + (1-f) D_b  : {f_gb * D_GB + (1 - f_gb) * D_B:.3e}"
    f"  (vs D_b = {D_B:.3e})"
)

beta = DELTA * (D_GB / D_B - 1) / (2 * np.sqrt(D_B * T_END))
print(f"  type-B parameter beta          : {beta:.0f}  (short circuit needs beta >> 1)")

bulk_z = cb_fast.function_space.tabulate_dof_coordinates()[:, 2]
c_grain_deep = cb_fast.x.array[bulk_z < junction_only_below].mean()

print("\njunction transport: no boundary touching the charged face reaches below")
print(f"z = {junction_only_below:.3f}, so everything the network holds there has")
print("crossed at least one triple line.")
print(f"  max c on the network there     : {c_deep:.4e}")
print(f"  mean c in the grains there     : {c_grain_deep:.4e}")
print(f"  ratio                          : x {c_deep / c_grain_deep:.0f}")

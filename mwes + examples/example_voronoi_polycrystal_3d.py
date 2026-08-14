"""Short-circuit diffusion through the grain-boundary network of a 3D Voronoi
polycrystal.

Same idea as the 2D example: the entire network is declared as **one** codim-1
subdomain with **one** species, so the submesh built from all the grain-boundary
entities is topologically connected and hydrogen crosses from one boundary to another
with no junction condition to write. In 3D "codim-1" means the network is made of
*facets* (triangles) rather than edges, so ``dim=2`` in a ``tdim == 3`` mesh, and the
junctions are triple *lines* (edges shared by three faces) plus quadruple *points*.
The submesh is connected through those shared edges exactly as the 2D one was connected
through shared vertices, so nothing about the FESTIM setup changes.

What does change relative to the 2D script:

* A Voronoi ridge in 3D is a convex *polygon*, not a segment, so Liang-Barsky is
  replaced by Sutherland-Hodgman clipping against the six half-spaces of the box.
* The network is tagged in gmsh with a physical group and picked up from the facet
  tags instead of being located geometrically. The geometric route still works (see
  ``near_faces`` / ``LOCATE_GEOMETRICALLY``) but a point-to-polygon distance is much
  more delicate than a point-to-segment distance, and the fragment operation already
  knows exactly which facets are grain boundaries.
* ``mouths`` is ``dim=1`` (curves where the network meets the charged face), because
  the locator runs on the network submesh and dim = mesh dimension - 2.
* The grain-boundary volume fraction is delta * area / L**3 instead of
  delta * length / L**2.
"""

from collections import Counter, defaultdict

from mpi4py import MPI

import dolfinx
import gmsh
import numpy as np
import ufl
from dolfinx.io.gmsh import model_to_mesh
from scipy.spatial import Voronoi

import festim as F

# parameters
L = 1.0  # specimen size
N_SEEDS = 12  # number of grains (a unit cube with 8 seeds gives grains ~L/2 across)
SEED = 3  # rng seed, so the microstructure is reproducible

D_B = 1e-3  # lattice diffusivity
D_GB = 30.0  # grain-boundary diffusivity
DELTA = 1e-3  # grain-boundary width
K_EX = 1.0  # bulk <-> grain-boundary exchange (see the Fisher example on units)
C0 = 1.0  # surface concentration

H_GB, H_BULK = 0.05, 0.15  # mesh size at the boundaries / in the grain interiors
T_END, DT = 3.0, 0.05

GB_TAG = 2  # physical group of the grain-boundary facets
MIN_FACE_AREA = 1e-8  # clipping can leave slivers at the walls; drop them
LOCATE_GEOMETRICALLY = False  # use near_faces instead of the facet tags


# microstructure
def order_ring(poly, normal):
    """Sort the vertices of a planar polygon into a ring by angle about the centroid.

    Qhull returns 3D ridge vertices in cyclic order in every case tested here, but the
    scipy documentation only describes ``ridge_vertices`` as the indices of the
    vertices forming the ridge, without promising an order, so sorting costs nothing
    and removes the assumption.
    """
    c = poly.mean(axis=0)
    u = poly[0] - c
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return poly[np.argsort(np.arctan2((poly - c) @ v, (poly - c) @ u))]


def clip_to_box(poly, size):
    """Sutherland-Hodgman clip of a convex polygon to ``[0, size]^3``."""
    planes = []
    for k in range(3):
        n = np.zeros(3)
        n[k] = 1.0
        planes.append((n, 0.0))  # keeps x_k >= 0
        planes.append((-n, -size))  # keeps x_k <= size
    out = poly
    for n, d in planes:
        if len(out) < 3:
            return None
        s = out @ n - d
        new = []
        for i in range(len(out)):
            j = (i + 1) % len(out)
            if s[i] >= -1e-12:
                new.append(out[i])
            if (s[i] > 0) != (s[j] > 0) and abs(s[i] - s[j]) > 1e-15:
                t = s[i] / (s[i] - s[j])
                new.append(out[i] + t * (out[j] - out[i]))
        out = np.array(new) if new else np.zeros((0, 3))
    if len(out) < 3:
        return None
    ring = [out[0]]
    for p in out[1:]:
        if np.linalg.norm(p - ring[-1]) > 1e-10:
            ring.append(p)
    if len(ring) > 1 and np.linalg.norm(ring[-1] - ring[0]) < 1e-10:
        ring.pop()
    return np.array(ring) if len(ring) >= 3 else None


def polygon_area(poly):
    """Area of a planar polygon given as an ordered ring, by the fan rule."""
    c = poly.mean(axis=0)
    return 0.5 * sum(
        float(np.linalg.norm(np.cross(poly[i] - c, poly[(i + 1) % len(poly)] - c)))
        for i in range(len(poly))
    )


def voronoi_faces(n_seeds, size, rng):
    """Voronoi ridges of a periodically tiled seed set, clipped to the box."""
    pts = rng.uniform(0, size, (n_seeds, 3))
    tiled = np.vstack(
        [
            pts + np.array([dx, dy, dz]) * size
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ]
    )
    vor = Voronoi(tiled)
    faces, seen = [], set()
    for ridge, (i, j) in zip(vor.ridge_vertices, vor.ridge_points):
        if -1 in ridge or len(ridge) < 3:
            continue
        # a ridge lies on the perpendicular bisector of its two seeds; projecting onto
        # that exact plane makes the polygon planar to machine precision, which is what
        # OCC's addPlaneSurface wants
        n = tiled[j] - tiled[i]
        n /= np.linalg.norm(n)
        d = 0.5 * (tiled[i] + tiled[j]) @ n
        poly = vor.vertices[ridge]
        poly = poly - np.outer(poly @ n - d, n)
        clipped = clip_to_box(order_ring(poly, n), size)
        if clipped is None or polygon_area(clipped) < MIN_FACE_AREA:
            continue
        key = tuple(np.round(clipped.mean(axis=0), 7))  # periodic images can coincide
        if key in seen:
            continue
        seen.add(key)
        faces.append(clipped)
    return faces


def near_faces(points, faces, tol=1e-7):
    """Vectorised test: is each point within ``tol`` of any polygon?

    Only needed if ``LOCATE_GEOMETRICALLY``. Distance to a convex polygon is the
    distance to its plane when the projection falls inside the ring, and the distance
    to the nearest edge otherwise.
    """
    p = np.asarray(points)
    hit = np.zeros(p.shape[1], dtype=bool)
    for poly in faces:
        n = np.cross(poly[1] - poly[0], poly[2] - poly[0])
        n /= np.linalg.norm(n)
        rel = p.T - poly[0]
        s = rel @ n
        proj = p.T - np.outer(s, n)
        inside = np.ones(p.shape[1], dtype=bool)
        best = np.full(p.shape[1], np.inf)
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            e = b - a
            inside &= (np.cross(np.broadcast_to(e, proj.shape), proj - a) @ n) > -tol
            t = np.clip(((proj - a) @ e) / (e @ e), 0.0, 1.0)
            best = np.minimum(best, np.linalg.norm(proj - (a + np.outer(t, e)), axis=1))
        dist = np.where(inside, np.abs(s), np.hypot(s, best))
        hit |= dist < tol
    return hit


def build_mesh(faces, size):
    """A tet mesh whose facets conform to every grain boundary, refined there."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("polycrystal3d")
    occ = gmsh.model.occ

    box = occ.addBox(0, 0, 0, size, size, size)
    surfaces = []
    for poly in faces:
        pts = [occ.addPoint(*p) for p in poly]
        lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        surfaces.append(occ.addPlaneSurface([occ.addCurveLoop(lines)]))
    # fragment forces the box to be split along every polygon, so the generated mesh
    # has facets lying exactly on the grain boundaries
    out, out_map = occ.fragment([(3, box)], [(2, s) for s in surfaces])
    occ.synchronize()

    gb_surfaces = sorted({t for entry in out_map[1:] for (d, t) in entry if d == 2})
    # every fragment of the box is the same material, so they all go in one group; the
    # network is then interior to a single volume subdomain, as in 2D
    gmsh.model.addPhysicalGroup(3, [t for (d, t) in out if d == 3], 1)
    gmsh.model.addPhysicalGroup(2, gb_surfaces, GB_TAG)

    field = gmsh.model.mesh.field
    field.add("Distance", 1)
    field.setNumbers(1, "SurfacesList", gb_surfaces)
    field.setNumber(1, "Sampling", 30)  # surface distance fields need sample points
    field.add("Threshold", 2)
    field.setNumber(2, "InField", 1)
    field.setNumber(2, "SizeMin", H_GB)
    field.setNumber(2, "SizeMax", H_BULK)
    field.setNumber(2, "DistMin", H_GB)
    field.setNumber(2, "DistMax", 5 * H_GB)
    field.setAsBackgroundMesh(2)
    for opt in (
        "Mesh.MeshSizeFromPoints",
        "Mesh.MeshSizeFromCurvature",
        "Mesh.MeshSizeExtendFromBoundary",
    ):
        gmsh.option.setNumber(opt, 0)

    gmsh.model.mesh.generate(3)
    result = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
    gmsh.finalize()
    if hasattr(result, "mesh"):
        return result.mesh, result.facet_tags
    return result[0], result[2]


class GrainBoundaryNetwork(F.VolumeSubdomain):
    """The whole network as one codim-1 subdomain: facets of a 3D mesh, so ``dim=2``.

    The facets come straight from the gmsh physical group. The 2D script instead
    located them geometrically and had to test facet midpoints, because
    ``locate_entities`` marks an entity when *all its vertices* satisfy the locator,
    which near a junction also catches short entities that merely touch two different
    boundaries. That problem is worse in 3D -- a triangle with all three vertices on a
    triple line lies on no grain boundary in particular -- so the tags are the safer
    route. ``near_faces`` is kept for meshes that arrive without tags.
    """

    def __init__(self, id, material, faces, facet_tags=None):
        super().__init__(id=id, material=material, dim=2)
        self.faces = faces
        self.facet_tags = facet_tags

    def locate_subdomain_entities(self, mesh):
        if not LOCATE_GEOMETRICALLY and self.facet_tags is not None:
            return self.facet_tags.find(GB_TAG).astype(np.int32)
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, 0)
        facet_to_vertex = mesh.topology.connectivity(tdim - 1, 0)
        candidates = dolfinx.mesh.locate_entities(
            mesh, tdim - 1, lambda x: near_faces(x, self.faces)
        )
        x = mesh.geometry.x
        midpoints = np.array(
            [x[facet_to_vertex.links(f)].mean(axis=0) for f in candidates]
        )
        keep = near_faces(midpoints.T, self.faces)
        return candidates[keep].astype(np.int32)


# build and solve
faces = voronoi_faces(N_SEEDS, L, np.random.default_rng(SEED))
mesh, facet_tags = build_mesh(faces, L)

grains = F.VolumeSubdomain(
    id=1,
    material=F.Material(D_0=D_B, E_D=0.0),
    locator=lambda x: np.full_like(x[0], True, dtype=bool),
)
network = GrainBoundaryNetwork(
    id=2,
    material=F.Material(D_0=D_GB, E_D=0.0),
    faces=faces,
    facet_tags=facet_tags,
)
top = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[2], L))
# every curve where a grain boundary meets the charged face, in one object: the locator
# runs on the network itself, so dim = mesh dimension - 2
mouths = F.SurfaceSubdomain(id=4, dim=1, locator=lambda x: np.isclose(x[2], L))

c_b = F.Species("c_b", subdomains=[grains])
c_gb = F.Species("c_gb", subdomains=[network])


def solve(d_gb):
    """Run the problem, returning the two solutions. ``d_gb == D_B`` is the reference
    case in which the boundaries are not short circuits at all."""
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
            F.VTXSpeciesExport("voronoi3d_grains.bp", field=c_b, subdomain=grains),
            F.VTXSpeciesExport("voronoi3d_network.bp", field=c_gb, subdomain=network),
        ]
        if d_gb != D_B
        else [],
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
def face_edges(poly):
    for i in range(len(poly)):
        a = tuple(np.round(poly[i], 7))
        b = tuple(np.round(poly[(i + 1) % len(poly)], 7))
        yield tuple(sorted((a, b)))


def connected_components(faces):
    """Union-find over shared polygon edges (the 2D version used shared endpoints)."""
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edge_to_face = defaultdict(list)
    for k, poly in enumerate(faces):
        find(k)
        for e in face_edges(poly):
            edge_to_face[e].append(k)
    for shared in edge_to_face.values():
        for k in shared[1:]:
            parent[find(shared[0])] = find(k)
    return len({find(k) for k in range(len(faces))})


edge_count = Counter(e for poly in faces for e in face_edges(poly))
triple_lines = [e for e, n in edge_count.items() if n >= 3]
triple_length = sum(
    float(np.linalg.norm(np.array(b) - np.array(a))) for a, b in triple_lines
)
corner_count = Counter(v for e in triple_lines for v in e)
quadruple = [v for v, n in corner_count.items() if n >= 4]
face_area = sum(polygon_area(poly) for poly in faces)

sub = network.submesh
tri = sub.geometry.dofmap.reshape(-1, 3)[: sub.topology.index_map(2).size_local]
x = sub.geometry.x
submesh_area = float(
    np.sum(
        0.5
        * np.linalg.norm(
            np.cross(x[tri[:, 1]] - x[tri[:, 0]], x[tri[:, 2]] - x[tri[:, 0]]), axis=1
        )
    )
)
submesh_area = mesh.comm.allreduce(submesh_area, op=MPI.SUM)

print(f"microstructure: {N_SEEDS} seeds, {len(faces)} boundary polygons")
print(
    f"  triple lines                    : {len(triple_lines)}"
    f" (total length {triple_length:.3f})"
)
print(f"  quadruple points                : {len(quadruple)}")
print(f"  connected components            : {connected_components(faces)}")
print(
    f"  mesh                            : "
    f"{mesh.topology.index_map(3).size_global} cells"
)
print(
    f"  network captured by the submesh  : {submesh_area:.4f} of {face_area:.4f}"
    f" ({100 * submesh_area / face_area:.2f} %)"
)
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
# The deepest a boundary touching the charged face reaches. Below this depth nothing is
# fed directly, so whatever arrives has crossed at least one triple line.
touching = [poly for poly in faces if poly[:, 2].max() > L - 1e-9]
junction_only_below = min((poly[:, 2].min() for poly in touching), default=L)

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

# For scale only: Hart's effective diffusivity is the upper bound you would get if every
# boundary ran straight along the gradient. A real network is tortuous and only partly
# connected to the source, so the observed enhancement is well below it.
f_gb = DELTA * face_area / L**3
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

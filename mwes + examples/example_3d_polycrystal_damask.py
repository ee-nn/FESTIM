"""Short-circuit diffusion through the grain-boundary network of a 3D polycrystal,
built with DAMASK's voxel-grid tessellation instead of a polygon-based one.

Same FESTIM problem as the other versions: one codim-1 subdomain, one species,
hydrogen crosses from one boundary to another with no junction condition to write.

Architecturally this version is different from the microgen and MicroStructPy
versions, not just a swapped generator:

* DAMASK's `damask.seeds` + `damask.GeomGrid.from_Laguerre_tessellation` are plain
  numpy/scipy.spatial -- no compiled Voronoi binding, no CAD kernel, so this avoids
  the pyvoro/cadquery version-freshness problems entirely. But the output is a
  structured voxel array of material (grain) IDs, not a polygonal tessellation.
* Instead of handing that to gmsh, this script builds a conforming tetrahedral mesh
  directly: each voxel is split into 6 tets along the same body diagonal (the
  "Kuhn"/"Freudenthal" triangulation of a cubic grid). Because every voxel uses the
  *same* diagonal, the triangulation of any face shared by two neighbouring voxels is
  identical from both sides, so the resulting mesh is conforming with no extra work.
  This means no gmsh, no OpenCASCADE, no Neper -- the whole pipeline is DAMASK +
  numpy + dolfinx.
* Grain-boundary facets are identified the same way as in the microgen version, just
  on a tet mesh instead of a CAD assembly: an interior facet is a grain-boundary facet
  if its two neighbouring cells carry different material IDs. Material IDs are looked
  up by cell midpoint after mesh creation (not by input order), so this is correct
  under parallel partitioning/reordering, unlike a naive index-based tagging would be.

See the caveats in the reply this was generated alongside: voxel-resolution
staircase boundaries instead of smooth polygons, uniform mesh resolution (no
`H_GB`/`H_BULK`-style grading), and a heuristic (not fitted) grain-size distribution.
This has not been run against a live install -- written against the DAMASK and
dolfinx APIs as currently documented, not tested.

Requirements beyond the earlier scripts: `pip install damask` (or via conda-forge).
No gmsh, cadquery, or Neper needed for this version.
"""

from mpi4py import MPI

import basix.ufl
import damask
import dolfinx
import numpy as np
import ufl

import festim as F

# parameters
L = 1.0  # specimen size
CELLS = [24, 24, 24]  # voxel grid resolution -- this is the only knob controlling how
# faithfully the (staircase) grain-boundary surface is resolved;
# raise it for a better GB shape at a roughly cubic cost in cells
N_GRAINS = 12  # target number of grains
MEAN_D, SIGMA = 0.5, 0.15  # target lognormal grain-diameter distribution -- see the
# module docstring: this drives Laguerre *weights*
# heuristically, it is not a calibrated fit
SEED = 3  # rng seed, so the microstructure is reproducible

D_B = 1e-3  # lattice diffusivity
D_GB = 30.0  # grain-boundary diffusivity
DELTA = 1e-3  # grain-boundary width
K_EX = 1.0  # bulk <-> grain-boundary exchange (see the Fisher example on units)
C0 = 1.0  # surface concentration

T_END, DT = 3.0, 0.05

GB_TAG = 2  # physical group of the grain-boundary facets
BULK_TAG = 1


# microstructure
def generate_grid(size, cells, n_grains, mean_d, sigma, seed):
    """DAMASK voxel grid of grain (material) IDs, via a Laguerre tessellation with
    weights biased toward a lognormal diameter spread.

    `periodic=False` is used deliberately even though DAMASK supports periodic
    tessellations natively (a genuine advantage over the earlier scripts, if you want
    an RVE) -- the FESTIM model here has a fixed top/bottom, not periodic, boundary
    condition, so periodicity would be inconsistent with the rest of the setup.
    """
    size = np.asarray(size, dtype=float)
    cells = np.asarray(cells, dtype=int)

    seeds = damask.seeds.from_random(size, n_grains, cells, rng_seed=seed)

    rng = np.random.default_rng(seed)
    target_diameters = rng.lognormal(mean=np.log(mean_d), sigma=sigma, size=n_grains)
    weights = (target_diameters / 2.0) ** 2  # power-diagram weight ~ radius^2

    grid = damask.GeomGrid.from_Laguerre_tessellation(
        cells, size, seeds, weights, periodic=False
    )
    return grid.material  # (nx, ny, nz) int array of grain IDs


def build_tet_mesh(size, cells, comm):
    """A conforming tet mesh over the domain, one voxel = 6 tets, via a fixed body
    diagonal (Kuhn triangulation) so neighbouring voxels' shared faces always agree.

    Pure-Python triple loop -- fine for a few tens of thousands of voxels (the default
    24^3 grid here is ~83k tets); vectorize with numpy indexing if you push `CELLS`
    much higher.
    """
    nx, ny, nz = (int(c) for c in cells)
    xs = np.linspace(0.0, size[0], nx + 1)
    ys = np.linspace(0.0, size[1], ny + 1)
    zs = np.linspace(0.0, size[2], nz + 1)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    def vid(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                v0 = vid(i, j, k)
                v1 = vid(i + 1, j, k)
                v2 = vid(i + 1, j + 1, k)
                v3 = vid(i, j + 1, k)
                v4 = vid(i, j, k + 1)
                v5 = vid(i + 1, j, k + 1)
                v6 = vid(i + 1, j + 1, k + 1)
                v7 = vid(i, j + 1, k + 1)
                # all 6 tets share the same body diagonal v0-v6: this is what keeps
                # the triangulation of every cube face identical from both
                # neighbouring voxels
                tets.extend(
                    (
                        (v0, v1, v2, v6),
                        (v0, v2, v3, v6),
                        (v0, v3, v7, v6),
                        (v0, v7, v4, v6),
                        (v0, v4, v5, v6),
                        (v0, v5, v1, v6),
                    )
                )
    tets = np.array(tets, dtype=np.int64)

    element = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
    domain = ufl.Mesh(element)
    return dolfinx.mesh.create_mesh(comm, cells=tets, e=domain, x=points)


def tag_grains_and_boundaries(mesh, size, cells, material):
    """Look up each cell's material ID by its midpoint (robust to whatever cell
    order/partitioning dolfinx ends up with, unlike tagging by input row index), then
    mark interior facets that separate two different materials as the network.
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1
    voxel_size = np.asarray(size, dtype=float) / np.asarray(cells, dtype=float)

    n_cells = mesh.topology.index_map(tdim).size_local
    midpoints = dolfinx.mesh.compute_midpoints(
        mesh, tdim, np.arange(n_cells, dtype=np.int32)
    )
    ijk = np.floor(midpoints / voxel_size).astype(int)
    ijk = np.clip(ijk, 0, np.asarray(cells, dtype=int) - 1)
    cell_material = material[ijk[:, 0], ijk[:, 1], ijk[:, 2]]

    mesh.topology.create_connectivity(fdim, tdim)
    conn = mesh.topology.connectivity(fdim, tdim)
    offsets, data = conn.offsets, conn.array
    counts = np.diff(offsets)
    interior = np.where(counts == 2)[0]
    c0 = data[offsets[interior]]
    c1 = data[offsets[interior] + 1]
    gb_facets = interior[cell_material[c0] != cell_material[c1]].astype(np.int32)

    facet_tags = dolfinx.mesh.meshtags(
        mesh, fdim, gb_facets, np.full(len(gb_facets), GB_TAG, dtype=np.int32)
    )
    cell_tags = dolfinx.mesh.meshtags(
        mesh,
        tdim,
        np.arange(n_cells, dtype=np.int32),
        np.full(n_cells, BULK_TAG, dtype=np.int32),
    )
    n_grains_realized = len(np.unique(cell_material))
    return facet_tags, cell_tags, gb_facets.size, n_grains_realized


class GrainBoundaryNetwork(F.VolumeSubdomain):
    """The whole network as one codim-1 subdomain: facets of a 3D mesh, so `dim=2`.
    Same pattern as the other versions -- only the facet tags differ in origin.
    """

    def __init__(self, id, material, facet_tags):
        super().__init__(id=id, material=material, dim=2)
        self.facet_tags = facet_tags

    def locate_subdomain_entities(self, mesh):
        return self.facet_tags.find(GB_TAG).astype(np.int32)


# build and solve
grain_material = generate_grid([L, L, L], CELLS, N_GRAINS, MEAN_D, SIGMA, SEED)
mesh = build_tet_mesh([L, L, L], CELLS, MPI.COMM_WORLD)
facet_tags, cell_tags, n_gb_facets, n_grains_realized = tag_grains_and_boundaries(
    mesh, [L, L, L], CELLS, grain_material
)

grains = F.VolumeSubdomain(
    id=1,
    material=F.Material(D_0=D_B, E_D=0.0),
    locator=lambda x: np.full_like(x[0], True, dtype=bool),
)
network = GrainBoundaryNetwork(
    id=2,
    material=F.Material(D_0=D_GB, E_D=0.0),
    facet_tags=facet_tags,
)
top = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[2], L))
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
            F.VTXSpeciesExport("damask3d_grains.bp", field=c_b, subdomain=grains),
            F.VTXSpeciesExport("damask3d_network.bp", field=c_gb, subdomain=network),
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
print(
    f"microstructure: {N_GRAINS} grains requested, {n_grains_realized} realized "
    f"on a {CELLS[0]}x{CELLS[1]}x{CELLS[2]} voxel grid"
)
print(f"  grain-boundary facets           : {n_gb_facets}")
print(
    f"  mesh                            : "
    f"{mesh.topology.index_map(3).size_global} cells"
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

# real (meshed) grain-boundary area, from the network submesh directly, same as the
# microgen version -- neither of these two scripts keeps a raw polygon list around
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

_, cb_ref, cgb_ref = solve(D_B)
ref = inventory(cb_ref, cgb_ref)

print(
    f"\nafter t = {T_END} (lattice diffusion alone reaches "
    f"~{2 * np.sqrt(D_B * T_END):.3g})"
)
print(f"  inventory with fast boundaries : {fast:.4e}")
print(f"  inventory with D_gb = D_b      : {ref:.4e}")
print(f"  enhancement                    : x {fast / ref:.1f}")

f_gb = DELTA * submesh_area / L**3
print(f"  boundary volume fraction f     : {f_gb:.3e}")
print(
    f"  Hart bound f D_gb + (1-f) D_b  : {f_gb * D_GB + (1 - f_gb) * D_B:.3e}"
    f"  (vs D_b = {D_B:.3e})"
)

beta = DELTA * (D_GB / D_B - 1) / (2 * np.sqrt(D_B * T_END))
print(f"  type-B parameter beta          : {beta:.0f}  (short circuit needs beta >> 1)")

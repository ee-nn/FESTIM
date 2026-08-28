"""Díaz-Rodríguez et al. (2022) OKMC permeation setup, as a FESTIM continuum model.

Their box: a single columnar grain, L x L lateral, d deep, with the four
*lateral* faces defined as grain boundary. Permeation runs along the column
axis (z), from a charged surface at z=0 to a sink at z=d. L = 50, 100, 150 nm;
d = 50, 100, 200 nm. They could not reach the experimental d = 2 um because
the OKMC times were prohibitive (their Sect. 3.3). That is the gap this script
fills: the continuum model has no such cost, so we run the experimental
thickness over many columns.

WHAT A REGULAR NX x NY TILING DOES AND DOES NOT TEST
---------------------------------------------------
With identical columns and no-flux lateral boundaries, every column sees the
same environment, so 100 x 100 columns give the same per-column answer as one
column. The regular tiling is therefore a *verification* run: if the flux
fraction moves when NX changes, something is wrong with the coupling or the
facet location. Set DISORDER=1 to make the array non-trivial -- it draws a
per-boundary GB migration energy from the 0.191-0.547 eV range Wei et al.
(2026, doi:10.1016/j.ijhydene.2025.152835) report across eight tungsten GBs,
which spans a factor of ~4e3 in D_gb at 500 K. That is the regime no single-
column OKMC box can reach and is the actual reason to run 10,000 grains.

PARAMETERS AND WHERE THEY COME FROM
-----------------------------------
From the paper (Sect. 2.4, 3.2, Figs. 3, 7, 8):
  E_m,bulk = 0.20 eV, E_m,GB = 0.12 eV, E_bind = 1.05 eV  (DFT, their Fig. 3)
  max GB occupation 0.25 - 6.25 H/nm^2, swept; 6.25 matched experiment
  charged surface held at 5.6 H/nm^3 over a 0.4 nm near-surface region
  temperatures 520-705 K

NOT in the paper, assumed here and flagged:
  * Arrhenius prefactors. The paper says the parametrisation is that of its
    ref. [9] and does not restate it. D0_BULK = 1.9e-7 m^2/s is the standard
    tungsten value whose barrier (0.20 eV) is exactly their E_m,bulk, so it is
    the consistent choice. D0_GB is taken as 1.5 x D0_BULK, the ratio of the
    2D to the 3D random-walk prefactor (1/4 vs 1/6) at equal jump distance and
    attempt frequency -- they state explicitly that GB migration is 2D and
    bulk migration 3D.
  * DELTA, the GB slab width. Their OKMC GB is a discrete atomic plane with an
    areal site density, so no width is defined. It matters here because the GB
    current is DELTA * D_gb * grad(c_gb): f_GB is close to linear in DELTA.
    0.5 nm is the order of their 0.4 nm near-surface region. Treat any f_GB
    from this script as carrying that proportionality.
  * The prefactor in s. Detailed balance gives s = (lam_b nu_f)/(lam_gb nu_r)
    exp(E_bind/kT); the ratio is O(1) and is set to 1.

  39 H/nm^2 is NOT the trap density. That is the 0 K DFT filling limit of
  their Fig. 6, which they say explicitly is not reached at temperature and is
  not what the OKMC used. The OKMC used 0.25-6.25 H/nm^2.

THE COUPLING LAW
----------------
One-way fluxes, not a symmetric driving force. Per unit GB area,

    J = K c_b (1 - c_gb/N_gb)  -  (K/s) c_gb

The forward coefficient K is kinetic and follows from the bulk->GB barrier
alone; s is thermodynamic and follows from E_bind alone; the reverse
coefficient K/s is fixed by detailed balance and is never specified
independently. N_gb is the capacity and is the only trap-like part.
Equilibrium gives the McLean isotherm, which is the isotherm Zhou et al. use.

Saturation is expected to dominate here. At 705 K, s = exp(1.05/kT) ~ 3e7 and
N_gb/s ~ 4e20 m^-3, against a surface concentration of 5.6e27 m^-3. The
boundaries are therefore full near the inlet and can only develop the gradient
that carries current deeper in. A saturated boundary carries no net flux, so
the max-occupancy sweep is not a detail -- it is the controlling physics, and
it is why the paper's own permeability rises with the occupancy limit.

Outputs diaz-flux-fraction.png, diaz-permeability.png and a CSV.
"""

from mpi4py import MPI

import dolfinx
import matplotlib
import numpy as np
import ufl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import festim as F

# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------
L = 100e-9  # column side, m                     (their L, 100 nm case)
D_THICK = 2e-6  # layer thickness, m             (the experimental 2 um)
NX = 7  # columns per side; 100 -> 10 um x 10 um
NY = NX
CPG = 3  # mesh cells across one column, per axis
NZ = 80  # mesh cells through the thickness

# CPG >= 2 is required for the facet locator below: with one cell per column a
# tetrahedron can have vertices on two different column planes and would be
# mislabelled as a boundary facet. However, it can be only a few across because
# is very fast. At 705 K, D_bulk ~ 7e-9 m^2/s, so (L/2)^2/D ~
# 4e-7 s across the column against d^2/D ~ 6e-4 s along it.

LX, LY = NX * L, NY * L

# ----------------------------------------------------------------------------
# material and coupling parameters
# ----------------------------------------------------------------------------
E_M_BULK = 0.20  # eV, DFT, their Fig. 3
E_M_GB = 0.12  # eV, DFT, their Fig. 3
E_BIND = 1.05  # eV, DFT, H binding to an empty GB
E_FORWARD = 0.145  # eV, bulk->GB approach barrier; Zhou 2010 gives 0.13-0.16

D0_BULK = 1.9e-7  # m^2/s, assumed (see header)
D0_GB = 1.5 * D0_BULK  # m^2/s, assumed: 2D vs 3D random-walk prefactor

DELTA = 5e-10  # GB slab width, m, assumed
N_GB_AREAL = 6.25e18  # m^-2, their max occupancy
N_GB = N_GB_AREAL / DELTA  # m^-3, volumetric capacity

LAMBDA_JUMP = 1.12e-10  # m, tetrahedral-site hop in bcc W, for K
NU_ATTEMPT = 1e13  # s^-1

C_SURF = 5.6e27  # m^-3, their 5.6 H/nm^3 charged-surface concentration

TEMPERATURES = [520.0, 570.0, 615.0, 660.0, 705.0]  # their experimental range
T_END = 1.0  # s; steady state is reached in
DT0 = 1e-6  # ~d^2/D ~ 6e-4 s at 705 K

DISORDER = 0
E_M_GB_RANGE = (0.191, 0.547)  # eV, Wei et al. 2026, eight tungsten GBs
SEED = 0


def s_partition(T):
    """Equilibrium GB/bulk concentration ratio. Thermodynamic, from E_bind."""
    return np.exp(E_BIND / (F.k_B * T))


def k_forward(T):
    """Interfacial mass transfer coefficient, m/s. Kinetic, from E_forward."""
    return LAMBDA_JUMP * NU_ATTEMPT * np.exp(-E_FORWARD / (F.k_B * T))


# ----------------------------------------------------------------------------
# mesh: a structured box whose node planes fall exactly on the column walls
# ----------------------------------------------------------------------------
comm = MPI.COMM_WORLD
mesh = dolfinx.mesh.create_box(
    comm,
    [np.array([0.0, 0.0, 0.0]), np.array([LX, LY, D_THICK])],
    [NX * CPG, NY * CPG, NZ],
    cell_type=dolfinx.mesh.CellType.tetrahedron,
)
if comm.rank == 0:
    ntet = NX * CPG * NY * CPG * NZ * 6
    print(f"domain {LX * 1e6:.1f} x {LY * 1e6:.1f} x {D_THICK * 1e6:.1f} um")
    print(f"{NX * NY} columns, L = {L * 1e9:.0f} nm, ~{ntet:.3e} tetrahedra")
    if ntet > 2e7:
        print("  WARNING: this is a cluster-sized problem. Reduce NX or CPG.")


class ColumnWalls(F.VolumeSubdomain):
    """The whole GB network as one codim-1 subdomain: interior facets, dim=2.

    Same construction as GrainBoundaryNetwork in ebsd_gb_diffusion.py -- one
    connected submesh carrying one species, so hydrogen crosses from one wall
    to another with no junction condition. Here the facets are found
    geometrically rather than from tags, because the geometry is a regular
    tiling and there is no tessellation to read.

    The x-normal and y-normal wall sets are located separately and unioned.
    Locating them with a single predicate would also catch facets that have
    some vertices on an x-wall and the rest on a y-wall, which near a column
    corner lie on no wall at all.
    """

    def __init__(self, id, material):
        super().__init__(id=id, material=material, dim=2)
        self.entity_axis = None  # 0 or 1 per located facet, for the D_gb field

    def locate_subdomain_entities(self, mesh):
        tol = 0.05 * L / CPG

        def on_walls(axis):
            def pred(x):
                r = np.remainder(x[axis] + tol, L)
                interior = (x[axis] > tol) & (x[axis] < (LX if axis == 0 else LY) - tol)
                return (r < 2 * tol) & interior

            return pred

        fx = dolfinx.mesh.locate_entities(mesh, 2, on_walls(0))
        fy = dolfinx.mesh.locate_entities(mesh, 2, on_walls(1))
        self.entity_axis = np.concatenate(
            [np.zeros(fx.size, dtype=np.int8), np.ones(fy.size, dtype=np.int8)]
        )
        return np.concatenate([fx, fy]).astype(np.int32)


def gb_diffusivity_field(network, T):
    """Per-boundary D_gb as a DG0 field on the network submesh.

    Same idea as gb_diffusivity_field in ebsd_gb_diffusion.py: the coefficient
    has to live on the one submesh, because splitting the network into two
    subdomains would disconnect it and reintroduce junction conditions. Here
    the per-boundary label is a random draw rather than a measured
    disorientation -- swap in theta when you point this at a real map.
    """
    submesh = network.submesh
    parent = None
    for name in ("submesh_to_mesh", "submesh_to_parent", "parent_to_submesh"):
        parent = getattr(network, name, None)
        if parent is not None:
            break
    if parent is None:
        raise NotImplementedError("cannot map submesh cells to parent facets")

    V = dolfinx.fem.functionspace(submesh, ("DG", 0))
    d = dolfinx.fem.Function(V, name="D_gb")
    n_local = submesh.topology.index_map(submesh.topology.dim).size_local

    # one barrier per wall, not per element: label walls by (axis, index) so a
    # whole boundary plane shares a value, which is what a boundary character
    # means. Facet midpoints give the wall index.
    mid = dolfinx.mesh.compute_midpoints(
        submesh, submesh.topology.dim, np.arange(n_local, dtype=np.int32)
    )
    ix = np.rint(mid[:, 0] / L).astype(int)
    iy = np.rint(mid[:, 1] / L).astype(int)
    on_x = np.isclose(mid[:, 0] / L, ix, atol=0.1)
    wall_id = np.where(on_x, ix, 10_000 + iy)

    rng = np.random.default_rng(SEED)
    lo, hi = E_M_GB_RANGE
    barrier = rng.uniform(lo, hi, size=20_001)[wall_id]
    d.x.array[:n_local] = D0_GB * np.exp(-barrier / (F.k_B * T))
    d.x.scatter_forward()
    return d


# ----------------------------------------------------------------------------
# flux accounting
# ----------------------------------------------------------------------------
def outlet_flux(c, D, z_out, weight=1.0):
    """Axial current leaving through z = z_out, as an assembled UFL form.

    Works for both the 3D bulk field (an area integral over the outlet face)
    and the 2D GB submesh field (a line integral over the outlet edge of the
    submesh), because in each case it is the exterior-facet measure of that
    field's own mesh restricted to the outlet plane. `weight` carries DELTA for
    the GB, which turns its per-unit-width current into a current.
    """
    m = c.function_space.mesh
    fdim = m.topology.dim - 1
    facets = dolfinx.mesh.locate_entities_boundary(
        m, fdim, lambda x: np.isclose(x[2], z_out, atol=1e-12)
    )
    tags = dolfinx.mesh.meshtags(
        m, fdim, np.sort(facets), np.ones(facets.size, dtype=np.int32)
    )
    ds = ufl.Measure("ds", domain=m, subdomain_data=tags)
    n = ufl.FacetNormal(m)
    # the submesh of a plane set has no meaningful outward normal in 3D, so
    # take the axial component of the gradient directly rather than dot(.., n)
    form = -weight * D * ufl.grad(c)[2] * ds(1)
    local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(form))
    return m.comm.allreduce(local, op=MPI.SUM)


# ----------------------------------------------------------------------------
# one temperature
# ----------------------------------------------------------------------------
def run(T):
    D_b = D0_BULK * np.exp(-E_M_BULK / (F.k_B * T))
    D_g = D0_GB * np.exp(-E_M_GB / (F.k_B * T))
    s = s_partition(T)
    K = k_forward(T)

    grains = F.VolumeSubdomain(
        id=1,
        material=F.Material(D_0=D_b, E_D=0.0),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    network = ColumnWalls(id=2, material=F.Material(D_0=D_g, E_D=0.0))
    inlet = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[2], 0.0))
    outlet = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[2], D_THICK))
    # the lines where the walls meet the charged face and the sink face
    mouths = F.SurfaceSubdomain(id=5, dim=1, locator=lambda x: np.isclose(x[2], 0.0))
    drains = F.SurfaceSubdomain(
        id=6, dim=1, locator=lambda x: np.isclose(x[2], D_THICK)
    )

    c_b = F.Species("c_b", subdomains=[grains])
    c_gb = F.Species("c_gb", subdomains=[network])

    # the isotherm value at the charged face, so the mouth condition is the
    # equilibrium one rather than an arbitrary number
    beta = s * C_SURF / N_GB
    c_gb_surf = N_GB * beta / (1.0 + beta)

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[c_b, c_gb],
        subdomains=[grains, network, inlet, outlet, mouths, drains],
        sources=[
            F.ParticleSource(
                # two faces per slab -> 2/DELTA; one-way fluxes, blocking on
                # the forward term only (the bulk is dilute so reverse
                # blocking is negligible)
                value=lambda cb, cg: (
                    (2.0 / DELTA) * (K * cb * (1.0 - cg / N_GB) - (K / s) * cg)
                ),
                species=c_gb,
                volume=network,
                species_dependent_value={"cb": c_b, "cg": c_gb},
            )
        ],
        boundary_conditions=[
            F.ParticleFluxBC(
                subdomain=network,
                species=c_b,
                value=lambda cb, cg: (K / s) * cg - K * cb * (1.0 - cg / N_GB),
                species_dependent_value={"cb": c_b, "cg": c_gb},
            ),
            F.FixedConcentrationBC(subdomain=inlet, value=C_SURF, species=c_b),
            F.FixedConcentrationBC(subdomain=mouths, value=c_gb_surf, species=c_gb),
            F.FixedConcentrationBC(subdomain=outlet, value=0.0, species=c_b),
            F.FixedConcentrationBC(subdomain=drains, value=0.0, species=c_gb),
        ],
        temperature=T,
        settings=F.Settings(
            atol=1e-10,
            rtol=1e-10,
            transient=True,
            final_time=T_END,
            stepsize=F.Stepsize(
                initial_value=DT0, growth_factor=1.2, target_nb_iterations=5
            ),
        ),
        exports=[],
    )
    model.initialise()
    if DISORDER:
        network.material = F.Material(D_0=gb_diffusivity_field(network, T), E_D=0.0)
        model.initialise()
    model.run()

    cb = c_b.subdomain_to_post_processing_solution[grains]
    cg = c_gb.subdomain_to_post_processing_solution[network]

    j_bulk = outlet_flux(cb, D_b, D_THICK)
    j_gb = outlet_flux(cg, D_g, D_THICK, weight=DELTA)
    j_tot = j_bulk + j_gb

    # what fraction of the network is saturated, since that is what limits it
    theta = cg.x.array / N_GB
    sat = float(np.mean(theta > 0.99))
    sat = mesh.comm.allreduce(sat, op=MPI.SUM) / mesh.comm.size

    return dict(
        T=T,
        D_b=D_b,
        D_gb=D_g,
        s=s,
        K=K,
        j_bulk=j_bulk,
        j_gb=j_gb,
        j_tot=j_tot,
        f_gb=j_gb / j_tot if j_tot else float("nan"),
        # permeability in their Eq. (1) sense: flux x thickness / surface conc.
        perm=j_tot * D_THICK / C_SURF,
        saturated_fraction=sat,
    )


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    rows = []
    for T in TEMPERATURES:
        r = run(T)
        rows.append(r)
        if comm.rank == 0:
            print(
                f"T={r['T']:.0f} K  D_b={r['D_b']:.2e}  D_gb={r['D_gb']:.2e}  "
                f"s={r['s']:.2e}  K={r['K']:.2e} m/s  "
                f"f_GB={r['f_gb']:.4f}  sat={r['saturated_fraction']:.2f}"
            )

    if comm.rank == 0:
        import csv

        with open("diaz-columnar.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        Ts = np.array([r["T"] for r in rows])

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(Ts, [r["f_gb"] for r in rows], "o-")
        ax.set_xlabel("T (K)")
        ax.set_ylabel(r"$J_{GB}/J_{tot}$")
        ax.set_title(
            f"L={L * 1e9:.0f} nm, d={D_THICK * 1e6:.0f} um, L/d={L / D_THICK:.2f}"
        )
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig("diaz-flux-fraction.png", dpi=150)

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.semilogy(1000.0 / Ts, [r["perm"] for r in rows], "o-", label="model")
        ax.set_xlabel("1000/T (1/K)")
        ax.set_ylabel(r"permeability (m$^{-1}$s$^{-1}$)")
        ax.legend()
        fig.tight_layout()
        fig.savefig("diaz-permeability.png", dpi=150)
        print("wrote diaz-columnar.csv, diaz-flux-fraction.png, diaz-permeability.png")

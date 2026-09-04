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
From the paper:
  E_m,bulk = 0.20 eV, E_m,GB = 0.12 eV, E_bind = 1.05 eV   (DFT; Sect. 3.2, Fig. 3)
  max GB occupation 0.25-6.25 H/nm^2, swept; 6.25 matched experiment
                                                            (Sect. 2.4, Fig. 7)
  charged surface 5.6 H/nm^3 = 5.6e27 m^-3 over a 0.4 nm near-surface region
                                                            (Fig. 7 caption)
  constant-flux runs at 1e23 H2 m^-2 s^-1                   (Fig. 8, Fig. 9)
  lattice constant 3.165 A                                  (Sect. 2.3)
  experimental permeability fit, Eq. (6):
      phi_NW = 1.74e15 exp(-0.569 eV/kT)  H2 m^-1 s^-1 Pa^-1/2
  temperatures 520-705 K (experiment); OKMC shown for 1000/T = 1.3-1.75

Esteban et al. (2001) solubility, the one the paper says it used to convert
the OKMC surface concentration to a pressure (Sect. 3.3), taken from the
HTM database entry for tungsten:
  S = 1.75e22 exp(-0.28 eV/kT)  H m^-3 Pa^-1/2     (4.99e20 at 910 K)

NOT in the paper, assumed here and flagged:
  * Arrhenius prefactors. The paper says the parametrisation is that of its
    ref. [9] and does not restate it. D0_BULK = 1.9e-7 m^2/s is Fernandez et
    al. 2015 (their ref. [53]), whose barrier (0.20 eV) is exactly their
    E_m,bulk. D0_GB is taken as 1.5 x D0_BULK, the ratio of the 2D to the 3D
    random-walk prefactor (1/4 vs 1/6) at equal jump distance and attempt
    frequency -- they state explicitly that GB migration is 2D and bulk
    migration 3D. See PERMEABILITY for what the Fig. 7 monocrystalline line
    says about D0_BULK.
  * E_FORWARD, the bulk->GB approach barrier: Zhou et al. 2010 (their ref.
    [18]) give 0.13-0.16 eV.
  * The bulk site density N_b in the isotherm. bcc has 2 atoms per cubic cell
    and 6 tetrahedral sites per atom, i.e. 12 per cell, so N_b = 12/a^3 =
    3.8e29 m^-3. H occupies tetrahedral sites in W (Heinola & Ahlgren 2010,
    their ref. [54]).

  39 H/nm^2 is NOT the trap density. That is the 0 K DFT filling limit of
  their Fig. 6, which they say explicitly is not reached at temperature and is
  not what the OKMC used. The OKMC used 0.25-6.25 H/nm^2.

THE GB VARIABLE IS AREAL; THERE IS NO SLAB WIDTH
------------------------------------------------
The GB species Gamma is an areal concentration (m^-2). That matches both the
paper's GB (a discrete plane with an areal site limit, Sect. 2.4) and the
codim-1 formulation on FESTIM PR #1216, in which a ParticleSource on a
manifold is integrated over the manifold measure with no thickness factor and
is the same quantity as the normal flux leaving the bulk (the PR's reference
test: K_left=2, K_right=3, D=1.5 -> c = 14/9, 8/9, 4/9 only closes with that
convention). The capacity is the paper's areal limit directly and the GB
current per unit length is D_gb * grad(Gamma). No slab width appears anywhere.

THE COUPLING LAW
----------------
One-way fluxes, per unit GB area and per side,

    J = K c_b (1 - Gamma/Gamma_max)  -  k_r Gamma

K [m/s] is kinetic, lambda nu exp(-E_forward/kT). k_r [1/s] is fixed by
detailed balance so that equilibrium is the McLean/Oriani site-fraction
isotherm (McLean 1957; Oriani, Acta Metall. 18, 147, 1970)

    theta/(1-theta) = (c_b/N_b) exp(E_bind/kT),

which gives k_r = K/ell_s with ell_s = (Gamma_max/N_b) exp(E_bind/kT), a
segregation length [m]. The GB receives 2J (two faces); the bulk loses the
same. The mass-balance check below confirmed (in-out)/in = -3e-5 with
N_FLUX_SIDES = 2, i.e. FESTIM applies the wall flux BC once per facet.

K IS CAPPED
-----------
The physical K (~60 m/s at 570 K) exceeds the rate at which diffusion can feed
the wall, D_b/h (~0.1 m/s for h = 25 nm), by orders of magnitude. The interface
is then in local equilibrium and K acts as a penalty parameter: raising it
changes the Jacobian's conditioning, not the answer. K is therefore capped at
DA_H_MAX * D_b/h. Doubling DA_H_MAX must leave f_GB unchanged.

THE SOLVE IS DIMENSIONLESS
--------------------------
    x = L xhat,   t = (L^2/D_b) that,   c_b = C u,   Gamma = Gamma_max theta

    bulk:  du/dt = lap(u)
           flux BC on the walls, per side: -(Da_f u (1-theta) - Da_r theta)
    GB:    dtheta/dt = (D_gb/D_b) lap(theta) + 2 (Ph_f u (1-theta) - Ph_r theta)

    Da_f = K L / D_b                  Ph_f = K C L^2 / (D_b Gamma_max)
    Da_r = k_r Gamma_max L / (D_b C)  Ph_r = k_r L^2 / D_b

The independent groups are {Da_f, beta, Lambda, D_gb/D_b, d/L} with
beta = (C/N_b) exp(E_bind/kT) (inlet GB occupancy, theta/(1-theta)) and
Lambda = C L/Gamma_max (bulk inventory across one column per unit GB
capacity). Lambda decides whether the GB can dominate: its carrying capacity
relative to the bulk is at most 2 (D_gb/D_b)/Lambda. The concentration scale
C depends on INLET_MODE. Currents convert back with j_b = D_b C L jhat_b and
j_gb = D_b Gamma_max jhat_gb (atoms/s).

INLET CONDITION (INLET_MODE)
----------------------------
"concentration": u = 1 with C = 5.6e27 m^-3. Their Fig. 7 method (constant
    concentration in the near-surface region). Lambda ~ 90: the GB saturates
    and the bulk carries most of the transport, so the exit fraction f_GB is
    high while D_eff/D_b is small -- the interpretation trap they describe
    for their 50x150x150 box (Sect. 3.3, p. 1085).

"implantation": the FESTIM plasma-implantation approximation (theory docs,
    Eq. 23): C = phi_imp R_p / D_b with R_p = 0.4 nm (their near-surface
    region) and phi_imp = 2e23 H m^-2 s^-1 (their 1e23 H2 m^-2 s^-1). Note
    what that equation assumes (theory, Eqs. 17-19): the diffusion depth is
    >> R_p so the flux into the bulk vanishes and essentially all of phi_imp
    leaves back through the front face by fast recombination. The permeating
    flux is then ~ phi_imp R_p/d = 2e-4 phi_imp, and C ~ 1e22 m^-3 = 1e-5
    H/nm^3. The OKMC constant-flux runs are the opposite surface limit: no
    surface model (Sect. 3.3), a reflecting front face, and the whole
    phi_imp permeates. Fig. 9's in-grain surface values (~0.005-0.01 H/nm^3
    for d = 200 nm at 705 K) match phi_imp d/D_b, not phi_imp R_p/D_b, by a
    factor ~500. This mode is provided because it was asked for; "flux" is
    the one that reproduces their Fig. 8/9 runs.

"flux": Neumann inlet, phi_imp into the bulk face, GB mouths no-flux,
    reflecting front face, Dirichlet 0 at the back. C = phi_imp d/D_b (the
    pure-Fick surface value), so u(0) <= 1 and the imposed dimensionless flux
    is 1/D_HAT. The inlet concentration is read off the solution.

PERMEABILITY
------------
Their Eq. (1): J = phi P^(1/2)/d, with P the pressure in equilibrium with the
inlet concentration through Esteban's solubility, P = (C_in/S)^2. Fig. 7 is
in H2 m^-1 s^-1 Pa^-1/2, so J_H2 = J_H/2 and

    phi = J_H2 d / sqrt(P) = S D_eff / 2,    phi_mono = S D_b / 2.

The absolute level is therefore S x D0_BULK and the GB effect is D_eff/D_b.
phi_mono is the pure-Fick reference for their dashed monocrystalline line;
with the HTM solubility and D0_BULK = 1.9e-7 it lies ~15x above that line,
whereas D0 = lambda^2 nu/6 = 2.1e-8 (the natural OKMC hop prefactor) lands
within a factor ~2. D0_BULK is the knob that line calibrates.

Also flagged: with these S values the equilibrium pressure for 5.6 H/nm^3 is
~1e15-1e16 Pa, not the 0.1-1 GPa the paper quotes for the same conversion
(Sect. 3.3). The script prints P_eq so this can be checked against whatever
Esteban fit they actually used.

OUTPUTS
-------
diaz-fig7.png (phi vs 1000/T: model, phi_mono, their Eq. (6) fit),
diaz-flux-fraction.png, diaz-columnar.csv.
"""

from mpi4py import MPI

import dolfinx
import matplotlib
import numpy as np
import ufl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import festim as F

# geometry (SI)
# ----------------------------------------------------------------------------
L = 100e-9  # column side, m                     (their L, 100 nm case)
D_THICK = 2e-6  # layer thickness, m             (the experimental 2 um)
NX, NY = 2, 2  # columns per side; 100 -> 10 um x 10 um
CPG = 4  # mesh cells across one column, per axis
NZ = 100  # mesh cells through the thickness

# CPG >= 2 is required for the facet locator below: with one cell per column a
# tetrahedron can have vertices on two different column planes and would be
# mislabelled as a boundary facet. It can be only a few across because
# transport across a column is fast: at 705 K, D_bulk ~ 7e-9 m^2/s, so
# (L/2)^2/D ~ 4e-7 s across the column against d^2/D ~ 6e-4 s along it.
LX, LY = NX * L, NY * L
AREA = LX * LY
H_WALL = L / CPG  # wall-normal cell size, sets the K cap

# the mesh FESTIM sees is in units of L
L_HAT = 1.0
LX_HAT, LY_HAT, D_HAT = NX * L_HAT, NY * L_HAT, D_THICK / L

# material and coupling parameters (SI)
# ----------------------------------------------------------------------------
E_M_BULK = 0.20  # eV, DFT, their Fig. 3
E_M_GB = 0.12  # eV, DFT, their Fig. 3
E_BIND = 1.05  # eV, DFT, H binding to an empty GB
E_FORWARD = 0.145  # eV, bulk->GB approach barrier; Zhou 2010 gives 0.13-0.16

D0_BULK = 1.9e-7  # m^2/s, Fernandez 2015; see PERMEABILITY in the header
D0_GB = 1.5 * D0_BULK  # m^2/s, assumed: 2D vs 3D random-walk prefactor

GAMMA_MAX = 6.25e18  # m^-2, their max occupancy; the GB capacity, areal

A_LAT = 3.165e-10  # m, lattice constant (Sect. 2.3)
N_B_SITES = 12.0 / A_LAT**3  # 3.8e29 m^-3, tetrahedral sites: 12 per bcc cell

LAMBDA_JUMP = 1.12e-10  # m, tetrahedral-site hop in bcc W, for K
NU_ATTEMPT = 1e13  # s^-1
DA_H_MAX = 5.0  # cap on K h / D_b (see header); double it and check f_GB

# Esteban et al. 2001 solubility, HTM database values (H per m^3 per Pa^1/2)
S0_ESTEBAN = 1.75e22
E_S_ESTEBAN = 0.28  # eV

# their experimental permeability fit, Eq. (6), H2 m^-1 s^-1 Pa^-1/2
PHI0_EXP = 1.74e15
EA_EXP = 0.569  # eV

# inlet
INLET_MODE = "implantation"  # "concentration" | "implantation" | "flux"
C_CONC = 5.6e27  # m^-3, their 5.6 H/nm^3 (Fig. 7 caption)
PHI_IMP = 2.0 * 1e23  # H m^-2 s^-1, their 1e23 H2 m^-2 s^-1 (Fig. 8)
R_P = 0.4e-9  # m, their near-surface region (Fig. 7 caption)

N_FLUX_SIDES = 2  # confirmed by the mass-balance check

TEMPERATURES = [520.0, 566.0, 615.0, 655.0, 705.0, 770.0]  # Fig. 4 + OKMC end

# time, in units of L^2/D_b. Steady state along the column needs
# that >> D_HAT^2 = 400; the GB relaxes far faster.
T_END_HAT = 3000.0
DT0_HAT = 1e-3  # first step must not swallow the whole inlet jump
DT_MAX_HAT = 25.0

DISORDER = 0
E_M_GB_RANGE = (0.191, 0.547)  # eV, Wei et al. 2026, eight tungsten GBs
SEED = 0

VERBOSE = 0  # SNES/KSP monitors, to tell stagnation from blow-up


def segregation_length(T):
    """ell_s [m]: equilibrium Gamma/c_b in the dilute limit, McLean prefactor N_s/N_b."""
    return (GAMMA_MAX / N_B_SITES) * np.exp(E_BIND / (F.k_B * T))


def k_forward(T, D_b):
    """Absorption coefficient K [m/s]. Kinetic, from E_forward; capped, see header."""
    k_phys = LAMBDA_JUMP * NU_ATTEMPT * np.exp(-E_FORWARD / (F.k_B * T))
    return min(k_phys, DA_H_MAX * D_b / H_WALL)


def solubility(T):
    """Esteban Sieverts constant, H m^-3 Pa^-1/2."""
    return S0_ESTEBAN * np.exp(-E_S_ESTEBAN / (F.k_B * T))


def inlet_scale(T, D_b):
    """Concentration scale C [m^-3] for the chosen INLET_MODE (header)."""
    if INLET_MODE == "concentration":
        return C_CONC
    if INLET_MODE == "implantation":
        return PHI_IMP * R_P / D_b  # FESTIM theory Eq. (23)
    if INLET_MODE == "flux":
        return PHI_IMP * D_THICK / D_b  # pure-Fick surface value; u(0) <= 1
    raise ValueError(INLET_MODE)


# mesh: a structured box, in units of L, whose node planes fall on the walls
comm = MPI.COMM_WORLD
mesh = dolfinx.mesh.create_box(
    comm,
    [np.array([0.0, 0.0, 0.0]), np.array([LX_HAT, LY_HAT, D_HAT])],
    [NX * CPG, NY * CPG, NZ],
    cell_type=dolfinx.mesh.CellType.tetrahedron,
)

ntet = NX * CPG * NY * CPG * NZ * 6
if comm.rank == 0:
    print(f"domain {LX * 1e6:.1f} x {LY * 1e6:.1f} x {D_THICK * 1e6:.1f} um")
    print(f"{NX * NY} columns, L = {L * 1e9:.0f} nm, ~{ntet:.3e} tetrahedra")
    print(f"inlet mode: {INLET_MODE}")


class ColumnWalls(F.VolumeSubdomain):
    """The whole GB network as one codim-1 subdomain: interior facets, dim=2.

    The x-normal and y-normal wall sets are located separately and unioned.
    Locating them with a single predicate would also catch facets that have
    some vertices on an x-wall and the rest on a y-wall, which near a column
    corner lie on no wall at all.
    """

    def __init__(self, id, material):
        super().__init__(id=id, material=material, dim=2)
        self.entity_axis = None  # 0 or 1 per located facet, for the D_gb field

    def locate_subdomain_entities(self, mesh):
        tol = 0.05 * L_HAT / CPG

        def on_walls(axis):
            extent = LX_HAT if axis == 0 else LY_HAT

            def pred(x):
                r = np.remainder(x[axis] + tol, L_HAT)
                interior = (x[axis] > tol) & (x[axis] < extent - tol)
                return (r < 2 * tol) & interior

            return pred

        fx = dolfinx.mesh.locate_entities(mesh, 2, on_walls(0))
        fy = dolfinx.mesh.locate_entities(mesh, 2, on_walls(1))
        self.entity_axis = np.concatenate(
            [np.zeros(fx.size, dtype=np.int8), np.ones(fy.size, dtype=np.int8)]
        )
        return np.concatenate([fx, fy]).astype(np.int32)


def gb_diffusivity_field(network, T, D_b):
    """Per-boundary D_gb/D_b as a DG0 field on the network submesh.

    The coefficient has to live on the one submesh, because splitting the
    network into two subdomains would disconnect it and reintroduce junction
    conditions. Here the per-boundary label is a random draw rather than a
    measured disorientation -- swap in theta when you point this at a real
    map. The value is dimensionless (divided by D_b) because the solve is.
    """
    submesh = network.submesh

    V = dolfinx.fem.functionspace(submesh, ("DG", 0))
    d = dolfinx.fem.Function(V, name="D_gb")
    n_local = submesh.topology.index_map(submesh.topology.dim).size_local

    # one barrier per wall, not per element: label walls by (axis, index) so a
    # whole boundary plane shares a value, which is what a boundary character
    # means. Facet midpoints give the wall index.
    mid = dolfinx.mesh.compute_midpoints(
        submesh, submesh.topology.dim, np.arange(n_local, dtype=np.int32)
    )
    ix = np.rint(mid[:, 0] / L_HAT).astype(int)
    iy = np.rint(mid[:, 1] / L_HAT).astype(int)
    on_x = np.isclose(mid[:, 0] / L_HAT, ix, atol=0.1)
    wall_id = np.where(on_x, ix, 10_000 + iy)

    rng = np.random.default_rng(SEED)
    lo, hi = E_M_GB_RANGE
    barrier = rng.uniform(lo, hi, size=20_001)[wall_id]
    d.x.array[:n_local] = D0_GB * np.exp(-barrier / (F.k_B * T)) / D_b
    d.x.scatter_forward()
    return d


# ----------------------------------------------------------------------------
# flux accounting
# ----------------------------------------------------------------------------
def _plane_measure(c, z):
    m = c.function_space.mesh
    fdim = m.topology.dim - 1
    facets = dolfinx.mesh.locate_entities_boundary(
        m, fdim, lambda x: np.isclose(x[2], z, atol=1e-10)
    )
    tags = dolfinx.mesh.meshtags(
        m, fdim, np.sort(facets), np.ones(facets.size, dtype=np.int32)
    )
    return ufl.Measure("ds", domain=m, subdomain_data=tags)(1)


def axial_current(c, D_hat, z):
    """Dimensionless current in +z through the plane z, assembled.

    Works for both the 3D bulk field (an area integral over the plane) and the
    2D GB submesh field (a line integral over the edge of the submesh in that
    plane), because in each case it is the exterior-facet measure of that
    field's own mesh restricted to the plane. Positive means flow towards the
    outlet at both the inlet and the outlet, so the two can be compared.
    """
    ds = _plane_measure(c, z)
    # the submesh of a plane set has no meaningful outward normal in 3D, so
    # take the axial component of the gradient directly rather than dot(.., n)
    form = -D_hat * ufl.grad(c)[2] * ds
    local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(form))
    return c.function_space.mesh.comm.allreduce(local, op=MPI.SUM)


def plane_mean(c, z, area_hat):
    """Mean of a 3D field over the plane z (dimensionless area area_hat)."""
    ds = _plane_measure(c, z)
    local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(c * ds))
    return c.function_space.mesh.comm.allreduce(local, op=MPI.SUM) / area_hat


def run(T):
    kT = F.k_B * T
    D_b = D0_BULK * np.exp(-E_M_BULK / kT)
    D_g = D0_GB * np.exp(-E_M_GB / kT)
    ell = segregation_length(T)
    K = k_forward(T, D_b)
    kr = K / ell
    tau = L**2 / D_b
    C = inlet_scale(T, D_b)
    S = solubility(T)

    # dimensionless groups (header, THE SOLVE IS DIMENSIONLESS)
    Da_f = K * L / D_b
    Da_r = kr * GAMMA_MAX * L / (D_b * C)
    Ph_f = K * C * L**2 / (D_b * GAMMA_MAX)
    Ph_r = kr * L**2 / D_b
    D_g_hat = D_g / D_b
    beta = C / N_B_SITES * np.exp(E_BIND / kT)  # inlet theta/(1-theta) at u = 1
    Lambda = C * L / GAMMA_MAX
    theta_surf = beta / (1.0 + beta)

    if comm.rank == 0:
        print(
            f"T={T:.0f} K  D_b={D_b:.2e}  D_gb/D_b={D_g_hat:.2f}  "
            f"K={K:.2e} m/s (Kh/D_b={K * H_WALL / D_b:.1f})  k_r={kr:.2e} 1/s  "
            f"tau={tau:.2e} s  C={C:.2e} m^-3 ({C * 1e-27:.2e} H/nm^3)"
        )
        print(
            f"  Da_f={Da_f:.2e}  Da_r={Da_r:.2e}  Ph_f={Ph_f:.2e}  Ph_r={Ph_r:.2e}  "
            f"beta={beta:.2e}  Lambda={Lambda:.2e}  S={S:.2e}"
        )

    grains = F.VolumeSubdomain(
        id=1,
        material=F.Material(D_0=1.0, E_D=0.0),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    network = ColumnWalls(id=2, material=F.Material(D_0=D_g_hat, E_D=0.0))
    inlet = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[2], 0.0))
    outlet = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[2], D_HAT))
    # the lines where the walls meet the charged face and the sink face
    mouths = F.SurfaceSubdomain(id=5, dim=1, locator=lambda x: np.isclose(x[2], 0.0))
    drains = F.SurfaceSubdomain(id=6, dim=1, locator=lambda x: np.isclose(x[2], D_HAT))

    u = F.Species("u", subdomains=[grains])  # c_b / C
    th = F.Species("theta", subdomains=[network])  # Gamma / GAMMA_MAX

    wall_bc = F.ParticleFluxBC(
        subdomain=network,
        species=u,
        value=lambda cb, cg: N_FLUX_SIDES * (Da_r * cg - Da_f * cb * (1.0 - cg)),
        species_dependent_value={"cb": u, "cg": th},
    )
    sink_bcs = [
        F.FixedConcentrationBC(subdomain=outlet, value=0.0, species=u),
        F.FixedConcentrationBC(subdomain=drains, value=0.0, species=th),
    ]
    if INLET_MODE == "flux":
        # reflecting front face carrying the whole implanted flux; the GB
        # mouths get the natural no-flux condition
        inlet_bcs = [F.ParticleFluxBC(subdomain=inlet, species=u, value=1.0 / D_HAT)]
    else:
        # Dirichlet on the bulk face and the isotherm value at the mouths, so
        # the mouth condition is the equilibrium one rather than an arbitrary
        # number
        inlet_bcs = [
            F.FixedConcentrationBC(subdomain=inlet, value=1.0, species=u),
            F.FixedConcentrationBC(subdomain=mouths, value=theta_surf, species=th),
        ]

    petsc_options = {
        "pc_factor_mat_solver_type": "superlu_dist",
        "snes_linesearch_type": "bt",
    }
    if VERBOSE:
        petsc_options.update(
            {
                "snes_monitor": None,
                "snes_converged_reason": None,
                "ksp_converged_reason": None,
            }
        )

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        species=[u, th],
        subdomains=[grains, network, inlet, outlet, mouths, drains],
        sources=[
            F.ParticleSource(
                # both faces feed the GB; blocking on the forward term only
                # (the bulk is dilute, c_b/N_b <= 0.015, so reverse blocking
                # is negligible)
                value=lambda cb, cg: 2.0 * (Ph_f * cb * (1.0 - cg) - Ph_r * cg),
                species=th,
                volume=network,
                species_dependent_value={"cb": u, "cg": th},
            )
        ],
        boundary_conditions=[wall_bc, *inlet_bcs, *sink_bcs],
        temperature=T,
        settings=F.Settings(
            atol=1e-8,
            rtol=1e-8,
            transient=True,
            final_time=T_END_HAT,
            max_iterations=30,
            stepsize=F.Stepsize(
                initial_value=DT0_HAT,
                growth_factor=1.2,
                cutback_factor=0.5,
                target_nb_iterations=5,
                max_stepsize=DT_MAX_HAT,
            ),
        ),
        exports=[],
        petsc_options=petsc_options,
    )
    model.initialise()
    if DISORDER:
        network.material = F.Material(
            D_0=gb_diffusivity_field(network, T, D_b), E_D=0.0
        )
        model.initialise()

    model.run()

    cb = u.subdomain_to_post_processing_solution[grains]
    cg = th.subdomain_to_post_processing_solution[network]

    # dimensionless currents, then back to atoms/s
    jb_out = D_b * C * L * axial_current(cb, 1.0, D_HAT)
    jg_out = D_b * GAMMA_MAX * axial_current(cg, D_g_hat, D_HAT)
    jb_in = D_b * C * L * axial_current(cb, 1.0, 0.0)
    jg_in = D_b * GAMMA_MAX * axial_current(cg, D_g_hat, 0.0)
    j_out = jb_out + jg_out
    j_in = jb_in + jg_in
    balance = (j_in - j_out) / j_in if j_in else float("nan")

    # inlet concentration: imposed in the Dirichlet modes, read off the
    # solution in flux mode
    u_in = plane_mean(cb, 0.0, LX_HAT * LY_HAT)
    C_in = C * u_in

    # how much of the network is saturated, and whether Newton overshot
    theta = cg.x.array
    n_loc = theta.size
    n_sat = int(np.sum(theta > 0.99))
    n_tot = mesh.comm.allreduce(n_loc, op=MPI.SUM)
    n_sat = mesh.comm.allreduce(n_sat, op=MPI.SUM)
    theta_max = mesh.comm.allreduce(float(theta.max()) if n_loc else 0.0, op=MPI.MAX)

    # their Eq. (1): D_eff = (J/A) d / C_in; phi = J_H2 d / sqrt(P) = S D_eff/2
    D_eff = (j_out / AREA) * D_THICK / C_in
    P_eq = (C_in / S) ** 2
    phi = (j_out / AREA / 2.0) * D_THICK / np.sqrt(P_eq)
    phi_mono = S * D_b / 2.0
    phi_exp = PHI0_EXP * np.exp(-EA_EXP / kT)

    return dict(
        T=T,
        inv_T_1000=1000.0 / T,
        inlet_mode=INLET_MODE,
        D_b=D_b,
        D_gb=D_g,
        K=K,
        k_r=kr,
        C_scale=C,
        C_in=C_in,
        beta=beta,
        Lambda=Lambda,
        Da_f=Da_f,
        Da_r=Da_r,
        Ph_f=Ph_f,
        Ph_r=Ph_r,
        S=S,
        P_eq=P_eq,
        j_bulk=jb_out,
        j_gb=jg_out,
        j_tot=j_out,
        j_in=j_in,
        mass_balance=balance,
        f_gb=jg_out / j_out if j_out else float("nan"),
        D_eff=D_eff,
        phi=phi,
        phi_mono=phi_mono,
        phi_exp=phi_exp,
        saturated_fraction=n_sat / n_tot if n_tot else float("nan"),
        theta_max=theta_max,
    )


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    rows = []
    for T in TEMPERATURES:
        r = run(T)
        rows.append(r)
        if comm.rank == 0:
            print(
                f"T={r['T']:.0f} K  f_GB={r['f_gb']:.4f}  D_eff/D_b={r['D_eff'] / r['D_b']:.3f}  "
                f"C_in={r['C_in'] * 1e-27:.2e} H/nm^3  P_eq={r['P_eq']:.2e} Pa\n"
                f"  phi={r['phi']:.3e}  phi_mono={r['phi_mono']:.3e}  "
                f"phi_exp={r['phi_exp']:.3e}  H2 m^-1 s^-1 Pa^-1/2\n"
                f"  sat={r['saturated_fraction']:.2f}  theta_max={r['theta_max']:.6f}  "
                f"(in-out)/in={r['mass_balance']:+.3e}"
            )
            if abs(r["mass_balance"]) > 1e-2:
                print(
                    "  WARNING: mass balance fails; not at steady state? raise T_END_HAT"
                )
            if r["theta_max"] > 1.0 + 1e-6:
                print("  WARNING: theta exceeded 1; Newton overshot the GB capacity.")

    if comm.rank == 0:
        import csv

        with open("diaz-columnar.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        x = np.array([r["inv_T_1000"] for r in rows])
        order = np.argsort(x)
        x = x[order]
        get = lambda k: np.array([r[k] for r in rows])[order]

        # Fig. 7 layout: permeability vs 1000/T, log y
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.semilogy(x, get("phi"), "s-", color="purple", label=f"model ({INLET_MODE})")
        ax.semilogy(x, get("phi_mono"), "k--", label=r"$S D_b/2$ (no GB)")
        ax.semilogy(
            x, get("phi_exp"), "-", color="purple", alpha=0.5, label="their Eq. (6) fit"
        )
        ax.set_xlabel("1000/T (1/K)")
        ax.set_ylabel(r"permeability (H$_2$ m$^{-1}$ s$^{-1}$ Pa$^{-1/2}$)")
        ax.set_title(f"L={L * 1e9:.0f} nm, d={D_THICK * 1e6:.0f} um, Esteban S")
        ax.legend()
        fig.tight_layout()
        fig.savefig("diaz-fig7.png", dpi=150)

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(get("T"), get("f_gb"), "o-")
        ax.set_xlabel("T (K)")
        ax.set_ylabel(r"$J_{GB}/J_{tot}$")
        ax.set_title(
            f"L={L * 1e9:.0f} nm, d={D_THICK * 1e6:.0f} um, L/d={L / D_THICK:.2f}"
        )
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig("diaz-flux-fraction.png", dpi=150)

"""The resolved model: one subdomain per grain, coupled through a codim-1 network.

Every Voronoi cell is its own :class:`festim.VolumeSubdomain` carrying its own
:class:`festim.Species`, so the lattice concentration is allowed to *jump* from
one grain to the next. That is what makes a grain boundary able to resist as well
as to conduct: the only path from one grain to another is through the network,
and the exchange rate ``k`` of a grain with the network is a real transfer
resistance in series with the lattice. A single bulk subdomain cannot represent
that -- there the lattice field is continuous across every boundary, so a
boundary can only ever be a short circuit in parallel with the lattice.

The network itself stays **one** codim-1 subdomain with one species. That is what
makes triple junctions work with no junction condition to write: the submesh
built from all the grain-boundary facets is connected, so a single continuous
field lives on it. FESTIM gives each adjacent grain an interior-facet integral of
its own on which that grain is the ``"+"`` side, so a facet shared by two grains
is integrated once per grain -- which is exactly the two-sided feeding the
physics asks for.

Units, and where ``delta`` goes
-------------------------------
The physical boundary is a slab of width ``delta`` with tangential diffusivity
``D_gb``, fed from the grain on each side. Per unit boundary *area*::

    delta dc_gb/dt = delta D_gb lap_s(c_gb) + sum_(sides s) k_s (c_s - c_gb)

FESTIM assembles the manifold equation without any thickness factor, so it is
divided by ``delta``: the manifold material carries ``D_gb`` itself and each side
contributes a source ``(k_s / delta) (c_s - c_gb)`` -- one term per grain, no
factor of two, because the sum over the two adjacent grains supplies it. Each
grain loses what it gives, ``k_s (c_gb - c_s)`` per unit area, on its own side.
``delta`` reappears in every post-processed quantity: the network holds
``delta * integral(c_gb)`` and carries ``delta * D_gb * grad_s(c_gb)``.
"""

from dataclasses import dataclass, field

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import numpy as np
import ufl

import festim as F

__all__ = [
    "Grain",
    "GrainBoundaryNetwork",
    "GrainSurface",
    "MicroModel",
    "Physics",
    "averages",
    "build",
    "check_network_covers_grain_boundaries",
    "inventory",
    "tune_direct_solver",
]

ATOL = 1e-25
"""Absolute tolerance on the Newton residual.

Far below FESTIM's default, because nothing here is scaled: with
``D ~ 1e-11 m2/s`` over a cell of area ``1e-11 m2``, the residual of a *converged*
transient step is itself of order ``1e-12``. At ``atol = 1e-12`` the solver then
declares convergence at zero iterations as soon as the uptake slows, and the
solution freezes at whatever it had reached -- a silent stall that looks exactly
like a steady state, at the wrong value. Nondimensionalising is the better fix;
this is the one-line one.
"""

NETWORK_ID = 1_000_000  # above every grain id
SURFACE_ID_0 = 2_000_000  # the per-grain boundary patches are numbered from here


@dataclass
class Physics:
    """Material data. Defaults are tungsten-like, in the short-circuit regime.

    ``E_D_bulk`` is Frauenfelder's lattice migration energy and ``E_D_gb`` a
    typical DFT boundary value; at 500 K that is a diffusivity contrast of ~800,
    which with a boundary area fraction of a few 1e-3 is what puts the material
    in the regime where the network decides the transport. A contrast of ten, or
    millimetre grains, would homogenise to the lattice value and there would be
    nothing to identify.
    """

    T: float = 500.0
    D_0_bulk: float = 1.9e-7  # m2/s
    E_D_bulk: float = 0.39  # eV
    D_0_gb: float = 2.85e-7  # m2/s, 1.5x the lattice prefactor: 2D vs 3D random walk
    E_D_gb: float = 0.12  # eV
    delta: float = 1e-9  # m, boundary width
    k_exchange: float = 3.0  # m/s, grain <-> boundary transfer coefficient
    crystal_anisotropy: float = 1.0
    """Ratio of the fast to the slow lattice diffusivity *in the crystal frame*.

    ``1.0`` -- the default -- is the only value a cubic crystal can have:
    Neumann's principle makes any second-rank tensor property of a cubic crystal
    isotropic, so in bcc tungsten grain orientation cannot affect lattice
    diffusion at all, and all the macroscopic anisotropy comes from the network.
    Set it above 1 for a hcp or tetragonal lattice (Zr, Ti, graphite), where the
    orientation of each grain does feed through.
    """

    @property
    def D_bulk(self):
        """The orientation average, ie. the isotropic lattice diffusivity."""
        return self.D_0_bulk * np.exp(-self.E_D_bulk / (F.k_B * self.T))

    @property
    def D_gb(self):
        return self.D_0_gb * np.exp(-self.E_D_gb / (F.k_B * self.T))

    @property
    def contrast(self):
        return self.D_gb / self.D_bulk

    def crystal_tensor(self, orientation):
        """The lattice diffusivity of a grain whose fast axis is at ``orientation``.

        The two principal values are ``D_bulk * sqrt(a)`` and ``D_bulk / sqrt(a)``,
        so their geometric mean is ``D_bulk`` whatever the anisotropy ``a``, and
        an untextured aggregate of them still averages to the lattice value.
        """
        root = np.sqrt(self.crystal_anisotropy)
        principal = np.diag([self.D_bulk * root, self.D_bulk / root])
        c, s = np.cos(orientation), np.sin(orientation)
        rotation = np.array([[c, -s], [s, c]])
        return rotation @ principal @ rotation.T

    @property
    def equilibration_length(self):
        """Distance along a boundary over which it equilibrates with the grains.

        Well below the grain size means local equilibrium, ``c_gb = c_grain``
        pointwise, which is the regime in which a single-field homogeneous model
        can exist at all.
        """
        return np.sqrt(self.delta * self.D_gb / (2 * self.k_exchange))

    def interface_resistance_ratio(self, grain_size):
        """Transfer resistance of one boundary crossing over the lattice
        resistance of one grain, ``(2/k) / (d/D_bulk)``.

        Below one the boundaries are transparent and the polycrystal behaves as
        if the lattice field were continuous; above one they throttle grain-to-
        grain transport and the network becomes the only way across.
        """
        return (2.0 / self.k_exchange) / (grain_size / self.D_bulk)

    def report(self, grain_size=None):
        lines = [
            f"physics at T = {self.T:g} K",
            f"  D_bulk (orientation average)   : {self.D_bulk:.3e} m2/s",
            f"  D_gb                           : {self.D_gb:.3e} m2/s",
            f"  contrast D_gb / D_bulk         : {self.contrast:.4g}",
            f"  crystal anisotropy             : {self.crystal_anisotropy:g}",
            f"  boundary width delta           : {1e9 * self.delta:g} nm",
            f"  exchange rate k                : {self.k_exchange:.3e} m/s",
            f"  equilibration length           : "
            f"{1e9 * self.equilibration_length:.3g} nm",
        ]
        if grain_size is not None:
            lines.append(
                f"  interface / lattice resistance : "
                f"{self.interface_resistance_ratio(grain_size):.3e}"
            )
        return "\n".join(lines)


class Grain(F.VolumeSubdomain):
    """One Voronoi cell, located from its gmsh physical group.

    A locator cannot separate one grain from the next -- they have no analytical
    description -- so the cells are read straight from the tags the mesh was
    generated with.
    """

    def __init__(self, id, material, cell_tags):
        super().__init__(id=id, material=material)
        self.cell_tags = cell_tags

    def locate_subdomain_entities(self, mesh):
        return self.cell_tags.find(self.id).astype(np.int32)


class GrainBoundaryNetwork(F.VolumeSubdomain):
    """The whole network as one codim-1 subdomain.

    ``locate_subdomain_entities`` is overridden rather than passing a ``locator``:
    ``locate_entities`` marks a facet when *all its vertices* satisfy the locator,
    which near a triple junction also catches short facets that merely touch two
    different boundaries. Testing the facet midpoint instead selects the network
    exactly.

    Facets on the outer boundary of the mesh are then dropped. A tessellation
    occasionally puts a ridge along the edge of the cell, and FESTIM requires a
    manifold to be wholly interior or wholly exterior -- it needs ``dS`` for one
    and ``ds`` for the other, and cannot have both in one form. Those facets sit
    on a surface that carries a boundary condition anyway.
    """

    def __init__(self, id, material, micro):
        super().__init__(id=id, material=material, dim=1)
        self.micro = micro
        self.n_dropped = 0

    def locate_subdomain_entities(self, mesh):
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, 0)
        mesh.topology.create_connectivity(tdim - 1, tdim)
        facet_to_vertex = mesh.topology.connectivity(tdim - 1, 0)
        facet_to_cell = mesh.topology.connectivity(tdim - 1, tdim)
        candidates = dolfinx.mesh.locate_entities(mesh, tdim - 1, self.micro.locator)
        x = mesh.geometry.x
        midpoints = np.array(
            [x[facet_to_vertex.links(f)].mean(axis=0) for f in candidates]
        )
        on_network = self.micro.locator(midpoints.T)
        interior = np.array(
            [len(facet_to_cell.links(f)) == 2 for f in candidates], dtype=bool
        )
        self.n_dropped = int((on_network & ~interior).sum())
        return candidates[on_network & interior].astype(np.int32)


class GrainSurface(F.SurfaceSubdomain):
    """The part of an outer surface that belongs to one grain.

    A :class:`festim.SurfaceSubdomain` belongs to exactly one volume subdomain, so
    a surface crossing several grains has to be declared once per grain.
    """

    def __init__(self, id, grain_id, cell_tags, locator):
        super().__init__(id=id, locator=locator)
        self.grain_id = grain_id
        self.cell_tags = cell_tags

    def locate_boundary_facet_indices(self, mesh):
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        facet_to_cell = mesh.topology.connectivity(tdim - 1, tdim)
        cells = set(self.cell_tags.find(self.grain_id).tolist())
        facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, self.locator)
        keep = [f for f in facets if any(c in cells for c in facet_to_cell.links(f))]
        return np.array(keep, dtype=np.int32)


def check_network_covers_grain_boundaries(micro):
    """Every interior facet separating two grains must be in the network.

    If one were missed it would carry no coupling at all, and since the grains are
    now separate subdomains nothing else joins them there: the model would have a
    perfectly sealed wall where the microstructure has a grain boundary, and would
    quietly under-predict the transport. With a single bulk subdomain the same
    mistake is invisible, because the lattice field is continuous anyway.
    """
    mesh, tags = micro.mesh, micro.cell_tags
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    mesh.topology.create_connectivity(tdim - 1, 0)
    facet_to_cell = mesh.topology.connectivity(tdim - 1, tdim)
    facet_to_vertex = mesh.topology.connectivity(tdim - 1, 0)
    index_map = tags.topology.index_map(tdim)
    values = np.zeros(index_map.size_local + index_map.num_ghosts, dtype=np.int32)
    values[tags.indices] = tags.values

    x = mesh.geometry.x
    n_facets = mesh.topology.index_map(tdim - 1).size_local
    missed = 0
    total = 0
    for f in range(n_facets):
        cells = facet_to_cell.links(f)
        if len(cells) != 2 or values[cells[0]] == values[cells[1]]:
            continue
        total += 1
        midpoint = x[facet_to_vertex.links(f)].mean(axis=0)
        if not micro.locator(midpoint.reshape(3, 1))[0]:
            missed += 1
    return total, missed


def crystal_diffusivity_field(micro, physics):
    """A DG0 tensor field on the parent mesh holding each grain's lattice tensor.

    ``festim.Material`` takes a ``fem.Function`` for ``D``, and the subdomain
    integrals are parent-mesh integrals indexed by the grain id, so one field
    shared by every grain is enough: each grain's term only ever reads the cells
    of that grain.
    """
    mesh = micro.mesh
    V = dolfinx.fem.functionspace(mesh, ("DG", 0, (2, 2)))
    D = dolfinx.fem.Function(V, name="D_lattice")
    tensors = {}
    for grain_id in micro.grain_ids:
        tensor = physics.crystal_tensor(micro.orientations[grain_id - 1])
        tensors[grain_id] = tensor
        cells = micro.cell_tags.find(grain_id)
        dofs = V.dofmap.list[cells].reshape(-1)
        for component in range(4):
            D.x.array[4 * dofs + component] = tensor.reshape(-1)[component]
    D.x.scatter_forward()
    return D, tensors


@dataclass
class MicroModel:
    """A built (not yet solved) resolved problem, and the handles to read it."""

    model: F.HydrogenTransportProblemDiscontinuous
    micro: object
    physics: Physics
    grains: list
    network: GrainBoundaryNetwork
    species: list
    c_gb: F.Species
    tensors: dict
    surfaces: dict = field(default_factory=dict)

    def run(self):
        self.model.initialise()
        tune_direct_solver(self.model)
        self.model.run()
        return self

    @property
    def grain_solutions(self):
        return [
            spe.subdomain_to_post_processing_solution[grain]
            for spe, grain in zip(self.species, self.grains, strict=True)
        ]

    @property
    def network_solution(self):
        return self.c_gb.subdomain_to_post_processing_solution[self.network]


def tune_direct_solver(model, icntl_14=400):
    """Give MUMPS more room for fill-in, after the solver exists.

    One subdomain per grain makes a wide block system, and the blocks are scaled
    very differently -- a lattice stiffness of order ``D_bulk`` against an
    exchange term of order ``k/delta``. MUMPS then delays many pivots, needs more
    working memory than it estimated, and stops with ``INFOG(1) = -9`` instead of
    reallocating. How bad it gets depends on the values, so the same mesh can
    factor at one exchange rate and fail at another.

    This cannot be done through ``petsc_options``: FESTIM removes those from the
    options database as soon as the solver is built (a workaround for PETSc issue
    1201), and ``mat_mumps_*`` is not read until ``PCSetUp`` runs at the first
    solve -- by which time the option is gone. Writing it back afterwards, under
    the solver's own prefix, is read at the right moment.

    Call it after ``initialise()`` and before the first solve; :meth:`MicroModel.run`
    already does, and code driving ``iterate()`` itself has to.
    """
    solver = getattr(model, "solver", None)
    snes = getattr(solver, "solver", None)
    if snes is None:
        return
    prefix = snes.getOptionsPrefix() or ""
    # only under the solver's own prefix: an unprefixed copy is never read, and
    # PETSc reports it as an unused option at the end of the run
    PETSc.Options()[f"{prefix}mat_mumps_icntl_14"] = icntl_14


def _window(mesh, window):
    """Indicator of an axis-aligned box, as a UFL expression on ``mesh``."""
    if window is None:
        return 1.0
    (x0, y0), (x1, y1) = window
    x = ufl.SpatialCoordinate(mesh)
    inside = ufl.And(ufl.And(x[0] > x0, x[0] < x1), ufl.And(x[1] > y0, x[1] < y1))
    return ufl.conditional(inside, 1.0, 0.0)


def _assemble(expr):
    form = dolfinx.fem.form(expr)
    local = dolfinx.fem.assemble_scalar(form)
    return form.mesh.comm.allreduce(local, op=MPI.SUM)


def averages(mm: MicroModel, window=None):
    """Volume averages of flux and gradient over ``window`` (the whole cell if None).

    Returns ``(q, grad_c, area)``. The flux sums every grain's lattice flux and
    the network's own ``delta D_gb grad_s c_gb`` -- leaving out the second term is
    the commonest way to get a homogenisation of this kind wrong, because it is
    exactly the short circuit one is trying to measure. Note that ``grad_c`` is
    the average over the grains of a field that is *discontinuous* between them;
    the jumps are carried by the network term, not by this average.
    """
    delta, D_gb = mm.physics.delta, mm.physics.D_gb
    area = 0.0
    q = np.zeros(2)
    grad_c = np.zeros(2)

    for c, grain in zip(mm.grain_solutions, mm.grains, strict=True):
        mesh = c.function_space.mesh
        dx = ufl.Measure("dx", domain=mesh)
        w = _window(mesh, window)
        D = ufl.as_matrix(mm.tensors[grain.id].tolist())
        flux = -D * ufl.grad(c)
        area += _assemble(w * dx)
        for i in range(2):
            q[i] += _assemble(w * flux[i] * dx)
            grad_c[i] += _assemble(w * ufl.grad(c)[i] * dx)

    cgb = mm.network_solution
    mesh_g = cgb.function_space.mesh
    dx_g = ufl.Measure("dx", domain=mesh_g)
    w_g = _window(mesh_g, window)
    for i in range(2):
        q[i] -= _assemble(w_g * delta * D_gb * ufl.grad(cgb)[i] * dx_g)

    return q / area, grad_c / area, area


def inventory(mm: MicroModel, window=None):
    """Hydrogen per unit area of the cell: the grains plus the boundary slabs."""
    total = 0.0
    for c in mm.grain_solutions:
        mesh = c.function_space.mesh
        total += _assemble(_window(mesh, window) * c * ufl.Measure("dx", domain=mesh))
    cgb = mm.network_solution
    mesh_g = cgb.function_space.mesh
    total += mm.physics.delta * _assemble(
        _window(mesh_g, window) * cgb * ufl.Measure("dx", domain=mesh_g)
    )
    return total


def build(
    micro,
    physics: Physics,
    bcs,
    transient=False,
    final_time=None,
    stepsize=None,
    exchange_rate=None,
    initial_conditions=(),
    exports=(),
    atol=ATOL,
    rtol=1e-10,
    petsc_options=None,
):
    """Assemble the resolved problem on ``micro``.

    Args:
        micro: a :class:`microstructure.Microstructure`.
        physics: the material data.
        bcs: an iterable of ``(name, locator, value)``. Each fixes the
            concentration on the part of the outer boundary picked by ``locator``
            -- on every grain that touches it, and on the points where a boundary
            meets that part of the surface (a codim-2 subdomain). Fixing only the
            grains would leave the network's mouths free and let the short
            circuits leak out of the cell. ``value`` is a float or a callable of
            ``x``.
        petsc_options: passed to the solver. The default is FESTIM's direct solve
            with a larger MUMPS working array: one subdomain per grain makes a
            wide block system whose fill-in MUMPS routinely under-estimates, and
            it fails with ``INFOG(1)=-9`` rather than reallocating. How much
            fill-in there is depends on the pivoting, so the same mesh can factor
            at one exchange rate and run out of memory at another.
        exchange_rate: ``grain_id -> k``, the transfer coefficient of that grain
            with the network. Defaults to ``physics.k_exchange`` everywhere. This
            is the hook for misorientation-dependent boundary properties: the
            exchange is a property of the (network, grain) pair, so it can be made
            to depend on the grain it faces.
        transient: if False, a steady solve.

    Returns:
        MicroModel
    """
    if petsc_options is None:
        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 400,  # % headroom over the estimated workspace
        }

    if exchange_rate is None:

        def exchange_rate(grain_id):
            return physics.k_exchange

    D_field, tensors = crystal_diffusivity_field(micro, physics)
    grains = [
        Grain(id=int(g), material=F.Material(D=D_field), cell_tags=micro.cell_tags)
        for g in micro.grain_ids
    ]
    network = GrainBoundaryNetwork(
        id=NETWORK_ID, material=F.Material(D_0=physics.D_gb, E_D=0.0), micro=micro
    )
    species = [F.Species(f"c_{g.id}", subdomains=[g]) for g in grains]
    c_gb = F.Species("c_gb", subdomains=[network])

    delta = physics.delta
    # one exchange per grain, each naming only that grain's species -- which is how
    # FESTIM works out which side of the network the term belongs to. No factor of
    # two here: the two grains adjacent to a facet each contribute their own term.
    sources = [
        F.ParticleSource(
            value=lambda c_g, c_b, k=exchange_rate(grain.id): (k / delta) * (c_b - c_g),
            species=c_gb,
            volume=network,
            species_dependent_value={"c_b": spe, "c_g": c_gb},
        )
        for spe, grain in zip(species, grains, strict=True)
    ]
    boundary_conditions = [
        F.ParticleFluxBC(
            subdomain=network,
            species=spe,
            value=lambda c_g, c_b, k=exchange_rate(grain.id): k * (c_g - c_b),
            species_dependent_value={"c_b": spe, "c_g": c_gb},
        )
        for spe, grain in zip(species, grains, strict=True)
    ]

    subdomains = [*grains, network]
    surfaces = {}
    next_id = SURFACE_ID_0
    for name, locator, value in bcs:
        patches = []
        for spe, grain in zip(species, grains, strict=True):
            patch = GrainSurface(next_id, grain.id, micro.cell_tags, locator)
            next_id += 1
            if len(patch.locate_boundary_facet_indices(micro.mesh)) == 0:
                continue  # this grain does not touch that part of the surface
            patches.append(patch)
            subdomains.append(patch)
            boundary_conditions.append(
                F.FixedConcentrationBC(subdomain=patch, value=value, species=spe)
            )
        mouths = F.SurfaceSubdomain(id=next_id, dim=0, locator=locator)
        next_id += 1
        subdomains.append(mouths)
        boundary_conditions.append(
            F.FixedConcentrationBC(subdomain=mouths, value=value, species=c_gb)
        )
        surfaces[name] = (patches, mouths)

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(micro.mesh),
        species=[*species, c_gb],
        subdomains=subdomains,
        sources=sources,
        boundary_conditions=boundary_conditions,
        initial_conditions=list(initial_conditions),
        temperature=physics.T,
        settings=F.Settings(
            atol=atol,
            rtol=rtol,
            transient=transient,
            final_time=final_time,
            stepsize=stepsize,
        ),
        exports=list(exports),
        petsc_options=petsc_options,
    )
    model.show_progress_bar = transient
    return MicroModel(
        model, micro, physics, grains, network, species, c_gb, tensors, surfaces
    )


def grain_areas(micro):
    """Area of every grain, by assembling the DG0 test function against ``dx``.

    That assembly gives one entry per cell holding the cell's own area, so the
    grains are summed from it without a form per grain.
    """
    mesh = micro.mesh
    V = dolfinx.fem.functionspace(mesh, ("DG", 0))
    volumes = dolfinx.fem.assemble_vector(
        dolfinx.fem.form(ufl.TestFunction(V) * ufl.Measure("dx", domain=mesh))
    ).array
    return {
        int(g): float(volumes[V.dofmap.list[micro.cell_tags.find(g)].reshape(-1)].sum())
        for g in micro.grain_ids
    }


def mean_lattice_tensor(mm: MicroModel):
    """Area-weighted average of the grains' lattice tensors."""
    areas = grain_areas(mm.micro)
    total = sum(areas.values())
    return sum(areas[g.id] * mm.tensors[g.id] for g in mm.grains) / total


def parent_field(mm: MicroModel, name="c"):
    """Every grain's solution gathered into one DG1 field on the parent mesh.

    Each grain lives on its own submesh, so a plain export writes one file per
    grain. Interpolating them all into a single discontinuous field instead gives
    one ParaView dataset in which the jump across each boundary is visible --
    which is the thing the per-grain formulation exists to represent.
    """
    V = dolfinx.fem.functionspace(mm.model.mesh.mesh, ("DG", 1))
    field = dolfinx.fem.Function(V, name=name)

    def update():
        for spe, grain in zip(mm.species, mm.grains, strict=True):
            parent_cells = mm.model.volume_meshtags.find(grain.id)
            sub_cells = grain.cell_map.sub_topology_to_topology(
                parent_cells, inverse=True
            )
            field.interpolate(
                spe.subdomain_to_post_processing_solution[grain],
                cells0=sub_cells,
                cells1=parent_cells,
            )

    return field, update


def equilibrium_error(mm: MicroModel):
    """max ``|c_grain - c_gb|`` on the network, over the largest concentration.

    Zero means the boundaries are transparent and a single-field homogeneous
    model can exist; order one means the grains are throttled off from the
    network and no single effective diffusivity will reproduce the microstructure.
    """
    import scipy.spatial

    cgb = mm.network_solution
    x_gb = cgb.function_space.tabulate_dof_coordinates()
    if x_gb.shape[0] == 0:
        return 0.0
    tree = scipy.spatial.cKDTree(x_gb)
    tol = mm.micro.tolerance
    worst, scale = 0.0, 1e-300
    for c in mm.grain_solutions:
        x = c.function_space.tabulate_dof_coordinates()
        scale = max(scale, float(np.abs(c.x.array).max()))
        distance, idx = tree.query(x)
        on_network = distance < tol
        if on_network.any():
            worst = max(
                worst,
                float(
                    np.abs(c.x.array[on_network] - cgb.x.array[idx[on_network]]).max()
                ),
            )
    return worst / scale

"""Numerical acceptance tests for the consolidated transport engine."""

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

import festim as F

from .tools import error_L2


def slab(n, transient=True):
    material = F.Material(
        D_0=1.0,
        E_D=0.0,
        K_S_0=2.0,
        E_K_S=0.0,
        thermal_conductivity=1.0,
        density=1.0,
        heat_capacity=1.0,
    )
    left = F.VolumeSubdomain1D(1, [0.0, 0.5], material)
    right = F.VolumeSubdomain1D(2, [0.5, 1.0], material)
    inlet, outlet = F.SurfaceSubdomain1D(3, 0.0), F.SurfaceSubdomain1D(4, 1.0)
    species = F.Species("H", subdomains=[left, right])
    model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D(np.linspace(0, 1, n + 1)),
        subdomains=[left, right, inlet, outlet],
        species=[species],
        temperature=400.0,
        interfaces=[
            F.Interface(5, [left, right], method="nitsche", penalty_term=100.0)
        ],
        settings=F.Settings(
            atol=1e-11,
            rtol=1e-11,
            transient=transient,
            stepsize=0.05 if transient else None,
            final_time=0.2 if transient else None,
        ),
    )
    model.show_progress_bar = False
    return model, left, right, inlet, outlet, species


def test_split_material_converges_to_continuous_cg_with_local_source():
    errors, continuous_errors, inventories, flux_errors = [], [], [], []
    for n in (16, 32, 64):
        model, left, right, inlet, outlet, species = slab(n, transient=False)
        model.boundary_conditions = [
            F.FixedConcentrationBC(s, 0.0, species) for s in (inlet, outlet)
        ]
        model.sources = [F.ParticleSource(1.0, volume=left, species=species)]
        totals = [F.TotalVolume(species, v) for v in (left, right)]
        fluxes = [F.SurfaceFlux(species, s) for s in (inlet, outlet)]
        model.exports = totals + fluxes
        model.initialise()
        model.run()

        def exact_left(x):
            return -0.5 * x[0] ** 2 + 0.375 * x[0]

        def exact_right(x):
            return 0.125 * (1.0 - x[0])

        errors.append(
            np.hypot(
                error_L2(
                    species.subdomain_to_post_processing_solution[left], exact_left
                ),
                error_L2(
                    species.subdomain_to_post_processing_solution[right], exact_right
                ),
            )
        )
        inventories.append(abs(sum(total.value for total in totals) - 1.0 / 24.0))
        flux_errors.append(
            abs(abs(fluxes[0].value) - 0.375) + abs(abs(fluxes[1].value) - 0.125)
        )
        # Independent implementation of the former shared, continuous P1 space.
        V = fem.functionspace(model.mesh.mesh, ("P", 1))
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        dofs = fem.locate_dofs_topological(
            V,
            0,
            np.concatenate(
                [model.facet_meshtags.find(3), model.facet_meshtags.find(4)]
            ),
        )
        problem = LinearProblem(
            ufl.inner(ufl.grad(u), ufl.grad(v)) * model.dx,
            v * model.dx(left.id),
            bcs=[fem.dirichletbc(0.0, dofs, V)],
            petsc_options_prefix=f"continuous_reference_{n}_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )
        continuous = problem.solve()
        continuous_errors.append(
            error_L2(
                continuous,
                lambda x: np.where(x[0] <= 0.5, exact_left(x), exact_right(x)),
            )
        )
    assert all(errors[i + 1] < 0.3 * errors[i] for i in range(2))
    assert all(continuous_errors[i + 1] < 0.3 * continuous_errors[i] for i in range(2))
    assert errors[-1] < 2 * continuous_errors[-1]
    assert inventories[-1] < 0.3 * inventories[-2]
    assert flux_errors[-1] < 0.6 * flux_errors[-2]

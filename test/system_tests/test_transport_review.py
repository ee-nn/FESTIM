"""Numerical acceptance tests for the consolidated transport engine."""

from mpi4py import MPI

import numpy as np
import pytest
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


@pytest.mark.parametrize("nonmatching", [False, True])
def test_multivolume_coupled_heat_manufactured_solution(nonmatching):
    model, left, right, inlet, outlet, species = slab(40)
    material = left.material
    material.E_D = 0.02

    def exact_temperature(x, t):
        return 400.0 + 20.0 * x[0] + 10.0 * t

    def exact_concentration(x, t):
        return 1.0 + x[0] + t

    def source(x, t):
        thermal = exact_temperature(x, t)
        D = ufl.exp(-material.E_D / (F.k_B * thermal))
        return 1.0 - D * material.E_D * 20.0 / (F.k_B * thermal**2)

    heat = F.HeatTransferProblem(
        mesh=F.Mesh1D(np.linspace(0, 1, 63)) if nonmatching else model.mesh,
        subdomains=[left, right, inlet, outlet],
        boundary_conditions=[
            F.FixedTemperatureBC(s, exact_temperature) for s in (inlet, outlet)
        ],
        initial_condition=F.InitialTemperature(
            lambda x: exact_temperature(x, 0.0), left
        ),
        sources=[F.HeatSource(10.0, volume=v) for v in (left, right)],
        settings=F.Settings(atol=1e-11, rtol=1e-11, stepsize=0.1, final_time=0.2),
    )
    model.boundary_conditions = [
        F.FixedConcentrationBC(s, exact_concentration, species) for s in (inlet, outlet)
    ]
    model.sources = [
        F.ParticleSource(source, volume=v, species=species) for v in (left, right)
    ]
    model.initial_conditions = [
        F.InitialConcentration(
            lambda x: exact_concentration(x, 0.0), volume=v, species=species
        )
        for v in (left, right)
    ]
    coupled = F.CoupledTransientHeatTransferHydrogenTransport(heat, model)
    coupled.initialise()
    assert (
        error_L2(model.temperature_fenics, lambda x: exact_temperature(x, 0.0)) < 1e-8
    )
    for volume in (left, right):
        assert error_L2(volume.sub_T, lambda x: exact_temperature(x, 0.0)) < 1e-8
    for _ in range(4):
        coupled.iterate()
        time = float(model.t)
        assert float(heat.t) == pytest.approx(time)
        assert float(heat.dt) == pytest.approx(float(model.dt))
        assert (
            error_L2(model.temperature_fenics, lambda x: exact_temperature(x, time))
            < 1e-8
        )
        for volume in (left, right):
            assert error_L2(volume.sub_T, lambda x: exact_temperature(x, time)) < 1e-8
            assert (
                error_L2(
                    species.subdomain_to_post_processing_solution[volume],
                    lambda x: exact_concentration(x, time),
                )
                < 1e-8
            )
        # Same physical flux on each side of the interface (ordered restrictions).
        D = material.get_diffusion_coefficient(
            model.mesh.mesh, model.temperature_fenics, species
        )
        fluxes = [
            -D(side) * ufl.grad(species.subdomain_to_solution[v])(side)[0]
            for v, side in zip((left, right), ("+", "-"))
        ]
        jump = fem.form(
            (fluxes[0] - fluxes[1]) ** 2 * model.interior_facet_measure(5),
            entity_maps=[left.cell_map, right.cell_map],
        )
        assert (
            model.mesh.mesh.comm.allreduce(fem.assemble_scalar(jump), op=MPI.SUM)
            < 1e-14
        )


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

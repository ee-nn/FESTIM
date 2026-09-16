"""Conservation and Jacobian checks for distributed mixed-dimensional assembly."""

import numpy as np
import pytest
import ufl
from dolfinx.fem.bcs import bcs_by_block
from dolfinx.fem.petsc import _extract_function_spaces, set_bc
from scifem import assemble_scalar

import festim as F

from .test_codim1_interface import build


@pytest.mark.parametrize("closed", [False, True])
def test_nonlinear_transient_assembly(closed):
    model, subdomains, species = build(n=8)
    left, right, gamma = subdomains
    c_left, c_right, c_gamma = species
    model.sources = []
    model.boundary_conditions = [] if closed else model.boundary_conditions[2:]
    for bulk, c_bulk, rate in ((left, c_left, 2.0), (right, c_right, 3.0)):

        def exchange(c_b, c_g, rate=rate):
            return rate * (c_b**2 - c_g**2)

        def influx(c_b, c_g, rate=rate):
            return rate * (c_g**2 - c_b**2)

        names = {"c_b": c_bulk, "c_g": c_gamma}
        model.sources.append(
            F.ParticleSource(
                value=exchange,
                volume=gamma,
                species=c_gamma,
                species_dependent_value=names,
            )
        )
        model.boundary_conditions.append(
            F.ParticleFluxBC(
                value=influx,
                subdomain=gamma,
                species=c_bulk,
                species_dependent_value=names,
            )
        )
    model.initial_conditions = [
        F.InitialConcentration(value=value, volume=sd, species=sp)
        for value, sd, sp in zip((2.0, 0.5, 0.25), subdomains, species)
    ]
    model.settings.transient = True
    model.settings.final_time = 0.1
    model.settings.stepsize = F.Stepsize(initial_value=0.02)
    model.show_progress_bar = False
    model.initialise()

    def inventories():
        return [
            assemble_scalar(sp.subdomain_to_solution[sd] * ufl.dx(domain=sd.submesh))
            for sd, sp in zip(subdomains, species)
        ]

    # Initial conditions populate the previous-timestep fields. The bulk areas are
    # 1/2 each and the manifold length is 1, giving these exact initial inventories.
    initial = (1.0, 0.25, 0.25)
    for _ in range(5):
        model.iterate()
        if closed:
            assert np.isclose(sum(inventories()), sum(initial), rtol=1e-10, atol=1e-12)

    # Post-processing copies must already contain synchronized ghost values.
    for sd, sp in zip(subdomains, species):
        solution = sp.subdomain_to_post_processing_solution[sd]
        before = solution.x.array.copy()
        solution.x.scatter_forward()
        assert np.allclose(before, solution.x.array, rtol=0, atol=1e-12)
    if closed:
        final = inventories()
        assert final[0] < initial[0]
        assert final[2] > initial[2]

    # Compare the assembled Jacobian action against a centered residual difference.
    # This probes every block, including tangential diffusion on the submesh and
    # off-process bulk/manifold coupling. Zero constrained components makes the
    # direction admissible for strong boundary conditions.
    solver = model.solver
    snes, x, A = solver.solver, solver.x, solver.A
    direction = x.duplicate()
    lo, hi = direction.getOwnershipRange()
    direction.array[:] = np.sin(np.arange(lo, hi) + 0.3)
    direction.setAttr("_blocks", solver.b.getAttr("_blocks"))
    set_bc(
        direction,
        bcs_by_block(_extract_function_spaces(model.forms), model.bc_forms),
        alpha=0,
    )
    xp, xm = x.copy(), x.copy()
    eps = 1e-6
    xp.axpy(eps, direction)
    xm.axpy(-eps, direction)
    bp, bm, action = solver.b.duplicate(), solver.b.duplicate(), solver.b.duplicate()
    snes.computeFunction(xp, bp)
    snes.computeFunction(xm, bm)
    snes.computeJacobian(x, A, A)
    A.mult(direction, action)
    bp.axpy(-1, bm)
    bp.scale(1 / (2 * eps))
    bp.axpy(-1, action)
    assert bp.norm() < 1e-7 * action.norm()

    # Assembly must clear the previous local and ghost contributions on every call.
    snes.computeFunction(x, bp)
    snes.computeFunction(x, bm)
    bp.axpy(-1, bm)
    assert bp.norm() < 1e-12
    for vector in (direction, xp, xm, bp, bm, action):
        vector.destroy()

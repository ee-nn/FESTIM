"""Connectivity, ownership and transfer regressions for transport consolidation."""

import numpy as np
import pytest

import festim as F

from .test_transport_consolidation import make_model


def test_species_reuse_resolves_new_volumes_and_resets_fields():
    first, volume1, species = make_model(explicit=False)
    with pytest.warns(DeprecationWarning, match="Inferring"):
        first.initialise()
    second, volume2, _ = make_model()
    second.species = [species]
    second.initial_conditions = [
        F.InitialConcentration(7.0, species=species, volume=volume2)
    ]
    with pytest.warns(DeprecationWarning, match="Inferring"):
        second.initialise()
    second.run()
    assert species.subdomains is None
    assert first.species_to_subdomains[species] == (volume1,)
    assert second.species_to_subdomains[species] == (volume2,)
    assert set(species.subdomain_to_solution) == {volume2}
    np.testing.assert_allclose(
        species.subdomain_to_post_processing_solution[volume2].x.array, 7.0
    )
    with pytest.warns(DeprecationWarning, match="Inferring"):
        second.initialise()
    assert float(second.t) == 0
    assert second.timesteps == []
    assert set(species.subdomain_to_solution) == {volume2}


def test_callable_tolerances_reach_petsc():
    model, _, _ = make_model()
    model.settings.atol = lambda t: 1e-10 * (1 + t)
    model.settings.rtol = lambda t: 2e-10 * (1 + t)
    model.initialise()
    initial_rtol, initial_atol, *_ = model.solver.solver.getTolerances()
    assert initial_rtol == pytest.approx(model.settings.rtol(0.0), rel=1e-12, abs=0)
    assert initial_atol == pytest.approx(model.settings.atol(0.0), rel=1e-12, abs=0)
    for time in (0.0, 0.1):
        model.iterate()
        rtol, atol, *_ = model.solver.solver.getTolerances()
        assert rtol == pytest.approx(model.settings.rtol(time), rel=1e-12, abs=0)
        assert atol == pytest.approx(model.settings.atol(time), rel=1e-12, abs=0)


def test_value_mesh_fallback_and_explicit_integration_mesh():
    model, volume, _ = make_model()
    model.initialise()
    V = volume.u.function_space
    for integration_mesh in (None, model.mesh.mesh):
        value = F.Value(2.0)
        value.convert_input_value(
            function_space=V, mesh=integration_mesh, up_to_ufl_expr=True
        )
        expected = V.mesh if integration_mesh is None else integration_mesh
        assert value.fenics_object.ufl_domain() == expected.ufl_domain()

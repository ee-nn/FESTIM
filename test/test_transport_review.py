"""Connectivity, ownership and transfer regressions for transport consolidation."""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

import festim as F

from .test_transport_consolidation import make_model, two_domain_model


@pytest.mark.parametrize("explicit", [False, True])
def test_shared_material_requires_complete_interfaces(explicit):
    model, _, _, _ = two_domain_model(explicit=explicit)
    with pytest.raises(ValueError, match=r"species H, volumes 1/2") as error:
        model.initialise()
    assert len(set(model.mesh.mesh.comm.allgather(str(error.value)))) == 1


def split_square():
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    material = F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)
    left = F.VolumeSubdomain(
        1, material=material, locator=lambda x: x[0] <= 0.5 + 1e-12
    )
    right = F.VolumeSubdomain(
        2, material=material, locator=lambda x: x[0] >= 0.5 - 1e-12
    )
    species = F.Species("H", subdomains=[left, right])
    model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        subdomains=[left, right],
        species=[species],
        temperature=300.0,
        settings=F.Settings(atol=1e-10, rtol=1e-10, stepsize=0.1, final_time=0.2),
    )
    model.show_progress_bar = False
    return model, left, right, species


def test_partial_facet_coverage_is_rejected():
    model, left, right, _ = split_square()
    model.interfaces = [F.Interface(3, [left, right])]
    model.define_meshtags_and_measures()
    tags = model.facet_meshtags
    points = dolfinx.mesh.compute_midpoints(model.mesh.mesh, 1, tags.indices)
    values = tags.values.copy()
    values[(values == 3) & (points[:, 1] > 0.5)] = 9
    model.facet_meshtags = dolfinx.mesh.meshtags(
        model.mesh.mesh, 1, tags.indices, values
    )
    with pytest.raises(ValueError, match="facet tag 9"):
        model.initialise()


def test_one_interface_does_not_cover_another_volume_pair():
    model, left, middle, species = two_domain_model(explicit=True)
    middle.borders[:] = [0.5, 0.8]
    right = F.VolumeSubdomain1D(3, [0.8, 1.0], left.material)
    model.subdomains.append(right)
    species.subdomains.append(right)
    model.interfaces = [F.Interface(4, [left, middle])]
    with pytest.raises(ValueError, match="volumes 2/3"):
        model.initialise()


def test_immobile_species_need_no_interfaces():
    model, _, _, species = two_domain_model(explicit=True)
    species.mobile = False
    model.initialise()
    model.run()


def test_nonadjacent_membership_needs_no_interface():
    model, left, middle, species = two_domain_model(explicit=True)
    middle.borders[:] = [0.5, 0.8]
    right = F.VolumeSubdomain1D(3, [0.8, 1.0], left.material)
    model.subdomains.append(right)
    species.subdomains = [left, right]
    model.species.append(F.Species("other", subdomains=[middle]))
    model.initialise()
    model.run()


def test_codimension_separation_preserves_independent_inventories():
    model, left, right, species = split_square()
    gamma = F.VolumeSubdomain(
        3, material=left.material, dim=1, locator=lambda x: np.isclose(x[0], 0.5)
    )
    model.subdomains.append(gamma)
    model.species.append(F.Species("surface", mobile=False, subdomains=[gamma]))
    model.initial_conditions = [
        F.InitialConcentration(1.0, species=species, volume=left),
        F.InitialConcentration(3.0, species=species, volume=right),
    ]
    totals = [F.TotalVolume(species, volume) for volume in (left, right)]
    model.exports = totals
    model.initialise()
    model.run()
    for total, expected in zip(totals, (0.5, 1.5)):
        np.testing.assert_allclose(total.data, expected, atol=1e-10)


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


def test_disconnected_solids_need_no_interface():
    model, left, right, species = two_domain_model(explicit=True)
    model.mesh = F.Mesh1D([np.linspace(0.0, 0.4, 5), np.linspace(0.6, 1.0, 5)])
    left.borders[:] = [0.0, 0.4]
    right.borders[:] = [0.6, 1.0]
    model.initialise()
    model.run()
    assert set(species.subdomain_to_solution) == {left, right}


@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="partition boundary regression")
def test_ghostless_custom_mesh_still_reports_missing_interfaces():
    model, _, _, _ = two_domain_model(explicit=True)
    model.mesh = F.Mesh(
        dolfinx.mesh.create_unit_interval(
            MPI.COMM_WORLD, 10, ghost_mode=dolfinx.mesh.GhostMode.none
        )
    )
    with pytest.raises(ValueError, match="Missing explicit interfaces"):
        model.initialise()

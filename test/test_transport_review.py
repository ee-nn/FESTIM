"""Connectivity, ownership and transfer regressions for transport consolidation."""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
from dolfinx import fem

import festim as F
from festim.helpers import FunctionTransfer

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


def test_interface_reaction_accepts_inferred_species_membership():
    model, left, right, species = two_domain_model(explicit=False)
    model.interfaces = [
        F.InterfaceReaction(
            3,
            [left, right],
            k_plus=2.0,
            k_minus=1.0,
            reactants=species,
            products=species,
        )
    ]
    model.initial_conditions = [
        F.InitialConcentration(1.0, species=species, volume=left),
        F.InitialConcentration(0.0, species=species, volume=right),
    ]
    inventories = [F.TotalVolume(species, v) for v in (left, right)]
    model.exports = inventories
    with pytest.warns(DeprecationWarning, match="Inferring"):
        model.initialise()
    model.run()
    assert species.subdomains is None
    np.testing.assert_allclose(
        np.sum([inventory.data for inventory in inventories], axis=0),
        0.5,
        atol=1e-10,
    )
    assert inventories[1].data[-1] > 0.0


@pytest.mark.parametrize("family,degree", [("P", 1), ("P", 2), ("DG", 0), ("DG", 2)])
def test_parent_temperature_transfer_needs_no_geometric_search(
    monkeypatch, family, degree
):
    model, volume, _ = make_model()
    V = fem.functionspace(model.mesh.mesh, (family, degree))
    temperature = fem.Function(V)
    temperature.interpolate(lambda x: 300.0 + x[0] ** degree)
    model.temperature = temperature

    def forbidden(*args, **kwargs):
        raise AssertionError("A parent/submesh transfer must not search geometrically")

    monkeypatch.setattr(fem, "create_interpolation_data", forbidden)
    model.initialise()
    for step in range(2):
        temperature.x.array[:] += 10.0
        model.iterate()
        expected = fem.Function(volume.sub_T.function_space)
        expected.interpolate(lambda x: 300.0 + 10.0 * (step + 1) + x[0] ** degree)
        np.testing.assert_allclose(volume.sub_T.x.array, expected.x.array)


def test_nonmatching_transfer_reuses_search_data(monkeypatch):
    source_mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    target_mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 7, 7)
    source = fem.Function(fem.functionspace(source_mesh, ("P", 2)))
    target = fem.Function(fem.functionspace(target_mesh, ("P", 2)))
    source.interpolate(lambda x: 300.0 + x[0] ** 2 + x[1])
    transfer = FunctionTransfer(target, source)

    def forbidden(*args, **kwargs):
        raise AssertionError("Geometric interpolation data should be cached")

    monkeypatch.setattr(fem, "create_interpolation_data", forbidden)
    for step in range(2):
        source.x.array[:] += 10.0
        transfer.update()
        expected = fem.Function(target.function_space)
        expected.interpolate(lambda x: 310.0 + step * 10.0 + x[0] ** 2 + x[1])
        np.testing.assert_allclose(target.x.array, expected.x.array)


def test_owned_static_temperature_is_transferred_once(monkeypatch):
    model, volume, _ = make_model()
    model.temperature = lambda x: 300.0 + x[0]
    model.initialise()

    def forbidden():
        raise AssertionError("Static temperature should not be transferred again")

    monkeypatch.setattr(model._temperature_transfers[volume], "update", forbidden)
    model.run()


def test_constant_temperature_export_needs_no_expression(monkeypatch):
    model, _, _ = make_model()
    model.temperature = fem.Constant(model.mesh.mesh, 300.0)
    model.initialise()

    def forbidden(*args, **kwargs):
        raise AssertionError("Constant temperature can be assigned directly")

    monkeypatch.setattr(fem, "Expression", forbidden)
    field = model._get_temperature_field_as_function()
    model.post_processing()
    model.temperature.value = 400.0
    model.post_processing()
    np.testing.assert_allclose(field.x.array, 400.0)


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


def test_manifold_temperature_transfer_reuses_search(monkeypatch):
    model, left, _, _ = split_square()
    gamma = F.VolumeSubdomain(
        3, material=left.material, dim=1, locator=lambda x: np.isclose(x[0], 0.5)
    )
    model.subdomains.append(gamma)
    model.species.append(F.Species("surface", mobile=False, subdomains=[gamma]))
    thermal = fem.Function(fem.functionspace(model.mesh.mesh, ("P", 2)))
    thermal.interpolate(lambda x: 300.0 + x[0] ** 2 + x[1])
    model.temperature = thermal
    model.initialise()
    transfer = model._temperature_transfers[gamma]
    assert transfer.parent_cells is None

    def forbidden(*args, **kwargs):
        raise AssertionError("The manifold interpolation search must be reused")

    monkeypatch.setattr(fem, "create_interpolation_data", forbidden)
    thermal.x.array[:] += 10.0
    model.iterate()
    expected = fem.Function(gamma.sub_T.function_space)
    expected.interpolate(lambda x: 310.0 + x[0] ** 2 + x[1])
    np.testing.assert_allclose(gamma.sub_T.x.array, expected.x.array)


def test_reinitialization_rebuilds_temperature_transfers():
    model, volume, _ = make_model()
    thermal = fem.Function(fem.functionspace(model.mesh.mesh, ("P", 2)))
    thermal.interpolate(lambda x: 300.0 + x[0] ** 2)
    model.temperature = thermal
    model.initialise()
    previous = model._temperature_transfers[volume]
    thermal.x.array[:] += 10.0
    model.initialise()
    current = model._temperature_transfers[volume]
    assert current is not previous
    assert current.target is volume.sub_T
    assert current.target.function_space.mesh is not previous.target.function_space.mesh
    expected = fem.Function(volume.sub_T.function_space)
    expected.interpolate(lambda x: 310.0 + x[0] ** 2)
    np.testing.assert_allclose(volume.sub_T.x.array, expected.x.array)


@pytest.mark.parametrize("function_temperature", [False, True])
def test_boundary_expression_uses_submesh_space_time_and_temperature(
    function_temperature,
):
    model, volume, species = make_model()
    surface = F.SurfaceSubdomain1D(2, 0.0)
    model.subdomains.append(surface)
    if function_temperature:
        thermal = fem.Function(fem.functionspace(model.mesh.mesh, ("P", 1)))
        thermal.interpolate(lambda x: 300.0 + x[0])
    else:
        thermal = fem.Constant(model.mesh.mesh, 300.0)
    model.temperature = thermal
    bc = F.FixedConcentrationBC(
        surface, lambda x, t, T: 1.0 + x[0] + t + T / 300.0, species
    )
    model.boundary_conditions = [bc]
    model.initialise()
    for step in range(3):
        if step:
            if function_temperature:
                thermal.x.array[:] += 10.0
            else:
                thermal.value += 10.0
            model.iterate()
        expected = fem.Function(bc.value_fenics.function_space)
        expected.interpolate(
            lambda x: (
                1.0
                + x[0]
                + float(model.t)
                + (300.0 + 10.0 * step + (x[0] if function_temperature else 0.0))
                / 300.0
            )
        )
        np.testing.assert_allclose(bc.value_fenics.x.array, expected.x.array)
        assert bc.value_fenics.function_space.mesh is volume.submesh

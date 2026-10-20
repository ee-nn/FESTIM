"""Contracts for the single mixed-submesh hydrogen transport engine."""

import importlib

import numpy as np
import pytest
from dolfinx import fem

import festim as F


def make_model(*, explicit=True, immobile=False, element_immobile="CG", degree=1):
    material = F.Material(D_0=1.0, E_D=0.0, K_S_0=1.0, E_K_S=0.0)
    volume = F.VolumeSubdomain1D(1, borders=[0.0, 1.0], material=material)
    species = F.Species(
        "H", subdomains=[volume] if explicit else None, mobile=not immobile
    )
    model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D(np.linspace(0.0, 1.0, 11)),
        subdomains=[volume],
        species=[species],
        temperature=300.0,
        settings=F.Settings(
            atol=1e-12, rtol=1e-12, stepsize=0.1, final_time=0.2, element_degree=degree
        ),
        initial_conditions=[
            F.InitialConcentration(value=2.0, volume=volume, species=species)
        ],
        element_immobile=element_immobile,
    )
    model.show_progress_bar = False
    return model, volume, species


def test_deprecated_class_name_is_the_same_class():
    for module in (F, importlib.import_module("festim.hydrogen_transport_problem")):
        with pytest.warns(
            DeprecationWarning, match="HydrogenTransportProblemDiscontinuous"
        ):
            alias = module.HydrogenTransportProblemDiscontinuous
        assert alias is F.HydrogenTransportProblem
        assert not hasattr(module, "HydrogenTransportProblemDiscontinuousChangeVar")
        namespace = {}
        with pytest.warns(DeprecationWarning):
            exec(f"from {module.__name__} import *", namespace)
        assert namespace["HydrogenTransportProblemDiscontinuous"] is alias
    assert F.HydrogenTransportProblem.__bases__ == (F.ProblemBase,)


def test_inferred_subdomains_warn_and_preserve_single_domain_setup():
    model, volume, species = make_model(explicit=False)
    with pytest.warns(DeprecationWarning, match="Inferring subdomains"):
        model.initialise()
    assert species.subdomains == [volume]
    model.run()
    assert np.allclose(
        species.subdomain_to_post_processing_solution[volume].x.array, 2.0
    )
    assert np.allclose(model.timesteps, [0.0, 0.1])


@pytest.mark.parametrize(
    "membership, error", [([], ValueError), ("invalid", TypeError)]
)
def test_explicit_invalid_subdomains_are_not_inferred(membership, error):
    model, _, species = make_model()
    species.subdomains = membership
    with pytest.raises(error, match="subdomains"):
        model.initialise()


def two_domain_model(*, explicit):
    model, first, species = make_model()
    first.borders[:] = [0.0, 0.5]
    second = F.VolumeSubdomain1D(2, borders=[0.5, 1.0], material=first.material)
    model.subdomains = [first, second]
    species.subdomains = [first, second] if explicit else None
    model.initial_conditions = []
    return model, first, second, species


def test_inferred_multimaterial_membership_requires_interfaces():
    model, _, _, _ = two_domain_model(explicit=False)
    with pytest.raises(ValueError, match="explicit interfaces"):
        model.initialise()


def test_inferred_multimaterial_membership_with_explicit_interfaces():
    model, first, second, species = two_domain_model(explicit=False)
    model.interfaces = [F.Interface(3, [first, second])]
    with pytest.warns(DeprecationWarning, match="Inferring subdomains"):
        model.initialise()
    assert species.subdomains == [first, second]


def test_single_domain_aliases_reference_live_functions():
    model, volume, species = make_model()
    model.initialise()
    aliases = {
        "solution": species.subdomain_to_solution[volume],
        "prev_solution": species.subdomain_to_prev_solution[volume],
        "test_function": species.subdomain_to_test_function[volume],
        "sub_function": species.subdomain_to_sub_function[volume],
        "sub_function_space": species.subdomain_to_function_space[volume],
        "collapsed_function_space": species.subdomain_to_collapsed_function_space[
            volume
        ][0],
        "map_sub_to_main_solution": species.subdomain_to_collapsed_function_space[
            volume
        ][1],
        "post_processing_solution": species.subdomain_to_post_processing_solution[
            volume
        ],
    }
    for name, target in aliases.items():
        with pytest.warns(DeprecationWarning):
            assert getattr(species, name) is target
    for name, target in {
        "u": volume.u,
        "u_n": volume.u_n,
        "function_space": volume.u.function_space,
    }.items():
        with pytest.warns(DeprecationWarning):
            assert getattr(model, name) is target
    model.run()
    assert np.allclose(aliases["post_processing_solution"].x.array, 2.0)


def test_ambiguous_solution_aliases_fail():
    model, first, second, species = two_domain_model(explicit=True)
    model.interfaces = [F.Interface(3, [first, second])]
    model.initialise()
    for obj, names in [
        (model, ["u", "u_n", "function_space"]),
        (
            species,
            [
                "solution",
                "prev_solution",
                "test_function",
                "post_processing_solution",
                "sub_function",
            ],
        ),
    ]:
        for name in names:
            with pytest.raises(ValueError, match=r"ambiguous.*per-subdomain"):
                getattr(obj, name)


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("element_immobile", ["CG", "P", "DG"])
def test_allocated_elements_honor_degree_and_immobile_family(degree, element_immobile):
    model, volume, species = make_model(
        immobile=True, degree=degree, element_immobile=element_immobile
    )
    mobile = F.Species("mobile", subdomains=[volume])
    model.species.append(mobile)
    model.initialise()
    element = species.subdomain_to_function_space[volume].element.basix_element
    assert element.degree == degree
    assert element.discontinuous == (element_immobile == "DG")
    assert not mobile.subdomain_to_function_space[
        volume
    ].element.basix_element.discontinuous
    model.run()
    assert np.allclose(
        species.subdomain_to_post_processing_solution[volume].x.array, 2.0
    )


def test_trap_generated_species_has_explicit_membership():
    model, volume, species = make_model()
    trap = F.Trap(
        "trapped", species, k_0=1.0, E_k=0.0, p_0=0.1, E_p=0.0, n=5.0, volume=volume
    )
    model.traps = [trap]
    model.initialise()
    assert trap.trapped_concentration.subdomains == [volume]
    model.run()
    mobile = species.subdomain_to_post_processing_solution[volume].x.array
    trapped = trap.trapped_concentration.subdomain_to_post_processing_solution[
        volume
    ].x.array
    assert np.allclose(mobile + trapped, 2.0)


def test_trap_mobile_species_must_exist_on_its_volume():
    model, volume, _ = make_model()
    model.traps = [
        F.Trap(
            "trapped",
            F.Species("absent"),
            k_0=1.0,
            E_k=0.0,
            p_0=0.1,
            E_p=0.0,
            n=5.0,
            volume=volume,
        )
    ]
    with pytest.raises(ValueError, match="Mobile species of trap"):
        model.initialise()


@pytest.mark.parametrize("temperature_kind", ["constant", "callable", "external"])
def test_custom_field_uses_current_mesh_local_time_temperature_and_species(
    tmp_path, temperature_kind
):
    model, volume, species = make_model()
    if temperature_kind == "callable":
        model.temperature = lambda t: 300.0 + t
    elif temperature_kind == "external":
        V = fem.functionspace(model.mesh.mesh, ("P", 1))
        model.temperature = fem.Function(V)
        model.temperature.x.array[:] = 300.0
    export = F.CustomFieldExport(
        tmp_path / "field.bp",
        expression=lambda c, x, t, T: c + x[0] + t + T,
        species_dependent_value={"c": species},
    )
    model.exports = [export]
    model.initialise()
    assert export.subdomain is volume
    if temperature_kind == "external":
        model.temperature.x.array[:] = 350.0
    model.iterate()
    coords = export.function.function_space.tabulate_dof_coordinates()[:, 0]
    temperature = {"constant": 300.0, "callable": 300.1, "external": 350.0}[
        temperature_kind
    ]
    np.testing.assert_allclose(
        export.function.x.array, 2.0 + coords + 0.1 + temperature
    )


def test_constant_temperature_exports_follow_updates(tmp_path):
    model, _, _ = make_model()
    model.temperature = lambda t: 300.0 + 10.0 * t
    exports = [F.TemperatureExport(tmp_path / f"temperature_{i}.bp") for i in range(2)]
    model.exports = exports
    model.initialise()
    model.iterate()
    np.testing.assert_allclose(model._temperature_as_function.x.array, 301.0)


def test_ambiguous_field_export_requires_a_subdomain(tmp_path):
    model, first, second, species = two_domain_model(explicit=True)
    model.interfaces = [F.Interface(3, [first, second])]
    model.exports = [F.SpeciesExport(tmp_path / "field.bp", field=species)]
    with pytest.raises(ValueError, match="explicit subdomain"):
        model.initialise()


def test_surface_flux_diffusivity_tracks_temperature_without_a_global_cache():
    model, volume, species = make_model()
    volume.material = F.Material(D_0=2.0, E_D=0.01)
    left = F.SurfaceSubdomain1D(1, x=0.0)
    right = F.SurfaceSubdomain1D(2, x=1.0)
    model.subdomains.extend([left, right])
    model.temperature = lambda t: 300.0 + 100.0 * t
    model.initial_conditions = [F.InitialConcentration(lambda x: x[0], volume, species)]
    model.boundary_conditions = [
        F.FixedConcentrationBC(left, lambda x: x[0], species),
        F.FixedConcentrationBC(right, lambda x: x[0], species),
    ]
    flux = F.SurfaceFlux(field=species, surface=right)
    model.exports = [flux]
    model.initialise()
    for _ in range(2):
        model.iterate()
        expected = 2.0 * np.exp(-0.01 / F.k_B / (300.0 + 100.0 * float(model.t)))
        assert np.isclose(abs(flux.value), expected)


def test_ready_made_diffusivity_can_be_used_for_each_volume():
    model, first, second, species = two_domain_model(explicit=True)
    V = fem.functionspace(model.mesh.mesh, ("P", 1))
    D = fem.Function(V)
    D.interpolate(lambda x: 1.0 + x[0])
    first.material.D = D
    second.material.D = D
    model.initialise()
    assert model.diffusion_coefficient(first, species) is D
    assert model.diffusion_coefficient(second, species) is D


def test_boundary_conditions_accept_species_names():
    model, volume, species = make_model()
    left = F.SurfaceSubdomain1D(1, x=0.0)
    right = F.SurfaceSubdomain1D(2, x=1.0)
    model.subdomains.extend([left, right])
    concentration = F.FixedConcentrationBC(left, value=2.0, species="H")
    flux = F.ParticleFluxBC(right, value=0.0, species="H")
    model.boundary_conditions = [concentration, flux]
    model.initialise()
    assert concentration.species is species
    assert flux.species is species
    model.run()
    np.testing.assert_allclose(
        species.subdomain_to_post_processing_solution[volume].x.array, 2.0
    )


def test_quadratic_external_temperature_is_preserved_on_the_submesh():
    model, volume, species = make_model(degree=2)
    V = fem.functionspace(model.mesh.mesh, ("P", 2))
    temperature = fem.Function(V)
    temperature.interpolate(lambda x: 300.0 + x[0] ** 2)
    model.temperature = temperature
    model.initial_conditions = [F.InitialConcentration(lambda T: T, volume, species)]
    model.initialise()
    initial = volume.u_n.sub(0).collapse()
    x = initial.function_space.tabulate_dof_coordinates()[:, 0]
    np.testing.assert_allclose(initial.x.array, 300.0 + x**2, atol=1e-12)
    temperature.interpolate(lambda x: 400.0 + 2.0 * x[0] ** 2)
    model.update_time_dependent_values()
    x = volume.sub_T.function_space.tabulate_dof_coordinates()[:, 0]
    np.testing.assert_allclose(volume.sub_T.x.array, 400.0 + 2.0 * x**2, atol=1e-12)

from collections.abc import Sequence

import numpy as np
import scifem
from dolfinx import fem

import festim as F


def evaluate_at_point(u: fem.Function, point: float | Sequence[float]):
    """Evaluate a finite element function at a physical point.

    Thin wrapper around :func:`scifem.evaluate_function`. Evaluating by
    coordinate rather than indexing ``u.x.array`` by position makes the result
    independent of DOF ordering (which changes between DOLFINx versions) and
    works for any element family and degree (Lagrange, DG, ...). ``scifem``
    handles locating the cell and, in parallel, gathering the value from the
    rank that owns it.

    Args:
        u: the function to evaluate.
        point: the coordinate, as a float (1D) or a sequence of ``gdim`` floats.

    Returns:
        The value of ``u`` at ``point``: a float for a scalar function, otherwise
        an array of its components.
    """
    gdim = u.function_space.mesh.geometry.dim
    coords = np.asarray(point, dtype=np.float64).reshape(1, gdim)
    values = np.ravel(scifem.evaluate_function(u, coords))
    return values[0] if values.size == 1 else values


def prepare_subdomains(model, element_degree=1):
    """Allocate the mesh-local fields needed by focused coefficient tests."""
    if not model.volume_subdomains:
        coords = model.mesh.mesh.geometry.x[:, 0]
        material = F.Material(D_0=1, E_D=0)
        if isinstance(model.mesh, F.Mesh1D):
            model.subdomains.append(
                F.VolumeSubdomain1D(1, [coords.min(), coords.max()], material)
            )
        else:
            model.subdomains.append(F.VolumeSubdomain(id=1, material=material))
    if model.settings is None:
        model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)
    if not hasattr(model, "t"):
        model.t = fem.Constant(model.mesh.mesh, 0.0)
    model._normalise_species_subdomains()
    model.define_meshtags_and_measures()
    model.manifold_to_volumes = {}
    model.surface_to_volume = F.map_surface_to_volume_subdomains(
        ft=model.facet_meshtags,
        ct=model.volume_meshtags,
        facet_to_cell=model.mesh.mesh.topology.connectivity(
            model.mesh.vdim - 1, model.mesh.vdim
        ),
        volume_subdomains=model.volume_subdomains,
        surface_subdomains=model.surface_subdomains,
        comm=model.mesh.mesh.comm,
    )
    for volume in model.volume_subdomains:
        volume.create_subdomain(model.mesh.mesh, model.volume_meshtags)
        volume.transfer_meshtag(model.mesh.mesh, model.facet_meshtags)
        model.define_function_spaces(volume, element_degree=element_degree)
    model._spaces_defined = True
    model.create_submesh_time_constants()
    if model.temperature_fenics is not None:
        model._define_submesh_temperatures()

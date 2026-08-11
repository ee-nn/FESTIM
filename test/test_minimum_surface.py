import numpy as np
import pytest
from dolfinx import fem

import festim as F


def test_minimum_surface_export_compute_1D():
    """Test that the minimum surface export computes the correct value."""

    # BUILD
    L = 4.0
    D = 1.5
    my_mesh = F.Mesh1D(np.linspace(0, L, 10000))
    dummy_surface = F.SurfaceSubdomain1D(id=1, x=4)

    # create mesh tags
    ft, _ct = my_mesh.define_meshtags(
        surface_subdomains=[dummy_surface],
        volume_subdomains=[
            F.VolumeSubdomain1D(id=1, material=F.Material(D_0=1, E_D=0), borders=[0, L])
        ],
        interfaces=None,
    )

    # give function to species
    V = fem.functionspace(my_mesh.mesh, ("Lagrange", 1))
    c = fem.Function(V)
    c.interpolate(lambda x: (x[0] - 2) ** 2)

    my_species = F.Species("H")
    my_species.post_processing_solution = c

    my_export = F.MinimumSurface(field=my_species, surface=dummy_surface)
    my_export.D = D
    my_export.facet_meshtags = ft

    # RUN: the meshtags set on the export are used when none are passed
    my_export.compute()

    # TEST
    expected_value = 4.0
    computed_value = my_export.value

    assert np.isclose(computed_value, expected_value, rtol=1e-2)


def test_minimum_surface_meshtags_argument_wins():
    """Meshtags passed to `compute` override the ones set on the export, which is how
    the discontinuous problem points the extremum at a submesh."""
    L = 4.0
    my_mesh = F.Mesh1D(np.linspace(0, L, 10000))
    dummy_surface = F.SurfaceSubdomain1D(id=1, x=4)
    ft, _ct = my_mesh.define_meshtags(
        surface_subdomains=[dummy_surface],
        volume_subdomains=[
            F.VolumeSubdomain1D(id=1, material=F.Material(D_0=1, E_D=0), borders=[0, L])
        ],
        interfaces=None,
    )

    V = fem.functionspace(my_mesh.mesh, ("Lagrange", 1))
    c = fem.Function(V)
    c.interpolate(lambda x: (x[0] - 2) ** 2)

    my_export = F.MinimumSurface(field=F.Species("H"), surface=dummy_surface)
    my_export.facet_meshtags = None
    my_export.compute(u=c, facet_meshtags=ft)

    assert np.isclose(my_export.value, 4.0, rtol=1e-2)


def test_minimum_surface_unmatched_id_raises():
    """An id that matches no facet used to come back as a +/-inf sentinel, which is
    worse than a traceback: it lands in the exported csv as if it were a result."""
    L = 4.0
    my_mesh = F.Mesh1D(np.linspace(0, L, 100))
    dummy_surface = F.SurfaceSubdomain1D(id=1, x=4)
    ft, _ct = my_mesh.define_meshtags(
        surface_subdomains=[dummy_surface],
        volume_subdomains=[
            F.VolumeSubdomain1D(id=1, material=F.Material(D_0=1, E_D=0), borders=[0, L])
        ],
        interfaces=None,
    )

    V = fem.functionspace(my_mesh.mesh, ("Lagrange", 1))
    c = fem.Function(V)

    my_export = F.MinimumSurface(
        field=F.Species("H"), surface=F.SurfaceSubdomain1D(id=99, x=4)
    )
    with pytest.raises(ValueError, match="matches no facet"):
        my_export.compute(u=c, facet_meshtags=ft)


def test_minimum_surface_without_meshtags_raises():
    """No meshtags anywhere is a mistake, not a reason to fall through."""
    my_export = F.MinimumSurface(
        field=F.Species("H"), surface=F.SurfaceSubdomain1D(id=1, x=4)
    )
    with pytest.raises(ValueError, match="no facet meshtags"):
        my_export.compute()

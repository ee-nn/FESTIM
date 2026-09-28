import math

import ufl
from dolfinx import fem
from scifem import assemble_scalar

from festim.exports.surface_quantity import SurfaceQuantity
from festim.helpers import restrict
from festim.mesh import CoordinateSystem
from festim.species import Species
from festim.subdomain.surface_subdomain import SurfaceSubdomain
from festim.subdomain.volume_subdomain import VolumeSubdomain


class SurfaceFlux(SurfaceQuantity):
    """Computes the flux of a field on a given surface.

    The total flux ``-D grad(c) . n + c v . n`` is computed, so a drift term acting on
    the field -- advection, Soret, electromigration -- contributes.

    Args:
        field: species for which the surface flux is computed
        surface: surface subdomain. See `festim.SurfaceQuantity` for the codimensional
            cases -- a manifold, or the boundary of one.
        filename: name of the file to which the surface flux is exported

    Attributes:
        drift_velocity: the summed velocity of the drift terms acting on ``field`` in
            the subdomain this surface bounds, set by the problem during
            ``initialise()``. ``None`` when there is none
        see `festim.SurfaceQuantity` for the rest
    """

    field: Species
    surface: SurfaceSubdomain | VolumeSubdomain
    filename: str
    coordinate_system: CoordinateSystem

    title: str
    value: float
    data: list[float]
    drift_velocity = None

    @property
    def title(self):
        return f"{self.field.name} flux surface {self.surface.id}"

    def __init__(
        self,
        field: Species,
        surface: SurfaceSubdomain | VolumeSubdomain,
        filename: str | None = None,
    ) -> None:
        super().__init__(field=field, surface=surface, filename=filename)

    def compute(
        self,
        u: fem.Function | ufl.indexed.Indexed,
        ds: ufl.Measure,
        entity_maps=None,
        restriction: str | None = None,
        subdomain_id: int | None = None,
    ):
        """Computes the value of the flux at the surface.

        Args:
            u: field for which the flux is computed
            ds: surface measure of the model
            entity_maps: entity maps relating parent mesh and submesh
            restriction: which side of an interior facet to evaluate the flux on, when
                ``ds`` is an interior facet measure. The whole integrand is restricted,
                so the normal is the one pointing out of that side and the sign
                convention matches an exterior surface.
            subdomain_id: the id to index ``ds`` with, when it is not the surface's own.
                One side of a manifold adjacent to more than two volume subdomains is
                integrated under an id of its own
        """

        # obtain mesh normal from integration domain
        mesh = ds.ufl_domain()
        n = ufl.FacetNormal(mesh)

        match self.coordinate_system:
            case CoordinateSystem.CARTESIAN:
                weight = 1
            case CoordinateSystem.CYLINDRICAL:
                r = ufl.SpatialCoordinate(mesh)[0]
                # TODO: full coverage assumed; expose as a constructor parameter
                # (e.g. azimuth_range) to support partial coverage
                coverage = 2 * math.pi  # radians
                weight = coverage * r
            case CoordinateSystem.SPHERICAL:
                r = ufl.SpatialCoordinate(mesh)[0]
                # TODO: full coverage assumed; expose as constructor parameters
                # (e.g. azimuth_range, polar_range) to support partial coverage
                coverage = 4 * math.pi  # steradians
                weight = coverage * r**2
            case _:
                raise NotImplementedError(
                    f"Unknown coordinate system {self.coordinate_system!s}"
                )

        integrand = -self.D * ufl.dot(ufl.grad(u), n)
        if subdomain_id is None:
            subdomain_id = self.surface.id

        if self.drift_velocity is not None:
            integrand += u * ufl.dot(self.drift_velocity, n)

        # the weight carries the metric of the coordinate system, so it multiplies the
        # whole flux density, drift included
        integrand *= weight

        self.value = assemble_scalar(
            fem.form(
                restrict(integrand, restriction) * ds(subdomain_id),
                entity_maps=entity_maps,
            )
        )
        self.data.append(self.value)

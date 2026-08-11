from mpi4py import MPI

import dolfinx
import numpy as np

from festim.exports.surface_quantity import SurfaceQuantity


class MaximumSurface(SurfaceQuantity):
    """Computes the maximum value of a field on a given surface.

    Args:
        field (festim.Species): species for which the maximum surface is computed
        surface (festim.SurfaceSubdomain): surface subdomain
        filename (str, optional): name of the file to which the maximum surface
        is exported

    Attributes:
        see `festim.SurfaceQuantity`
        facet_meshtags: the facet meshtags of the mesh the field is defined on
    """

    facet_meshtags: dolfinx.mesh.MeshTags | None = None

    @property
    def title(self):
        return f"Maximum {self.field.name} surface {self.surface.id}"

    def compute(
        self,
        u: dolfinx.fem.Function | None = None,
        facet_meshtags: dolfinx.mesh.MeshTags | None = None,
    ):
        """Computes the maximum value of the field on the defined surface subdomain, and
        appends it to the data list.

        Args:
            u: the field the maximum is computed from. Defaults to
                ``self.field.post_processing_solution``
            facet_meshtags: the facet meshtags used to locate the facets of the
                surface subdomain. Defaults to ``self.facet_meshtags``. For the
                discontinuous problem these are the facet meshtags of the submesh
                the field lives on (``VolumeSubdomain.ft``)

        Raises:
            ValueError: if no facet meshtags are available, or if the surface id is
                absent from them on every process
        """
        solution = self.field.post_processing_solution if u is None else u
        meshtags = self.facet_meshtags if facet_meshtags is None else facet_meshtags
        if meshtags is None:
            raise ValueError(
                f"Cannot compute {self.title}: no facet meshtags were passed to "
                "`compute` and none are set on the export."
            )

        if isinstance(solution, dolfinx.fem.Function):
            V = solution.function_space
        else:
            V = self.field.sub_function_space
        mesh = V.mesh
        fdim = mesh.topology.dim - 1

        entities = meshtags.find(self.surface.id)
        mesh.topology.create_connectivity(fdim, mesh.topology.dim)
        dofs = dolfinx.fem.locate_dofs_topological(
            V=V, entity_dim=fdim, entities=entities
        )
        values = solution.x.array[dofs]

        # a process may hold no dof of the surface at all, np.max would then raise
        local_max = np.max(values) if values.size > 0 else -np.inf

        # ... but if *no* process holds one the surface id is not in these meshtags,
        # and the allreduce below would return the sentinel as if it were a result
        if mesh.comm.allreduce(values.size, op=MPI.SUM) == 0:
            raise ValueError(
                f"Cannot compute {self.title}: surface id {self.surface.id} matches "
                "no facet of the given meshtags, so there is no value to take the "
                "maximum of."
            )

        self.value = mesh.comm.allreduce(local_max, op=MPI.MAX)
        self.data.append(self.value)

Transport consolidation (unreleased)
===================================

This is a breaking numerical and API change. ``HydrogenTransportProblem`` now
uses a separate finite-element field on each volume subdomain. The former
``HydrogenTransportProblemDiscontinuous`` name remains available as a deprecated
explicit import; wildcard imports expose only the canonical name. The
change-of-variable implementation has been removed.

Multiple volumes, including one material
---------------------------------------

Splitting a material into several volumes to localize a trap or source now
requires explicit interface conditions wherever a mobile species occupies both
adjacent volumes. This applies even when both volumes reference the same
``Material`` object. Initialization checks the actual shared facets, so one
interface does not excuse a missing interface elsewhere, or an incompletely
marked boundary. Errors identify the affected species, volume IDs and facet tags.
Disconnected volumes and immobile species do not need interface conditions.
For custom MPI meshes, create adjacent ghost cells with
``dolfinx.mesh.GhostMode.shared_facet`` so interface integrals can see both sides
of partition boundaries. ``Mesh1D`` configures this automatically.

Use explicit species membership and, where concentration continuity is intended,
a Nitsche interface between materials with matching physical solubility laws and
coefficients. For example::

    H = F.Species("H", subdomains=[left, right])
    model.interfaces = [
        F.Interface(interface_tag, [left, right], method="nitsche")
    ]

The shared continuous space previously imposed concentration continuity strongly.
The new formulation imposes its interface law weakly, so finite-mesh solutions
may differ. Check mesh convergence, inventory and flux when migrating a model.
Different solubilities impose a partition law rather than concentration continuity.
Do not replace physical solubility coefficients with arbitrary unit values when
those coefficients also determine pressure boundary conditions.

Intentional separation and codimension settings
-----------------------------------------------

Existing codimension-one volume subdomains can explicitly separate ordinary
volumes. Their facets do not require an additional direct volume-to-volume
interface. Exchange through a codimension-one subdomain is determined by its
configured fluxes, sources and reactions; declaring the subdomain alone does
not automatically create transport across it. Without exchange terms the
separate bulk fields retain their natural no-flux boundary behavior.

No new impermeable-boundary declaration or flag is introduced. Use the existing
codimension settings to express the intended separation and coupling, as described
in :doc:`userguide/subdomains`. Direct interface conditions and codimension-one
subdomains must not overlap on the same facets.

Species membership and reuse
----------------------------

Declare ``Species.subdomains`` explicitly for new models. Omitted membership is
still inferred from the ordinary volumes, with a ``DeprecationWarning``; it does
not bypass interface coverage checks. The inferred membership is stored in
``model.species_to_subdomains`` and does not modify ``Species.subdomains``.
Manifold membership must be explicit.

Species instances hold their allocated solution fields for one model at a time.
Sequential reuse and reinitialization reset those fields. Use distinct Species
instances for models that must remain usable concurrently. Read fields through
``species.subdomain_to_solution[volume]`` or
``species.subdomain_to_post_processing_solution[volume]``. Deprecated solution
aliases are available only when the target is unique.

Coupled heat transfer
---------------------

Heat coupling supports multiple hydrogen volumes and matching or nonmatching
heat meshes. The prescribed initial temperature is transferred before hydrogen
initialization. After each heat step, temperature updates reach the hydrogen
parent mesh and its submeshes before temperature-dependent inputs are evaluated.
Parent/submesh transfers use the cell correspondence; other transfers reuse
geometric interpolation data until meshes or spaces are reinitialized. Externally
supplied temperature Functions remain refreshable even when the callable
``temperature_time_dependent`` flag is false.

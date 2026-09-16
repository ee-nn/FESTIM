===================
Running in parallel
===================

FESTIM can be run in parallel on N cores using the command: 

.. code::
    
    mpirun -np N python your_FESTIM_script.py


The mesh is partitioned between the processes by DOLFINx and each process solves its share of the problem.
The quantities computed by the exports (the ``value`` attribute of :class:`festim.SurfaceFlux`, :class:`festim.TotalVolume`, :class:`festim.MaximumVolume`...) are global: they are the same on every process.
Derived-quantity CSV files are written only by rank zero of the simulation's mesh
communicator. Their ``value``, ``data`` and ``t`` attributes remain available on all
processes.
On the other hand, the arrays of the solution (``species.solution.x.array``) contain local owned and ghost degrees of freedom, so any post-processing done directly in the script needs the appropriate MPI reduction, for example ``mesh.comm.allreduce(local_max, op=MPI.MAX)``. Sums must exclude ghost degrees of freedom to avoid counting them twice.

.. warning::

    Not every feature of FESTIM supports parallel runs yet. In particular:

    * Parallel support for codimensional (manifold) subdomains lying *inside* the
      mesh is experimental (see :ref:`Codimensional (manifold) Subdomains`). The
      MPI solver tests cover steady and transient coupling, multiple manifolds,
      grain-boundary networks, manifold boundary conditions, and nonlinear
      conservation and Jacobian checks. Derived quantities include totals, averages,
      extrema, fluxes and custom integrals on manifolds and their boundaries.
      Internal manifold facets require both adjacent cells on
      their owning rank: create or read the mesh with
      ``dolfinx.mesh.GhostMode.shared_facet`` ghosting. Missing adjacent cells
      cause ``initialise()`` to raise an error. Manifolds lying on the outer
      boundary of the domain are not affected by this requirement.
    * :class:`festim.Profile1DExport` only collects the part of the profile owned by each process, without raising an error.

    When in doubt, compare the results of a small case run in serial and in parallel.

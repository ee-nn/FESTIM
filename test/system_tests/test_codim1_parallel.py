"""Distributed topology and a minimal internal-manifold coupling regression.

Run with ``mpirun -np 2 python -m pytest test/system_tests/test_codim1_parallel.py``.
The same tests also run in serial and on four ranks.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl
from scifem import assemble_scalar

import festim as F
from festim.helpers import meshtags_with_ghosts

from .test_codim1_interface import build


@pytest.mark.parametrize("dim", [1, 2])
def test_sparse_owner_tags_are_forwarded_to_ghosts(dim):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    mesh.topology.create_entities(dim)
    index_map = mesh.topology.index_map(dim)
    indices = np.arange(index_map.size_local + index_map.num_ghosts, dtype=np.int32)
    global_ids = index_map.local_to_global(indices)
    # Leave holes and include tag zero: absence must not be encoded as a tag value.
    selected = (indices < index_map.size_local) & (global_ids % 3 != 0)
    tags = dolfinx.mesh.meshtags(
        mesh, dim, indices[selected], (global_ids[selected] % 2).astype(np.int32)
    )
    tags.name = "sparse tags"
    synced = meshtags_with_ghosts(mesh, tags)
    expected = global_ids % 3 != 0
    assert synced.name == tags.name
    assert np.array_equal(synced.indices, indices[expected])
    assert np.array_equal(synced.values, global_ids[expected] % 2)


@pytest.mark.parametrize("owner_only_tags", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_internal_manifold_coupling(owner_only_tags, reverse):
    model, (left, right, gamma), species = build(n=12, swap_declaration_order=reverse)
    mesh = model.mesh.mesh
    if owner_only_tags:
        # Mimic tags loaded from a file: only owners carry markers. Sparse cell
        # tags are exercised separately below.
        model.define_meshtags_and_measures()
        for name in ("facet_meshtags", "volume_meshtags"):
            tags = getattr(model, name)
            owned = tags.indices < mesh.topology.index_map(tags.dim).size_local
            setattr(
                model,
                name,
                dolfinx.mesh.meshtags(
                    mesh, tags.dim, tags.indices[owned], tags.values[owned]
                ),
            )
    model.initialise()
    assert model.manifold_is_interior(gamma)

    # The facet is integrated once globally, with a stable choice of bulk side.
    assert np.isclose(assemble_scalar(1 * model.facet_measure(gamma)(gamma.id)), 1)
    ct = model.volume_meshtags
    facet_data = model._manifold_integration_data(gamma)[0][1].reshape(-1, 4)
    lookup = np.full(
        mesh.topology.index_map(2).size_local + mesh.topology.index_map(2).num_ghosts,
        -1,
        dtype=np.int32,
    )
    lookup[ct.indices] = ct.values
    plus = model.manifold_to_volumes[gamma][0]
    assert np.all(lookup[facet_data[:, 0]] == plus.id)

    model.run()
    # Compare the entire field with the analytic solution, including ranks with no
    # local dofs. Different exchange rates make accidental side swaps observable.
    exact = (
        lambda x: 2 - 8 / 9 * x,
        lambda x: 8 / 9 * (1 - x),
        lambda x: np.full_like(x, 8 / 9),
    )
    for sp, subdomain, expression in zip(species, (left, right, gamma), exact):
        solution = sp.subdomain_to_post_processing_solution[subdomain]
        x = solution.function_space.tabulate_dof_coordinates()[:, 0]
        assert np.allclose(solution.x.array, expression(x), atol=1e-8)


def test_ordering_with_sparse_cell_tags():
    model, (left, right, gamma), _ = build(n=12)
    model.define_meshtags_and_measures()
    mesh = model.mesh.mesh
    mesh.topology.create_connectivity(1, 2)
    facets = model.facet_meshtags.find(gamma.id)
    # Only the cells touching Gamma are tagged, so local cell indices cannot be
    # used as offsets into MeshTags.values.
    cells = dolfinx.mesh.compute_incident_entities(mesh.topology, facets, 1, 2)
    ct = model.volume_meshtags
    keep = np.isin(ct.indices, cells)
    sparse = dolfinx.mesh.meshtags(mesh, 2, ct.indices[keep], ct.values[keep])
    tag, data = F.subdomain.compute_ordered_interior_facet_data(
        sparse, model.facet_meshtags, gamma.id, right, left
    )
    dS = ufl.Measure("dS", domain=mesh, subdomain_data=[(tag, data)])
    # On the right-hand bulk's outward normal, n_x = -1.
    assert np.isclose(assemble_scalar(ufl.FacetNormal(mesh)("+")[0] * dS(tag)), -1)


def test_coupling_with_empty_ranks():
    def partition_on_rank_zero(comm, nparts, tdim, cells):
        # 0.11 passes flattened vertex arrays per cell type; 0.10 passes adjacency.
        num_cells = len(cells[0]) // 3 if isinstance(cells, list) else cells.num_nodes
        return dolfinx.cpp.graph.AdjacencyList_int32(
            np.zeros((num_cells, 1), dtype=np.int32)
        )

    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 8, 8, partitioner=partition_on_rank_zero
    )
    if mesh.comm.rank != 0:
        assert mesh.topology.index_map(2).size_local == 0
    model, (_, _, gamma), (_, _, species) = build(mesh=mesh)
    model.initialise()
    model.run()
    values = species.subdomain_to_post_processing_solution[gamma].x.array
    assert np.allclose(values, 8 / 9, atol=1e-8)
    assert np.isclose(assemble_scalar(1 * model.facet_measure(gamma)(gamma.id)), 1)


@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="needs a partition boundary")
def test_missing_facet_ghosting_is_rejected_collectively():
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 8, 8, ghost_mode=dolfinx.mesh.GhostMode.none
    )
    mesh.topology.create_connectivity(1, 2)
    owned = np.arange(mesh.topology.index_map(1).size_local, dtype=np.int32)
    internal = np.setdiff1d(owned, dolfinx.mesh.exterior_facet_indices(mesh.topology))
    gamma = F.VolumeSubdomain(id=3, material=F.Material(D_0=1, E_D=0), dim=1)
    model = F.HydrogenTransportProblemDiscontinuous(mesh=F.Mesh(mesh))
    model.facet_meshtags = dolfinx.mesh.meshtags(mesh, 1, internal, 3)
    model._manifold_is_interior = {}
    with pytest.raises(ValueError, match=r"GhostMode\.shared_facet"):
        model.manifold_is_interior(gamma)

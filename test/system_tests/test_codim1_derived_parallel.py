"""Manifold reductions and single-writer output under MPI."""

import csv

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import festim as F

from .test_codim1_exports import exterior_model
from .test_codim1_interface import build
from .test_codim1_polycrystal import strips


def test_manifold_extrema_and_custom_quantities():
    model, (left, right, gamma), (cl, cr, cg) = build(n=8)
    end = F.SurfaceSubdomain(id=20, dim=0, locator=lambda x: np.isclose(x[1], 1))
    quantities, expected = [], []
    for cls in (F.MinimumSurface, F.MaximumSurface):
        for sp, surface, value in (
            (cl, gamma, 14 / 9),
            (cr, gamma, 4 / 9),
            (cg, end, 8 / 9),
        ):
            quantities.append(cls(field=sp, surface=surface))
            expected.append(value)
    for cls in (F.MinimumVolume, F.MaximumVolume):
        quantities.append(cls(field=cg, volume=gamma))
        expected.append(8 / 9)
    quantities.extend(
        [
            F.CustomQuantity(
                expr=lambda **kw: (
                    kw["H_g"] + kw["x"][1] + kw["t"] + kw["T"] / 500 + kw["D"] / 0.7
                ),
                subdomain=gamma,
            ),
            F.CustomQuantity(expr=lambda **kw: kw["H_g"], subdomain=end, volume=gamma),
            F.CustomQuantity(
                expr=lambda **kw: -kw["D"] * ufl.dot(ufl.grad(kw["H_l"]), kw["n"]),
                subdomain=gamma,
                volume=left,
            ),
            F.CustomQuantity(
                expr=lambda **kw: kw["H_r"], subdomain=gamma, volume=right
            ),
        ]
    )
    expected.extend([8 / 9 + 2.5, 8 / 9, 4 / 3, 4 / 9])
    model.exports = quantities
    model.initialise()
    model.run()
    values = [q.value for q in quantities]
    assert np.allclose(values, expected, atol=1e-8)
    assert np.allclose(model.mesh.mesh.comm.allgather(values), expected, atol=1e-8)


@pytest.mark.parametrize("self_comm", [False, True])
def test_derived_csv_written_once_on_model_communicator(tmp_path, self_comm):
    comm = MPI.COMM_SELF if self_comm else MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, 4, 4)
    model, (_, _, gamma), (_, _, cg) = build(mesh=mesh)
    # Every world rank is rank zero of its COMM_SELF simulation and must write.
    suffix = str(MPI.COMM_WORLD.rank) if self_comm else "shared"
    quantity = F.TotalVolume(
        field=cg, volume=gamma, filename=str(tmp_path / f"q_{suffix}.csv")
    )
    custom = F.CustomQuantity(expr=lambda **kw: kw["H_g"] + kw["t"], subdomain=gamma)
    model.exports = [quantity, custom]
    model.settings.transient = True
    model.settings.final_time = 0.06
    model.settings.stepsize = F.Stepsize(initial_value=0.02)
    model.initialise()
    model.run()
    comm.barrier()
    with open(quantity.filename, newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["t(s)", quantity.title]
    assert len(rows) == len(quantity.t) + 1
    assert len(quantity.t) >= 3
    assert np.allclose(
        np.asarray(rows[1:], dtype=float), np.column_stack((quantity.t, quantity.data))
    )
    assert np.allclose(custom.data, np.asarray(quantity.data) + quantity.t)
    assert np.allclose(comm.allgather(quantity.data), quantity.data)


def test_custom_quantity_rejects_unrelated_volume():
    model, (left, right, _), (cl, _, _) = build(n=4)
    model.exports = [
        F.CustomQuantity(expr=lambda **kw: kw[cl.name], subdomain=left, volume=right)
    ]
    with pytest.raises(ValueError, match="adjacent"):
        model.initialise()


def test_custom_quantities_on_network_sides():
    model, grains, network, species, _ = strips(n=12)
    pairs = []
    for grain, sp in zip(grains, species):
        custom = F.CustomQuantity(
            expr=lambda sp=sp, **kw: kw[sp.name], subdomain=network, volume=grain
        )
        standard = F.TotalSurface(field=sp, surface=network)
        pairs.append((custom, standard))
    model.exports = [quantity for pair in pairs for quantity in pair]
    model.initialise()
    model.run()
    for custom, standard in pairs:
        assert np.isclose(custom.value, standard.value, atol=1e-10)


def test_continuous_csv_uses_simulation_communicator(tmp_path):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_SELF, 4, 4)
    volume = F.VolumeSubdomain(id=1, material=F.Material(D_0=1, E_D=0))
    wall = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[0], 0))
    species = F.Species("H")
    quantity = F.TotalVolume(
        field=species,
        volume=volume,
        filename=str(tmp_path / f"continuous_{MPI.COMM_WORLD.rank}.csv"),
    )
    model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        subdomains=[volume, wall],
        species=[species],
        temperature=300,
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=wall, species=species, value=2)
        ],
        settings=F.Settings(transient=False, atol=1e-10, rtol=1e-10),
        exports=[quantity],
    )
    model.initialise()
    model.run()
    with open(quantity.filename, newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == len(quantity.t) + 1
    assert np.isclose(float(rows[-1][1]), 2)


def test_nonuniform_extrema_on_manifold_and_its_boundary():
    model, (_, gamma, _), (bulk_species, manifold_species) = exterior_model(n=64)
    end = F.SurfaceSubdomain(id=20, dim=0, locator=lambda x: np.isclose(x[1], 1))
    model.exports = [
        F.MinimumSurface(field=bulk_species, surface=gamma),
        F.MaximumSurface(field=bulk_species, surface=gamma),
        F.MinimumVolume(field=manifold_species, volume=gamma),
        F.MaximumVolume(field=manifold_species, volume=gamma),
        F.MinimumSurface(field=manifold_species, surface=end),
        F.MaximumSurface(field=manifold_species, surface=end),
        F.CustomQuantity(expr=lambda **kw: kw["H_gam"], subdomain=end),
    ]
    model.initialise()
    model.run()
    assert np.allclose(
        [q.value for q in model.exports],
        [0, 2, 0.75, 1.25, 0.75, 0.75, 0.75],
        atol=1e-3,
    )


def test_derived_quantities_with_empty_ranks():
    def rank_zero_partitioner(comm, nparts, tdim, cells):
        num_cells = len(cells[0]) // 3 if isinstance(cells, list) else cells.num_nodes
        return dolfinx.cpp.graph.AdjacencyList_int32(
            np.zeros((num_cells, 1), dtype=np.int32)
        )

    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 4, 4, partitioner=rank_zero_partitioner
    )
    model, (_, _, gamma), (cl, _, cg) = build(mesh=mesh)
    end = F.SurfaceSubdomain(id=20, dim=0, locator=lambda x: np.isclose(x[1], 1))
    model.exports = [
        F.TotalVolume(field=cg, volume=gamma),
        F.AverageVolume(field=cg, volume=gamma),
        F.MinimumVolume(field=cg, volume=gamma),
        F.MaximumVolume(field=cg, volume=gamma),
        F.MinimumSurface(field=cg, surface=end),
        F.MaximumSurface(field=cg, surface=end),
        F.CustomQuantity(expr=lambda **kw: kw["H_g"], subdomain=gamma),
        F.SurfaceFlux(field=cl, surface=gamma),
    ]
    model.initialise()
    model.run()
    assert np.allclose(
        [q.value for q in model.exports], [8 / 9] * 7 + [4 / 3], atol=1e-9
    )


def test_partition_point_is_not_a_manifold_dirichlet_boundary():
    model, _, (_, species) = exterior_model(n=8)
    inside = F.SurfaceSubdomain(id=20, dim=0, locator=lambda x: np.isclose(x[1], 0.5))
    model.subdomains.append(inside)
    model.boundary_conditions.append(
        F.FixedConcentrationBC(subdomain=inside, species=species, value=1)
    )
    with pytest.raises(ValueError, match="matched no boundary entity"):
        model.initialise()

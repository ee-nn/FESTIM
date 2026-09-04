r"""Identify an anisotropic diffusivity for the grain-boundary network.

Two steady cell problems are solved on the resolved microstructure, one per
direction of an imposed macroscopic gradient ``G``. Each gives an average flux,
and the two together give every entry of the ``2 x 2`` tensor:

    q_bar = - D_eff . grad_c_bar

with, over an averaging window ``W``,

    q_bar     = (1/|W|) [ sum_g int_(Omega_g /\ W) -D_g grad(c_g) dx
                          + int_(Gamma /\ W) -delta D_gb grad_s(c_gb) ds ]
    grad_c_bar = (1/|W|) sum_g int_(Omega_g /\ W) grad(c_g) dx

The second flux term is the whole point: it is what is carried *along* the
boundaries, and it is what makes ``D_eff`` differ from the lattice value at all.

Boundary conditions, and why there are two estimates
----------------------------------------------------
Without ``dolfinx_mpc`` there are no periodic constraints available, so the cell
problems use the uniform-gradient (Taylor) condition ``c = G.x`` on the entire
outer boundary -- on every grain that touches it and on the network at its
mouths. That is a *constraint* on the fluctuation, so it stiffens the cell: it
clamps the short circuits to the macroscopic field exactly where they leave, and
the resulting ``D_eff`` is an upper estimate that only relaxes to the true one as
the cell grows.

The cheap fix is to average over an interior window instead, where the clamping
has been forgotten. Then ``grad_c_bar`` is no longer ``G``, so both averages are
collected into matrices and

    D_eff = - Q H^-1,   Q = [q^(x) q^(y)],  H = [grad_c_bar^(x) grad_c_bar^(y)]

The gap between the two estimates, and their drift with cell size, is the honest
error bar on the identification. Both are printed.

When a single ``D_eff`` exists at all
-------------------------------------
Because each grain is its own subdomain, the lattice concentration may jump from
grain to grain, and a homogeneous single-field model can only be equivalent to
the microstructure if those jumps are small -- if the boundaries are transparent
enough that ``c_gb = c_grain`` locally. ``equilibrium error`` in the report
measures exactly that, and ``--k-sweep`` walks the exchange rate from the regime
where it holds to the regime where it does not.
"""

import argparse
import json
from dataclasses import asdict, dataclass

import dolfinx
import micromodel as mm
import numpy as np
from microstructure import Microstructure, network_tensor

__all__ = ["Identification", "hart_bound", "identify", "make_microstructure"]


def make_microstructure(size, grain_size, aspect=1.0, seed=0, cells_per_grain=10):
    """A microstructure whose Voronoi seeds are spaced by ``grain_size``."""
    n_seeds = max(2, round((size / grain_size) ** 2))
    return Microstructure.create(
        size=size,
        n_seeds=n_seeds,
        aspect=aspect,
        seed=seed,
        cells_per_grain=cells_per_grain,
    )


def hart_bound(model: mm.MicroModel):
    """``<D_lattice> + (delta D_gb / A) sum_i l_i t_i (x) t_i``.

    The parallel (Hart / Voigt) estimate: every boundary conducts along its own
    tangent as if the macroscopic gradient reached it undisturbed, and the grains
    conduct in parallel with it. It ignores tortuosity, connectivity and the
    transfer resistance at the boundaries, so it is an upper bound; how much of it
    the identification reaches says how much of the network's raw conductance the
    microstructure actually delivers.
    """
    micro, physics = model.micro, model.physics
    tensor = network_tensor(micro.segments)
    return mm.mean_lattice_tensor(model) + physics.delta * physics.D_gb / micro.area * (
        tensor
    )


@dataclass
class Identification:
    """The outcome of one identification."""

    D_cell: list  # 2x2, averaged over the whole cell (upper estimate)
    D_window: list  # 2x2, averaged over an interior window
    D_hart: list  # 2x2, the parallel bound
    D_bulk: float
    size: float
    grain_size: float
    aspect: float
    seed: int
    n_seeds: int
    n_grains: int
    n_cells: int
    k_exchange: float
    equilibrium_error: float
    gb_facets_missed: int

    @staticmethod
    def _principal(D):
        sym = 0.5 * (np.asarray(D) + np.asarray(D).T)
        evals, evecs = np.linalg.eigh(sym)
        order = np.argsort(evals)[::-1]
        return evals[order], evecs[:, order]

    def report(self, name="D_eff"):
        lines = []
        for label, D in (("whole cell", self.D_cell), ("window", self.D_window)):
            D = np.asarray(D)
            evals, evecs = self._principal(D)
            asym = abs(D[0, 1] - D[1, 0]) / max(abs(D).max(), 1e-300)
            lines += [
                f"  {name} ({label}), m2/s",
                f"    [[{D[0, 0]:.4e}, {D[0, 1]:+.4e}],",
                f"     [{D[1, 0]:+.4e}, {D[1, 1]:.4e}]]",
                f"    principal values / D_bulk   : "
                f"{evals[0] / self.D_bulk:.3f}, {evals[1] / self.D_bulk:.3f}",
                f"    anisotropy ratio            : {evals[0] / evals[1]:.3f}",
                f"    strong axis                 : "
                f"({evecs[0, 0]:+.3f}, {evecs[1, 0]:+.3f})",
                f"    non-symmetry of the estimate: {asym:.2e}",
            ]
        hart = np.asarray(self.D_hart)
        lines += [
            f"  Hart bound / D_bulk           : "
            f"{hart[0, 0] / self.D_bulk:.3f} (xx), {hart[1, 1] / self.D_bulk:.3f} (yy)",
            f"  fraction of the bound reached : "
            f"{np.asarray(self.D_cell)[0, 0] / hart[0, 0]:.3f} (xx), "
            f"{np.asarray(self.D_cell)[1, 1] / hart[1, 1]:.3f} (yy)"
            "   [whole-cell estimate, the one the bound applies to]",
            f"  local equilibrium error       : {self.equilibrium_error:.2e}",
            f"  grain-boundary facets missed  : {self.gb_facets_missed}",
        ]
        return "\n".join(lines)


def identify(micro, physics, window_fraction=0.5, export_prefix=None, verbose=True):
    """Solve the two cell problems and assemble the effective tensor."""
    half = 0.5 * (1.0 - window_fraction) * micro.size
    window = ((half, half), (micro.size - half, micro.size - half))
    _, missed = mm.check_network_covers_grain_boundaries(micro)

    Q_cell, H_cell, Q_win, H_win = (np.zeros((2, 2)) for _ in range(4))
    eq_error = 0.0
    hart = None
    for j, G in enumerate((np.array([1.0, 0.0]), np.array([0.0, 1.0]))):
        model = mm.build(
            micro,
            physics,
            bcs=[
                (
                    "outer",
                    lambda x: np.full_like(x[0], True, dtype=bool),
                    (lambda x, G=G: G[0] * x[0] + G[1] * x[1]),
                )
            ],
        )
        model.run()

        q, g, _ = mm.averages(model)
        Q_cell[:, j], H_cell[:, j] = q, g
        q_w, g_w, _ = mm.averages(model, window=window)
        Q_win[:, j], H_win[:, j] = q_w, g_w
        eq_error = max(eq_error, mm.equilibrium_error(model))
        hart = hart_bound(model)

        if export_prefix is not None:
            _export(model, f"{export_prefix}_{'xy'[j]}")
        if verbose:
            print(f"    solved cell problem G = e_{'xy'[j]}", flush=True)

    return Identification(
        D_cell=(-Q_cell @ np.linalg.inv(H_cell)).tolist(),
        D_window=(-Q_win @ np.linalg.inv(H_win)).tolist(),
        D_hart=hart.tolist(),
        D_bulk=physics.D_bulk,
        size=micro.size,
        grain_size=np.sqrt(micro.area / micro.n_grains),
        aspect=micro.aspect,
        seed=micro.seed,
        n_seeds=micro.n_seeds,
        n_grains=micro.n_grains,
        n_cells=micro.mesh.topology.index_map(2).size_global,
        k_exchange=physics.k_exchange,
        equilibrium_error=eq_error,
        gb_facets_missed=missed,
    )


def _export(model, prefix):
    """Write the corrector fields: the grains as one discontinuous parent field
    (so the jumps show), and the network as a line dataset."""
    field, update = mm.parent_field(model, name="c_lattice")
    update()
    comm = model.micro.mesh.comm
    with dolfinx.io.VTXWriter(comm, f"{prefix}_grains.bp", [field], "BP5") as w:
        w.write(0.0)
    network = model.network_solution
    with dolfinx.io.VTXWriter(comm, f"{prefix}_network.bp", [network], "BP5") as w:
        w.write(0.0)


def _dump(path, results, sweep):
    """Rewrite the results file. Called after every solve, so a run that dies
    late still leaves everything that had already been computed."""
    with open(path, "w") as f:
        json.dump({"identifications": results, "k_sweep": sweep}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grain-size", type=float, default=0.6e-6)
    parser.add_argument(
        "--sizes",
        type=float,
        nargs="+",
        default=[3e-6, 5e-6, 8e-6],
        help="cell sides to run, in metres: the RVE convergence study",
    )
    parser.add_argument("--aspect", type=float, default=4.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--temperature", type=float, default=500.0)
    parser.add_argument("--crystal-anisotropy", type=float, default=1.0)
    parser.add_argument(
        "--k-sweep",
        type=float,
        nargs="+",
        default=None,
        help="exchange rates (m/s) to run on the first cell size, to show where a "
        "single effective diffusivity stops existing",
    )
    parser.add_argument("--cells-per-grain", type=int, default=8)
    parser.add_argument("--window-fraction", type=float, default=0.5)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--out", type=str, default="identified.json")
    args = parser.parse_args()

    results, sweep = [], []
    for size in args.sizes:
        for seed in args.seeds:
            micro = make_microstructure(
                size, args.grain_size, args.aspect, seed, args.cells_per_grain
            )
            grain_size = np.sqrt(micro.area / micro.n_grains)
            physics = mm.Physics(
                T=args.temperature, crystal_anisotropy=args.crystal_anisotropy
            )
            print(physics.report(grain_size=grain_size))
            print(micro.report())
            ident = identify(
                micro,
                physics,
                window_fraction=args.window_fraction,
                export_prefix="corrector" if args.export else None,
            )
            print(ident.report())
            print(flush=True)
            results.append(asdict(ident))
            _dump(args.out, results, sweep)

    if args.k_sweep:
        micro = make_microstructure(
            args.sizes[0],
            args.grain_size,
            args.aspect,
            args.seeds[0],
            args.cells_per_grain,
        )
        grain_size = np.sqrt(micro.area / micro.n_grains)
        print(f"exchange-rate sweep on the {1e6 * args.sizes[0]:.1f} um cell")
        for k in args.k_sweep:
            physics = mm.Physics(
                T=args.temperature,
                crystal_anisotropy=args.crystal_anisotropy,
                k_exchange=k,
            )
            ident = identify(
                micro, physics, window_fraction=args.window_fraction, verbose=False
            )
            D = np.asarray(ident.D_window)
            # past a few percent the grains no longer sit at the boundary value, the
            # macroscopic field is not one field any more, and the number in the
            # D columns is an artefact: the flux still runs along the network while
            # <grad c> inside the isolated grains collapses, so the ratio diverges.
            # It is reported because the divergence is the diagnosis.
            verdict = "ok" if ident.equilibrium_error < 0.05 else "NO SINGLE D_eff"
            print(
                f"  k = {k:9.3e} m/s  R_int/R_grain = "
                f"{physics.interface_resistance_ratio(grain_size):9.3e}  "
                f"Dxx/D_b = {D[0, 0] / ident.D_bulk:8.3f}  "
                f"Dyy/D_b = {D[1, 1] / ident.D_bulk:8.3f}  "
                f"eq.err = {ident.equilibrium_error:.2e}  {verdict}",
                flush=True,
            )
            sweep.append(asdict(ident))
            _dump(args.out, results, sweep)

    print(f"written to {args.out}")


if __name__ == "__main__":
    main()

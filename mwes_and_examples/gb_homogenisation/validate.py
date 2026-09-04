r"""Does the identified tensor actually reproduce the microstructure?

The identification fits ``D_eff`` to two steady cell problems driven by a uniform
gradient. Reporting it and stopping would be circular, so this script asks the
tensor to predict things it was not fitted to.

Test A -- a different boundary condition
    Permeation across the cell: concentration fixed on two opposite faces and
    *no flux* on the other two, which is a different constraint on the
    fluctuation than the uniform-gradient condition the tensor was fitted under.
    The resolved model gives both the mean flux and the mean gradient, and the
    constitutive law it is supposed to obey is

        q_bar = -D_eff . grad_c_bar

    so the residual of that identity is a direct measure of the fit, with no
    homogeneous solve involved.

Test B -- a transient, against an actual homogeneous model
    Uptake from one face into an initially empty cell, resolved microstructure
    against a plain rectangle carrying the tensor. This is the end-to-end
    question: same geometry, same boundary condition, one field instead of a few
    hundred, and the inventory histories should lie on top of each other. It
    tests something the steady identification never sees -- the storage. The
    network holds ``delta * |Gamma|`` worth of volume, so the effective capacity
    is ``1 + delta |Gamma| / A``, and the report prints it; it is a fraction of a
    percent for a 1 nm boundary and micron grains, which is why the homogeneous
    model can be a plain diffusion equation.

Run ``python validate.py`` after ``homogenise.py``, or on its own -- it does the
identification itself.
"""

import argparse
import json
from dataclasses import asdict

from mpi4py import MPI

import dolfinx
import micromodel as mm
import numpy as np
import ufl
from homogenise import identify, make_microstructure

import festim as F

__all__ = ["homogeneous_model", "permeation_bcs", "steady_consistency", "uptake"]

ATOL = mm.ATOL  # see the note there: an unscaled problem stalls at a loose atol


def permeation_bcs(size, direction, c_in=1.0, c_out=0.0):
    """Concentration fixed on the two faces normal to ``direction``.

    The other two faces carry nothing, which in a finite element formulation is
    the natural condition -- zero flux.
    """
    axis = "xy".index(direction)
    return [
        ("inlet", lambda x, a=axis: np.isclose(x[a], 0.0), c_in),
        ("outlet", lambda x, a=axis: np.isclose(x[a], size), c_out),
    ]


def steady_consistency(micro, physics, candidates, verbose=True):
    """Test A: how well ``q_bar = -D_eff grad_c_bar`` holds under permeation.

    ``candidates`` is a ``{label: tensor}`` mapping, scored side by side. Which of
    the two estimators predicts better is a result, not something to assume: the
    whole-cell one is biased upward by the Taylor condition, the window one is
    unbiased but noisier on a small cell.
    """
    rows = {}
    for direction in ("x", "y"):
        model = mm.build(
            micro, physics, bcs=permeation_bcs(micro.size, direction)
        ).run()
        q, grad_c, _ = mm.averages(model)
        # the driven component is the one the test is about; the transverse one is
        # near zero on both sides and its relative error is meaningless
        i = "xy".index(direction)
        if verbose:
            print(f"  driven along {direction}: q_{direction} = {q[i]:+.5e}")
        for label, D in candidates.items():
            predicted = -np.asarray(D) @ grad_c
            error = (predicted[i] - q[i]) / max(abs(q[i]), 1e-300)
            rows[(direction, label)] = error
            if verbose:
                print(
                    f"    {label:<11s} predicts {predicted[i]:+.5e}"
                    f"   ({100 * error:+6.2f} %)",
                    flush=True,
                )
    return rows


def homogeneous_model(
    size,
    D_eff,
    physics,
    bcs,
    n=48,
    transient=False,
    final_time=None,
    stepsize=None,
    atol=ATOL,
):
    """A plain rectangle carrying the anisotropic tensor.

    ``festim.Material`` takes ``D`` as a ``fem.Function``, so the constant tensor
    is stored in a DG0 tensor-valued field -- which is also how a spatially
    varying effective tensor (a graded microstructure) would be supplied.
    """
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([size, size])], [n, n]
    )
    V = dolfinx.fem.functionspace(mesh, ("DG", 0, (2, 2)))
    D = dolfinx.fem.Function(V, name="D_eff")
    for component in range(4):
        D.x.array[component::4] = D_eff.reshape(-1)[component]

    volume = F.VolumeSubdomain(
        id=1,
        material=F.Material(D=D),
        locator=lambda x: np.full_like(x[0], True, dtype=bool),
    )
    c = F.Species("c", subdomains=[volume])
    subdomains = [volume]
    boundary_conditions = []
    for i, (name, locator, value) in enumerate(bcs):
        surface = F.SurfaceSubdomain(id=10 + i, locator=locator)
        subdomains.append(surface)
        boundary_conditions.append(
            F.FixedConcentrationBC(subdomain=surface, value=value, species=c)
        )
    model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        species=[c],
        subdomains=subdomains,
        boundary_conditions=boundary_conditions,
        temperature=physics.T,
        settings=F.Settings(
            atol=atol,
            rtol=1e-10,
            transient=transient,
            final_time=final_time,
            stepsize=stepsize,
        ),
    )
    model.show_progress_bar = False
    return model, c, volume, mesh


def uptake(micro, physics, D_eff, n_steps=60, verbose=True):
    """Test B: inventory during uptake from the top face, resolved vs homogeneous.

    The end time is set from the tensor itself -- about the time the slow axis
    needs to cross the cell -- so that the comparison covers the whole transient
    rather than its first moments.
    """
    size = micro.size
    slow = min(np.linalg.eigvalsh(0.5 * (D_eff + D_eff.T)))
    final_time = 0.35 * size**2 / slow
    dt = final_time / n_steps
    bcs = [("top", lambda x: np.isclose(x[1], size), 1.0)]

    resolved = mm.build(
        micro,
        physics,
        bcs=bcs,
        transient=True,
        final_time=final_time,
        stepsize=F.Stepsize(initial_value=dt),
    )
    resolved.model.show_progress_bar = False  # the stepping is driven here
    resolved.model.initialise()

    homogeneous, c, _, mesh_h = homogeneous_model(
        size,
        D_eff,
        physics,
        bcs,
        transient=True,
        final_time=final_time,
        stepsize=F.Stepsize(initial_value=dt),
    )
    homogeneous.show_progress_bar = False
    homogeneous.initialise()
    dx_h = ufl.Measure("dx", domain=mesh_h)

    def homogeneous_inventory():
        form = dolfinx.fem.form(c.post_processing_solution * dx_h)
        return mesh_h.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)

    times, resolved_inventory, model_inventory = [0.0], [0.0], [0.0]
    while resolved.model.t.value < final_time - 0.5 * dt:
        resolved.model.iterate()
        homogeneous.iterate()
        times.append(float(resolved.model.t))
        resolved_inventory.append(mm.inventory(resolved))
        model_inventory.append(homogeneous_inventory())
        if verbose and len(times) % 10 == 0:
            print(
                f"    t = {times[-1]:.3e} s  resolved {resolved_inventory[-1]:.4e}"
                f"  homogeneous {model_inventory[-1]:.4e}",
                flush=True,
            )

    return (
        np.array(times),
        np.array(resolved_inventory),
        np.array(model_inventory),
        final_time,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=float, default=3e-6)
    parser.add_argument("--grain-size", type=float, default=0.6e-6)
    parser.add_argument("--aspect", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=500.0)
    parser.add_argument("--cells-per-grain", type=int, default=8)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--skip-transient", action="store_true")
    parser.add_argument("--out", type=str, default="validation.json")
    args = parser.parse_args()
    record = {}

    def dump():
        with open(args.out, "w") as f:
            json.dump(record, f, indent=2)

    micro = make_microstructure(
        args.size, args.grain_size, args.aspect, args.seed, args.cells_per_grain
    )
    physics = mm.Physics(T=args.temperature)
    print(micro.report())
    print()

    ident = identify(micro, physics)
    print(ident.report())
    D_eff = np.asarray(ident.D_window)
    capacity = 1.0 + physics.delta * micro.ridge_length / micro.area
    print(f"  effective capacity 1 + delta|Gamma|/A : {capacity:.5f}")
    print()

    record["identification"] = asdict(ident)
    record["capacity"] = capacity
    dump()

    print("test A -- permeation, a boundary condition the tensor was not fitted to")
    errors = steady_consistency(
        micro,
        physics,
        {
            "whole cell": np.asarray(ident.D_cell),
            "window": np.asarray(ident.D_window),
            "Hart bound": np.asarray(ident.D_hart),
            "D_bulk only": ident.D_bulk * np.eye(2),
        },
    )
    record["permeation"] = {f"{d}|{label}": e for (d, label), e in errors.items()}
    dump()
    print()

    if args.skip_transient:
        dump()
        return
    print("test B -- uptake transient, resolved microstructure vs homogeneous model")
    times, resolved, homogeneous, final_time = uptake(
        micro, physics, D_eff, n_steps=args.steps
    )
    record["uptake"] = {
        "times": times.tolist(),
        "resolved": resolved.tolist(),
        "homogeneous": homogeneous.tolist(),
        "final_time": final_time,
    }
    dump()
    saturation = resolved[-1]
    deviation = np.abs(resolved - homogeneous) / max(saturation, 1e-300)
    print(f"  final time                     : {final_time:.3e} s")
    print(
        f"  inventory at the end           : "
        f"resolved {resolved[-1]:.4e}, homogeneous {homogeneous[-1]:.4e}"
    )
    print(
        f"  max deviation over the history : {100 * deviation.max():.2f} % of the "
        f"final inventory"
    )
    print(f"  deviation at the end           : {100 * deviation[-1]:.2f} %")

    half = 0.5 * saturation
    t_resolved = np.interp(half, resolved, times)
    t_homogeneous = np.interp(half, homogeneous, times)
    print(
        f"  time to half saturation        : resolved {t_resolved:.3e} s, "
        f"homogeneous {t_homogeneous:.3e} s "
        f"({100 * abs(t_homogeneous - t_resolved) / t_resolved:.1f} % apart)"
    )
    record["half_saturation"] = {
        "resolved": float(t_resolved),
        "homogeneous": float(t_homogeneous),
    }
    dump()
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()

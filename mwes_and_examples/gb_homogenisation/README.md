# Grain-boundary short circuits, and the anisotropic diffusivity they add up to

A workflow in three steps:

1. **Build a polycrystal** whose grain boundaries the mesh conforms to
   (`microstructure.py`).
2. **Solve the resolved problem** — one FESTIM subdomain per grain, coupled
   through the boundary network declared as a single codim-1 subdomain
   (`micromodel.py`).
3. **Identify an anisotropic `D_eff`** for a homogeneous model, and then check
   that it predicts things it was not fitted to (`homogenise.py`, `validate.py`).

Voronoi cells stand in for the Neper microstructure. Nothing above the mesh
generator depends on that choice: `Microstructure` only has to supply a list of
boundary segments, per-grain cell tags and a per-grain orientation, so a Neper
`.tess` or an EBSD map drops in at the same place.

![The microstructure and the flux its boundaries carry](fig_microstructure.png)

*Left: the polycrystal. Middle and right: the flux running along each boundary,
on a shared scale, expressed as the width of lattice that would carry the same
flux. Driven along the grain elongation the network lights up -- single
boundaries move more hydrogen than a 0.46 um grain of lattice beside them --
and driven across it, far fewer boundaries are usefully oriented and the same
scale stays dim. That difference is the anisotropy, before any number is put
on it.*

## What the resolved model is

Every Voronoi cell is its own `festim.VolumeSubdomain` with its own
`festim.Species`, so the lattice concentration may **jump** from grain to grain.
That is what lets a boundary resist as well as conduct: the only route between
two grains is through the network, and the exchange coefficient `k` is a genuine
transfer resistance in series with the lattice. With a single bulk subdomain the
lattice field is continuous across every boundary, so a boundary can only ever be
a short circuit in parallel with it.

The network stays **one** codim-1 subdomain carrying one species. That is what
makes triple junctions work with no junction condition to write: the submesh
built from every grain-boundary facet is connected, so a single continuous field
lives on it. FESTIM gives each adjacent grain an interior-facet integral of its
own on which that grain is the `"+"` side, so a facet shared by two grains is
integrated once per grain — exactly the two-sided feeding the physics asks for.
This needs a FESTIM in which a manifold may be adjacent to more than two volume
subdomains (`main`, after the codim-1 work).

Where `delta` goes, since it is never meshed: the boundary is a slab of width
`delta` and tangential diffusivity `D_gb`, and per unit boundary area

```
delta dc_gb/dt = delta D_gb lap_s(c_gb) + sum over sides s of k_s (c_s - c_gb)
```

FESTIM assembles the manifold equation with no thickness factor, so it is divided
by `delta`: the manifold material carries `D_gb` itself and each grain
contributes a source `(k/delta) (c_grain - c_gb)` — one term per grain, with no
factor of two, because summing the two sides supplies it. `delta` comes back in
every post-processed quantity: the network holds `delta * integral(c_gb)` and
carries `delta * D_gb * grad_s(c_gb)`.

## The identification

Two steady cell problems, one per direction of an imposed macroscopic gradient,
give the columns of

```
q_bar = -D_eff . grad_c_bar
```

where `q_bar` sums every grain's lattice flux **and** the network's own
`delta D_gb grad_s(c_gb)`. Dropping that second term is the commonest way to get
this wrong: it is precisely the short circuit being measured.

Two estimates are reported because there are no periodic constraints available
without `dolfinx_mpc`. The cell problems use the uniform-gradient condition
`c = G.x` on the whole outer boundary, which clamps the short circuits where they
leave the cell and biases `D_eff` upward; averaging over an interior window
instead relaxes that. The gap between the two, and its drift with cell size, is
the error bar. The analytic Hart bound `<D_lattice> + (delta D_gb / A) sum_i l_i t_i (x) t_i`
is printed alongside as the no-tortuosity ceiling.

**A single `D_eff` only exists when the boundaries are transparent.** If `k` is
small the grains decouple and no one-field model can reproduce the
microstructure. `equilibrium error` measures the departure from `c_gb = c_grain`,
and `--k-sweep` walks across the transition.

## Running it

```bash
python microstructure.py                      # geometry only, no solve
python homogenise.py --sizes 2e-6 3e-6 4e-6 --out rve.json   # + RVE convergence
python homogenise.py --k-sweep 1e-6 1e-4 1e-2 3 --out rve.json  # the transition
python validate.py --out validation.json      # predict, do not just fit
python plots.py --rve rve.json --validation validation.json  # the figures above
```

Useful flags: `--aspect` (grain elongation — an equiaxed tessellation homogenises
to a nearly isotropic tensor, so the anisotropy needs elongated grains or
texture), `--crystal-anisotropy` (anisotropic *lattice* diffusion per grain; it
must be 1 for a cubic crystal, where Neumann's principle forbids a grain
orientation from affecting a second-rank property), `--export` (ParaView files:
the grains as one discontinuous field, so the jumps show, plus the network as a
line dataset).

## What comes out

Tungsten-like defaults at 500 K (`D_gb/D_b = 790`, `delta = 1 nm`, grains ~0.45 um
elongated 4:1, `k = 3 m/s`), one realisation per size:

![Diffusivity by direction](fig_anisotropy.png)

`D_eff ~ 4.0 D_b` along the grain elongation and `~1.3 D_b` across it, an
anisotropy of about 3, with the strong axis on the elongation to better than a
degree, and the whole lobe sitting just inside the no-tortuosity ceiling. The
tensor comes out symmetric to 1e-8 without symmetry being imposed anywhere -- a
good check that the network flux term is assembled correctly -- and no
grain-boundary facet is missed by the locator, so no grain is accidentally
sealed off.

How far to trust that depends on how many grains the cell holds:

![Estimator convergence with cell size](fig_rve.png)

| cell | grains | `Dxx/D_b` cell / window | `Dyy/D_b` cell / window |
|------|--------|-------------------------|-------------------------|
| 2 um | 24     | 4.205 / 3.390           | 1.364 / 1.116           |
| 3 um | 43     | 4.122 / 4.057           | 1.339 / 1.486           |
| 4 um | 64     | 4.031 / 4.056           | 1.325 / 1.274           |

On the strong axis the two estimators close from 24 % apart at 2 um to 0.6 % at
4 um. On the weak axis the window estimate is still wandering at 64 grains, so it
has not reached an RVE there -- which is exactly why both estimates are reported
rather than one.

`validate.py` on the 3 um cell, predicting what was never fitted:

![Validation against a transient and a different boundary condition](fig_validation.png)

The tensor carries transport that the lattice value misses by a factor four, and
it predicts a different boundary-value problem to a few percent. The window
estimate wins on the strong axis (+0.9 %) and loses on the weak one (+14.4 %) at
this cell size -- it is the unbiased but noisier of the two, and 43 grains is not
many. The uptake transient tracks to 5.3 % of the final inventory, with the
homogeneous model slightly ahead throughout, as an upper estimate should be.

The exchange-rate sweep is where the per-grain formulation earns itself:

![Effective diffusivity against the exchange rate](fig_sweep.png)

Above `k ~ 1e-2 m/s` the boundaries are transparent and `D_eff` sits on a plateau
-- and there it reproduces the single-bulk-subdomain model to four figures, as it
must. As `k` falls past the point where one boundary crossing costs as much as
crossing a grain, the grains decouple: the apparent tensor runs away and even
inverts its anisotropy, because the flux still runs along the network while
`<grad c>` inside the now-isolated grains collapses. That divergence is the
diagnosis, not a number to use -- the material has become dual-porosity and wants
a two-field model, which is exactly what the resolved problem already is.

## Numerics worth knowing

Two traps, both from the problem being unscaled (`D ~ 1e-11 m2/s`, cell area
`~1e-11 m2`), both fixed here and both worth remembering:

* **A silent solver stall.** The Newton residual of a *converged* transient step
  is itself of order 1e-12, so at FESTIM's default `atol = 1e-12` the solver
  declares convergence at zero iterations as soon as the uptake slows, and the
  solution freezes while `t` keeps advancing. It looks exactly like a steady
  state, at the wrong value -- here it saturated at 43 % of the right inventory.
  `ATOL = 1e-25` in `micromodel.py`; nondimensionalising is the better fix.
* **A tessellation that did not tessellate.** `snap_segments` rounds ridge
  endpoints onto a grid to merge near-degenerate junctions, and `size / tol` is
  not a whole number, so endpoints that the clip had put *exactly* on the edge of
  the cell were nudged off it. Nudged inward, they left a two-nanometre gap;
  OpenCASCADE will not split a face across a gap, so the ridge ended up embedded
  inside a face instead of dividing it, and every grain it should have separated
  merged into one piece holding a quarter of the cell. Nothing else noticed: the
  ridge was still meshed, still in the network, the total curve length still
  matched exactly, and the check that every grain-grain facet is in the network
  still passed -- because those facets now had the same tag on both sides. Only
  the *size* of the piece gave it away, which is why `Microstructure.report` now
  prints the largest piece as a multiple of an average grain and flags anything
  above three. (In the transparent-boundary regime it changed the answer by
  nothing at all, because the lattice field is continuous across a boundary
  whether or not the two sides are separate subdomains. It would matter for
  per-grain orientations, per-grain traps, or any run at low exchange rate.)
* **`petsc_options` cannot set a MUMPS ICNTL.** FESTIM deletes its PETSc options
  from the database as soon as the solver is built, and `mat_mumps_*` is not read
  until `PCSetUp` at the first solve. One subdomain per grain makes a wide, badly
  scaled block system that MUMPS routinely under-estimates the fill-in of, and it
  stops with `INFOG(1) = -9` -- at some exchange rates and not others, on the same
  mesh. `tune_direct_solver()` writes the option back after `initialise()`.

## Caveats

- Two-dimensional. In 3D the boundaries are surfaces, the area fraction scales as
  `delta/d` the same way, but the connectivity of the network is quite different.
- The Taylor boundary condition biases the estimate upward; the window estimate
  and the RVE convergence are how far that is trusted, not a proof.
- One realisation per cell size is scatter, not a trend — pass several `--seeds`
  before believing a number.
- Traps are not in the model. They do not change the steady identification at
  all, only the transient capacity.

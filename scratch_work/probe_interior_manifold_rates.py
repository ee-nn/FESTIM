"""Probe 3: interior manifold (dS + restriction) with x- and T-dependent rates.

Analytical solution from the repo's own test: with K_LEFT=2, K_RIGHT=3, a manifold at
x=0.5, c=2 at x=0 and c=0 at x=1, steady state gives c_L=14/9, c_G=8/9, c_R=4/9.

Here the rates are written as functions of x (and of T, which itself varies in x) that
evaluate to exactly 2 and 3 on the plane x=0.5, so the analytical solution must be
recovered if and only if x/T are resolved at the right physical location.
"""

import sys

from mpi4py import MPI

import dolfinx
import numpy as np

import festim as F

VARIANT = sys.argv[1]
D_BULK, D_GAMMA = 1.5, 0.7
PLANE = 0.5

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 20, 20)

left = F.VolumeSubdomain(
    id=1,
    material=F.Material(D_0=D_BULK, E_D=0.0),
    locator=lambda x: x[0] <= PLANE + 1e-14,
)
right = F.VolumeSubdomain(
    id=2,
    material=F.Material(D_0=D_BULK, E_D=0.0),
    locator=lambda x: x[0] >= PLANE - 1e-14,
)
gamma = F.VolumeSubdomain(
    id=3,
    material=F.Material(D_0=D_GAMMA, E_D=0.0),
    dim=1,
    locator=lambda x: np.isclose(x[0], PLANE),
)
outer_l = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[0], 0.0))
outer_r = F.SurfaceSubdomain(id=5, locator=lambda x: np.isclose(x[0], 1.0))

H_l = F.Species("H_l", subdomains=[left])
H_r = F.Species("H_r", subdomains=[right])
H_g = F.Species("H_g", subdomains=[gamma])

temperature = 500

if VARIANT == "const":
    kL = lambda: 2.0
    kR = lambda: 3.0
    src_l = lambda c_g, c_b: 2.0 * (c_b - c_g)
    src_r = lambda c_g, c_b: 3.0 * (c_b - c_g)
    flx_l = lambda c_g, c_b: 2.0 * (c_g - c_b)
    flx_r = lambda c_g, c_b: 3.0 * (c_g - c_b)
elif VARIANT == "x_dep":
    # 4*x = 2 and 6*x = 3 on the plane x = 0.5
    src_l = lambda x, c_g, c_b: 4.0 * x[0] * (c_b - c_g)
    src_r = lambda x, c_g, c_b: 6.0 * x[0] * (c_b - c_g)
    flx_l = lambda x, c_g, c_b: 4.0 * x[0] * (c_g - c_b)
    flx_r = lambda x, c_g, c_b: 6.0 * x[0] * (c_g - c_b)
elif VARIANT == "T_dep":
    # T = 500 + 100 x, so T = 550 on the plane; k = k0 * 550 / T there
    temperature = lambda x: 500.0 + 100.0 * x[0]
    src_l = lambda T, c_g, c_b: (2.0 * 550.0 / T) * (c_b - c_g)
    src_r = lambda T, c_g, c_b: (3.0 * 550.0 / T) * (c_b - c_g)
    flx_l = lambda T, c_g, c_b: (2.0 * 550.0 / T) * (c_g - c_b)
    flx_r = lambda T, c_g, c_b: (3.0 * 550.0 / T) * (c_g - c_b)
elif VARIANT == "xT_dep":
    temperature = lambda x: 500.0 + 100.0 * x[0]
    src_l = lambda x, T, c_g, c_b: (4.0 * x[0] * 550.0 / T) * (c_b - c_g)
    src_r = lambda x, T, c_g, c_b: (6.0 * x[0] * 550.0 / T) * (c_b - c_g)
    flx_l = lambda x, T, c_g, c_b: (4.0 * x[0] * 550.0 / T) * (c_g - c_b)
    flx_r = lambda x, T, c_g, c_b: (6.0 * x[0] * 550.0 / T) * (c_g - c_b)
elif VARIANT == "x_src_only":
    src_l = lambda x, c_g, c_b: 4.0 * x[0] * (c_b - c_g)
    src_r = lambda x, c_g, c_b: 6.0 * x[0] * (c_b - c_g)
    flx_l = lambda c_g, c_b: 2.0 * (c_g - c_b)
    flx_r = lambda c_g, c_b: 3.0 * (c_g - c_b)
elif VARIANT == "x_flx_only":
    src_l = lambda c_g, c_b: 2.0 * (c_b - c_g)
    src_r = lambda c_g, c_b: 3.0 * (c_b - c_g)
    flx_l = lambda x, c_g, c_b: 4.0 * x[0] * (c_g - c_b)
    flx_r = lambda x, c_g, c_b: 6.0 * x[0] * (c_g - c_b)
elif VARIANT == "k_parent_fn":
    # the proposed workaround: fold the spatial dependence into a parent-mesh Function
    from dolfinx import fem as _fem

    _V = _fem.functionspace(mesh, ("CG", 1))
    kL = _fem.Function(_V)
    kR = _fem.Function(_V)
    kL.interpolate(lambda x: 4.0 * x[0])
    kR.interpolate(lambda x: 6.0 * x[0])
    src_l = lambda c_g, c_b: kL * (c_b - c_g)
    src_r = lambda c_g, c_b: kR * (c_b - c_g)
    flx_l = lambda c_g, c_b: kL * (c_g - c_b)
    flx_r = lambda c_g, c_b: kR * (c_g - c_b)
else:
    raise SystemExit(f"unknown variant {VARIANT}")

sources = [
    F.ParticleSource(
        value=src_l,
        species=H_g,
        volume=gamma,
        species_dependent_value={"c_b": H_l, "c_g": H_g},
    ),
    F.ParticleSource(
        value=src_r,
        species=H_g,
        volume=gamma,
        species_dependent_value={"c_b": H_r, "c_g": H_g},
    ),
]
bcs = [
    F.ParticleFluxBC(
        subdomain=gamma,
        species=H_l,
        value=flx_l,
        species_dependent_value={"c_b": H_l, "c_g": H_g},
    ),
    F.ParticleFluxBC(
        subdomain=gamma,
        species=H_r,
        value=flx_r,
        species_dependent_value={"c_b": H_r, "c_g": H_g},
    ),
    F.FixedConcentrationBC(subdomain=outer_l, value=2.0, species=H_l),
    F.FixedConcentrationBC(subdomain=outer_r, value=0.0, species=H_r),
]

model = F.HydrogenTransportProblemDiscontinuous(
    mesh=F.Mesh(mesh),
    species=[H_l, H_r, H_g],
    subdomains=[left, right, gamma, outer_l, outer_r],
    sources=sources,
    boundary_conditions=bcs,
    temperature=temperature,
    exports=[],
)
model.settings = F.Settings(atol=1e-12, rtol=1e-12, transient=False)
model.initialise()
model.run()

c_l = H_l.subdomain_to_post_processing_solution[left].x.array
c_r = H_r.subdomain_to_post_processing_solution[right].x.array
c_g = H_g.subdomain_to_post_processing_solution[gamma].x.array

print(
    f"{VARIANT}: c_L_min={c_l.min():.8f} (14/9={14 / 9:.8f})  "
    f"c_G={c_g.mean():.8f} (8/9={8 / 9:.8f})  "
    f"c_R_max={c_r.max():.8f} (4/9={4 / 9:.8f})  "
    f"OK={np.isclose(c_g.mean(), 8 / 9, atol=1e-8) and np.isclose(c_l.min(), 14 / 9, atol=1e-8)}"
)
print(f"   c_g spread: {c_g.min():.8f} .. {c_g.max():.8f}")
print(f"   J_left = {3 * (2 - c_l.min()):.8f}   J_right = {3 * c_r.max():.8f}")

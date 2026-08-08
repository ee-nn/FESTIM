"""
Analytical solution from the repo's own test: with K_LEFT=2, K_RIGHT=3, a manifold at
x=0.5, c=2 at x=0 and c=0 at x=1, steady state gives c_L=14/9, c_G=8/9, c_R=4/9.

Here the rates are written as functions of x that evaluate to exactly 2 and 3 on the plane
x=0.5, so the analytical solution must be recovered iff x is resolved at the right location.
"""

from mpi4py import MPI

import dolfinx
import numpy as np

import festim as F

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

# 4*x = 2 and 6*x = 3 on the plane x = 0.5
src_l = lambda x, c_g, c_b: 4.0 * x[0] * (c_b - c_g)
src_r = lambda x, c_g, c_b: 6.0 * x[0] * (c_b - c_g)
flx_l = lambda x, c_g, c_b: 4.0 * x[0] * (c_g - c_b)
flx_r = lambda x, c_g, c_b: 6.0 * x[0] * (c_g - c_b)

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
    temperature=500,
    exports=[],
)
model.settings = F.Settings(atol=1e-12, rtol=1e-12, transient=False)
model.initialise()
model.run()

c_l = H_l.subdomain_to_post_processing_solution[left].x.array
c_r = H_r.subdomain_to_post_processing_solution[right].x.array
c_g = H_g.subdomain_to_post_processing_solution[gamma].x.array

print(
    f"c_left: {c_l.min():.8f} (14/9={14 / 9:.8f})  "
    f"c_g: {c_g.mean():.8f} (8/9={8 / 9:.8f})  "
    f"c_right: {c_r.max():.8f} (4/9={4 / 9:.8f})  "
)
print(f"   J_left = {3 * (2 - c_l.min()):.8f}   J_right = {3 * c_r.max():.8f}")

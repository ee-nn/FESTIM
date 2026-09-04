"""Periodic Voronoi microstructures and the meshes that conform to them.

The grain boundaries are *lines*, not thin bands: nothing here resolves the
physical boundary width ``delta``. That width only ever appears as a
coefficient, which is what lets a 1 nm boundary sit inside a 20 um cell. What
the mesh must do is put facets exactly on the ridges, so that a codim-1 subdomain
can be built from them.

Grains can be elongated (``aspect``): the seeds are Voronoi-tessellated in a
coordinate stretched along x, so the grains come out with an aspect ratio while
staying a proper periodic tessellation. An equiaxed tessellation homogenises to a
nearly isotropic tensor, so the elongation is what makes the anisotropy
identification worth doing.
"""

from collections import Counter
from dataclasses import dataclass

from mpi4py import MPI

import gmsh
import numpy as np
from dolfinx.io.gmsh import model_to_mesh
from scipy.spatial import Voronoi

__all__ = [
    "Microstructure",
    "build_mesh",
    "near_segments",
    "network_tensor",
    "voronoi_segments",
]


def clip_to_box(p, q, box):
    """Liang-Barsky clip of the segment ``pq`` to ``[0, box[0]] x [0, box[1]]``."""
    d = q - p
    t0, t1 = 0.0, 1.0
    for num, den in (
        (-d[0], p[0]),
        (d[0], box[0] - p[0]),
        (-d[1], p[1]),
        (d[1], box[1] - p[1]),
    ):
        if abs(num) < 1e-15:
            if den < 0:
                return None
            continue
        r = den / num
        if num < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
    return None if t0 > t1 else (p + t0 * d, p + t1 * d)


def voronoi_segments(n_seeds, size, rng, aspect=1.0):
    """Ridges of a periodic Voronoi tessellation of the box ``[0, size]^2``.

    The seeds are tiled over the 3x3 neighbourhood before tessellating, so the
    ridges that reach the box edge are the ones a periodic tessellation would
    have, and the microstructure tiles.

    Args:
        n_seeds: number of grains in the box.
        size: box side (m).
        rng: a ``numpy`` generator, so the microstructure is reproducible.
        aspect: grain elongation along x. The tessellation is built in the
            coordinate ``(x / aspect, y)`` and mapped back, which stretches every
            grain by ``aspect`` along x without breaking periodicity.

    Returns:
        list of ``(p, q)`` endpoint pairs, in metres.
    """
    box = np.array([size / aspect, size])
    pts = np.column_stack(
        [rng.uniform(0, box[0], n_seeds), rng.uniform(0, box[1], n_seeds)]
    )
    tiled = np.vstack(
        [pts + np.array([dx, dy]) * box for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    )
    vor = Voronoi(tiled)
    stretch = np.array([aspect, 1.0])
    segments = []
    for a, b in vor.ridge_vertices:
        if a < 0 or b < 0:
            continue
        clipped = clip_to_box(vor.vertices[a], vor.vertices[b], box)
        if clipped is None:
            continue
        p, q = (clipped[0] * stretch, clipped[1] * stretch)
        if np.linalg.norm(q - p) > 1e-12 * size:
            segments.append((p, q))
    return segments


def snap_segments(segments, tol):
    """Snap ridge endpoints to a grid of size ``tol`` and drop what collapses.

    A Voronoi tessellation produces the occasional ridge far shorter than any
    mesh size -- a quadruple junction that random seeds resolved into two triple
    junctions a nanometre apart. Meshing those is hopeless and dropping them
    outright would disconnect the network at that junction, which is exactly the
    thing the codim-1 formulation exists to get right. Snapping merges the two
    junctions into one instead, so connectivity survives.
    """
    snapped = []
    for p, q in segments:
        a = np.round(p / tol) * tol
        b = np.round(q / tol) * tol
        if np.linalg.norm(b - a) > 0.5 * tol:
            snapped.append((a, b))
    return snapped


def near_segments(points, segments, tol):
    """Vectorised test: is each point within ``tol`` of any segment?

    ``points`` is the ``(3, n)`` array DOLFINx hands to a locator.
    """
    px, py = points[0], points[1]
    hit = np.zeros(px.shape, dtype=bool)
    for p, q in segments:
        d = q - p
        t = np.clip(((px - p[0]) * d[0] + (py - p[1]) * d[1]) / (d @ d), 0.0, 1.0)
        dx, dy = px - (p[0] + t * d[0]), py - (p[1] + t * d[1])
        hit |= dx * dx + dy * dy < tol * tol
    return hit


def network_tensor(segments):
    """The second-moment tensor of the network, ``sum_i length_i * t_i (x) t_i``.

    This is the only geometric information a first-order (no-tortuosity) estimate
    of the effective diffusivity needs: a boundary conducts along its own tangent
    and not at all across it, so each segment contributes a rank-1 tensor. Its
    trace is the total ridge length, and its anisotropy is the anisotropy the
    homogenised tensor would have if the network were perfectly connected.

    Returns:
        ``(2, 2)`` array with units of length (m).
    """
    tensor = np.zeros((2, 2))
    for p, q in segments:
        d = q - p
        length = np.linalg.norm(d)
        if length == 0.0:
            continue
        t = d / length
        tensor += length * np.outer(t, t)
    return tensor


def connected_components(segments, size):
    """Number of connected components of the network, by union-find over endpoints."""
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    scale = 1e-9 * size
    for p, q in segments:
        a = tuple(np.round(p / scale).astype(np.int64))
        b = tuple(np.round(q / scale).astype(np.int64))
        parent[find(a)] = find(b)
    return len({find(k) for k in parent})


def triple_junctions(segments, size):
    """Interior points where three or more ridges meet."""
    scale = 1e-9 * size
    ends = Counter(
        tuple(np.round(p / scale).astype(np.int64)) for seg in segments for p in seg
    )
    tol = 1e-9 * size
    return [
        np.array(k, dtype=float) * scale
        for k, n in ends.items()
        if n >= 3
        and tol < k[0] * scale < size - tol
        and tol < k[1] * scale < size - tol
    ]


def build_mesh(segments, size, h_gb, h_bulk, comm=MPI.COMM_WORLD):
    """A triangular mesh of ``[0, size]^2`` whose facets lie on every ridge.

    Returns ``(mesh, cell_tags, n_grains)``, the tags marking each Voronoi cell
    with its own id, numbered from 1.

    ``occ.fragment`` is what guarantees that: it splits the rectangle along every
    segment, so the segments become model edges the mesher is obliged to follow.
    The mesh is then graded from ``h_gb`` on the ridges to ``h_bulk`` in the grain
    interiors -- the concentration field varies fastest next to a short circuit.

    The model is built in units of ``size`` and the mesh scaled back afterwards.
    OpenCASCADE compares points against an absolute tolerance of about 1e-7, so a
    geometry stated in metres and tens of microns across is entirely below its
    resolution -- every point would be a duplicate of every other.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("polycrystal")
    occ = gmsh.model.occ

    rect = occ.addRectangle(0, 0, 0, 1.0, 1.0)
    points = {}

    def point(xy):
        # one gmsh point per ridge endpoint, so that segments meeting at a
        # junction share it and fragment has nothing to reconcile
        key = tuple(np.round(xy / size, 9))
        if key not in points:
            points[key] = occ.addPoint(key[0], key[1], 0)
        return points[key]

    lines = [occ.addLine(point(p), point(q)) for p, q in segments]
    _, out_map = occ.fragment([(2, rect)], [(1, ln) for ln in lines])
    occ.synchronize()

    # one physical group per Voronoi cell, so that every grain becomes a volume
    # subdomain of its own. ``fragment`` splits the rectangle along every ridge
    # either way; tagging the pieces separately is the whole difference between a
    # polycrystal whose lattice field is continuous and one whose grains can each
    # hold their own concentration, orientation and traps.
    grain_surfaces = [tag for (dim, tag) in out_map[0] if dim == 2]
    for i, tag in enumerate(grain_surfaces):
        gmsh.model.addPhysicalGroup(2, [tag], i + 1)

    gb_curves = [tag for entry in out_map[1:] for (dim, tag) in entry if dim == 1]

    field = gmsh.model.mesh.field
    field.add("Distance", 1)
    field.setNumbers(1, "CurvesList", gb_curves)
    field.add("Threshold", 2)
    field.setNumber(2, "InField", 1)
    field.setNumber(2, "SizeMin", h_gb / size)
    field.setNumber(2, "SizeMax", h_bulk / size)
    field.setNumber(2, "DistMin", 2 * h_gb / size)
    field.setNumber(2, "DistMax", 12 * h_gb / size)
    field.setAsBackgroundMesh(2)
    for opt in (
        "Mesh.MeshSizeFromPoints",
        "Mesh.MeshSizeFromCurvature",
        "Mesh.MeshSizeExtendFromBoundary",
    ):
        gmsh.option.setNumber(opt, 0)

    gmsh.model.mesh.generate(2)
    result = model_to_mesh(gmsh.model, comm, 0, gdim=2)
    gmsh.finalize()
    mesh = result.mesh if hasattr(result, "mesh") else result[0]
    cell_tags = result.cell_tags if hasattr(result, "cell_tags") else result[1]
    mesh.geometry.x[:, :2] *= size
    return mesh, cell_tags, len(grain_surfaces)


@dataclass
class Microstructure:
    """A Voronoi polycrystal, the mesh that conforms to it, and the grain tags.

    ``n_grains`` is generally larger than ``n_seeds``: the ridges of the
    periodically tiled seed set cut the box into more pieces than there are seeds
    inside it, and each piece is a grain in its own right.
    """

    size: float
    n_seeds: int
    aspect: float
    seed: int
    segments: list
    mesh: object
    cell_tags: object
    n_grains: int
    orientations: np.ndarray  # one angle per grain, radians
    h_gb: float

    @classmethod
    def create(
        cls,
        size,
        n_seeds,
        aspect=1.0,
        seed=0,
        cells_per_grain=14,
        bulk_coarsening=8.0,
        comm=MPI.COMM_WORLD,
    ):
        """Generate the tessellation and mesh it.

        Args:
            size: box side (m).
            n_seeds: number of Voronoi seeds.
            aspect: grain elongation along x.
            seed: rng seed.
            cells_per_grain: mesh cells across the *short* axis of a grain, which
                sets ``h_gb``.
            bulk_coarsening: ``h_bulk / h_gb``.
        """
        rng = np.random.default_rng(seed)
        segments = voronoi_segments(n_seeds, size, rng, aspect)
        # the short axis of a grain: equal-area grains of aspect ratio `aspect`
        short_axis = np.sqrt(size**2 / n_seeds / aspect)
        h_gb = short_axis / cells_per_grain
        segments = snap_segments(segments, 0.1 * h_gb)
        mesh, cell_tags, n_grains = build_mesh(
            segments, size, h_gb, bulk_coarsening * h_gb, comm=comm
        )
        # an untextured polycrystal: orientations uniform on [0, pi). Replace this
        # with the measured Euler angles to drive the model from EBSD or Neper.
        orientations = rng.uniform(0.0, np.pi, n_grains)
        return cls(
            size,
            n_seeds,
            aspect,
            seed,
            segments,
            mesh,
            cell_tags,
            n_grains,
            orientations,
            h_gb,
        )

    @property
    def grain_ids(self):
        return np.arange(1, self.n_grains + 1)

    def misorientation(self, grain_a, grain_b):
        """Misorientation angle between two grains, folded into ``[0, pi/2]``.

        The hook for misorientation-dependent boundary properties: a boundary's
        diffusivity or its exchange rate with the grains can be made a function of
        this instead of a constant.
        """
        delta = abs(self.orientations[grain_a - 1] - self.orientations[grain_b - 1])
        delta %= np.pi
        return min(delta, np.pi - delta)

    @property
    def area(self):
        return self.size**2

    @property
    def ridge_length(self):
        return float(np.trace(network_tensor(self.segments)))

    @property
    def tolerance(self):
        """Distance below which a point counts as lying on the network."""
        return 0.05 * self.h_gb

    def locator(self, points):
        return near_segments(points, self.segments, self.tolerance)

    def report(self):
        tensor = network_tensor(self.segments)
        evals, evecs = np.linalg.eigh(tensor)
        lines = [
            f"microstructure: {self.n_seeds} grains in a {1e6 * self.size:.1f} um box,"
            f" aspect {self.aspect:g}, seed {self.seed}",
            f"  ridge segments                 : {len(self.segments)}",
            f"  triple junctions (interior)    : "
            f"{len(triple_junctions(self.segments, self.size))}",
            f"  connected components           : "
            f"{connected_components(self.segments, self.size)}",
            f"  ridge length / area            : "
            f"{self.ridge_length / self.area:.4g} 1/m",
            f"  network tensor eigenvalues     : "
            f"{evals[1] / self.ridge_length:.3f}, {evals[0] / self.ridge_length:.3f}"
            f" (of the trace)",
            f"  strong direction               : "
            f"({evecs[0, 1]:+.3f}, {evecs[1, 1]:+.3f})",
            f"  grains (tagged pieces)         : {self.n_grains}",
            f"  mesh                           : "
            f"{self.mesh.topology.index_map(2).size_global} cells,"
            f" h_gb = {1e9 * self.h_gb:.0f} nm",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    for aspect in (1.0, 4.0):
        micro = Microstructure.create(
            size=20e-6, n_seeds=48, aspect=aspect, seed=3, cells_per_grain=10
        )
        print(micro.report())
        print()

"""Figures for the README.

Reads the JSON that ``homogenise.py`` and ``validate.py`` write, and solves one
extra pair of cell problems for the field maps. Run it after those two::

    python homogenise.py --sizes 2e-6 3e-6 4e-6 --k-sweep 1e-6 1e-2 --out rve.json
    python validate.py --out validation.json
    python plots.py --rve rve.json --validation validation.json

Colour is assigned by the job it does, not by taste: one hue ramp light-to-dark
for the flux maps (a magnitude) and a fixed categorical order for series
identity. The
categorical slots used here were checked with the palette validator -- worst
adjacent colour-vision separation dE 9.1, worst normal-vision dE 22.9, both above
their floors. Two of the four sit below 3:1 contrast on this surface, so every
series is directly labelled rather than identified by colour alone.
"""

import argparse
import json

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

# --- design tokens -------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEQUENTIAL = [
    "#cde2fb",
    "#9ec5f4",
    "#5598e7",
    "#2a78d6",
    "#256abf",
    "#184f95",
    "#0d366b",
]
BLUES = LinearSegmentedColormap.from_list("blues", SEQUENTIAL)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK_2,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
    }
)


def tidy(ax, grid_axis="y"):
    """Recessive chrome: no top/right spines, a hairline grid on one axis only."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


def label_end(ax, x, y, text, color, dx=6, dy=0, ha="left", va="center"):
    """Direct label at the end of a series, so identity never rests on colour."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        color=color,
        fontsize=8.5,
        fontweight="bold",
        ha=ha,
        va=va,
    )


def place_labels(ax, x, items, dx=8, min_gap_px=13):
    """Direct-label several series at the same x without letting them collide.

    Labels are nudged apart in display space and the block re-centred, so their
    order still matches the order of the curves.
    """
    fig = ax.figure
    fig.canvas.draw()
    items = sorted(items, key=lambda t: t[0])
    original = [ax.transData.transform((x, y))[1] for y, _, _ in items]
    adjusted = list(original)
    for i in range(1, len(adjusted)):
        adjusted[i] = max(adjusted[i], adjusted[i - 1] + min_gap_px)
    shift = (sum(original) - sum(adjusted)) / len(adjusted)
    for (y, text, color), y0, y1 in zip(items, original, adjusted, strict=True):
        label_end(ax, x, y, text, color, dx=dx, dy=(y1 + shift - y0) * 72.0 / fig.dpi)


def title(ax, headline, sub=None):
    ax.set_title(headline, loc="left", color=INK, pad=22 if sub else 6)
    if sub:
        ax.annotate(
            sub,
            xy=(0, 1),
            xytext=(0, 6),
            xycoords="axes fraction",
            textcoords="offset points",
            color=INK_2,
            fontsize=8.5,
            va="bottom",
        )


# --- figure 1: the microstructure and the corrector fields ---------------
def draw_network(ax, segments, color=INK, lw=0.9, alpha=1.0):
    for p, q in segments:
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, alpha=alpha, zorder=3)


def network_flux_segments(model, scale):
    """Each boundary segment, and the tangential flux it carries.

    The flux is reported as the *width of lattice carrying the same flux*,
    ``delta D_gb |ds c_gb| / (D_bulk |G|)``. That turns a quantity with awkward
    units into a length that can be held against the grain size: a boundary whose
    number exceeds the grain width is moving more hydrogen than a whole grain of
    lattice beside it.
    """
    c = model.network_solution
    V = c.function_space
    coords = V.tabulate_dof_coordinates()
    cells = V.dofmap.list
    p0, p1 = coords[cells[:, 0], :2], coords[cells[:, 1], :2]
    length = np.linalg.norm(p1 - p0, axis=1)
    keep = length > 0
    slope = np.abs(c.x.array[cells[:, 1]] - c.x.array[cells[:, 0]])[keep] / length[keep]
    physics = model.physics
    width = physics.delta * physics.D_gb * slope / physics.D_bulk
    return np.stack([p0[keep] * scale, p1[keep] * scale], axis=1), width * scale


def colourbar(fig, mappable, ax):
    bar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7.5, color=AXIS)
    return bar


def figure_microstructure(micro, physics, path, size_um):
    """The polycrystal, and the flux its boundaries carry under each gradient.

    The two flux panels share one colour scale, which is the point: driven along
    the grain elongation the network lights up, driven across it far fewer
    boundaries are usefully oriented and the same colour scale stays dim. That
    difference *is* the anisotropy the identification puts a number on.

    There is deliberately no map of the bulk fluctuation here. It is very nearly
    zero: the network is a set of measure-zero lines, so it carries its extra flux
    without needing to disturb the lattice field much, and a map of it is a blank
    sheet of paper.
    """
    import micromodel as mm

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3))
    scale = 1e6  # metres -> microns
    segments_um = [(p * scale, q * scale) for p, q in micro.segments]
    grain_um = scale * np.sqrt(micro.area / micro.n_grains)

    ax = axes[0]
    ax.add_patch(
        plt.Rectangle(
            (0, 0), size_um, size_um, facecolor="#f0efec", edgecolor="none", zorder=1
        )
    )
    draw_network(ax, segments_um, lw=1.0)
    title(
        ax,
        "The microstructure",
        f"{micro.n_grains} grains, elongated {micro.aspect:g}:1 along x",
    )

    cases = []
    for G, name in (
        (np.array([1.0, 0.0]), "along x"),
        (np.array([0.0, 1.0]), "across x"),
    ):
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
        cases.append((name, *network_flux_segments(model, scale)))

    ceiling = max(width.max() for _, _, width in cases)
    for ax, (name, lines, width) in zip(axes[1:], cases, strict=True):
        order = np.argsort(width)
        collection = LineCollection(
            lines[order],
            array=width[order],
            cmap=BLUES,
            linewidths=0.8 + 3.2 * width[order] / ceiling,
            zorder=3,
        )
        collection.set_clim(0.0, ceiling)
        ax.add_collection(collection)
        colourbar(fig, collection, ax)
        title(
            ax,
            f"Flux along the boundaries, driven {name}",
            "width of lattice carrying the same flux (um)",
        )
        ax.annotate(
            f"one grain is {grain_um:.2f} um across",
            xy=(0.5, 0.015),
            xycoords="axes fraction",
            ha="center",
            color=INK_2,
            fontsize=8,
        )

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(0, size_um)
        ax.set_ylim(0, size_um)
        ax.set_xlabel("x (um)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("y (um)")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- figure 2: the identified anisotropy ---------------------------------
def directional(D, theta):
    """``n . D . n``: the diffusivity felt along the direction ``theta``."""
    n = np.stack([np.cos(theta), np.sin(theta)])
    return np.einsum("in,ij,jn->n", n, D, n)


def figure_anisotropy(ident, path):
    theta = np.linspace(0, 2 * np.pi, 721)
    D_b = ident["D_bulk"]
    curves = [
        (np.asarray(ident["D_hart"]), SERIES[2], "Hart bound"),
        (np.asarray(ident["D_window"]), SERIES[0], "identified"),
    ]

    fig, ax = plt.subplots(figsize=(5.4, 5.2), subplot_kw={"projection": "polar"})
    ax.plot(
        theta,
        np.ones_like(theta),
        color=MUTED,
        lw=1.6,
        ls=(0, (4, 3)),
        zorder=2,
    )
    label_end(ax, np.deg2rad(70), 1.0, "lattice alone", MUTED, dx=4, dy=8)

    for D, color, name in curves:
        radius = directional(D, theta) / D_b
        ax.plot(theta, radius, color=color, lw=2.0, zorder=3)
        # labelled off the shoulder of each lobe rather than at 0 deg, where the
        # two curves are a hair apart and the angular tick already sits
        angle = np.deg2rad(20 if name == "identified" else -20)
        label_end(
            ax,
            angle,
            directional(D, np.array([angle]))[0] / D_b,
            name,
            color,
            dx=10,
            dy=6 if name == "identified" else -6,
        )

    ax.set_theta_zero_location("E")
    ax.set_rlabel_position(103)
    ax.set_rticks([1, 2, 3, 4])
    ax.set_yticklabels(["1", "2", "3", "4"], fontsize=8)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"])
    ax.grid(color=GRID, lw=0.8)
    ax.spines["polar"].set_color(AXIS)
    fig.text(
        0.01,
        0.985,
        "Diffusivity by direction, in units of the lattice value",
        color=INK,
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.01,
        0.935,
        f"n . D_eff . n / D_bulk, {1e6 * ident['size']:.0f} um cell, "
        f"grains elongated {ident['aspect']:g}:1 along x",
        color=INK_2,
        fontsize=8.5,
        va="top",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- figure 3: does the cell contain enough grains? ----------------------
def figure_rve(identifications, path):
    sizes = np.array([1e6 * i["size"] for i in identifications])
    D_b = identifications[0]["D_bulk"]
    order = np.argsort(sizes)
    sizes = sizes[order]
    grains = [identifications[k]["n_grains"] for k in order]

    def component(key, i, j):
        return np.array(
            [np.asarray(identifications[k][key])[i, j] / D_b for k in order]
        )

    # the two components differ by a factor three, so a shared scale would squash
    # the weak axis into the bottom tenth of the frame and hide the convergence
    # that is the whole point of the figure. Each panel gets its own.
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    for ax, (i, j), name in zip(
        axes, ((0, 0), (1, 1)), ("along x", "across x"), strict=True
    ):
        labels = []
        for key, color, label in (
            ("D_hart", SERIES[2], "Hart bound"),
            ("D_cell", SERIES[1], "whole cell"),
            ("D_window", SERIES[0], "window"),
        ):
            y = component(key, i, j)
            ax.plot(sizes, y, color=color, marker="o", markersize=7, zorder=3)
            labels.append((y[-1], label, color))
        tidy(ax)
        ax.set_xlabel("cell side (um)")
        ax.set_xticks(sizes)
        ax.set_xticklabels(
            [f"{s:g}\n{n} grains" for s, n in zip(sizes, grains, strict=True)]
        )
        ax.set_xlim(sizes[0] - 0.3, sizes[-1] + 1.9)
        ax.set_ylabel("D_eff / D_bulk")
        title(ax, f"D_eff {name}", None)
        place_labels(ax, sizes[-1], labels)
    fig.suptitle(
        "How far the estimate can be trusted, against cell size",
        x=0.008,
        ha="left",
        color=INK,
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.008,
        0.905,
        "the two estimators close on the strong axis; on the weak one the window "
        "estimate is still scattering, so 45 grains is not yet an RVE for it",
        color=INK_2,
        fontsize=8.5,
        va="top",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- figure 4: where a single effective diffusivity stops existing -------
def figure_sweep(sweep, path, threshold=0.05):
    k = np.array([s["k_exchange"] for s in sweep])
    order = np.argsort(k)
    k = k[order]
    D_b = sweep[0]["D_bulk"]
    dxx = np.array([np.asarray(sweep[i]["D_window"])[0, 0] / D_b for i in order])
    dyy = np.array([np.asarray(sweep[i]["D_window"])[1, 1] / D_b for i in order])
    err = np.array([sweep[i]["equilibrium_error"] for i in order])

    broken = err > threshold
    edge = None
    if broken.any() and (~broken).any():
        # the shading ends midway (in log) between the last failing run and the
        # first passing one -- the transition was not resolved more finely
        edge = np.sqrt(k[broken].max() * k[~broken].min())

    fig, axes = plt.subplots(
        2, 1, figsize=(7.6, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.35, 1]}
    )
    for ax in axes:
        if edge is not None:
            ax.axvspan(k.min() / 2, edge, color="#f0efec", zorder=0)

    ax = axes[0]
    series = ((dxx, SERIES[0], "D_eff along x"), (dyy, SERIES[1], "D_eff across x"))
    for y, color, label in series:
        ax.plot(k, y, color=color, marker="o", markersize=7, zorder=3)
        label_end(ax, k[-1], y[-1], label, color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("D_eff / D_bulk")
    tidy(ax)
    title(
        ax,
        "The plateau is the answer; the runaway is the diagnosis",
        "identified tensor against the grain-to-boundary exchange rate",
    )
    if edge is not None:
        ax.annotate(
            "grains decoupled:\nno single D_eff exists",
            xy=(k.min() * 1.6, max(dxx.max(), dyy.max())),
            color=INK_2,
            fontsize=8.5,
            va="top",
        )

    ax = axes[1]
    ax.plot(k, err, color=SERIES[3], marker="o", markersize=7, zorder=3)
    label_end(ax, k[-1], err[-1], "equilibrium\nerror", SERIES[3])
    ax.axhline(threshold, color=MUTED, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.annotate(
        f"{threshold:g} -- above this the grains no longer sit at the boundary value",
        xy=(k.max(), threshold * 1.35),
        color=MUTED,
        fontsize=8,
        va="bottom",
        ha="right",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("exchange rate k (m/s)")
    ax.set_ylabel("max |c_grain - c_gb| / max c")
    tidy(ax)
    ax.set_xlim(k.min() / 2, k.max() * 6)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- figure 5: does the tensor predict what it was not fitted to? --------
def figure_validation(validation, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    uptake = validation["uptake"]
    t = np.array(uptake["times"])
    # normalised by the inventory of a fully charged cell, so the axis is a
    # fraction rather than an integral in units nobody carries in their head
    full = validation["identification"]["size"] ** 2
    labels = []
    for key, color, label in (
        ("resolved", SERIES[0], "resolved microstructure"),
        ("homogeneous", SERIES[1], "homogeneous D_eff"),
    ):
        y = np.array(uptake[key]) / full
        ax.plot(t, y, color=color, zorder=3)
        labels.append((y[-1], label, color))
    tidy(ax)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("inventory, as a fraction of a fully charged cell")
    ax.set_xlim(0, t[-1] * 1.3)
    ax.set_ylim(0, None)
    place_labels(ax, t[-1], labels)
    title(
        ax,
        "Uptake from one face",
        "a transient the steady identification never saw",
    )

    ax = axes[1]
    labels = ["whole cell", "window", "Hart bound", "D_bulk only"]
    directions = ["x", "y"]
    height = 0.36
    positions = np.arange(len(labels))
    for d, (direction, hatch) in enumerate(zip(directions, ("", "///"), strict=True)):
        values = [
            100 * validation["permeation"][f"{direction}|{label}"] for label in labels
        ]
        offset = (d - 0.5) * (height + 0.03)
        bars = ax.barh(
            positions + offset,
            values,
            height=height,
            color=SERIES[0],
            hatch=hatch,
            edgecolor=SURFACE,
            linewidth=2.0,
            zorder=3,
        )
        for bar, value in zip(bars, values, strict=True):
            ax.annotate(
                f"{value:+.1f}%",
                xy=(value, bar.get_y() + bar.get_height() / 2),
                xytext=(5 if value >= 0 else -5, 0),
                textcoords="offset points",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=8,
                color=INK_2,
            )
    ax.axvline(0, color=AXIS, lw=1.2, zorder=2)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", labelcolor=INK_2)
    ax.set_xlabel("error in the predicted permeation flux (%)")
    ax.set_xlim(-88, 30)
    tidy(ax, grid_axis="x")
    ax.annotate(
        "solid: driven along x      hatched: driven across x",
        xy=(0, 1),
        xytext=(0, 6),
        xycoords="axes fraction",
        textcoords="offset points",
        color=INK_2,
        fontsize=8.5,
        va="bottom",
    )
    ax.set_title("Permeation, with no-flux sides", loc="left", color=INK, pad=22)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rve", default="rve.json")
    parser.add_argument("--validation", default="validation.json")
    parser.add_argument("--field-size", type=float, default=3e-6)
    parser.add_argument("--grain-size", type=float, default=0.6e-6)
    parser.add_argument("--aspect", type=float, default=4.0)
    parser.add_argument("--cells-per-grain", type=int, default=8)
    parser.add_argument("--skip-fields", action="store_true")
    args = parser.parse_args()

    with open(args.rve) as f:
        rve = json.load(f)
    written = []

    if rve["identifications"]:
        biggest = max(rve["identifications"], key=lambda i: i["size"])
        written.append(figure_anisotropy(biggest, "fig_anisotropy.png"))
        if len(rve["identifications"]) > 1:
            written.append(figure_rve(rve["identifications"], "fig_rve.png"))
    if rve.get("k_sweep"):
        written.append(figure_sweep(rve["k_sweep"], "fig_sweep.png"))

    try:
        with open(args.validation) as f:
            validation = json.load(f)
    except FileNotFoundError:
        validation = None
    if validation and "uptake" in validation and "permeation" in validation:
        written.append(figure_validation(validation, "fig_validation.png"))

    if not args.skip_fields:
        import micromodel as mm
        from homogenise import make_microstructure

        micro = make_microstructure(
            args.field_size, args.grain_size, args.aspect, 0, args.cells_per_grain
        )
        written.append(
            figure_microstructure(
                micro, mm.Physics(), "fig_microstructure.png", 1e6 * args.field_size
            )
        )

    for path in written:
        print(f"  -> {path}")


if __name__ == "__main__":
    main()

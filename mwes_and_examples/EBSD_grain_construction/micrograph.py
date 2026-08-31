"""Micrograph housekeeping shared by the EBSD pipeline scripts.

    from micrograph import annotate_png, scale_bar_ax

    annotate_png("check-ori.png", width_units=160, unit="um", trim_border=True)
    scale_bar_ax(ax, nx * voxsize, "um")

Trims the uniform border neper -V leaves around a 2D raster (its camera frames
a 3D scene, so a flat map lands in the middle of a 1200 x 900 canvas) and draws
a scale bar in the lower right. After trimming, the image width *is* the
raster width, so the bar length in pixels follows from --width, the physical
width of the content in --unit. The same bar is available for matplotlib axes
(`scale_bar_ax`) so every picture the pipeline writes carries one.

Bar length is the largest of 1, 2, 5 x 10^k not exceeding a quarter of the
image width, the usual micrograph convention.
"""

import numpy as np


def nice_length(width, fraction=0.25):
    """Largest 1/2/5 x 10^k not exceeding `fraction` of `width`."""
    target = width * fraction
    k = np.floor(np.log10(target))
    for m in (5.0, 2.0, 1.0):
        if m * 10**k <= target:
            return m * 10**k
    return 10 ** (k - 1) * 5.0


def unit_label(unit):
    return {"um": "um", "micron": "um", "microns": "um"}.get(unit, unit)


def format_length(value, unit):
    v = f"{value:g}"
    return f"{v} {unit_label(unit)}"


# --- PIL (rendered PNGs) -----------------------------------------------------
def trim(img, tol=6, margin=0):
    """Crop away a border of (near-)uniform colour equal to the corner colour."""

    arr = np.asarray(img.convert("RGB")).astype(int)
    bg = arr[0, 0]
    diff = np.abs(arr - bg).max(axis=2) > tol
    rows, cols = np.flatnonzero(diff.any(axis=1)), np.flatnonzero(diff.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return img
    y0, y1 = max(rows[0] - margin, 0), min(rows[-1] + 1 + margin, arr.shape[0])
    x0, x1 = max(cols[0] - margin, 0), min(cols[-1] + 1 + margin, arr.shape[1])
    return img.crop((x0, y0, x1, y1))


def scale_bar_image(img, width_units, unit="um", length=None, inset=0.03):
    """Draw a scale bar (white on a translucent dark box) in the lower right."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(24)
    img = img.convert("RGBA")
    w, h = img.size
    px_per_unit = w / width_units
    length = nice_length(width_units) if length is None else length
    bar_px = round(length * px_per_unit)
    thick = max(round(0.012 * h), 3)
    pad = max(round(inset * w), 6)
    label = format_length(length, unit)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    tw, th = draw.textbbox((0, 0), label, font=font)[2:]
    box_w = max(bar_px, tw) + 2 * pad
    box_h = thick + th + 3 * pad
    x1, y1 = w - pad, h - pad
    x0, y0 = x1 - box_w, y1 - box_h
    draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0, 120))
    bx = x1 - pad - bar_px
    by = y1 - pad - thick
    draw.rectangle((bx, by - thick, bx + bar_px, by), fill=(255, 255, 255, 255))
    draw.text((x1 - pad - tw, y0 + pad), label, font=font, fill=(255, 255, 255, 255))
    return Image.alpha_composite(img, overlay).convert("RGB")


# --- matplotlib --------------------------------------------------------------
def scale_bar_ax(ax, width_units, unit="um", length=None, color="white"):
    """Scale bar in the lower right of a matplotlib axes in data units."""
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    length = nice_length(width_units) if length is None else length
    bar = AnchoredSizeBar(
        ax.transData,
        length,
        format_length(length, unit),
        "lower right",
        pad=0.5,
        borderpad=0.8,
        sep=4,
        color=color,
        frameon=True,
        size_vertical=0.012 * width_units,
        fontproperties={"size": 16},
    )
    bar.patch.set_facecolor("black")
    bar.patch.set_alpha(0.45)
    bar.patch.set_edgecolor("none")
    ax.add_artist(bar)
    return bar


def annotate_png(
    path, width_units, unit="um", trim_border=False, length=None, output=None, log=print
):
    """Trim a rendered PNG and draw a scale bar on it. Returns the output path.

    `width_units` is the physical width of the *content*, so after trimming the
    image width and that number describe the same span. `output` defaults to
    overwriting `path`. Needs Pillow; the matplotlib path (`scale_bar_ax`) does
    not.
    """
    from PIL import Image

    img = Image.open(path)
    if trim_border:
        img = trim(img)
    img = scale_bar_image(img, width_units, unit, length)
    out = output or path
    img.save(out)
    if log:
        bar = format_length(length or nice_length(width_units), unit)
        log(f"  wrote {out} ({img.size[0]} x {img.size[1]} px, bar {bar})")
    return out

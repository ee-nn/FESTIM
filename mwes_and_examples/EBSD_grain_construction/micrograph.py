"""Scale bars and border trimming, shared by the pipeline's figures.

    from micrograph import annotate_png, scale_bar_ax
    annotate_png("check-ori.png", width_units=160, unit="um", trim_border=True)
    scale_bar_ax(ax, nx * voxsize, "um")

neper -V frames a flat map in the middle of a 1200 x 900 canvas, so a rendered
PNG needs its uniform border trimmed; after that the image width *is* the
raster width and the bar length in pixels follows from `width_units`. The same
bar is available for matplotlib axes, so every picture carries one. Bar length
is the largest of 1, 2, 5 x 10^k under a quarter of the width.
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


def format_length(value, unit):
    return f"{value:g} {'um' if unit in ('micron', 'microns') else unit}"


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


# Corner labels of Neper's standard stereographic triangle, as fractions of the
# key image. The Neper tutorial draws them at (180, 390), (545, 390), (505, 30)
# on the 800 x 400 render, and these are those positions divided through, so
# they survive the trim and the rescale below.
IPF_KEY_LABELS = (("[001]", 0.22, 0.96), ("[011]", 0.68, 0.96), ("[111]", 0.63, 0.08))


def annotate_png(
    path, width_units, unit="um", trim_border=False, length=None, output=None, log=print
):
    """Trim a rendered PNG and draw a scale bar on it; returns the output path.

    `width_units` is the physical width of the *content*, so after trimming it
    and the image width describe the same span. `output` defaults to
    overwriting `path`. Needs Pillow, which `scale_bar_ax` does not.
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


def append_key(png, key_png, output=None, labels=("001", "011", "111"), log=print):
    """Paste an IPF colour key to the right of a rendered map; returns the path.

    `key_png` is the standard stereographic triangle as Neper prints it (the
    ipf-key stage of ebsd_to_mesh.sh), so the key comes out of the same
    colouring code as the map rather than approximating it. It arrives with the
    uniform border every neper -V render has, and unlabelled, so it is trimmed,
    scaled to the map and its corners labelled here; the documented recipe uses
    ImageMagick for the labels, which the pipeline would not otherwise need.

    Corner positions follow the projection: [001] is at the origin, [011] at
    (sqrt(2) - 1, 0) and [111] at (sqrt(3) - 1)/2 twice over, and the 011-111
    arc only moves inwards, so after trimming the bounding box is exactly
    [001]-[011] wide and the apex sits at 0.884 of that width.
    """
    from PIL import Image, ImageDraw, ImageFont

    base = Image.open(png).convert("RGB")
    key = trim(Image.open(key_png).convert("RGB"))
    h = round(0.62 * base.height)
    w = round(key.width * h / key.height)
    key = key.resize((w, h), Image.LANCZOS)

    size = max(round(0.10 * h), 11)
    m = max(round(0.22 * h), 3 * size // 2)  # margin the corner labels live in
    font = ImageFont.load_default(size)
    panel = Image.new("RGB", (w + 2 * m, h + 2 * m), "white")
    panel.paste(key, (m, m))
    draw = ImageDraw.Draw(panel)
    apex = ((np.sqrt(3) - 1) / 2) / (np.sqrt(2) - 1)
    corners = ((m, m + h, 1), (m + w, m + h, 1), (m + round(apex * w), m, -1))
    for label, (x, y, below) in zip(labels, corners):
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        y = y + m // 4 if below > 0 else y - m // 4 - th
        draw.text((x - tw / 2, y), label, font=font, fill="black")

    gap = round(0.02 * base.width)
    out_img = Image.new(
        "RGB", (base.width + gap + panel.width, max(base.height, panel.height)), "white"
    )
    out_img.paste(base, (0, (out_img.height - base.height) // 2))
    out_img.paste(panel, (base.width + gap, (out_img.height - panel.height) // 2))
    out = output or png
    out_img.save(out)
    if log:
        w, h = out_img.size
        log(f"  wrote {out} with the IPF key ({w} x {h} px)")
    return out

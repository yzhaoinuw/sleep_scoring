"""Compose the two JOSS paper figures from the original application snapshots.

Run from any directory:

    python paper/figures/build_figures.py

The five source PNGs are never modified. All snapshots show the same
viewport (9155-10090 s at 2 px/s); the "after" snapshots sit 13 px to the
right because the window layout shifted, so crops are aligned by time.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent

VIEW_START_S = 9155.0
PX_PER_S = 2.0
PLOT_LEFT = {"before": 83, "after": 96}
ROW_SHIFT = {"before": 0, "after": 1}  # "after" snapshots sit 1 px lower
PLOT_TOP, PLOT_BOTTOM = 366, 1084  # EEG plot top, NE plot bottom ("before" layout)
ZOOM_TOP, ZOOM_BOTTOM = 330, 1135  # EEG title through the time-axis label
# Stage legend beside the EEG title; panel A keeps it, zoomed crops clip it.
LEGEND_BOX = (1232, 343, 1630, 362)

# Window shared by the zoomed panels; covers every prediction change near
# the expert label.
ZOOM_START_S, ZOOM_END_S = 9420.0, 9915.0

# Stage colors as rendered in the score overlay.
STAGES = {"Wake": (128, 128, 255), "NREM": (255, 128, 128), "REM": (128, 255, 128)}

WHITE = (255, 255, 255)
INK = (20, 20, 20)
GAP = 40


def font(size, bold=False):
    # Windows resolves "arial*.ttf" by name; macOS keeps Arial in Supplemental.
    names = (
        ["arialbd.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]
        if bold
        else ["arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def load(name):
    return Image.open(HERE / name).convert("RGB")


def x_of(t, layout):
    return round(PLOT_LEFT[layout] + (t - VIEW_START_S) * PX_PER_S)


def zoom_crop(img, layout, top, bottom):
    shift = ROW_SHIFT[layout]
    x0 = x_of(ZOOM_START_S, layout)
    crop = img.crop((x0, top + shift, x_of(ZOOM_END_S, layout), bottom + shift))
    dx = PLOT_LEFT[layout] - PLOT_LEFT["before"]
    l, t, r, b = LEGEND_BOX
    ImageDraw.Draw(crop).rectangle(
        (l + dx - x0, t - top, r + dx - x0, b - top), fill=WHITE
    )
    return crop


def stage_row(img, layout, y=885):
    """Stage per pixel column of the zoom window, read from a trace-free row."""
    a = np.asarray(img).astype(int)[y + ROW_SHIFT[layout]]
    a = a[x_of(ZOOM_START_S, layout) : x_of(ZOOM_END_S, layout)]
    colors = np.array(list(STAGES.values()))
    dist = np.abs(a[:, None, :] - colors[None, :, :]).sum(2)
    return dist.argmin(1)


def text_segments(strip, gap=14):
    """Split a one-line text strip into word groups at wide blank gaps."""
    a = np.asarray(strip.convert("L")) < 200
    inked = np.where(a.any(0))[0]
    breaks = np.where(np.diff(inked) > gap)[0]
    starts = np.r_[inked[0], inked[breaks + 1]]
    ends = np.r_[inked[breaks], inked[-1]] + 1
    return list(zip(starts, ends))


def wrap_strip(strip, width, split_after_chars=(";", ".")):
    """Re-flow a screenshot text line into several lines no wider than width.

    Lines break only at word gaps, so the application's own text is kept.
    """
    segs = text_segments(strip, gap=6)
    lines, line_start = [], segs[0][0]
    for i, (s, e) in enumerate(segs):
        if e - line_start > width:
            prev_end = segs[i - 1][1]
            lines.append(strip.crop((line_start, 0, prev_end, strip.height)))
            line_start = s
    lines.append(strip.crop((line_start, 0, segs[-1][1], strip.height)))
    out = Image.new("RGB", (width, strip.height * len(lines)), WHITE)
    for i, line in enumerate(lines):
        out.paste(line, (0, i * strip.height))
    return out


def stack(*parts, gap=0):
    w = max(p.width for p in parts)
    h = sum(p.height for p in parts) + gap * (len(parts) - 1)
    out = Image.new("RGB", (w, h), WHITE)
    y = 0
    for p in parts:
        out.paste(p, (0, y))
        y += p.height + gap
    return out


def side_by_side(*parts, gap=GAP):
    w = sum(p.width for p in parts) + gap * (len(parts) - 1)
    h = max(p.height for p in parts)
    out = Image.new("RGB", (w, h), WHITE)
    x = 0
    for p in parts:
        out.paste(p, (x, 0))
        x += p.width + gap
    return out


def with_header(panel, letter, title, extra=0):
    """Add a panel letter and short title above a panel."""
    head = 60 + extra
    out = Image.new("RGB", (panel.width, panel.height + head), WHITE)
    out.paste(panel, (0, head))
    d = ImageDraw.Draw(out)
    d.text((0, 6), letter, fill=INK, font=font(46, bold=True))
    d.text((52, 16), title, fill=INK, font=font(30))
    return out, head


def bracket(draw, x0, x1, y, label, color=INK, above=True):
    tick = 12 if above else -12
    draw.line([(x0, y + tick), (x0, y), (x1, y), (x1, y + tick)], fill=color, width=4)
    f = font(26, bold=True)
    w = draw.textlength(label, font=f)
    tx = min(max((x0 + x1) / 2 - w / 2, 0), draw.im.size[0] - w)
    draw.text((tx, y - 34), label, fill=color, font=f)


def outline_rows(draw, x0, x1, head):
    """Outline an interval across the EEG, EMG, and NE plots of a zoom panel."""
    draw.rectangle(
        (x0, head + PLOT_TOP - ZOOM_TOP, x1, head + PLOT_BOTTOM - ZOOM_TOP), outline=INK, width=3
    )


def figure1():
    before = load("app_figure.png")
    labeled = load("app_figure_correction.png")
    after = load("app_figure_after_correction.png")

    # A: interface overview without window chrome or file controls.
    overview = before.crop((15, 150, 2020, 1135))
    panel_a, _ = with_header(overview, "A", "Predicted stages with spectrogram, EEG, EMG, and NE")

    # B and C: matching time windows of the EEG, EMG, and NE rows.
    zoom_b = zoom_crop(labeled, "before", ZOOM_TOP, ZOOM_BOTTOM)
    zoom_c = zoom_crop(after, "after", ZOOM_TOP, ZOOM_BOTTOM)

    msg_b = labeled.crop((10, 1167, 912, 1192))
    msg_c = wrap_strip(after.crop((22, 1145, 1815, 1172)), zoom_c.width)
    body_b = stack(zoom_b, msg_b, gap=8)
    body_c = stack(zoom_c, msg_c, gap=8)

    panel_b, head_b = with_header(body_b, "B", "Expert labels an interval as NREM", extra=50)
    panel_c, head_c = with_header(body_c, "C", "Predictions after calibration", extra=50)

    # Callouts: the expert interval in B and C, and where C changed from B.
    label_x0 = (9751 - ZOOM_START_S) * PX_PER_S
    label_x1 = (9806 - ZOOM_START_S) * PX_PER_S
    db = ImageDraw.Draw(panel_b)
    bracket(db, label_x0, label_x1, head_b - 8, "expert NREM label")

    changed = stage_row(labeled, "before") != stage_row(after, "after")
    dc = ImageDraw.Draw(panel_c)
    bar_y = head_c - 18
    runs = np.flatnonzero(np.diff(np.r_[0, changed.astype(int), 0]))
    for s, e in zip(runs[::2], runs[1::2]):
        dc.rectangle((s, bar_y, e - 1, bar_y + 12), fill=INK)
    dc.text((0, bar_y - 34), "stage changed from (B)", fill=INK, font=font(26, bold=True))
    outline_rows(dc, label_x0, label_x1, head_c)

    bottom_row = side_by_side(panel_b, panel_c)
    fig = stack(panel_a, bottom_row, gap=30)
    return pad(fig)


def figure2():
    selected = load("app_figure_right_click_select.png")
    video = load("app_figure_video.png")

    zoom = zoom_crop(selected, "after", ZOOM_TOP, ZOOM_BOTTOM)
    msg = selected.crop((22, 1168, 924, 1193))
    body = stack(zoom, msg, gap=8)
    panel_a, head_a = with_header(body, "A", "Right-click selects a whole bout", extra=50)

    da = ImageDraw.Draw(panel_a)
    x0 = (9609 - ZOOM_START_S) * PX_PER_S
    x1 = (9622 - ZOOM_START_S) * PX_PER_S
    bracket(da, x0 - 4, x1 + 4, head_a - 8, "selected bout, 9609-9622 s")
    outline_rows(da, x0 - 2, x1 + 2, head_a)

    dialog = video.crop((533, 82, 1523, 1209))
    panel_b, _ = with_header(dialog, "B", "Behavior clip for the selection", extra=50)

    return pad(side_by_side(panel_a, panel_b))


def pad(img, margin=24):
    out = Image.new("RGB", (img.width + 2 * margin, img.height + 2 * margin), WHITE)
    out.paste(img, (margin, margin))
    return out


def main():
    figure1().save(HERE / "figure1_adaptation.png", optimize=True)
    figure2().save(HERE / "figure2_video_review.png", optimize=True)


if __name__ == "__main__":
    main()

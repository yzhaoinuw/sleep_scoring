"""Compose the two JOSS paper figures from the original application snapshots.

Run from any directory:

    python paper/figures/build_figures.py

The five source PNGs are never modified. The replacement snapshots are
1980 by 1202 pixels. Crops align to their shared 9200/9400-second tick
centres, with small vertical offsets between captured application states.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent

TIME_ANCHOR_S, TIME_ANCHOR_X = 9200.0, 299.0
PX_PER_S = (610.5 - TIME_ANCHOR_X) / 200.0
ROW_SHIFT = {"before": 0, "after": -2, "selected": 1}
PLOT_TOP, PLOT_BOTTOM = 316, 1034
ZOOM_TOP, ZOOM_BOTTOM = 286, 1070
WHITE = (255, 255, 255)
INK = (20, 20, 20)
GAP = 40
PLOT_LEFT, PLOT_RIGHT = 97, 1872  # overlay columns shared by every snapshot
OVERVIEW_BOX = (15, 95, 1940, 1072)
AXIS_TITLE_BOX = (943, 1038, 1031, 1066)  # "Time (s)", level with the ticks
MODEBAR_BOX = (1690, 95, 1940, 125)  # Plotly hover toolbar in the overview

# B and C together span exactly the overview's plot width, so both rows of
# Figure 1 share left and right plot edges. The window still covers every
# prediction change near the expert label (about 9378-9888 s).
ZOOM_WIDTH_PX = (PLOT_RIGHT - PLOT_LEFT - GAP) // 2
ZOOM_GAP = PLOT_RIGHT - PLOT_LEFT - 2 * ZOOM_WIDTH_PX
ZOOM_START_S = 9345.0
VIDEO_START_S, VIDEO_END_S = 9070.0, 9675.0
LABEL_START_S, LABEL_END_S = 9752.0, 9802.0
SELECT_START_S, SELECT_END_S = 9176.0, 9272.0

# Stage colors as rendered in the score overlay.
STAGES = {"Wake": (128, 128, 255), "NREM": (255, 128, 128), "REM": (128, 255, 128)}


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
    img = Image.open(HERE / name).convert("RGB")
    if img.size != (1980, 1202):
        raise ValueError(f"Update screenshot geometry for {name}: {img.size}")
    return img


def x_of(t):
    return round(TIME_ANCHOR_X + (t - TIME_ANCHOR_S) * PX_PER_S)


def axis_title(width, center):
    strip = Image.new("RGB", (width, 30), WHITE)
    ImageDraw.Draw(strip).text((center, 2), "Time (s)", anchor="mt", fill=INK, font=font(20))
    return strip


def zoom_crop(img, layout, start=ZOOM_START_S, width=ZOOM_WIDTH_PX):
    shift = ROW_SHIFT[layout]
    x0 = x_of(start)
    crop = img.crop((x0, ZOOM_TOP + shift, x0 + width, ZOOM_BOTTOM + shift))
    draw = ImageDraw.Draw(crop)
    # Re-centre subplot titles in the crops; the overview keeps the original
    # titles and full stage legend. Only inter-plot whitespace is replaced.
    for title, y0, y1 in [("EEG", 286, 315), ("EMG", 540, 569), ("NE", 794, 823)]:
        draw.rectangle((0, y0 - ZOOM_TOP, crop.width, y1 - ZOOM_TOP), fill=WHITE)
        draw.text((crop.width / 2, y0 - ZOOM_TOP + 7), title, anchor="mt", fill=INK, font=font(20))
    # Keep the screenshot's tick values and place a common axis title below.
    l, t, r, b = AXIS_TITLE_BOX
    draw.rectangle((l - x0, t - ZOOM_TOP, r - x0, b - ZOOM_TOP), fill=WHITE)
    return stack(crop, axis_title(crop.width, crop.width / 2))


def stage_row(img, layout, y=885):
    """Stage per pixel column of the zoom window, read from a trace-free row."""
    a = np.asarray(img).astype(int)[y + ROW_SHIFT[layout]]
    a = a[x_of(ZOOM_START_S) : x_of(ZOOM_START_S) + ZOOM_WIDTH_PX]
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


def wrap_strip(strip, width):
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


def with_header(panel, letter, title, extra=0, indent=0):
    """Add a panel letter and short title above a panel indented by indent."""
    head = 60 + extra
    out = Image.new("RGB", (indent + panel.width, panel.height + head), WHITE)
    out.paste(panel, (indent, head))
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

    # A: interface overview without window chrome, file controls, or the
    # hover toolbar; the axis title moves below the ticks it overlapped.
    overview = before.crop(OVERVIEW_BOX)
    ox, oy = OVERVIEW_BOX[:2]
    draw = ImageDraw.Draw(overview)
    for l, t, r, b in (MODEBAR_BOX, AXIS_TITLE_BOX):
        draw.rectangle((l - ox, t - oy, r - ox, b - oy), fill=WHITE)
    plot_center = (PLOT_LEFT + PLOT_RIGHT) / 2 - ox
    overview = stack(overview, axis_title(overview.width, plot_center))
    panel_a, _ = with_header(overview, "A", "Predicted stages with spectrogram, EEG, EMG, and NE")

    # B and C: matching time windows of the EEG, EMG, and NE rows.
    zoom_b = zoom_crop(labeled, "before")
    zoom_c = zoom_crop(after, "after")

    msg_c = wrap_strip(after.crop((25, 1091, 1940, 1118)), zoom_c.width)
    body_b = zoom_b
    body_c = stack(zoom_c, msg_c, gap=8)

    # B starts at A's left plot edge while its letter stays under A's.
    indent = PLOT_LEFT - ox
    panel_b, head_b = with_header(
        body_b, "B", "Expert labels an interval as NREM", extra=50, indent=indent
    )
    panel_c, head_c = with_header(body_c, "C", "Predictions after calibration", extra=50)

    # Callouts: the expert interval in B and C, and where C changed from B.
    label_x0 = x_of(LABEL_START_S) - x_of(ZOOM_START_S)
    label_x1 = x_of(LABEL_END_S) - x_of(ZOOM_START_S)
    db = ImageDraw.Draw(panel_b)
    bracket(db, indent + label_x0, indent + label_x1, head_b - 8, "expert NREM label")

    changed = stage_row(labeled, "before") != stage_row(after, "after")
    dc = ImageDraw.Draw(panel_c)
    bar_y = head_c - 18
    runs = np.flatnonzero(np.diff(np.r_[0, changed.astype(int), 0]))
    for s, e in zip(runs[::2], runs[1::2]):
        dc.rectangle((s, bar_y, e - 1, bar_y + 12), fill=INK)
    dc.text((0, bar_y - 34), "stage changed from (B)", fill=INK, font=font(26, bold=True))
    outline_rows(dc, label_x0, label_x1, head_c)

    bottom_row = side_by_side(panel_b, panel_c, gap=ZOOM_GAP)
    fig = stack(panel_a, bottom_row, gap=30)
    return pad(fig)


def figure2():
    selected = load("app_figure_right_click_select.png")
    video = load("app_figure_video.png")

    zoom = zoom_crop(
        selected, "selected", VIDEO_START_S, x_of(VIDEO_END_S) - x_of(VIDEO_START_S)
    )
    msg = wrap_strip(selected.crop((25, 1117, 1930, 1143)), zoom.width)
    body = stack(zoom, msg, gap=8)
    panel_a, head_a = with_header(body, "A", "Right-click selects a whole bout", extra=50)

    da = ImageDraw.Draw(panel_a)
    x0 = x_of(SELECT_START_S) - x_of(VIDEO_START_S)
    x1 = x_of(SELECT_END_S) - x_of(VIDEO_START_S)
    bracket(da, x0 - 4, x1 + 4, head_a - 8, "selected bout, 9176–9272 s")
    outline_rows(da, x0 - 2, x1 + 2, head_a)

    # Scale the dialog (aspect ratio kept) so both panels have equal height.
    dialog = video.crop((481, 73, 1480, 1169))
    dialog = dialog.resize(
        (round(dialog.width * body.height / dialog.height), body.height), Image.LANCZOS
    )
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

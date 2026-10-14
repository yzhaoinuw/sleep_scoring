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
PLOT_BOTTOM = 1034
WHITE = (255, 255, 255)
INK = (20, 20, 20)
GAP = 40
PLOT_LEFT, PLOT_RIGHT = 97, 1872  # overlay columns shared by every snapshot
AXIS_TITLE_BOX = (943, 1038, 1031, 1066)  # "Time (s)", level with the ticks
MODEBAR_BOX = (1690, 95, 1940, 125)  # Plotly hover toolbar in panel 1A

# Figure 1 stacks three full-width views of the same viewport, so they share
# one time axis; tick labels appear only under the bottom panel.
PANEL_LEFT, PANEL_RIGHT = 15, 1940
NAME_TOP = 95  # recording name and colour bar, kept on 1A only
SPEC_TITLE_TOP, SPEC_TOP = 131, 152  # "EEG Spectrogram" title, plot top
CBAR_TICKS_BOX = (1250, 131, 1560, 142)  # colour-bar ticks beside the title
TICKS_BOTTOM = 1060
STACK_GAP = 16
# Figure 2A keeps the controls row and spectrogram but ends at this time, so
# the selected bout and Check Video button stay large next to the dialog.
CONTROLS_TOP = 40
FIG2_END_S = 9675.0
CHECK_VIDEO_BOX = (256, 46, 383, 80)
# Subplot title bands; cropping the time axis off-centres the app's titles.
TITLE_BANDS = [("EEG Spectrogram", 129, 151), ("EEG", 286, 315), ("EMG", 540, 569), ("NE", 794, 823)]
LABEL_START_S, LABEL_END_S = 9752.0, 9802.0
SELECT_START_S, SELECT_END_S = 9176.0, 9272.0


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


def full_view(img, layout, top, bottom, blank=()):
    """Crop the full viewport from top to bottom, whitening boxes (source coordinates)."""
    shift = ROW_SHIFT[layout]
    crop = img.crop((PANEL_LEFT, top + shift, PANEL_RIGHT, bottom + shift))
    draw = ImageDraw.Draw(crop)
    for l, t, r, b in blank:
        draw.rectangle((l - PANEL_LEFT, t - top, r - PANEL_LEFT, b - top), fill=WHITE)
    return crop


def figure1():
    before = load("app_figure.png")
    labeled = load("app_figure_correction.png")
    after = load("app_figure_after_correction.png")

    # A keeps the recording name and colour bar; the hover toolbar goes.
    body_a = full_view(before, "before", NAME_TOP, PLOT_BOTTOM + 2, [MODEBAR_BOX])
    # B and C start at the spectrogram title. C alone carries the time axis,
    # with its title moved below the ticks it overlapped. C omits the app's
    # calibration message: unreadable at print size, and the caption states
    # its labeled-second totals.
    body_b = full_view(labeled, "before", SPEC_TITLE_TOP, PLOT_BOTTOM + 2, [CBAR_TICKS_BOX])
    body_c = full_view(
        after, "after", SPEC_TITLE_TOP, TICKS_BOTTOM, [CBAR_TICKS_BOX, AXIS_TITLE_BOX]
    )
    body_c = stack(body_c, axis_title(body_c.width, (PLOT_LEFT + PLOT_RIGHT) / 2 - PANEL_LEFT))

    panel_a, _ = with_header(body_a, "A", "Predicted stages with spectrogram, EEG, EMG, and NE")
    panel_b, head_b = with_header(body_b, "B", "Expert labels an interval as NREM")
    panel_c, head_c = with_header(body_c, "C", "Same interval after adaptive calibration")

    # Callouts: the expert interval in B and C. Readers compare the stages
    # down the shared time axis themselves.
    label_x0 = x_of(LABEL_START_S) - PANEL_LEFT
    label_x1 = x_of(LABEL_END_S) - PANEL_LEFT
    bracket(ImageDraw.Draw(panel_b), label_x0, label_x1, head_b - 8, "expert NREM label")
    ImageDraw.Draw(panel_c).rectangle(
        (
            label_x0,
            head_c + SPEC_TOP - SPEC_TITLE_TOP,
            label_x1,
            head_c + PLOT_BOTTOM - SPEC_TITLE_TOP,
        ),
        outline=INK,
        width=3,
    )

    return pad(stack(panel_a, panel_b, panel_c, gap=STACK_GAP))


def figure2():
    selected = load("app_figure_right_click_select.png")
    video = load("app_figure_video.png")

    # A: the window from the controls row through the time axis, cut at
    # FIG2_END_S, with subplot titles and the axis title recentred on the
    # remaining plot area.
    right = x_of(FIG2_END_S)
    view = full_view(selected, "selected", CONTROLS_TOP, TICKS_BOTTOM, [AXIS_TITLE_BOX])
    view = view.crop((0, 0, right - PANEL_LEFT, view.height))
    center = (PLOT_LEFT + right) / 2 - PANEL_LEFT
    draw = ImageDraw.Draw(view)
    for title, y0, y1 in TITLE_BANDS:
        top, bottom = y0 - CONTROLS_TOP, y1 - CONTROLS_TOP
        draw.rectangle((PLOT_LEFT - PANEL_LEFT, top, view.width, bottom), fill=WHITE)
        draw.text((center, top + (bottom - top) / 2), title, anchor="mm", fill=INK, font=font(20))
    msg = wrap_strip(selected.crop((25, 1117, 1930, 1143)), view.width)
    body = stack(view, axis_title(view.width, center), msg, gap=0)

    # Callouts: the selected bout and the button that opens its clip.
    x0 = x_of(SELECT_START_S) - PANEL_LEFT
    x1 = x_of(SELECT_END_S) - PANEL_LEFT
    draw = ImageDraw.Draw(body)
    draw.rectangle(
        (x0 - 2, SPEC_TOP - CONTROLS_TOP, x1 + 2, PLOT_BOTTOM - CONTROLS_TOP), outline=INK, width=3
    )
    l, t, r, b = CHECK_VIDEO_BOX
    draw.rectangle(
        (l - PANEL_LEFT - 5, t - CONTROLS_TOP - 5, r - PANEL_LEFT + 5, b - CONTROLS_TOP + 5),
        outline=INK,
        width=3,
    )
    panel_a, _ = with_header(body, "A", "Right-click selects a whole bout")

    # Scale the dialog (aspect ratio kept) so both panels have equal height.
    dialog = video.crop((481, 73, 1480, 1169))
    dialog = dialog.resize(
        (round(dialog.width * body.height / dialog.height), body.height), Image.LANCZOS
    )
    panel_b, _ = with_header(dialog, "B", "Behavior clip for the selection")

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

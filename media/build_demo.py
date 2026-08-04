#!/usr/bin/env python
"""Build the captioned README demo assets from the raw screen recordings.

Each demo is declared in DEMOS below: which recording it comes from, which
stretches of it to keep and at what speed, and the caption cues in *source*
time. Rebuild one with, for example:

    python build_demo.py --demo annotation --kind gif
    python build_demo.py --demo check_video --kind mp4

Recordings are not tracked in this repository; point --src at your copy, or
drop them on the Desktop under the names in DEMOS.

Captions are rendered as transparent PNG strips with PIL and composited with
overlay filters, because Homebrew's ffmpeg is built without libfreetype and so
has no drawtext filter. Check `ffmpeg -filters` before reaching for drawtext.
"""

import argparse
import os
import shlex
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

# The three recordings were made at the same window size and position.
CROP = "crop=2880:1806:112:76"  # drop the black desktop border / window shadow
BAND_BG = (0x11, 0x14, 0x18, 255)
FADE = 0.30

DESKTOP = os.path.expanduser("~/Desktop")

# segments: (source_start, source_end, speed) kept in order; anything between
#           two segments is dropped. captions: (source_start, source_end, text).
# tail_freeze: seconds to hold the final frame, for demos whose payoff lands
#           just before the recording stops.
DEMOS = {
    "annotation": {
        "src": os.path.join(DESKTOP, "sleep_scoring_app_annotation_demo.mov"),
        # The lead-in is navigation only, so it runs at 1.5x to reach the
        # annotation work sooner.
        "segments": [(0.0, 5.95, 1.5), (5.95, 45.01, 1.0)],
        "tail_freeze": 0.0,
        "captions": [
            (0.30, 3.20, "Zoom in and out by scrolling"),
            (3.30, 5.90, "Pan by dragging"),
            (5.95, 7.45, "Switch to annotation mode"),
            (7.50, 13.00, "Select a region by drawing a box"),
            (17.00, 21.50, "Select a whole segment by right-clicking"),
            (23.00, 25.80, "Select a thin strip with a single click"),
            (26.00, 30.00, "Undo an annotation"),
            (31.00, 41.00, "Annotate continuously by dragging"),
        ],
    },
    "check_video": {
        "src": os.path.join(DESKTOP, "check_video_demo.mov"),
        # 4.3-10.3 is a frozen file-picker dialog while the video is chosen in
        # Finder, which the recording does not capture.
        "segments": [(0.0, 4.30, 1.0), (10.30, 22.99, 1.0)],
        "tail_freeze": 0.0,
        "captions": [
            (0.30, 3.10, "Select a region, then click Check Video"),
            (3.55, 11.50, "Point the app to the matching video file"),
            (12.30, 17.50, "The clip plays aligned to your selection"),
        ],
    },
    "auto_scores": {
        "src": os.path.join(DESKTOP, "generate_automatic_scores.mov"),
        "segments": [(0.0, 6.0, 1.0)],
        # The recording stops half a second after the scored trace appears.
        "tail_freeze": 2.5,
        "captions": [
            (0.30, 2.60, "Click Generate Predictions"),
            (2.90, 5.20, "Confirm that existing scores will be overwritten"),
            (5.60, 8.30, "The whole recording is scored in one pass"),
        ],
    },
}

# Output geometry. The caption band is padded on below the frame so the app's
# own status bar stays readable.
PRESETS = {
    "gif": {"width": 720, "band": 46, "font": 24, "fps": 10, "colors": 64},
    "mp4": {"width": 1920, "band": 116, "font": 60, "fps": 30, "crf": 23},
}

FONT_CANDIDATES = [
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
    ("/System/Library/Fonts/HelveticaNeue.ttc", 1),
    ("/System/Library/Fonts/Supplemental/Verdana Bold.ttf", 0),
]


def load_font(size):
    for path, index in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size, index=index)
            except OSError:
                continue
    raise SystemExit("no usable font found")


def render_strips(captions, outdir, width, band, font_size):
    os.makedirs(outdir, exist_ok=True)
    font = load_font(font_size)
    paths = []
    for i, (_, _, text) in enumerate(captions):
        img = Image.new("RGBA", (width, band), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        if right - left > width - 24:
            raise SystemExit(f"caption too wide for a {width}px band: {text!r}")
        x = (width - (right - left)) / 2 - left
        y = (band - (bottom - top)) / 2 - top
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
        path = os.path.join(outdir, f"cap_{i:02d}.png")
        img.save(path)
        paths.append(path)
    return paths


def remap(t, segments):
    """Map a source timestamp onto the output timeline.

    Time inside a dropped gap collapses onto the cut point, and time inside a
    sped-up segment is divided by that segment's speed.
    """
    out = 0.0
    last_end = 0.0
    for start, end, speed in segments:
        if t < start:
            return out
        if t <= end:
            return out + (t - start) / speed
        out += (end - start) / speed
        last_end = end
    # Past the final segment, time keeps running at 1x so that captions can be
    # placed over a frozen tail.
    return out + (t - last_end)


def build_filter(demo, width, band, fps):
    bg = "0x%02X%02X%02X" % BAND_BG[:3]
    segments = demo["segments"]
    head = f"[0:v]{CROP},fps={fps},scale={width}:-2:flags=lanczos"

    if len(segments) == 1 and segments[0][2] == 1.0:
        parts = [f"{head}[cat]"]
    else:
        # Cut the source into its kept segments, retime each one, then
        # re-normalize the frame rate so they share one cadence.
        labels = "".join(f"[p{i}]" for i in range(len(segments)))
        parts = [f"{head},split={len(segments)}{labels}"]
        for i, (start, end, speed) in enumerate(segments):
            pts = "PTS-STARTPTS" if speed == 1.0 else f"(PTS-STARTPTS)/{speed}"
            parts.append(f"[p{i}]trim={start}:{end},setpts={pts}[g{i}]")
        joined = "".join(f"[g{i}]" for i in range(len(segments)))
        parts.append(f"{joined}concat=n={len(segments)}:v=1:a=0,fps={fps}[cat]")

    tail = demo["tail_freeze"]
    freeze = f"tpad=stop_mode=clone:stop_duration={tail}," if tail > 0 else ""
    parts.append(f"[cat]{freeze}pad=iw:ih+{band}:0:0:color={bg}[base]")

    cur = "base"
    for i, (raw_start, raw_end, _) in enumerate(demo["captions"]):
        start = remap(raw_start, segments)
        end = remap(raw_end, segments)
        parts.append(
            f"[{i + 1}:v]format=rgba,"
            f"fade=t=in:st={start:.3f}:d={FADE}:alpha=1,"
            f"fade=t=out:st={end - FADE:.3f}:d={FADE}:alpha=1[c{i}]"
        )
        nxt = f"o{i}"
        # shortest=1: the caption inputs are infinite (-loop 1), so without it
        # the graph keeps emitting frames long after the source video ends.
        parts.append(
            f"[{cur}][c{i}]overlay=0:main_h-{band}:shortest=1:"
            f"enable='between(t,{start:.3f},{end:.3f})'[{nxt}]"
        )
        cur = nxt
    return ";".join(parts), cur


def run(cmd, dry):
    print("\n$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", choices=sorted(DEMOS), required=True)
    ap.add_argument("--kind", choices=["mp4", "gif"], required=True)
    ap.add_argument("--out")
    ap.add_argument("--src")
    ap.add_argument("--workdir")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    demo = DEMOS[args.demo]
    preset = PRESETS[args.kind]
    src = args.src or demo["src"]
    out = args.out or f"sleep_scoring_{args.demo}_demo.{args.kind}"
    workdir = args.workdir or f".strips_{args.demo}_{args.kind}"

    strips = render_strips(
        demo["captions"], workdir, preset["width"], preset["band"], preset["font"]
    )
    fchain, last = build_filter(demo, preset["width"], preset["band"], preset["fps"])
    duration = remap(demo["segments"][-1][1], demo["segments"]) + demo["tail_freeze"]

    inputs = ["-i", src]
    for p in strips:
        inputs += ["-loop", "1", "-framerate", str(preset["fps"]), "-i", p]

    if args.kind == "mp4":
        run(
            ["ffmpeg", "-hide_banner", "-y"]
            + inputs
            + [
                "-filter_complex",
                fchain,
                "-map",
                f"[{last}]",
                "-an",
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                str(preset["crf"]),
                "-preset",
                "medium",
                "-movflags",
                "+faststart",
                "-t",
                f"{duration:.3f}",
                out,
            ],
            args.dry_run,
        )
    else:
        palette = os.path.join(workdir, "palette.png")
        run(
            ["ffmpeg", "-hide_banner", "-y"]
            + inputs
            + [
                "-filter_complex",
                f"{fchain};[{last}]palettegen=max_colors={preset['colors']}" ":stats_mode=diff[p]",
                "-map",
                "[p]",
                "-frames:v",
                "1",
                "-t",
                f"{duration:.3f}",
                palette,
            ],
            args.dry_run,
        )
        run(
            ["ffmpeg", "-hide_banner", "-y"]
            + inputs
            + ["-i", palette]
            + [
                "-filter_complex",
                f"{fchain};[{last}][{len(strips) + 1}:v]"
                "paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                "-loop",
                "0",
                "-t",
                f"{duration:.3f}",
                out,
            ],
            args.dry_run,
        )

    if not args.dry_run and os.path.exists(out):
        print(f"\n{out}: {os.path.getsize(out) / 1e6:.2f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()

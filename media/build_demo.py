#!/usr/bin/env python
"""Build the captioned README demo assets from a raw screen recording.

Regenerates media/sleep_scoring_demo.gif and media/sleep_scoring_demo.mp4:

    python build_demo.py --kind gif --width 720 --band 46 --font 24 \
        --fps 10 --colors 64 --out sleep_scoring_demo.gif --src RECORDING.mov
    python build_demo.py --kind mp4 --width 1920 --band 116 --font 60 \
        --fps 30 --crf 23 --out sleep_scoring_demo.mp4 --src RECORDING.mov

Captions are rendered as transparent PNG strips with PIL and composited with
overlay filters, because Homebrew's ffmpeg is built without libfreetype and so
has no drawtext filter.
"""

import argparse
import os
import shlex
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

DEFAULT_SRC = os.path.expanduser("~/Desktop/sleep_scoring_app_annotation_demo.mov")
CROP = "crop=2880:1806:112:76"  # drop the black desktop border / window shadow
BAND_BG = (0x11, 0x14, 0x18, 255)
FADE = 0.30

# (start, end, text)
CAPTIONS = [
    (0.30, 3.20, "Zoom in and out by scrolling"),
    (3.30, 5.90, "Pan by dragging"),
    (5.95, 7.45, "Switch to annotation mode"),
    (7.50, 13.00, "Select a region by drawing a box"),
    (17.00, 21.50, "Select a whole segment by right-clicking"),
    (23.00, 25.80, "Select a thin strip with a single click"),
    (26.00, 30.00, "Undo an annotation"),
    (31.00, 41.00, "Annotate continuously by dragging"),
]

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


def render_strips(outdir, width, band, font_size):
    os.makedirs(outdir, exist_ok=True)
    font = load_font(font_size)
    paths = []
    for i, (_, _, text) in enumerate(CAPTIONS):
        img = Image.new("RGBA", (width, band), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        x = (width - (right - left)) / 2 - left
        y = (band - (bottom - top)) / 2 - top
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
        path = os.path.join(outdir, f"cap_{i:02d}.png")
        img.save(path)
        paths.append(path)
    return paths


def build_filter(width, band, fps, strips):
    bg = "0x%02X%02X%02X" % BAND_BG[:3]
    parts = [
        f"[0:v]{CROP},fps={fps},scale={width}:-2:flags=lanczos,"
        f"pad=iw:ih+{band}:0:0:color={bg}[base]"
    ]
    cur = "base"
    for i, (start, end, _) in enumerate(CAPTIONS):
        parts.append(
            f"[{i + 1}:v]format=rgba,"
            f"fade=t=in:st={start}:d={FADE}:alpha=1,"
            f"fade=t=out:st={end - FADE:.2f}:d={FADE}:alpha=1[c{i}]"
        )
        nxt = f"o{i}"
        # shortest=1: the caption inputs are infinite (-loop 1), so without it
        # the graph keeps emitting frames long after the source video ends.
        parts.append(
            f"[{cur}][c{i}]overlay=0:main_h-{band}:shortest=1:"
            f"enable='between(t,{start},{end})'[{nxt}]"
        )
        cur = nxt
    return ";".join(parts), cur


def run(cmd, dry):
    print("\n$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["mp4", "gif"], required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--band", type=int, required=True)
    ap.add_argument("--font", type=int, required=True)
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crf", type=int, default=30)
    ap.add_argument("--colors", type=int, default=128)
    ap.add_argument("--duration", type=float, default=45.01)
    ap.add_argument("--workdir", default="strips")
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    strips = render_strips(args.workdir, args.width, args.band, args.font)
    fchain, last = build_filter(args.width, args.band, args.fps, strips)

    inputs = ["-i", args.src]
    for p in strips:
        inputs += ["-loop", "1", "-framerate", str(args.fps), "-i", p]

    if args.kind == "mp4":
        cmd = (
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
                str(args.crf),
                "-preset",
                "medium",
                "-movflags",
                "+faststart",
                "-t",
                str(args.duration),
                args.out,
            ]
        )
        run(cmd, args.dry_run)
    else:
        palette = os.path.join(args.workdir, "palette.png")
        run(
            ["ffmpeg", "-hide_banner", "-y"]
            + inputs
            + [
                "-filter_complex",
                f"{fchain};[{last}]palettegen=max_colors={args.colors}" ":stats_mode=diff[p]",
                "-map",
                "[p]",
                "-frames:v",
                "1",
                "-t",
                str(args.duration),
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
                str(args.duration),
                args.out,
            ],
            args.dry_run,
        )

    if not args.dry_run and os.path.exists(args.out):
        mb = os.path.getsize(args.out) / 1e6
        print(f"\n{args.out}: {mb:.2f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()

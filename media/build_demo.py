#!/usr/bin/env python
"""Turn a raw screen recording into a captioned README demo.

Two subcommands:

    build_demo.py inspect RECORDING.mov     # probe it, write a contact sheet
    build_demo.py build DEMO --kind gif     # render from demos.toml

`inspect` reports duration, resolution, and the auto-detected crop, then
writes a grid of timestamped frames. Read the caption times off that grid and
put them in demos.toml; nothing demo-specific belongs in this file.

Captions are drawn as transparent PNG strips with PIL and composited with
`overlay` rather than with `drawtext`, because Homebrew's ffmpeg is built
without libfreetype and has no drawtext filter. Run `ffmpeg -filters | grep
drawtext` before assuming otherwise.
"""

import argparse
import collections
import os
import re
import shlex
import subprocess
import sys
import tomllib

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPEC = os.path.join(HERE, "demos.toml")

BAND_BG = (0x11, 0x14, 0x18)
FADE = 0.30  # caption cross-fade, seconds

FONT_CANDIDATES = [
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
    ("/System/Library/Fonts/HelveticaNeue.ttc", 1),
    ("/System/Library/Fonts/Supplemental/Verdana Bold.ttf", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
]


def run(cmd, dry=False, quiet=False):
    if not quiet:
        print("\n$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True)


def probe(src):
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=0",
            src,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)


def detect_crop(src, sample_start=1.0, sample_len=10.0):
    """Return the most frequent cropdetect result over a sample of the file."""
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-ss",
            str(sample_start),
            "-t",
            str(sample_len),
            "-i",
            src,
            "-vf",
            "cropdetect=limit=24:round=2",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    found = re.findall(r"crop=(\d+:\d+:\d+:\d+)", proc.stderr)
    if not found:
        raise SystemExit(f"could not detect a crop for {src}")
    return collections.Counter(found).most_common(1)[0][0]


def load_font(size):
    for path, index in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size, index=index)
            except OSError:
                continue
    raise SystemExit("no usable bold TrueType font found; add one to FONT_CANDIDATES")


def render_strips(captions, outdir, width, band, font_size):
    """Draw each caption as a transparent strip the width of the frame."""
    os.makedirs(outdir, exist_ok=True)
    font = load_font(font_size)
    paths = []
    for i, (_, _, text) in enumerate(captions):
        img = Image.new("RGBA", (width, band), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        if right - left > width - 24:
            raise SystemExit(f"caption does not fit a {width}px band, shorten it: {text!r}")
        x = (width - (right - left)) / 2 - left
        y = (band - (bottom - top)) / 2 - top
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
        path = os.path.join(outdir, f"cap_{i:02d}.png")
        img.save(path)
        paths.append(path)
    return paths


def remap(t, segments):
    """Map a source timestamp onto the output timeline.

    A timestamp inside a dropped gap collapses onto the cut point, one inside a
    retimed segment is divided by that segment's speed, and one past the final
    segment keeps running at 1x so captions can sit over a frozen tail.
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
    return out + (t - last_end)


def output_duration(demo):
    segments = demo["segments"]
    return remap(segments[-1][1], segments) + demo.get("tail_freeze", 0.0)


def build_filter(demo, crop, width, band, fps):
    """Assemble the filter graph: retime, pad a caption band, overlay cues."""
    bg = "0x%02X%02X%02X" % BAND_BG
    segments = demo["segments"]
    captions = demo["captions"]
    head = f"[0:v]crop={crop},fps={fps},scale={width}:-2:flags=lanczos"

    if len(segments) == 1 and segments[0][2] == 1.0:
        parts = [f"{head}[cat]"]
    else:
        # Cut the source into the segments to keep, retime each, then
        # re-normalize the frame rate so they share one cadence.
        labels = "".join(f"[p{i}]" for i in range(len(segments)))
        parts = [f"{head},split={len(segments)}{labels}"]
        for i, (start, end, speed) in enumerate(segments):
            pts = "PTS-STARTPTS" if speed == 1.0 else f"(PTS-STARTPTS)/{speed}"
            parts.append(f"[p{i}]trim={start}:{end},setpts={pts}[g{i}]")
        joined = "".join(f"[g{i}]" for i in range(len(segments)))
        parts.append(f"{joined}concat=n={len(segments)}:v=1:a=0,fps={fps}[cat]")

    tail = demo.get("tail_freeze", 0.0)
    freeze = f"tpad=stop_mode=clone:stop_duration={tail}," if tail > 0 else ""
    parts.append(f"[cat]{freeze}pad=iw:ih+{band}:0:0:color={bg}[base]")

    cur = "base"
    for i, (raw_start, raw_end, _) in enumerate(captions):
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


def contact_sheet(src, crop, times, out_path, cols=3, tile_width=400):
    """Write a grid of timestamped frames, for reading caption cues off."""
    tmp = os.path.join(HERE, ".inspect_frames")
    os.makedirs(tmp, exist_ok=True)
    tiles = []
    for t in times:
        frame = os.path.join(tmp, f"t{t:07.2f}.png")
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{t:.2f}",
                "-i",
                src,
                "-frames:v",
                "1",
                "-vf",
                f"crop={crop},scale={tile_width}:-2",
                "-y",
                frame,
            ],
            quiet=True,
        )
        tiles.append((t, frame))

    ims = [(t, Image.open(f)) for t, f in tiles if os.path.exists(f)]
    if not ims:
        raise SystemExit("no frames could be extracted")
    w, h = ims[0][1].size
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    draw = ImageDraw.Draw(sheet)
    label_font = load_font(max(12, tile_width // 28))
    for i, (t, im) in enumerate(ims):
        x, y = (i % cols) * w, (i // cols) * h
        sheet.paste(im, (x, y))
        text = f"{t:.2f}s"
        box = draw.textbbox((0, 0), text, font=label_font)
        draw.rectangle([x, y, x + box[2] + 10, y + box[3] + 8], fill="red")
        draw.text((x + 5, y + 3), text, font=label_font, fill="white")
    sheet.save(out_path)
    return out_path


def load_spec(path):
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def cmd_inspect(args):
    src = os.path.expanduser(args.recording)
    info = probe(src)
    crop = args.crop or detect_crop(src)
    duration = float(info["duration"])
    print(f"\nsource      {src}")
    print(f"resolution  {info['width']}x{info['height']}")
    print(f"duration    {duration:.2f} s")
    print(f"frame rate  {info['avg_frame_rate']} (average)")
    print(f"crop        {crop}")

    step = duration / args.frames
    times = [min(duration - 0.05, step * (i + 0.5)) for i in range(args.frames)]
    out = args.out or os.path.join(HERE, "inspect_sheet.png")
    contact_sheet(src, crop, times, out)
    print(f"\ncontact sheet -> {out}")
    print("Read caption start/end times off the tiles, then edit demos.toml.")


def cmd_build(args):
    spec = load_spec(args.spec)
    if args.demo not in spec["demos"]:
        raise SystemExit(
            f"unknown demo {args.demo!r}; {args.spec} defines: " + ", ".join(sorted(spec["demos"]))
        )
    demo = spec["demos"][args.demo]
    preset = spec["presets"][args.kind]
    src = os.path.expanduser(args.src or demo["src"])
    if not os.path.exists(src):
        raise SystemExit(f"recording not found: {src}")
    crop = args.crop or demo.get("crop") or detect_crop(src)
    default_name = demo.get("out", {}).get(args.kind, f"sleep_scoring_{args.demo}.{args.kind}")
    out = args.out or os.path.join(HERE, default_name)
    workdir = args.workdir or os.path.join(HERE, f".strips_{args.demo}_{args.kind}")

    strips = render_strips(
        demo["captions"], workdir, preset["width"], preset["band"], preset["font"]
    )
    fchain, last = build_filter(demo, crop, preset["width"], preset["band"], preset["fps"])
    duration = output_duration(demo)

    inputs = ["-i", src]
    for path in strips:
        inputs += ["-loop", "1", "-framerate", str(preset["fps"]), "-i", path]

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
        # Two passes: build a palette from the finished frames, then apply it.
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
        size = os.path.getsize(out) / 1e6
        print(f"\n{out}: {duration:.1f} s, {size:.2f} MB", file=sys.stderr)
        if args.kind == "mp4" and size > 10:
            print(
                "warning: over GitHub's 10 MB upload limit; lower `width` or "
                "raise `crf` in demos.toml",
                file=sys.stderr,
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    ins = sub.add_parser("inspect", help="probe a recording, write a contact sheet")
    ins.add_argument("recording")
    ins.add_argument("--frames", type=int, default=12)
    ins.add_argument("--crop", help="override the auto-detected crop")
    ins.add_argument("--out", help="contact sheet path")
    ins.set_defaults(func=cmd_inspect)

    bld = sub.add_parser("build", help="render a demo declared in demos.toml")
    bld.add_argument("demo")
    bld.add_argument("--kind", choices=["gif", "mp4"], default="mp4")
    bld.add_argument("--spec", default=DEFAULT_SPEC)
    bld.add_argument("--src", help="override the recording path")
    bld.add_argument("--crop", help="override the crop")
    bld.add_argument("--out")
    bld.add_argument("--workdir")
    bld.add_argument("--dry-run", action="store_true")
    bld.set_defaults(func=cmd_build)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

# README Demo Media

How the demo clips in the repository README are made. Rebuild them whenever the
app interface changes enough that the recordings look dated.

## What Is Tracked Here

| File | Tracked | Why |
| --- | --- | --- |
| `sleep_scoring_demo.gif` | yes | Linked from the top of the repository README by relative path, so it has to live in the repository |
| `build_demo.py` | yes | The builder |
| `demos.toml` | yes | What each demo is made of — the file you edit |
| `*.mp4` | **no** | GitHub hosts the players; see [Publishing](#publishing) |

Raw `.mov` recordings are not tracked either. They are large, and everything
needed to rebuild from them is in `demos.toml`.

## Recording

Record the app window at its normal size, with no audio. The three current
recordings are 3104x2030 on a Retina display; anything of that order is fine,
since the builder crops and scales.

**Record the full screen, not a single window, for any demo where the app opens
a system dialog.** A window-scoped capture excludes the macOS file picker, so
the "Check Video" demo shows the app's own prompt and then several seconds of a
frozen dialog while the file is chosen in a window that was never captured.
That stretch has to be cut, and the step gets explained by a caption instead of
shown. If you re-record that demo full-screen, drop the `segments` cut for it.

Then put the file where `demos.toml` expects it, or pass `--src`.

## Rebuilding

```bash
conda activate sleep_scoring_dash3.0

# 1. Probe the recording and get a grid of timestamped frames.
python media/build_demo.py inspect ~/Desktop/my_new_demo.mov --frames 16

# 2. Read caption times off media/inspect_sheet.png, then edit demos.toml.

# 3. Render.
python media/build_demo.py build annotation --kind gif
python media/build_demo.py build check_video --kind mp4

# 4. Check the result, sampling the moments where captions start and end.
python media/build_demo.py inspect media/sleep_scoring_check_video.mp4 --frames 12
```

Step 4 matters more than it sounds. Every caption-timing bug so far was found
by looking at a contact sheet of the finished file, not by reasoning about the
numbers.

## Editing `demos.toml`

All times are in **source** seconds — timestamps as they appear in the raw
recording. The builder maps them onto the output timeline, so cutting or
speeding up a stretch does not mean renumbering every caption.

- `segments` — `[start, end, speed]` for each stretch to **keep**, in order.
  Anything between two entries is dropped. Use this to cut dead air, and to
  run a slow lead-in faster.
- `tail_freeze` — seconds to hold the final frame, for a recording that stops
  right after the payoff appears.
- `captions` — `[start, end, "text"]`. Give every cue at least ~1.5 s, which
  usually means extending it into the idle gap that follows rather than ending
  it exactly when the gesture does. Captions render in a band padded on below
  the frame, so they never cover the app's own status bar.
- `crop` — optional. Omit it and the app window is detected automatically;
  pin it only if detection is fooled, e.g. by a dark opening frame.

Keep the wording imperative and short. A caption that does not fit the band
fails the build rather than rendering clipped.

## Publishing

**The GIF** is committed and referenced with a relative path:

```html
<img src="media/sleep_scoring_demo.gif" width="720" alt="...">
```

Keep the filename stable — `demos.toml` pins it via `out.gif`.

**The mp4s are not committed.** GitHub's markdown sanitizer strips `<video>`
tags, so a file in the repository cannot render an inline player no matter how
it is linked; only a `user-attachments` URL can. To publish one:

1. Build it.
2. Drag the file into any GitHub issue or pull request comment — do not submit
   the comment.
3. Copy the `https://github.com/user-attachments/assets/...` URL it produces.
4. Paste that URL on its own line in the README, where GitHub renders it as a
   player.

Since GitHub then hosts the file, committing it too would only add weight to
the repository. The current players live under "Use The App", "Check Aligned
Video", and "Generate Automatic Scores".

## Sizing

GitHub's drag-and-drop upload limit is **10 MB**, which is the real constraint
on the mp4 preset. At 1920 px and CRF 23 the 45 s walkthrough lands around
6 MB, so there is room; raise `crf` or lower `width` in `demos.toml` if a
longer recording overruns. The builder warns when an mp4 exceeds the limit.

The GIF is the page-weight concern instead, since it loads for everyone who
opens the repository. 720 px at 10 fps with 64 colors is about 4 MB for 43 s
and keeps the axis labels legible. Measured alternatives: 800 px at 12 fps was
7.7 MB, and 640 px at 8 fps was 3.4 MB but visibly steppier.

## Requirements

`ffmpeg`, `ffprobe`, and Pillow. Note that the Homebrew ffmpeg build used here
has **no `drawtext` filter**, because it is compiled without libfreetype —
captions are drawn as PNG strips with Pillow and composited with `overlay`
instead. Check `ffmpeg -filters | grep drawtext` before assuming you can
simplify that.

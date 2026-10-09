# Manual latency measurements — 2026-10-08/09

Codex assessment of two maintainer-run, instrumented Windows sessions using
v0.17.4, plus one macOS session on 2026-10-09 (see
[macOS session](#macos-session--2026-10-09)). These support descriptive
navigation measurements for the first manuscript pass. They do not establish
a before/after speedup or a comparison of hardware. The manuscript body has
not yet been revised with these results.

## Results selected for the manuscript

| Interaction | Local run | Viewport (s) | Recorded | Retained | Median (ms) | p95 (ms) | Range (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Keyboard navigation | 20261008-183612 | 379.9 | 37 | 34 | 329.3 | 371.2 | 299.2–374.0 |
| Custom mouse drag navigation | 20261008-213414 | 311.0 | 51 | 48 | 212.2 | 238.7 | 183.1–243.6 |

The first three observations in each event/source/mode/viewport group were
excluded as warm-up. Medians and linearly interpolated 95th percentiles were
independently checked with Python's `statistics.median` and
`statistics.quantiles(method='inclusive')` for the follow-up mouse block.
Counts are logged completed refresh observations, not independent recordings
or verified physical gesture counts. Neither block needs more repetitions to
meet this tooling's initial target of 20 retained ordinary-navigation updates.

The follow-up mouse block replaces the first run's exploratory mouse result
(15 retained updates at 343.7 seconds; median 187.5 ms, p95 199.7 ms) as the
preferred descriptive mouse result. Keep the two runs separate; their
viewport widths and recording segments differ. The new, slightly higher
median does not imply a regression because application code was unchanged
and the conditions were not matched for a comparison.

## Measurement boundaries and components

Ordinary navigation `browser_total` starts when the coalescer receives the
request that produces the logged update (`inputPerformanceTime`) and ends
at the app's Plotly completion/profiler event. For custom mouse panning the
request is issued after release; this is not the timestamp of physical input.
It includes
configured coalescing, callback delivery and the trace-update completion path.
It measures neither physical input-device latency nor exact screen presentation
time. Server callback work is already included and must not be added to it.

| Retained block | Median coalescing (ms) | Median callback/delivery/update phase (ms) | Median server callback (ms) |
| --- | ---: | ---: | ---: |
| Keyboard | 160.6 | 167.3 | 12.1 |
| Mouse drag follow-up | 2.8 | 206.0 | 12.6 |

Component medians need not sum to the median total. In these sessions, server
work is a small part of logged update latency; the remaining time includes
coalescing and interface delivery/update work, rather than isolated painting.
Every logged server callback in the follow-up had `active_at_start=1`; no
`Traceback` or `Error:` indicators were found in its log. Those observations
do not establish an exhaustive UI error check.

The first run also captured selection auto-pan merge updates at a 311-second
viewport: 58 retained refreshes, median 467.4 ms, p95 553.2 ms. Those updates
are correlated within drags, and their timer ends after issuing
`Plotly.restyle` without awaiting rendering completion. Keep this exploratory
metric separate; it is not needed for the first ordinary-navigation claim.

## Recording and machine context

The maintainer identified the first recording and reported default Sampling
Level. MAT variable shapes and scalar rate fields verify:

- EEG/EMG: 6,286,400 samples each at 610.3515625 Hz.
- NE: 104,774 samples at 10.172526245117188 Hz.
- Sample-count duration: 10,299.63776 s for EEG/EMG (~2.86 hours),
  10,299.70309 s for NE. The app rounds scoring duration to 10,300 seconds.
- Default display Sampling Level: x1, with 2,048 samples per resampled trace.

The mouse follow-up was requested on the same recording at x1. Its generated
notes are unfilled, and the logger does not independently record selected MAT
identity or Sampling Level; that context is carried from the requested protocol.
Retain this distinction when preparing final submission evidence.

Both runs automatically recorded source commit
`8d4b04036d44984cfca7fb0e12ce3dcc6a5a05c7`, Windows build 26200, an Intel
family 6/model 141 processor with 16 logical CPUs, 31.71 GiB RAM, Python 3.11.14,
Dash 3.3.0, Plotly 6.5.0, Plotly Resampler 0.11.0 and pywebview 6.1.
Slot 0, direct restyle and all three background profiling flags were enabled.
GPU, display scaling/window size, WebView2 version, power mode and background
apps remain to be recorded before the final measurement description is archived.

## Suggested manuscript wording

> In local instrumented Windows tests using v0.17.4, median logged update
> latency was 329 ms for keyboard navigation and 212 ms for mouse panning,
> with 95th percentiles of 371 and 239 ms across 34 and 48 retained updates,
> respectively. The corresponding viewport widths were 380 and 311 seconds;
> three initial updates per block were excluded. The metric includes
> navigation coalescing, callback delivery and trace-update completion.
> These manual measurements describe the current application rather than a
> controlled comparison of versions or hardware.

Add the recording duration/default display setting to this wording once the
follow-up run context is finalized. Do not combine these numbers with the
historical 935-ms baseline or Apple M4 measurements as if conditions matched.

## Local evidence and verification

The raw logs, metadata, profiler flags, CSV samples, summaries and per-run
assessments remain under ignored `paper/latency_runs/<run>/` directories. They
may include local paths; use reviewed/redacted evidence when sharing results.
This summary omits the recording filename.

Raw-log SHA-256 values:

- `20261008-183612`:
  `c36f7f1ce488155d645e6f06ba1d848a61c99dd06c2d72683d2791398f8a5fb3`.
- `20261008-213414`:
  `d3de0541829193b4bd28db5fcc18b3c0a52e05c72385d74759130c92fec0dbb5`.

The first summary was rebuilt after fixing case-sensitive auto-pan boolean
parsing (`True` versus `true`); no recording rerun was needed for that fix.
The follow-up already used the fixed parser. No application source changed
between the two measurements. Both raw logs remain unchanged.

## macOS session — 2026-10-09

One maintainer-run instrumented session (local run `20261009-011356`) on the
same recording at Sampling Level x1, both confirmed by the maintainer. It was
the first use of `measure_latency.py` on macOS; capture, parsing and summary
worked without changes. Same commit family as the Windows runs (recorded
source commit `5ba8193d257b2825618ce32863b69a9863759d6b`, which adds only the
capture tooling); same warm-up rule (first three per group excluded).

| Interaction | Viewport (s) | Recorded | Retained | Median (ms) | p95 (ms) | Range (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Keyboard navigation | 331.9 | 72 | 69 | 181.0 | 190.6 | 175–194 |
| Custom mouse drag navigation | 255.9 | 61 | 58 | 64.0 | 69.2 | 58–75 |

| Retained block | Median coalescing (ms) | Median callback/delivery/update phase (ms) | Median server callback (ms) |
| --- | ---: | ---: | ---: |
| Keyboard | 122.0 | 58.0 | 6.6 |
| Mouse drag | 1.0 | 63.0 | 6.7 |

Medians and inclusive-method p95 were recomputed from `samples.csv` with
Python's `statistics` module and match the generated summary. WebKit reports
`performance.now()` at 1-ms resolution, so macOS browser times are whole
milliseconds; this is immaterial at these magnitudes. The viewports were
narrower than the Windows blocks (379.9 and 311.0 s) and were not matched to
them.

Selection auto-pan merge refreshes were also captured in three blocks
(338.6, 311.0 and 312.6 s; 153, 226 and 295 retained; medians 115.0, 133.0
and 136.0 ms; p95 129.4, 143.8 and 146.0 ms). As on Windows, these are
correlated within drags and end after issuing `Plotly.restyle`; keep them
exploratory. Single-observation groups are setup zooms between blocks, and two
short keyboard bursts (6 and 4 retained) are not reported.

No `Traceback` or `Error:` lines were found; all 197 logged server callbacks
had `active_at_start=1`.

Machine context: Apple M4 (Mac16,13), 10 cores, integrated GPU, 16 GiB RAM,
macOS 26.3.1, built-in 2880×1864 Retina display, WKWebView through pywebview
6.1 (WebKit 21623.2.7.111.2), Low Power Mode off. Python 3.11.0, Dash 3.3.0,
Plotly 6.5.0, Plotly Resampler 0.11.0, NumPy 2.4.0, SciPy 1.16.3. Window size
and background applications were not recorded.

Raw-log SHA-256 for `20261009-011356`:
`f707f7f3b2c64e1ed49d4815e4d022e67f98fdd29683ba960e44815bddb91e88`.

Possible manuscript addition, kept separate from the Windows sentence:

> On an Apple M4 MacBook Air (macOS 26.3.1), the corresponding medians were
> 181 ms for keyboard navigation and 64 ms for mouse panning (p95 191 and
> 69 ms; 69 and 58 retained updates at 332- and 256-second viewports).

Do not present the Windows-to-Mac difference as a speedup or platform
ranking; machines, viewports and renderers differ. Do not combine it with the
historical May 2026 M4 numbers, which used different code and tasks.

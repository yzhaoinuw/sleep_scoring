---
title: 'sleep_scoring: Interactive review and correction of rodent sleep annotations with synchronized electrophysiology, photometry, and behavior video'
# TODO(name): the software name in the title is provisional; decide whether
# to keep `sleep_scoring` or adopt a name such as "Sleep Scoring App", and
# align the repository, CITATION.cff, and Zenodo metadata with the choice.
tags:
  - Python
  - sleep
  - sleep scoring
  - electroencephalography
  - EEG
  - EMG
  - fiber photometry
  - norepinephrine
  - rodent
  - neuroscience
  - annotation
authors:
  - name: Yue Zhao
    orcid: 0000-0002-0819-5012
    corresponding: true
    affiliation: 1
  # TODO: add co-authors (lab members who contributed code, models, or data)
affiliations:
  - name: University of Rochester, Rochester, NY, USA   # TODO: confirm/expand
    index: 1
date: 7 October 2026
bibliography: paper.bib
---

# Summary

Sleep research in rodents depends on *sleep scoring*: labeling each moment
of a recording as wakefulness, non-rapid-eye-movement (NREM) sleep, or REM
sleep, plus brief microarousals (MA), from brain electrical activity (EEG)
and muscle activity (EMG). `sleep_scoring` is a local desktop application
for reviewing and correcting these annotations at one-second resolution. It
brings EEG, EMG, an EEG spectrogram, an optional fiber-photometry
norepinephrine (NE) signal, and behavior-video clips of a selected interval
into one shared scoring workflow. Users can label intervals manually or
generate automatic proposals, then revise individual seconds or whole bouts
while their explicit labels stay protected from later predictions. The
default automatic scorer is a small, interpretable rule set that adapts to
the user's labels on the current recording. Completed scores are saved back
to the recording and exported as bout tables with stage and transition
statistics.

# Statement of need

`sleep_scoring` was developed for researchers studying how brain state
regulates cerebrospinal-fluid transport during sleep, in Project 2 of the
NIH BRAIN Initiative U19 program at the University of Rochester. That work
relates sleep states to cortical NE dynamics, which oscillate during NREM
sleep and shape sleep architecture [@kjaerby2022norepinephrine] and drive
the vasomotion underlying glymphatic clearance [@hauglund2025norepinephrine].
<!-- TODO(authors): confirm with the Kjaerby/Hauglund co-authors that these
two citations frame the program fairly, and whether either study's scoring
used this application. -->
Questions at this level depend on brief events, such as microarousals and
state transitions, that span a few seconds. Scoring them requires reading
the same short interval against several kinds of evidence, and correcting
boundaries, often across multi-hour recordings.

Without an integrated tool, this review means moving among a signal viewer,
a classifier's output, and separate video software, and redrawing boundaries
that an automatic scorer or a previous pass got wrong. `sleep_scoring` is
intended for experimenters scoring rodent EEG/EMG, particularly those with
aligned NE photometry and behavior video.

Its input contract is deliberately narrow. Recordings are MATLAB files with
a documented field layout, produced by a companion preprocessing pipeline;
EEG and EMG share a sampling rate and the optional photometry signal carries
its own. Other laboratories need a thin adapter to that layout; the
application is not a general acquisition-format importer. One-second
annotation is a design requirement of the motivating research, not a claim
that physiological boundaries or model estimates are accurate to one second.

# State of the field

Several open tools support rodent sleep scoring. AccuSleep and its Python
successor AccuSleePy [@barger2019accusleep] combine a manual-labeling
interface with a neural-network classifier for EEG/EMG and configurable
brain states. SPINDLE [@miladinovic2019spindle] provides end-to-end learned
scoring across laboratories and species. Somnotate
[@brodersen2024somnotate] classifies vigilance states with linear
discriminant analysis and a hidden Markov model, with simple interfaces for
refining annotations. Visbrain Sleep [@combrisson2019visbrain] offers a
general hypnogram viewer and editor for polysomnography.
<!-- TODO(verify): confirm each characterization against the cited paper and
current documentation, including epoch-length options, and add commercial
packages only with a citable source. As of 2026-10-07 the AccuSleep,
AccuSleePy, and somnotate documentation describe EEG/EMG(/LFP) inputs and do
not mention photometry or video. -->

These tools center on electrophysiology and, for most, on classification.
Our need was the inverse emphasis: a correction workspace in which NE
photometry is scoring evidence rather than an extra trace, video checks are
tied to the selected interval, and automatic proposals never override
explicit decisions. Extending an existing tool would have meant changing its
central data model (a single label stream over EEG/EMG inputs) as well as
its interaction layer. A focused application was the smaller change, and its
interaction design is documented for reuse.
<!-- TODO(authors): this build-vs-contribute rationale is inferred from the
design, not from project history. Confirm or replace it with the actual
reasons a new application was built in 2023. -->


# Software design

The application is a Dash/Plotly interface [@plotly] hosted locally in a
native `pywebview` window, so recordings open through native file dialogs by
path, without browser uploads. Its design is organized around three tasks.

**Inspecting and correcting at the needed scale.** EEG spectrogram with a
theta/delta ratio, EEG, EMG, and NE share one time axis, and the score is
drawn as an overlay on every signal row, so a boundary can be read against
each kind of evidence. Users switch between navigation and annotation with
one key. Selections can be a zoom-adaptive click, a dragged box, or a
right-click that selects the whole contiguous bout under the cursor; when a
drag reaches the viewport edge, the view pans automatically and newly
revealed signal is streamed in, so long selections keep boundary detail.
All selection forms feed one labeling step, keys `1`–`4` and `0` for clear.

Long signals are decimated on demand by Plotly Resampler
[@vanderdonckt2022plotlyresampler]. The application uses that library only
to compute what to draw for a range, and owns how updates flow. The main
tradeoff is to keep interaction state in the browser and full-resolution
data on the local server. Gestures, panning, and labeling run in the
browser; per-frame navigation events are coalesced into one refresh after
the view settles; refreshes overtaken by newer navigation are discarded;
updates are applied as trace patches rather than figure rebuilds; and
auto-pan fetches data through a dedicated endpoint outside the Dash callback
graph. Applying a label repaints the overlay in the browser without a server
round trip.

**Using predictions while retaining human decisions.** The application keeps
a sparse layer of explicit user labels separate from the displayed scores.
Any backend's output is placed beneath that layer, so regenerating
predictions preserves the user's labels. The tradeoff is provenance: labels
already stored in an opened file seed the layer and are protected too, and
users clear intervals they want regenerated.

The default backend is a deliberately small rule set that needs no GPU or
deep-learning runtime. A normalized 1–7 Hz EEG spectral feature identifies
Wake, with rules for minimum bout length, and candidate bouts become REM
when NE falls below recording-wide and within-bout percentile thresholds.
Without usable NE, it identifies only Wake and NREM. When the user has
labeled a few Wake, NREM, or REM seconds, the app calibrates the five
exposed controls for this recording by a small deterministic search that
minimizes disagreement with those examples and breaks ties toward the
defaults; the tuned values are displayed. We chose recording-specific,
inspectable calibration over persistent training: it is fast and does not
change the next recording's defaults, and agreement on supplied examples is
not a claim of held-out accuracy. The externally developed sDREAMER model
[@chen2023sdreamer] can be selected instead, with optional PyTorch
[@paszke2019pytorch] dependencies.

**Checking evidence and completing a recording.** For a selected interval of
up to five minutes, the application cuts the matching clip from the behavior
video with ffmpeg, validates the recording-to-video offset, and plays it in
the same window. One-step undo and filesystem-backed recovery protect
in-progress work; up to three isolated windows allow side-by-side
comparison, refusing the same file twice. Saving reports the first unscored
gap; a complete recording also exports bout, stage, and transition
statistics, including microarousals. Packaged Windows installations receive
compatible updates that preserve supported settings.

# Research impact statement

Researchers in Project 2 of the University of Rochester U19 program have been
the application's main users for about four years, using it to score and
correct their rodent EEG/EMG/NE recordings.
<!-- TODO(authors): git history begins June 2023; confirm "about four
years". Add the users' rough estimate of total recording hours scored. The
opt-in usage counter (README badge) is recent and covers only opted-in
copies, so it undercounts total use. Name studies, preprints, or datasets
scored with the app once confirmed. -->
It integrates with a companion preprocessing pipeline that converts raw
acquisition output into its input format, and the repository provides
packaged Windows releases, macOS source installation, documentation, tests,
demonstration videos, and archived versions [@sleepscoring_zenodo].

Manual timing tests on a 4.25-hour recording show the interaction design is
practical on long data. On a Windows laptop, optimizing the update pipeline
reduced the time from the end of a navigation gesture to the refreshed traces
from about 935 ms to about 300–370 ms; server work was 14–17 ms of that, and
the remainder is browser redraw. On an Apple M4 laptop, refreshes after drag
panning took about 190–300 ms, and live auto-pan updates during a selection
about 260–310 ms. These are local manual measurements rather than a
controlled benchmark.
<!-- Source: ui_response_time_optimization_progress.txt (2026-05-23/25).
The measured pipeline matches the shipped code (the later Dash-store bypass
was reverted). Consider rerunning on the current release before submission. -->

# AI usage disclosure

Generative AI coding agents (OpenAI Codex and Anthropic Claude Code) assisted
with software development, tests, documentation, release audits, and drafting
of this manuscript, as recorded in the repository's work log. The authors
reviewed and tested the code changes and reviewed every manuscript claim.
<!-- TODO(authors): confirm the scope, and that author review is complete,
before submission. -->

# Acknowledgments

This work was supported by the BRAIN Initiative of the US National Institutes
of Health (U19NS128613).

<!-- TODO: thank the lab PI, data contributors, and model contributors by
name. The funding sentence above is the PI-approved wording and should be
kept verbatim. -->

# References

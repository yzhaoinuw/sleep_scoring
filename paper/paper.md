---
title: 'Sleep Scoring App: Visualizing and annotating rodent sleep recordings with synchronized electrophysiology, photometry, and behavior video'
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
affiliations:
  - name: University of Rochester, Rochester, NY, USA
    index: 1
date: 14 October 2026
bibliography: paper.bib
---

# Summary

Sleep Scoring App
([`sleep_scoring`](https://github.com/yzhaoinuw/sleep_scoring)) is a local
desktop application for sleep scoring rodent recordings: labeling each
second as wakefulness (Wake), non-rapid-eye-movement (NREM) sleep, REM
sleep, or a brief microarousal (MA), from brain electrical activity (EEG)
and muscle activity (EMG). It
brings EEG, EMG, an EEG spectrogram, an optional fiber-photometry
norepinephrine (NE) signal, and behavior-video clips of a selected interval
into one shared scoring workflow. Users can label intervals manually or
start from sleep scores predicted by an automatic scoring model, then revise
individual seconds or whole bouts; predicted sleep scores never overwrite
the user's own labels. The
default scoring model is a rule-based algorithm that adapts to
expert-labeled examples: it fits the user's labels on the current recording
while changing its other predictions as little as possible.
Completed scores are saved back
to the recording and exported as bout tables with stage and transition
statistics.

# Statement of need

Sleep Scoring App was developed for researchers studying how brain state
regulates cerebrospinal-fluid transport during sleep, in
[Project 2](https://www.urmc.rochester.edu/research/u19/project-2) of the
BrainFlowZZZ program at the University of Rochester. That work
relates sleep states to cortical NE dynamics, which oscillate during NREM
sleep and shape sleep architecture [@kjaerby2022norepinephrine] and may drive
the vasomotion underlying glymphatic clearance [@hauglund2025norepinephrine].
<!-- TODO(authors): confirm with the Kjaerby/Hauglund co-authors that these
two citations frame the program fairly. Neither study's scoring used this
application: Kjaerby et al. scored in Viewpoint's SleepScore (Methods), and
users report no publications from the app's scores yet (2026-10-15). -->
Such questions depend on brief events, such as microarousals and state
transitions, that span a few seconds. Scoring them means examining the same
short interval across several physiological signals and the animal's
movement, and correcting boundaries across multi-hour recordings; without an
integrated tool, this means moving among a signal viewer, predicted scores,
and separate video software. One-second annotation is a requirement of this
research, not a claim that boundaries or model estimates are accurate to
one second. Sleep Scoring App is intended for
experimenters scoring rodent EEG/EMG, particularly with aligned NE
photometry and behavior video.

The current version takes as input a MATLAB file produced from the raw
recordings by a companion preprocessing pipeline,
[`preprocess_sleep_data`](https://github.com/yzhaoinuw/preprocess_sleep_data).
Other laboratories may need a thin adapter to this file layout.

# State of the field

Several open tools support rodent sleep scoring.
[AccuSleep](https://github.com/zekebarger/AccuSleep) and its Python
successor [AccuSleePy](https://github.com/zekebarger/AccuSleePy)
[@barger2019accusleep] pair a manual-labeling interface with a
neural-network scoring model. [SPINDLE](https://sleeplearning.ethz.ch)
[@miladinovic2019spindle] provides end-to-end learned scoring across
laboratories and species.
[Somnotate](https://github.com/paulbrodersen/somnotate)
[@brodersen2024somnotate] scores with linear discriminant analysis and a
hidden Markov model. [Visbrain Sleep](https://github.com/EtienneCmb/visbrain)
[@combrisson2019visbrain] is a general polysomnography viewer and hypnogram
editor.
<!-- TODO(verify): confirm each characterization against the cited paper and
current documentation, including epoch-length options, and add commercial
packages only with a citable source. As of 2026-10-07 the AccuSleep,
AccuSleePy, and somnotate documentation describe EEG/EMG(/LFP) inputs and do
not mention photometry or video. Scoring methods checked 2026-10-14:
AccuSleep CNN on EEG/EMG (PLOS ONE 2019); SPINDLE CNN+HMM on 2 EEG + 1 EMG,
4 s epochs (PLoS Comput Biol 2019); somnotate LDA+HMM on EEG/LFP/EMG
spectrograms (repo README); Visbrain detects spindles/K-complexes/slow
waves/REMs but does not stage. Viewpoint SleepScore uses an undisclosed
"probability method" on EEG/EMG (product page); check its manual before
extending the NE claim to it. sDREAMER's NE-aware variant does use NE, so
keep the claim scoped to the tools listed here. -->

These tools center on electrophysiology and, for most, on automatic sleep
scoring, which draws on EEG, EMG, or LFP features alone. None of their
scoring methods uses NE photometry, although NE dynamics have been proposed
as markers for distinguishing wake, microarousals, NREM, and REM sleep
[@andersen2023photometry]. Beyond automatic scoring, our work needed a
workspace for labeling and correcting scores, in which NE photometry and
video checks inform scoring. Project 2 previously used Viewpoint's commercial
[SleepScore](https://www.viewpoint.fr/product/rodent/sleep-monitoring/sleepscore), which did not integrate NE
photometry well; that gap motivated a dedicated application.
<!-- History from the Project 2 users (2026-10-14); they confirm the wording
in the full-draft review. SleepScore supports synchronized video, so the
stated gap is NE photometry integration only. -->


# Software design

The application is designed around five tasks: (1) zero-setup installation
and automatic updates, (2) visualizing and annotating at scale, (3)
automatic scoring that keeps and learns from expert labels, (4) checking
movement against video, and (5) saving, recovery, and export. Smooth
interactive visualization is central to these tasks, so the application is
built with [Dash](https://dash.plotly.com) and
[Plotly](https://plotly.com/python/) [@plotly]: a local web page, shown in a
native window via [pywebview](https://pywebview.flowrl.com) and backed by a
Python server on the same computer.

## 1. Zero-setup installation and automatic updates

Many experimenters do not program, so we packaged the Windows application
with [PyInstaller](https://pyinstaller.org) to install like conventional
desktop software: users download a release, extract it, and double-click to
open it, with no Python installation, environment setup, or command line.
Our first users ran Windows; macOS support came later, and because a
comparable package would need Apple's code signing and notarization, macOS
users currently install from source. Packaging alone would have required a
rebuild and a fresh download for every change, so we kept the application's
own source code outside the packaged runtime. We further built and
integrated into the application an updater, released as the reusable,
dependency-free package
[`desktop_app_source_updater`](https://github.com/yzhaoinuw/desktop_app_source_updater)
[@desktop_app_source_updater], which automatically fetches and applies
updates, including bug fixes, new features, and other improvements, at
startup.[^deps]

[^deps]: Updates that change the packaged dependencies still require
downloading a new release.

## 2. Visualizing and annotating at scale

The EEG spectrogram (with a theta/delta ratio), EEG, EMG, and NE share one
time axis, and the score is drawn as an overlay on every signal row
(\autoref{fig:adaptation}A), so each signal can be used on its own to
examine or label sleep states. We built shortcut gestures so that navigation
and annotation take as few steps as possible. For example, a click selects
an interval too narrow to drag at the current zoom, a right-click selects a
whole contiguous bout, and a drag that reaches the edge of a zoomed-in view
keeps panning, so a selection can extend arbitrarily far without losing
detail. A selection is then labeled with a single key press.

In a Dash application, the interface (the embedded web page) draws the
figures while the data and computation stay on the server: each interaction
is sent to the server, which returns updated figure data. Plotly alone
becomes unresponsive when asked to draw the millions of samples in a
multi-hour recording, so the application decimates long signals on demand
with [Plotly Resampler](https://github.com/predict-idlab/plotly-resampler)
[@vanderdonckt2022plotlyresampler]: the server keeps the full-resolution
signals and sends the interface only enough points to draw the visible
range, so zooming in or panning needs new data from the server.

Because each round trip to the server adds delay, we followed one principle
for responsiveness: interactions run in the interface, and the server is
asked only for signal data the interface does not yet have. Selecting and
labeling therefore run in the interface. Zooming and panning must involve
the server, because the refreshed traces come from there. Plotly Resampler's
stock update path requests refreshed traces on every zoom or pan event,
which compounds the delay during continuous movement. Users of earlier
versions reported lags of one to two seconds while navigating, which
disrupted scoring. We replaced that path: while the user zooms or pans
continuously, the application waits for the movement to pause before
requesting new traces, drops requests already overtaken by newer movement,
and swaps in only the changed trace data rather than redrawing the figure.

In manual timing tests on a 2.86-hour recording with v0.17.4, the median time
from a navigation request to refreshed traces on a Windows laptop was 212 ms
for mouse panning and 329 ms for keyboard navigation, down roughly 65–75%
from about 935 ms in an earlier version; on an Apple M4 laptop it was 64 and
181 ms. Much of the keyboard time is deliberate waiting for key repeats to
settle. None of these is a controlled benchmark.
<!-- Source: paper/latency_measurements.md (v0.17.4, 2026-10-08/09; same
recording at Sampling Level x1; first three updates per task excluded;
34–69 updates per task at 256- to 380-second views; 95th percentiles
Windows 371/239 ms, Mac 191/69 ms for keyboard/mouse). Windows and Mac runs
differ in machine and viewport; do not frame the difference as a speedup or
platform ranking. The 935 ms figure is from
ui_response_time_optimization_progress.txt (May 2026, Windows i9 laptop): an
earlier app version, not stock Plotly Resampler, under unmatched conditions;
keep it framed as a rough comparison, not a before/after benchmark. The
65-75% range is 1 - 329/935 and 1 - 212/935. The
one-to-two-second lags were seen by the maintainer while watching users
navigate over screen share in feedback meetings (no written record). -->

This interaction design is not specific to sleep; we extracted it into a
runnable template for general multichannel time series
[@timeseries_app_cookbook].

## 3. Automatic scoring that keeps and learns from expert labels

We implemented the default scoring model as a rule-based algorithm that
encodes the stage criteria used by Project 2 researchers
[@kjaerby2022norepinephrine]: it identifies Wake from EEG spectral power and
REM from sustained low NE, and without usable NE it identifies only Wake and
NREM. Its thresholds adapt to each recording. Each run treats the user's
labels (a single labeled interval is enough) as ground truth and chooses the
settings that best fit them while changing the rest of the predictions as
little as possible (\autoref{fig:adaptation}). Adaptation therefore needs no
multi-recording training set. Agreement with the user's labels shows only
that the fit succeeded, not how accurate the predictions are elsewhere in
the recording.
<!-- Rule origin: Project 2 researchers agreed the rules as a group. Kjaerby
et al. (Methods) define Wake/NREM/REM from EEG and EMG in 1 s epochs and MA
as wake bouts under 15 s, and note NE is lowest in REM; the app's numeric
thresholds are not in that paper, so keep the citation at the level of
stage criteria. Detail deliberately left to the README (2026-10-15): Wake
uses a normalized 1-7 Hz EEG feature; REM needs bouts of at least 30 s
below recording-wide and within-bout NE percentiles; five settings are
fitted (Wake threshold, minimum Wake and REM durations, two NE percentiles),
scored by average per-stage error plus a penalty for changing unlabeled
predictions; MA labels are kept but not fitted. -->

Alternatively, the deep-learning model sDREAMER [@chen2023sdreamer], in a
version adapted by the author
([`sdreamer_flow`](https://github.com/yzhaoinuw/sdreamer_flow)), runs when
the optional [PyTorch](https://pytorch.org) [@paszke2019pytorch]
dependencies are installed; it does not adapt to user labels. With either
model, the application keeps user labels separate from the predictions and
places them on top, so a prediction never overwrites a label.

![**Expert-guided adaptation.** The same interval at three successive steps:
**(A)** the initial predicted stages over the spectrogram, EEG, EMG, and NE;
**(B)** an expert relabels 50 s as NREM; **(C)** the model, recalibrated on
228 s of expert labels (178 s Wake outside this view, 50 s NREM), updates
its predictions, including beyond the outlined interval.\label{fig:adaptation}](figures/figure1_adaptation.png)

## 4. Checking movement against video

Users sometimes check the behavior video to corroborate the movements
indicated by EMG, which help distinguish sleep states. The application
therefore links video to the selection: for a selected interval of up to
five minutes, enough for events that span seconds and quick to extract, it
cuts the matching clip with [FFmpeg](https://ffmpeg.org) and plays it in the
same window (\autoref{fig:video}).

![**Selection-linked video review.** **(A)** A right-click selects the
bout at 9176–9272\ s, and Check Video (outlined) opens its clip. **(B)** The
96-second clip plays beside the signals.\label{fig:video}](figures/figure2_video_review.png)

## 5. Saving, recovery, and export

Besides saving the scores back to the recording, the application can undo
the most recent annotation or prediction. Saving a completely scored
recording also exports bout, stage, and transition statistics to a
spreadsheet; otherwise, saving reports what is left unscored, especially
short gaps hard to spot when zoomed out, so the user can complete it. If the
computer crashes before a save, filesystem-backed recovery restores the
unsaved scores when the file is reopened.

# Research impact statement

Researchers in Project 2 of the University of Rochester BrainFlowZZZ program
have been the application's main users since 2023, using it to score their
rodent EEG/EMG/NE recordings. The program's channel for the application has
about 30 members; five users who reported their use have together scored
about 1,200 hours of recordings, a lower bound on total use. Results scored
with the application have been presented in conference posters.
<!-- Source: replies from five users, 2026-10-15/16: ~300 h, ~300 h, ~250 h,
~200 h, and 50-100 recordings at a typical ~3 h (150-300 h); about
1,200-1,350 h in total. No publications or datasets from these scores yet, and no known users
outside the program. As of 2026-10-18 there are no published studies,
preprints, or datasets to name; add them only if one appears before
submission. -->
The repository provides releases, documentation, tests, demonstration
videos, and archived versions [@sleepscoring_zenodo].

# AI usage disclosure

From 2023 through 2025, the author asked web-based ChatGPT and Claude
coding questions (model versions not recorded). Since 2026, coding agents
have assisted with code, tests, documentation, and release checks: Claude
Code (Claude Opus 4.5–5.5 and Fable 5) and OpenAI Codex (GPT-5 and GPT-6),
with sessions recorded in the repository's work log (`treaty_docs/`), kept
under the author's
[Agent Collab Treaty](https://github.com/yzhaoinuw/agent_collab_treaty)
conventions for human–agent collaboration. Claude Code and Codex
also helped plan, draft, and review this manuscript. The author made the
architectural and design decisions, reviewed and tested all AI-assisted code
before release, and reviewed, revised, and verified the manuscript against
the software and cited sources.
<!-- TODO(author): confirm review is complete before submission. -->

# Acknowledgments

This work was supported by the BRAIN Initiative of the US National Institutes
of Health (U19NS128613).

We thank Mie Andersen, Jingyuan Chen, Natalie Linea Hauglund, Shadi Sartipi,
Anastasia Tsopanidou, Verena Untiet, and Yuan Yao for helping define the sleep-stage
rules that the rule-based model encodes. We especially thank Mie Andersen,
Rebeka Kovács, Klaudia Anna Tokarska, and Anastasia Tsopanidou, who were among
the application's earliest users and gave extensive one-on-one feedback.
We also thank the other users whose feedback and feature requests shaped
the application: at the University
of Copenhagen, Katharina Franziska Baumgart, Antonia Benita Beiersdorfer,
Julia Johanna Berger, Zuzanna Bojarowska, Saiyue Deng, Stefanie Gregoriades,
Natalie Linea Hauglund, Jaspreet Kaur, Elise Schiøler Nielsen, Tessa Radovanovic,
Laurence Vince Riesselmann, Katia Soud, and Yingshan Wang; and at the University of
Rochester, Ziru Deng, Tianyi Tong, and Jianbo Wang. We thank Douglas Kelley
for supervising this work, Verena Untiet for leading the Copenhagen group, and
Maiken Nedergaard for leading the BrainFlowZZZ program.

<!-- The funding sentence above is the PI-approved wording and should be
kept verbatim. Names (2026-10-16) are from the user consensus so far; the
Copenhagen users are running a poll, so the user list may grow. Names are
alphabetical by surname within each group. The "especially thank" group
must meet both criteria: among the earliest users (a matter of record; some
have since left for other groups) AND extensive one-on-one feedback (only
Yue can confirm this). Don't add anyone on one alone (assumed surname:
Schiøler Nielsen); check spelling and diacritics with each person. -->

<!-- TODO: thank data and model contributors by name, if separate from the
people above. -->

# References

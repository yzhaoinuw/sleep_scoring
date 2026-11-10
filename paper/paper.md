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
date: 20 October 2026
bibliography: paper.bib
---

# Summary

Sleep Scoring App
([`sleep_scoring`](https://github.com/yzhaoinuw/sleep_scoring)) is a local
desktop application for sleep scoring rodent recordings: labeling each
second as wakefulness (Wake), non-rapid-eye-movement (NREM) sleep, REM
sleep, or a brief microarousal (MA), from brain electrical activity (EEG)
and muscle activity (EMG). It brings EEG, EMG, an EEG spectrogram, an
optional norepinephrine (NE) signal measured by fiber photometry, and
behavior-video clips of a selected interval into one scoring workflow.
Users can label intervals manually or start from predicted sleep scores,
then revise individual seconds or whole bouts; predictions never overwrite
the user's own labels. The default rule-based scoring model adapts to the
user's labels on each recording, and completed scores are exported as bout
tables with stage and transition statistics.

# Statement of need

Sleep Scoring App was developed for researchers studying how brain state
regulates cerebrospinal-fluid transport during sleep, in
[Project 2](https://www.urmc.rochester.edu/research/u19/project-2) of the
BrainFlowZZZ program at the University of Rochester. That work
relates sleep states to cortical NE dynamics, which oscillate during NREM
sleep and shape sleep architecture [@kjaerby2022norepinephrine] and may drive
the vasomotion underlying glymphatic clearance [@hauglund2025norepinephrine].
Such questions depend on brief events, such as microarousals and state
transitions, that span a few seconds. Scoring them means examining the same
short interval across several signals and the animal's movement, and
correcting boundaries across multi-hour recordings; without an integrated
tool, this means moving among a signal viewer, predicted scores, and
separate video software. One-second annotation is a requirement of this
research, not a claim that boundaries or predictions are accurate to one
second. The application is intended for experimenters scoring rodent
EEG/EMG, particularly with aligned NE photometry and behavior video.

The current version reads a MATLAB file produced by a companion
preprocessing pipeline,
[`preprocess_sleep_data`](https://github.com/yzhaoinuw/preprocess_sleep_data);
other laboratories may need a thin adapter to this file layout.

# State of the field

Several open tools support rodent sleep scoring.
[AccuSleep](https://github.com/zekebarger/AccuSleep) and its Python
successor [AccuSleePy](https://github.com/zekebarger/AccuSleePy)
[@barger2019accusleep] pair a manual-labeling interface with a
neural-network scoring model. [SPINDLE](https://sleepyrat.ai)
[@miladinovic2019spindle] provides end-to-end learned scoring across
laboratories and species, now as a web service.
[Somnotate](https://github.com/paulbrodersen/somnotate)
[@brodersen2024somnotate] scores with linear discriminant analysis and a
hidden Markov model. [Visbrain Sleep](https://github.com/EtienneCmb/visbrain)
[@combrisson2019visbrain] is a general polysomnography viewer and hypnogram
editor.

These tools center on electrophysiology and, for most, on automatic scoring
from EEG, EMG, or LFP features alone. None of their scoring methods uses NE
photometry, although NE dynamics have been proposed as markers for
distinguishing wake, microarousals, NREM, and REM sleep
[@andersen2023photometry]. Beyond automatic scoring, our work needed a
workspace for labeling and correcting scores, in which NE photometry and
video checks inform scoring. Project 2 previously used Viewpoint's
commercial
[SleepScore](https://www.viewpoint.fr/product/rodent/sleep-monitoring/sleepscore),
which did not integrate NE photometry well; that gap motivated a dedicated
application. It also had to suit many experimenters who are not
programmers, yet every open tool above requires either setup in a
programming environment or uploading recordings to a web service.

# Software design

The application is designed around five tasks: (1) zero-setup installation
and automatic updates, (2) visualizing and annotating at scale, (3)
automatic scoring that keeps and learns from expert labels, (4) checking
movement against video, and (5) saving, recovery, and export. Because
smooth interactive visualization is central to these tasks, it is built
with [Dash](https://dash.plotly.com) and
[Plotly](https://plotly.com/python/) [@plotly]: a local web page, shown in a
native window via [pywebview](https://pywebview.flowrl.com) and backed by a
Python server on the same computer.

## 1. Zero-setup installation and automatic updates

We packaged the Windows application with
[PyInstaller](https://pyinstaller.org) so that it installs like
conventional desktop software: users download a release, extract it, and
double-click to open it. macOS users currently install from source, because
a comparable package would need Apple's code signing and notarization. To
avoid a rebuild and a fresh download for every change, the application's
own source code stays outside the packaged runtime, and an integrated
updater, released as the reusable, dependency-free package
[`desktop_app_source_updater`](https://github.com/yzhaoinuw/desktop_app_source_updater)
[@desktop_app_source_updater], fetches and applies bug fixes and new
features at startup.[^deps]

[^deps]: Updates that change the packaged dependencies still require
downloading a new release.

## 2. Visualizing and annotating at scale

The EEG spectrogram (with a theta/delta ratio), EEG, EMG, and NE share one
time axis, and the score is drawn as an overlay on every signal row
(\autoref{fig:adaptation}A), so each signal can be used on its own to
examine or label sleep states. Shortcut gestures minimize steps: for
example, a right-click selects a whole contiguous bout, and a drag that
reaches the edge of a zoomed-in view keeps panning, so a selection can
extend arbitrarily far. A single key press then labels the selection.

Plotly alone becomes unresponsive when drawing the millions of samples in a
multi-hour recording, so the application decimates long signals on demand
with [Plotly Resampler](https://github.com/predict-idlab/plotly-resampler)
[@vanderdonckt2022plotlyresampler]: the server keeps the full-resolution
signals and sends the interface only enough points to draw the visible
range. Because each round trip to the server adds delay, selecting and
labeling run entirely in the interface, and only zooming and panning
request new data. Plotly Resampler's stock update path requests refreshed
traces on every zoom or pan event, which compounds the delay during
continuous movement; users of earlier versions reported lags of one to two
seconds that disrupted scoring. We replaced that path: during continuous
movement, the application waits for a pause before requesting new traces,
drops requests already overtaken by newer movement, and swaps in only the
changed trace data rather than redrawing the figure.

In manual timing tests on a 2.86-hour recording with v0.17.4, the median time
from a navigation request to refreshed traces on a Windows laptop was 212 ms
for mouse panning and 329 ms for keyboard navigation, down from about 935 ms
in an earlier version (not a controlled benchmark). This interaction design
is not specific to sleep; we extracted it into a runnable template for general
multichannel time series [@timeseries_app_cookbook].

## 3. Automatic scoring that keeps and learns from expert labels

The default scoring model is a rule-based algorithm that encodes the stage
criteria used by Project 2 researchers [@kjaerby2022norepinephrine]: it
identifies Wake from EEG spectral power and REM from deep NE valleys, and
without usable NE it identifies only Wake and NREM. Its thresholds adapt to
each recording. Each run treats the user's labels (a single labeled
interval is enough) as ground truth and chooses the settings that best fit
them while changing the rest of the predictions as little as possible
(\autoref{fig:adaptation}), so adaptation needs no multi-recording training
set. Agreement with the user's labels shows only that the fit succeeded,
not how accurate the predictions are elsewhere in the recording.

Alternatively, the deep-learning model sDREAMER [@chen2023sdreamer], in a
version adapted by the author
([`sdreamer_flow`](https://github.com/yzhaoinuw/sdreamer_flow)), runs when
the optional [PyTorch](https://pytorch.org) [@paszke2019pytorch]
dependencies are installed.

![**Expert-guided adaptation.** The same interval at three successive steps:
**(A)** the initial predicted stages over the spectrogram, EEG, EMG, and NE;
**(B)** an expert relabels 50 s as NREM; **(C)** the model, recalibrated on
228 s of expert labels (178 s Wake outside this view, 50 s NREM), updates
its predictions, including beyond the outlined interval.\label{fig:adaptation}](figures/figure1_adaptation.png)

## 4. Checking movement against video

Movements indicated by EMG help distinguish sleep states, and users
sometimes corroborate them with behavior video. For a selected interval of
up to five minutes, the application cuts the matching clip with
[FFmpeg](https://ffmpeg.org) and plays it in the same window
(\autoref{fig:video}).

![**Selection-linked video review.** **(A)** A right-click selects the
bout at 9176–9272\ s, and Check Video (outlined) opens its clip. **(B)** The
96-second clip plays beside the signals.\label{fig:video}](figures/figure2_video_review.png)

## 5. Saving, recovery, and export

Scores are saved back to the recording, and the most recent annotation or
prediction can be undone. Saving a completely scored recording also exports
bout, stage, and transition statistics to a spreadsheet; otherwise, saving
reports what is left unscored, including short gaps hard to spot when
zoomed out. After a crash, filesystem-backed recovery restores unsaved
scores when the file is reopened.

# Research impact statement

Researchers in Project 2 of the University of Rochester BrainFlowZZZ program
have been the application's main users since 2023, using it to score their
rodent EEG/EMG/NE recordings. The program's channel for the application has
about 30 members; five users who reported their use have together scored
about 1,200 hours, a lower bound on total use. The repository provides releases, documentation, tests, and demonstration
videos, and links to archived versions [@sleepscoring_zenodo] and an example
dataset [@sleepscoring_example_data].

# AI usage disclosure

From 2023 through 2025, the author asked web-based ChatGPT and Claude
coding questions (model versions not recorded). Since 2026, coding agents
have assisted with code, tests, documentation, and release checks: Claude
Code (Claude Opus 4.5–5.5 and Fable 5) and OpenAI Codex (GPT-5 and GPT-6),
with sessions recorded in the repository's work log (`agent_worklog/`), kept
under the author's
[Agent Collab Treaty](https://github.com/yzhaoinuw/agent_collab_treaty)
conventions for human–agent collaboration. Claude Code and Codex
also helped plan, draft, and review this manuscript. The author made the
architectural and design decisions, reviewed and tested all AI-assisted code
before release, and reviewed, revised, and verified the manuscript against
the software and cited sources.

# Acknowledgments

This work was supported by the BRAIN Initiative of the US National Institutes
of Health (U19NS128613).

We thank Mie Andersen, Jingyuan Chen, Natalie Linea Hauglund, Shadi Sartipi,
Anastasia Tsopanidou, Verena Untiet, and Yuan Yao for helping define the sleep-stage
rules that the rule-based model encodes. We especially thank Mie Andersen,
Rebeka Kovács, Klaudia Anna Tokarska, and Anastasia Tsopanidou, who were among
the application's earliest users, gave extensive one-on-one feedback, and
shared recordings for testing.
We also thank the other users whose feedback and feature requests shaped
the application: at the University
of Copenhagen, Katharina Franziska Baumgart, Antonia Benita Beiersdorfer,
Julia Johanna Berger, Zuzanna Bojarowska, Saiyue Deng, Stefanie Gregoriades,
Natalie Linea Hauglund, Jaspreet Kaur, Elise Schiøler Nielsen, Tessa Radovanovic,
Laurence Vince Riesselmann, Katia Soud, and Yingshan Wang; and at the University of
Rochester, Ziru Deng, Tianyi Tong, and Jianbo Wang. We thank Douglas Kelley
for supervising this work, Verena Untiet for leading the Copenhagen group, and
Maiken Nedergaard for leading the BrainFlowZZZ program.

# References

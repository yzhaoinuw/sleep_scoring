# JOSS manuscript layout and feature assessment

First-pass editorial plan, 2026-10-06. Reviewed against shipped v0.17.4 and
`main` commit `4353b76`, after fast-forwarding the local `publication` branch
from `d336ff7`. This is a planning document; `paper.md` still needs revision.

## Recommended argument

Lead with a practical research problem: a scorer must inspect several kinds
of evidence, correct brief sleep-state boundaries, and retain those decisions
while reviewing a long recording. The application's contribution is the
integrated **inspect → select → label → predict → check → correct → export**
workflow at one-second resolution, including optional NE photometry and
selection-linked behavior video.

The most distinctive application-owned work is the interaction design:
whole-bout selection, drag selection that continues across viewport edges
while traces refresh, and explicit user labels that survive regenerated
predictions. The adaptive statistical scorer is another contribution worth
describing, particularly its recording-specific, inspectable tuning. These
are defensible design contributions; claiming they are unprecedented or
improve scientific accuracy requires comparison and validation evidence.

Synchronized views, undo/recovery, native file access, side-by-side windows,
and complete exports support this argument by reducing interruptions and
protecting work. They deserve space in proportion to their practical value,
without presenting each as a separate innovation.

Credit Plotly Resampler once for on-demand display of long signals. The app
adds gesture coordination, selection, and correction behavior around that
dependency; resampling algorithms and generic large-data plotting are not
the central contribution. Likewise, credit sDREAMER as an upstream model
integration, ffmpeg for encoding, and Dash/Plotly/pywebview for the framework.
Keep opt-in usage tracking out of the first-pass feature narrative. It is a
possible source of later impact evidence, not an experimenter-facing reason
to choose the application.

Suggested title:

> sleep_scoring: Interactive review and correction of rodent sleep annotations
> with synchronized electrophysiology, photometry, and behavior video

This removes the old title's emphasis on optional deep learning and leaves
room for manual scoring and the default statistical backend.

## Assessment of every cookbook recipe

Priority: **Core** gets substantive manuscript space; **Support** gets a brief
mention tied to the workflow; **Credit** identifies a dependency;
**Docs** stays in developer documentation; **Optional** is deferred.

| Recipe | Priority | Manuscript use and contribution boundary |
| --- | --- | --- |
| 1. Desktop shell | Support | Native file access and a familiar local desktop workflow; pywebview is an upstream dependency. |
| 2. Layout/component model | Docs | Ordinary UI plumbing; describe mode-appropriate controls only where useful. |
| 3. Server-side cache | Support | Explain recovery and local data briefly, not cache keys or serialization. |
| 4. File loading/dialogs/validation | Support | Open large local MAT recordings without browser upload; state the fixed input contract and need for adapters. |
| 5. Resampler figure | Support + Credit | Synchronized EEG/spectrogram/EMG/optional NE and scores are central context; credit Plotly Resampler for signal downsampling. |
| 6. EventListener bridge | Docs | Implementation mechanism, useful only as part of a short design explanation. |
| 7. Relayout coalescer | Support | An application design choice that avoids redundant/stale refreshes during navigation; no latency claim without measurements. |
| 8. Patch/direct-restyle pipeline | Support | Explain why navigation and label edits avoid rebuilding the whole figure; do not sell the upstream resampler as ours. |
| 9. Keyboard panning | Support | A small part of efficient keyboard-led review. |
| 10. Custom pointer pan | Support | Navigation tailored to synchronized signals; omit low-level pointer/axis details. |
| 11. Mode switching | Support | One-key navigation/annotation switching keeps review and correction in the same workspace. |
| 12. Box/click/whole-bout selection | Core | Select a narrow interval or a complete scored/unscored bout without drawing every boundary by hand. Click width depends on zoom; it is not always exactly one epoch. |
| 13. Auto-pan selection/live refresh | Core | Extend a selection beyond the viewport without zooming away from boundary detail; this is application-owned interaction work. |
| 14. Keypress annotation/overlays | Core | Immediate shared score display, one-second labels, manual MA, and clearing selected ranges. Explain behavior, not trace patch syntax. |
| 15. Undo/crash recovery | Support | One-step undo and same-file, same-slot recovery protect decisions. Do not call it an unlimited undo stack or a backup system. |
| 16. Saving/export | Support | Partial MAT saves identify remaining gaps; complete saves offer bout, stage, and transition statistics, including MA. |
| 17. Selection-linked video | Core | Inspect behavior for the selected ambiguous interval without leaving the scoring workflow; credit ffmpeg and native playback. |
| 18. Performance instrumentation | Docs | A way to produce future evidence, not a first-pass feature. Existing instrumentation alone is not a benchmark. |
| 19. Multiple desktop instances | Support | Up to three isolated windows for comparing different recordings; the same MAT path is refused in a peer window. This is not simultaneous collaborative scoring. |
| 20. Opt-in aggregate reporting | Optional | Omit from the first pass. If used later, app-copy totals cannot establish distinct users/labs, accuracy, or time saved. |
| 21. Adaptive statistical calibration | Core | Explicit examples tune a small rule set for this recording and remain protected in predictions. Not persistent training or established accuracy improvement. |
| 22. Prediction backends/correction | Core | One correction workflow supports manual work, the default statistical scorer, and optional upstream sDREAMER. Attribute the model correctly. |
| 23. Compatible startup updates | Support | One sentence on maintaining packaged installations and supported settings; updater mechanics stay in the cookbook. |

## Proposed JOSS structure

Use the current [JOSS paper guidance](https://joss.readthedocs.io/en/latest/paper.html)
and [review checklist](https://joss.readthedocs.io/en/latest/review_checklist.html),
checked on 2026-10-06. The current guidance gives a 750–1750-word range and
requires Summary, Statement of need, State of the field, Software design,
Research impact statement, and AI usage disclosure, plus acknowledgments and
references. Aim for roughly 1,400–1,600 words of prose; the planning tables
here belong in documentation, not the submitted paper.

### Summary — about 150 words

Introduce rodent sleep annotation for a non-specialist. Describe the local
application, synchronized signal evidence, one-second manual correction,
optional prediction, behavior-video checks, and saved analysis outputs.
Mention EEG/EMG and explain optional photometry in plain language. Avoid a
package list and save the details of calibration for Software design.

Possible opening:

> sleep_scoring is a local desktop application for reviewing and correcting
> sleep-state annotations in rodent recordings. It brings brain electrical
> activity, muscle activity, an optional norepinephrine photometry signal,
> and selected behavior-video clips into a shared scoring workflow. Users
> can label one-second intervals manually or inspect automatic proposals,
> revise individual intervals or whole bouts, and export completed scores
> for subsequent analysis.

### Statement of need — about 180 words

Explain why short events and boundaries matter to the intended NE/sleep
research workflow, and why switching among signal, prediction, and video
tools makes review cumbersome. Identify experimenters doing rodent EEG/EMG
scoring, especially those with aligned NE photometry and video. Describe
BrainFlowZZZ as the motivating application, with comparable laboratories as
the intended audience; do not assert external adoption without evidence.

State the constraint explicitly: MAT recordings follow the documented field
contract; EEG and EMG share a sampling rate; optional photometry carries its
own rate. This is not yet a general-purpose EDF/acquisition-format importer.
One-second annotation resolution is a design requirement, not proof that
physiological boundaries or model estimates are accurate to one second.

### State of the field — about 180 words

Compare a small number of directly relevant tools using their primary papers
and current documentation. Reuse verified bibliography entries where
appropriate. Compare actual annotation granularity, manual correction,
photometry/video integration, and installation/data contracts. Build a
source-backed comparison before claiming an unmet gap or explaining why a
new application was preferable to extending an existing tool.

The current assertions that commercial tools are vendor-locked, most tools
assume 4–10-second epochs, alternatives rarely combine viewing and scoring,
and none accept NE are too broad to carry forward without checking. Remove
uncited SleepEEGpy unless a relevant source and fair comparison are supplied.
Do not make a first/only claim from this cookbook audit.

### Software design — about 580 words

Organize this around three user tasks, with tradeoffs woven into each:

1. **Inspect and correct at the needed scale** (about 230 words). Shared time
   axes and score overlays let users read the same interval against multiple
   signals. Narrow, box, and whole-bout selections share a labeling step;
   edge auto-pan retains local detail during long selections. Explain why
   browser-side interaction and incremental updates avoid waiting for a
   full figure redraw. Credit Plotly Resampler here in one sentence.
2. **Use predictions while retaining human decisions** (about 230 words).
   Separate displayed predictions from explicit user evidence. Describe the
   lightweight statistical backend, its five configurable controls,
   recording-specific calibration, and protected manual overrides. Note
   that existing MAT scores seed the evidence layer and their provenance is
   not distinguished. Credit optional sDREAMER separately. Generated scores
   remain reviewable; avoid promising quality gains from sparse examples.
3. **Check evidence and complete a recording** (about 120 words). Show
   selection-linked video as an ambiguity check. Briefly mention one-step
   undo, recovery, isolated comparison windows, partial saves with gap
   feedback, and complete exports. Put distribution/settings preservation
   into one sentence if space permits.

Key tradeoffs: one-second labels versus continuous-time signal display;
local files/native dialogs versus a fixed MAT contract; sparse protected
evidence versus undifferentiated saved-score provenance; small deterministic
calibration versus persistent training; process-isolated windows versus
collaborative editing. These explain research-relevant decisions better
than listing modules and libraries.

### Research impact statement — about 180 words

Supply specific evidence: an identifiable research workflow using the app,
publications or integrations if documented, and ideally a public example or
repeatable task demonstration. The repository has releases, documentation,
tests, demo clips, and an archived software version; these are readiness
signals, not evidence of external users or measured scientific benefit.

Useful next measurements are correction-task completion time and error rate
on reviewed boundaries, and held-out scoring performance before/after
calibration. Record hardware, recording length/rates, task, and backend.
Without such results, describe implemented capabilities and concrete
current use; omit quantified speed, accuracy, and time-saving claims.
External adoption strengthens this section but the current criteria also
allow credible near-term significance backed by concrete evidence.

### AI usage disclosure — about 70 words

Disclose the actual assistance and verification, including this first-pass
Codex release/code audit, cookbook revision, and manuscript planning. The
repository work log also records AI-assisted development; authors should
confirm its full scope before writing the final disclosure. Describe code
review and tests where performed, and author review of manuscript claims.
Do not claim this session validated scientific performance or completed
human author review.

### Acknowledgments — about 60 words; References

Keep the existing approved funding sentence verbatim:

> This work was supported by the BRAIN Initiative of the US National Institutes
> of Health (U19NS128613).

Confirm authors, affiliations, ORCIDs, and contributor acknowledgments with
the maintainer. Cite directly relevant alternatives, Plotly Resampler,
sDREAMER, and the software archive. Recheck any revised bibliographic claims.

## Recommended figure

One compact, annotated workflow figure is more useful than a feature gallery:
show synchronized signals and score overlays; mark a whole-bout or
cross-viewport selection; show its aligned video check; and indicate manual
labels retained after prediction. Use a redistributable recording or an
explicitly identified synthetic example. Capture a current application
session before finalizing the caption. A still image cannot demonstrate
latency or prove accuracy; link the existing demos as supplementary usage
material where appropriate.

## Release-to-cookbook audit

Scope: all release entries from v0.16.3–v0.17.4, plus the older interaction,
timing, and statistical-backend changes relevant to the existing draft.
Checked changelog claims against current source and targeted test coverage;
this is not a fresh release/package validation.

Evidence entry points: [interaction callbacks](../app_src/assets/clientsideCallbacks.js),
[secondary-click handling](../app_src/assets/graphContextMenu.js),
[figure construction](../app_src/make_figure.py),
[spectral timing](../app_src/get_fft_plots.py),
[prediction integration](../app_src/callbacks/prediction.py),
[statistical rules/calibration](../app_src/run_inference_stats_model.py),
[saving](../app_src/callbacks/saving.py),
[video](../app_src/callbacks/video.py), and
[startup/update boundaries](../run_desktop_app.py).
Focused verification in this session passed 116 Python tests covering helpers,
spectral timing, exports, score layers, metadata aliases, windows, and startup,
plus all 51 clientside JavaScript tests. These checks establish implemented
behavior within their test scope, not scientific accuracy or human task speed.

| Change | Cookbook action |
| --- | --- |
| v0.15.5 fractional-rate timing | Recipe 5 now explains anchored spectral columns and true EEG/EMG sample times; removed the stale ShortTimeFFT description. |
| v0.16.0 statistical scoring/styling | Added baseline backend behavior in Recipe 22; expanded Recipe 5 display customization. |
| v0.16.1 selection/auto-pan | Recipes 12–13 already cover the workflow; Recipe 12 now explains the later secondary-click race fix. |
| v0.16.2 signed video offsets/bounds | Recipe 17 now explains range validation and unavailable-video feedback. |
| v0.16.3 fp_frequency alias | Recipe 4 now covers the alias and saved-metadata preservation. |
| v0.16.4 incomplete-save feedback | Recipe 16 now covers the first gap and feedback even on dialog cancellation. |
| v0.16.5 multiple windows/optional Torch | Recipe 19 already covers isolation/peer exclusion; Recipe 22 covers optional model dependencies. |
| v0.16.6 path-safe recovery | Corrected Recipe 15's remaining basename-based adaptation guidance. |
| v0.16.7/v0.17.0 colors/config retention | Recipe 5 covers stage/legend colors; added Recipe 23 for updates and settings. |
| v0.16.8 MA export fix | Recipe 16 now explicitly covers MA statistics and regenerating existing workbooks. |
| v0.17.0–v0.17.1 updater behavior | Recipe 23 covers daily checks, current full base, graceful failures, and full-download guidance. |
| v0.17.2 tracking/calibration | Recipe 20 already covers opt-in tracking. Recipe 21 already covers calibration; Recipe 22 supplies the missing backend/correction context. |
| v0.17.3 tuned controls/feedback | Recipe 21 already documents five controls and separate 60-second feedback; retained that coverage. |
| v0.17.4 semantic overlay identity | Updated Recipes 5, 12, 14 and the gotcha catalog to use role/type lookup instead of last-three-trace assumptions. |
| v0.17.4 native video/nonfatal cleanup | Updated Recipe 17 to the shipped player and cleanup behavior, with remaining collision/freeze limits explicit. |

## Corrections needed in the existing paper

- Replace SWS with NREM and document Wake/NREM/REM/MA accurately. Keyboard
  keys `1–4` map to stored codes `0–3`; key `0` clears selected labels.
- Add the default statistical scorer and calibration. Without usable NE,
  that backend identifies Wake/NREM and does not automatically detect REM.
  Do not imply all backends support every modality equally.
- Replace the claimed undo stack with one-step undo. Predictions and manual
  changes use the same display history. Qualify recovery by file path,
  window slot, and cache availability.
- Replace `dash-player` playback with native `html.Video`. The package is
  still listed in dependencies, but no longer implements clip playback.
- Replace the ShortTimeFFT implementation claim with the current anchored
  FFT-based spectrogram, or simply say spectral analysis without naming the
  former implementation.
- Remove the claim that the public repository ships representative MAT
  recordings: private/local test data and checkpoints are excluded. Add a
  public reproducible example before claiming one is supplied.
- Describe MAT saving as a user-selected path, which may overwrite the
  original; Excel export is offered separately only for complete scores.
- Replace blanket competitor and physiological-validity claims with
  attributed, verified statements. Rule-based cleanup is a heuristic.
- Do not include experimental Active/Quiet Wake or the pending full-path
  video-association/clip-identity fix as stable-release features. The
  original frozen-frame report is not established as resolved.
- Update the manuscript date when its revision is prepared; verify all
  author metadata. Preserve attribution and the approved funding text.

## Next revision pass

Rewrite `paper.md` around this layout; verify a short related-tool comparison
from primary sources; select concrete use/impact evidence; provide a public
example and current workflow figure; complete author metadata and AI
disclosure; then render and review the JOSS PDF. The first pass should stand
without usage-tracking material or unmeasured performance claims.

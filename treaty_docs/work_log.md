# Work Log

Prepend new session notes to this file. Record project decisions, reusable
evidence, and shared-state changes—not routine content already explained by the
diff. Each session ends with a `- Verification:` subsection. See
[`treaty_conventions.md`](treaty_conventions.md#work-log-discipline).

If today's date is already at the top, add a new `###` subsection beneath it.
The live log holds at most five unique dates; when a new date would exceed that
limit, move the oldest five-date chunk to
`work_log_archive/work_log_<earliest>_to_<latest>.md`.

Historical commands can contain machine-specific paths. When replaying them,
keep the `sleep_scoring` folder and `sleep_scoring_dash3.0` environment names
but adapt the user prefix and clone location. Default to the two newest dates;
search older entries by date anchor rather than reading every archive.

## 2026-10-12

### Figure 1 caption names the calibration input (Claude Code, Opus 5.5)

- Reversed the 2026-10-11 "no labeling-history qualifications" choice for
  the Figure 1 caption: panel C's message reports calibration from 358
  labeled seconds while panel B shows only the 55 s NREM label, so the
  caption now says "this and earlier expert labels (358 s in total)".
- Maintainer confirmed the snapshots were captured after the per-stage
  calibration-tolerance fix (`c2e3bb6`), so they need not be retaken.

## 2026-10-11

### Dev calibration fix merged into publication (Codex, GPT-6; effort/token budget not reported)

- Maintainer authorized bringing the committed dev change into `publication`.
  Merged dev `7129f49` (calibration-overfit fix and per-stage tolerance review)
  into publication `c7abf74`, preserving the figure sources and manuscript work.
- Calibration now favors parameters closest to defaults among near-best
  candidates, with a separate 10%-of-labeled-seconds allowance per stage.
  Updated the paper, layout assessment, and cookbook to match this behavior;
  stage-confined adaptation remains a future idea, as recorded on dev.
- Resolved the work-log conflict by retaining the October 11 and October 8
  publication entries and inserting dev's October 9 calibration entry between
  them. Dev's older October 6 entries already exist in publication's archive.
  The original dev checkout's uncommitted changes remain untouched.
- Verification:
  - Full merged Python suite: `conda run -n sleep_scoring_dash3.0 python -m
    pytest --basetemp .pytest_tmp\publication_merge -p no:cacheprovider -q`:
    229 passed (one existing Flask-Caching deprecation warning).
  - The first run encountered a missing `.pytest_tmp` parent in the fresh
    worktree; creating that directory resolved the fixture-setup errors.
  - `git diff --exit-code dev -- app_src/run_inference_stats_model.py
    tests/test_sleep_score_layers.py` passed; no unmerged entries or conflict
    markers remain. `git diff --check` passed. Approximate manuscript body
    word count is 1,685, excluding front matter and comments.
  - Commit hooks remain enabled; targeted branch ancestry and remote refs
    will be verified after the authorized push.

### Paper figure plan and publication delivery (Codex, GPT-6; effort/token budget not reported)

- Maintainer chose adaptability as the figure's purpose: the caption follows
  prediction, expert NREM labeling, and regenerated predictions without
  labeling-history qualifications. Changes beyond the labeled interval
  illustrate adaptation within this recording, not held-out accuracy or
  cross-recording generalizability.
- Agreed two figures: three adaptation panels with one larger interface
  overview, and two selection/video panels. Preserved the five supplied
  screenshots and recorded the panel mapping, proposed captions, and crop
  guidance. Final assembly, manuscript placement, and PDF review remain open.
- Prepared the authorized commit/push on `publication` from remote commit
  `15fdf38` in an isolated managed worktree. The original `dev` checkout's
  local work-log/next-step edits, archive, and latency runs are preserved.
- Verification:
  - Visually inspected all five source snapshots during the figure discussion;
    the selected 9609–9622 s bout matches the video clip's 13-second duration.
  - Verified all five copied PNG SHA256 hashes against the originals and 27
    local Markdown link targets; `git diff --check` passed.
  - `conda run -n sleep_scoring_dash3.0 treaty validate .` cannot validate the
    configured `treaty_docs` layout: installed CLI 0.3.2 expects flat files,
    while `.copier-answers.yml` pins template v0.9.0. Trying the docs directory
    instead misses root AGENTS/overview files. No tooling update was made.
  - No runtime code changed; application tests are not needed for this
    documentation-and-screenshot commit. Commit hooks remain enabled.

## 2026-10-09

### Stats-model calibration overfit to single-stage labels (Claude Opus 5.5)

- Report: on `35_app13.mat`, 202 s of Wake-only expert labels made the
  adaptive stats model call ~78% of the recording Wake and none NREM.
- Cause: calibration minimized mismatches on labelled seconds only, so the
  maximum Wake threshold (0.95) matched Wake-only labels perfectly and the
  closest-to-default tie-break never applied.
- Fix: each coordinate step accepts candidates that, for every labelled
  stage, miss at most `CALIBRATION_MISMATCH_TOLERANCE_FRACTION` (10%) of that
  stage's labelled seconds more than the strict best fit, then picks the one
  closest to the defaults. Tolerance floors to zero below 10 labels per
  stage, so one-label calibration is unchanged.
- Review follow-up (Codex on PR #17): a first version pooled the tolerance
  over all labels, so 100 correct NREM labels let defaults win while missing
  all 10 Wake examples. Per-stage tolerances fix this; regression tests use
  the real predictor for rare-Wake and rare-REM (NE path) cases.
- Evidence (ground truth from the wake-ne-analysis copy, codes 4/5 counted as
  Wake): defaults 84.3% agreement; old calibration 0.95, 17.6%; new
  calibration 0.85, 86.5%. A 0.80 threshold scores 92% but misses 68 labelled
  seconds, outside the tolerance.
- Open: Wake-only labels still change REM, because REM bouts are relabelled
  from Wake candidate bouts and so depend on the Wake threshold.
- Maintainer's later labels on 35_app13 (175 s Wake + 138 s NREM) calibrate
  to threshold 0.80, 92.0% agreement, under both the pooled and per-stage rules.
- Verification: the Wake-only test fails on the original code (picks 0.95);
  the minority-stage tests fail under the pooled rule; all pass with the fix.
  Full suite 218 passed, 2 skipped; `git diff --check` clean. Not yet
  exercised in the running app.

## 2026-10-08

### Latency tooling delivery (Codex, GPT-6; effort/token budget not reported)

- Maintainer authorized committing and pushing the independent capture
  launcher, reviewed measurement summary, timing-history update and Mac
  instructions to `publication`. Raw runs remain local/ignored. App runtime
  code, desktop launcher and manuscript body are unchanged by this delivery.
- Fetched `origin/publication` before staging; both refs were `8d4b040` and
  no collaborator changes needed integration. Prepared the documentation
  handoff around completed Windows trials and pending final context/Mac work.
  Remote-ref verification follows the authorized push.
- Verification:
  - Candidate Python code already passed the focused 9-test suite, pinned
    Black hook and logging preflight in this session; only documentation was
    adjusted afterward. Commit hooks remain enabled with isolated local caches.

### Mac capture audit and timing-history update (Codex, GPT-6; effort/token budget not reported)

- Code inspection supports using the same latency script on macOS in the
  existing working source environment: capture uses portable Python APIs,
  and `run_desktop_app.py` selects the native renderer outside Windows.
  This new capture launcher has not been executed on a Mac. Documented
  preflight, logging/summary checks and manual Mac hardware/runtime context;
  generalized generated run-note wording beyond Windows WebView2.
- Updated the timing-history file with a dated October section and current
  keyboard/mouse medians, p95, retained counts, actual viewport widths and
  measurement boundaries. The original May Windows and M4 results remain
  historical; no new Mac numbers or cross-platform speedup were inferred.
  Current tool changes are local and still uncommitted/unpushed.
- Clarified the ordinary timing start from source: `inputPerformanceTime`
  is set when the coalescer receives the request producing the logged
  update; custom pointer pan issues that request after release. It is not
  a physical input timestamp. Server time is included in browser total.
- Verification:
  - `python -m pytest --basetemp .pytest_tmp\paper_latency -p no:cacheprovider
    -q tests/test_paper_latency.py`: 9 passed. Pinned Black hook passed.
  - `python paper/measure_latency.py record --check`: slot 0 and all three
    profiling flags true on Windows. Inspected native-renderer selection,
    source-run setup, profiling environment overrides and timing formulas.

### Mouse latency follow-up reviewed (Codex, GPT-6; effort/token budget not reported)

- Run `20261008-213414` captured 51 custom-drag navigation updates at a fixed
  311-second viewport. After three warm-ups, 48 retained observations had
  median 212.2 ms, p95 238.7 ms, range 183.1–243.6 ms. This meets the
  first-pass sample target; no more repetitions are needed. Use this block
  instead of the first session's exploratory 15-update mouse result, without
  pooling the sessions or interpreting their difference as a regression.
- Saved a durable measurement summary in `paper/latency_measurements.md` and
  a local per-run assessment. Current application/source/package metadata
  matches the first run; ordinary mouse/keyboard viewport widths differ.
  Follow-up MAT/Sampling Level context follows the requested protocol but
  is not logged independently. Final machine/display context and manuscript
  integration remain open; manuscript body is unchanged. No commit or push.
- Verification:
  - Parsed all 126 events: 63 browser navigation and 63 server events, with
    63 unique browser profile IDs. Independently checked mouse median and
    inclusive p95 with `statistics`; reviewed saved component statistics.
  - All logged server callbacks had `active_at_start=1`; no `Traceback` or
    `Error:` indicators found. Captured profiling flags all true, slot 0.
    Raw-log SHA-256 recorded in the assessment; capture was not modified.

### First maintainer latency run reviewed (Codex, GPT-6; effort/token budget not reported)

- Run `20261008-183612` on v0.17.4/source `8d4b040` produced 37 keyboard
  updates at a 379.9-second viewport; 34 retained after three warm-ups had
  median 329.3 ms, p95 371.2 ms (range 299.2–374.0 ms). Custom mouse drag
  navigation had 15 retained updates at 343.7 seconds (median 187.5 ms,
  p95 199.7 ms), still exploratory. Viewport widths differ from the proposed
  300-second protocol; report actual widths rather than silently renaming them.
- Real logs revealed a parser defect: Python logging emits `applied=True`,
  but the original parser accepted only lowercase `true`. Case-insensitive
  matching recovered all 88 auto-pan samples from the unchanged raw log.
  There are now 201 valid browser samples. Auto-pan's largest merge group
  has 58 retained refreshes at 311 seconds (median 467.4 ms, p95 553.2 ms),
  correlated within drags and measured only through issuing the restyle call.
- Keyboard median coalescing delay was 160.6 ms versus 12.1 ms server work;
  do not attribute the entire non-server remainder to browser drawing. This
  session supports descriptive current-version latency, not a before/after
  speedup, comparison of hardware, or independent auto-pan gesture trials.
- Saved a local assessment with proposed manuscript wording and raw-log
  provenance. Maintainer identified the recording and reported default
  Sampling Level. MAT shapes/scalar rate fields verify duration ~2.86 h,
  EEG/EMG 610.3515625 Hz (6,286,400 samples each), NE 10.172526245117188 Hz
  (104,774 samples). UI default is x1/2,048 display samples per trace. Filled
  local run notes; display/runtime details remain open. The manuscript
  itself is unchanged. No commit or push performed.
- Verification:
  - Focused pytest suite: 9 passed, including both `True`/`False` and
    lowercase auto-pan forms. Pinned Black hook passed.
  - Rebuilt summary/CSV/events using `summarize ... --warmup 3`; reviewed all
    event counts and checked medians with `statistics.median`. All profiler
    flags true in captured preflight; no `Traceback`/`Error:` log indicators.
  - Verified recording context with `scipy.io.whosmat` and scalar-only
    `loadmat`, and default Sampling Level in components/loading callbacks.
    Signal arrays were not loaded. Documentation links and whitespace passed;
    raw-log SHA-256 remained unchanged after rebuilding derived reports.

### JOSS manual latency capture (Codex, GPT-6; effort/token budget not reported)

- Maintainer narrowed this pass to latency tooling and will choose the MAT
  file and perform the measurements. The manuscript figure and new timing
  claims remain pending; no research recording was opened or benchmarked.
- Use `python paper/measure_latency.py record` in the project environment.
  The launcher sets the existing server/browser profiling overrides for its
  child process, checks the effective flags, and requires free window slots.
  It captures stdout/stderr continuously and summarizes when the app closes.
  App configuration defaults are unchanged; profiling is enabled for this
  measurement launch. Run outputs remain local and ignored by Git.
- Metric boundary found in the current implementation: ordinary navigation
  ends at the Plotly completion/profiler event, while auto-pan issues
  `Plotly.restyle` without awaiting completed rendering. Report them
  separately; callback time is part of browser total, never added to it.
  Repeated drag refreshes are correlated, not independent human trials.
- `publication` was fast-forwarded to Claude's revised manuscript commit
  `8d4b040`; this tooling is currently uncommitted. No commit, push, release,
  or figure capture was performed in this pass.
- Verification:
  - `python -m pytest --basetemp .pytest_tmp\paper_latency -p no:cacheprovider
    -q tests/test_paper_latency.py`: 8 passed, including a real child-process
    stdout/stderr capture and automatic-summary test using synthetic events.
  - `python paper/measure_latency.py record --check`: slot 0 and all three
    profiling flags true; direct restyle true. `metadata()` collected the
    actual commit, OS, CPU/RAM, Python and package versions successfully.
  - Pinned Black 25.12.0 hook passed for both new Python files with an isolated
    repo-local pre-commit cache. The installed environment's standalone Black
    import failed (`AssertionError: LAZY`); its packages were not changed.
  - Documentation checks passed (45 local links, 23 cookbook recipes and
    manuscript assessments); `git diff --check` passed. Work-log rotation
    preserved the previous five-date chunk in its archive.

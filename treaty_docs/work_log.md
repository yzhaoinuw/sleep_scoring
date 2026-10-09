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

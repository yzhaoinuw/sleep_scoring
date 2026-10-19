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

## 2026-10-19

### Manuscript title, acknowledgments, and branch sync (Claude Code, Claude Opus 5.5; effort not reported; 15M token budget)

- Retitled the paper (2026-10-18) to "Visualizing and annotating rodent
  sleep recordings ..." to match the Software design wording; "review and
  correction" described only fixing existing labels. `CITATION.cff` follows
  the paper; the Zenodo bib entry keeps the archived title until the next
  release so the reference stays accurate to the record.
- Acknowledgments: users in alphabetical order by surname within each site,
  so the order implies no ranking. A separate "especially thank" clause names
  four users who meet two criteria: among the earliest users (a matter of
  record) and extensive one-on-one feedback (only the maintainer can
  confirm). Wording chosen to avoid disputes over who is included. Verena
  Untiet is thanked for the rules and for leading the Copenhagen group, not
  as a user. The list may grow after the Copenhagen users' poll.
- Authorship: after reviewing the JOSS criteria, the maintainer remains the
  sole person qualified as an author.
- `paszke2019pytorch`: Crossref and the NeurIPS proceedings page list no DOI;
  title, authors, volume, and year match the proceedings page, whose URL was
  added. Accepted without a DOI.
- Interface-server paragraph split by the maintainer's review: decimation
  first, then the responsiveness principle, the stock update path, the user
  lag, and the fix. The lag is placed beside the stock path, not claimed as
  its proven cause, because the 935 ms baseline was not stock Plotly
  Resampler.
- Fast-forwarded `dev` and `main` to `publication` at `f447167` and pushed;
  GitHub accepted the direct dev push under the maintainer bypass of its
  pull-request rule. This treaty-docs follow-up is synchronized the same way.
- Verification: CITATION packaging tests (`pytest -q
  tests/test_full_release_packaging.py tests/test_update_asset_packaging.py`:
  39 passed, 2 skipped); `git diff --check`; Crossref API search and the
  NeurIPS proceedings page for the PyTorch entry; `git ls-remote --heads`
  for `publication`, `dev`, and `main`.

## 2026-10-16

### PI manuscript comments and interval-review wording (Codex, GPT-6; effort/token budget not reported)

- Applied Douglas Kelley's annotated-PDF corrections to the manuscript source:
  use BrainFlowZZZ for the program name and soften the NE/vasomotion claim to
  "may drive." The existing author-verification TODO remains open; this wording
  change does not establish consensus or confirm either study used the app.
- The maintainer questioned whether "comparing ... to" changes the meaning of
  interval review. Used "examining the same short interval across several
  physiological signals and the animal's movement" to preserve the intended
  simultaneous review of aligned channels rather than imply a comparison of
  an interval with the signals. Other manuscript text and figures are unchanged.
- Fetched five newer publication commits through `61142fc` before delivery;
  replayed these four edits on top, preserving their reorganization,
  research-impact evidence, and "movement" wording. Maintainer authorized
  committing and pushing this review pass on `publication`.
- Verification: reviewed the four requested changes against the previously
  extracted annotations and current source; `git diff --check` passed before
  and after integration. The final diff contains only the four requested
  wording changes and this record; the pre-commit hook passed/skipped.

## 2026-10-15

### v0.17.6 published and archived; manuscript integrated (Codex, GPT-6; effort/token budget not reported)

- Final candidate `058fc94` passed the complete lightweight gate, then `dev`
  and `main` fast-forwarded to publication at the same commit. All six local
  and tracking refs plus targeted remote branch refs matched. GitHub accepted
  the direct dev push under its configured maintainer bypass; no PR created.
- Published annotated tag `v0.17.6` at `058fc94` and the latest GitHub release
  with the 52,030-byte automatic update ZIP and its checksum. GitHub's asset
  digest matches local SHA-256
  `21d54c6c0ecf8bc20f377b4325ac64432fbfb6277bbca12654c9e18d7a6a2c60`.
  Compatibility covers v0.17.1 through v0.17.5; schema 2 has 20 payload files.
- Zenodo record `23385811` is published as v0.17.6 with DOI
  `10.5281/zenodo.23385811` and a related identifier pointing to the exact
  GitHub tag. The general `sleep_scoring` query still omits the renamed
  records; `conceptrecid:21748494` finds this release. The paper's concept DOI
  therefore now resolves to software containing the described calibrator.
- Verification: `release_lightweight.ps1` on the final committed candidate:
  259 Python tests, 51 JavaScript tests, Black, compilation, source smoke,
  schema-2 validation, and fresh v0.17.1 frozen-install update/smoke passed
  with customized settings preserved. Transcript:
  `release_artifacts/v0.17.6_final_gate.log` (ignored). `gh release view`,
  release asset API digests, `git ls-remote --heads/--tags`, and Zenodo records
  API confirmed delivery. Candidate JOSS PDF workflow `38688072216` passed.
  This post-publication documentation follow-up leaves the released runtime,
  manuscript body, and figures unchanged; synchronize it across all three
  branches and verify their final refs before handoff.

### v0.17.6 candidate and branch consolidation (Codex, GPT-6; effort/token budget not reported)

- Maintainer requested a compatible calibration-overfit release and integration
  of the manuscript into `dev` and `main`, with all three branches synchronized.
  GitHub confirms v0.17.5 is still the latest release and predates objective
  commit `7156b93`. Fetched and fast-forwarded `publication` to `4e0c7ea`,
  retaining the latest collaborator revisions and figures.
- Both `dev` and `main` are ancestors of publication. Prepare one v0.17.6
  candidate containing the complete manuscript history, then fast-forward all
  three branches to the tested commit. No PR or history rewriting is needed.
  The automatic update contains only compatible `app_src/` changes; the tag
  also archives the manuscript. Metadata uses the verified 2026-10-15 date.
- Verification: `git fetch origin --prune --tags`, ancestry checks, authenticated
  `gh release list`, and branch-rule inspection completed. Latest manuscript
  PDF workflow `38687390320` passed at `4e0c7ea`. Targeted calibration/layer
  and manuscript-tool tests passed (22 tests); `git diff --check` passed.
  Initial release gate passed 259 Python tests, then Black reformatted one
  list in `paper/figures/build_figures.py`; no figure pixels or behavior changed.
  Rerun the complete gate on the committed formatting correction. Publication,
  remote-ref equality, and Zenodo DOI verification remain pending;
  record their actual results in a follow-up before the final handoff.

## 2026-10-14

### Figure layout alignment (Claude Code, Opus 5.5)

- Figure 1: B and C are sized from the overview's plot area (97–1872 px):
  each 867 px wide (9345–9902 s) with a 41 px gap, B indented so its plot
  starts at A's plot edge while its letter stays under A's. The window still
  covers every B→C change near the label (9378–9888 s); changes outside it
  (9118–9276 s, 10090–10162 s) were outside the previous crop too.
- Panel A blanks Plotly's hover toolbar (camera/pan/zoom icons captured in
  the new snapshot) and moves "Time (s)" below the tick row, where it had
  collided with the 9600 tick.
- Figure 2: the video dialog is scaled (LANCZOS, aspect kept) to panel A's
  body height, so both panels are equal in height; A stays wider.
- Dropped the computed "stage changed from (B)" bars from 1C at the
  maintainer's request: the large B→C changes are visible across the shared
  time axis, and the 1–5 s slivers distracted. C is now titled "Same interval
  after adaptive calibration" and keeps the outline of the labeled interval;
  the caption says C is the same interval as B.
- Cropped the calibration message from 1C (unreadable at print size; the
  caption states its totals). Moved the Figure 1 reference in `paper.md`
  from "The chosen values are displayed", which the figure no longer shows,
  to the sentence about labels as ground truth.
- Rebuilt Figure 1 as three stacked full views (A prediction, B expert
  label, C after calibration) at the maintainer's request: every panel now
  has the spectrogram, which the zoomed side-by-side B/C had dropped. JOSS
  has no page limit, but a figure plus caption must fit one page (~8.3 in of
  figure at 5.4 in wide); ticks only under C, name and colour bar only on A,
  16 px gaps, giving 1973 x 3065 px.
- Figure 2A now shows the same full view as Figure 1 (controls row with
  Check Video, spectrogram, EEG, EMG, NE), cut at 9675 s so it stays large
  beside the dialog. The bout and Check Video are outlined; subplot titles
  are recentred on the cropped plot area. Caption names Check Video.
- Cropped Figure 2A's selection message as well; the caption gives the range.

### Manuscript text review (Claude Code, Opus 5.5)

- Checked `paper.md` claims against the code before editing: 300 s clip
  limit (`clientsideCallbacks.js`), `MAX_SESSIONS = 3`, the 1–7 Hz band, five
  tuned parameters, and the calibration cost (stage-averaged mismatch plus
  0.25 x unlabeled-change fraction, ties to the defaults).
- Applied the maintainer-approved edits: removed the resolved title TODO;
  dated the draft 14 October 2026; "With v0.17.4" instead of "current
  release" (timings predate v0.17.5); "The author" in the AI disclosure;
  plainer Summary and calibration sentences; "(a single labeled interval is
  enough)"; Figure 1 caption notes the 178 s of Wake labels lie outside the
  view; Section 1 opens with a full sentence citing Figure 1A; US
  "unlabeled"; tightened the Software design opener and "Such questions";
  dropped the Windows-update sentence.
- Word count depends on the rule: 1,674 running words, 1,726 with headings,
  ~1,746 with headings and rendered citations. Codex's earlier ~1,748 used
  the strict rule, so headroom is only a few words; docs now give both.
- Replaced the inferred build-vs-contribute rationale in State of the field
  with the users' history: Project 2 previously scored with Viewpoint's
  SleepScore, which did not integrate NE photometry well. Kept the claim to
  photometry because SleepScore's product page lists synchronized video.
  Body now ~1,666 running / ~1,738 strict words.

### Integration delivery and refreshed PDF verified (Codex, GPT-6; effort/token budget not reported)

- Pushed `fix/calibration-overfit` and `dev` at `370a46c`; pushed the
  publication merge at `4cd6277`. Local refs, tracking refs, and targeted
  `git ls-remote --heads` results matched. Publication contains dev as an
  ancestor, with identical application runtime. GitHub accepted the
  maintainer's direct dev push under its configured owner bypass; no PR,
  main update, tag, or release was created.
- Dev CI run `38509728540` and CodeQL run `38509728215` succeeded. JOSS
  draft PDF run `38509732148` succeeded on publication `4cd6277`. Downloaded
  its seven-page artifact to ignored `.pytest_tmp/publication-sync-4cd6277/`,
  rendered all pages with bundled Poppler, and visually checked them.
  Figure 1 is on page 4 with the 228-second calibration breakdown; Figure 2
  is on page 5 with the 96-second selection. Text extraction confirms the
  updated values and no unresolved reference markers. The existing draft's
  small in-app text remains a zoom/readability decision, as noted in README.
- Verification: `gh run view` for all three successful runs; `pdfinfo`,
  `pdftoppm`, and bundled `pypdf` extraction. The runtime matches the tree
  that passed 259 tests. This follow-up changes verification/status notes
  only; manuscript body and figure PNGs are identical to the inspected PDF
  input. `git diff --check` passed before the follow-up commit/push.

### Dev integration and manuscript figure refresh (Codex, GPT-6; effort/token budget not reported)

- Maintainer authorized adding the replacement figures, merging the tested
  calibration branch into `dev`, then bringing `publication` up to date.
  Source assets committed as `370a46c`; local `dev` fast-forwarded to that
  commit, including implementation `7156b93` and cleanup `71a1d83`.
- Publication retains its manuscript history while merging all current dev
  changes. Resolved the citation/cookbook objective wording from dev and
  retained the five replacement PNGs byte-for-byte. Preserved both branches'
  live work-log entries; the five prior live dates are archived together in
  `work_log_archive/work_log_2026-10-08_to_2026-10-13.md` after midnight.
  Existing historical archive snapshots and note stashes remain intact.
- Rebuilt both composed figures for the 1980-by-1202-pixel sources. Crops
  align by shared time ticks, with the correct vertical offsets and complete
  time labels. Source signal pixels are preserved; only whitespace titles,
  callouts, crop boundaries, and message wrapping change in derived panels.
- The current cache confirms 228 labelled seconds: 178 Wake at [2873, 3051)
  and 50 NREM at [9752, 9802). Captions now name that total and correction;
  the video example is [9176, 9272] s, 96 seconds. Updated the paper's Summary,
  calibration description, figure plan, and status notes to match dev.
  The paper explicitly remains ahead of the v0.17.5 archived scorer and needs
  an appropriate release citation before submission; no release requested.
- Verification: merged-tree Python suite: 259 passed. Pinned Black passed;
  both derived PNGs visually inspected, including the corrected axis labels.
  Private `check_publication_refresh.py` verifies all five source hashes,
  PNG integrity, both figure paths, 12 citation keys, YAML metadata, absence
  of conflict markers, and an identical `app_src/` tree to dev. Body word
  count is approximately 1,748 including captions. `git diff --check` and
  cached-diff checks passed. Raw latency runs remain ignored on publication.
- Remote delivery and the new JOSS PDF workflow are verified after the merge
  commit; record the resulting refs/render outcome in the follow-up below.

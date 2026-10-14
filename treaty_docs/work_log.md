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

## 2026-10-14

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

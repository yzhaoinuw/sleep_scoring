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

## 2026-10-13

### Replacement manuscript snapshots prepared for integration (Codex, GPT-6; effort/token budget not reported)

- Maintainer supplied five replacement `app_figure*.png` snapshots and
  authorized adding them, merging `fix/calibration-overfit` into `dev`, then
  synchronizing `publication` from `dev`. No PR or release requested.
- Visually inspected all five and verified PNG integrity and hashes. Sources
  are 1980 by 1202 pixels. The calibration screenshot reports 228 labelled
  seconds; selection and video snapshots show [9176, 9272] s (96 seconds).
  Publication's existing composites/captions use old screenshot coordinates,
  a 358-second calibration total, and a different 13-second video selection;
  refresh those derived assets when synchronizing that branch.
- Raw `paper/latency_runs/` data and preserved Treaty-note stashes remain
  outside this asset commit. The calibration implementation passed 250 Python
  tests and startup smoke; the subsequent import cleanup passed 50 relevant
  tests and pinned Black, as recorded above.
- Verification: `git fetch origin dev publication fix/calibration-overfit`,
  targeted branch history, visual inspection, and private
  `python .pytest_tmp/check_manuscript_assets.py` in the project environment.
  `git diff --check` passed before committing the five source PNGs.

### Calibration dead-code audit (Codex, GPT-6; effort/token budget not reported)

- Maintainer reports smoother behavior with the new objective and requested a
  follow-up check for code left unused by the change. Traced the prediction
  callback through inference, calibration helpers, constants, and candidate
  selection; reviewed the changed tests and current documentation.
- The old tolerance constant, mismatch-count helpers, selector, and tolerance
  arguments have no remaining references in tracked Python. AST/reference
  inspection found no unused imports in the changed module/tests, unread
  calibration locals, or unreferenced top-level runtime helpers after removing
  one pre-existing unused `STAGE_COLORS` import from the statistical module.
- Private `experiment_wake33.py` still duplicated the prototype objective and
  patched a deleted selector. Removed that duplicate and use the production
  calibrator with a temporary weight override instead. All 15 private
  comparisons reran successfully; snapshots and recordings were preserved.
  Historical before/after source snapshots remain intentional diagnostic data.
- Verification: in `sleep_scoring_dash3.0`, ran the private AST audit
  `.pytest_tmp/audit_calibration_references.py`, cleaned experiment script,
  and `python -m pytest tests/test_sleep_score_layers.py
  tests/test_app_helpers.py tests/test_smoke.py --basetemp
  .pytest_tmp\calibration-dead-code-audit -p no:cacheprovider -q`: 50 passed.
  Pinned Black and `git diff --check` passed. The previously verified 250-test
  suite was not repeated for this unused-import cleanup. Changes stay local
  on `fix/calibration-overfit`.

### Calibration change penalty implemented on reused fix branch (Codex, GPT-6; effort/token budget not reported)

- Maintainer approved the stage-balanced label-error plus unlabelled-change
  objective and requested reuse of `fix/calibration-overfit`. Created its
  local tracking checkout from the existing remote branch, then fast-forwarded
  it to current `dev` (`9c823bb`) before editing. Dirty dev Treaty notes are
  preserved in stash `9d90a4348942ceefa5e8a66794ec0494149a4f1e`, named
  `Preserve dev notes before renewed calibration-overfit work`; private paper
  figures, latency runs, and recordings were left in place.
- Replaced hard 10% eligibility with mean per-labelled-stage mismatch rate
  plus `0.25` times the fraction of unlabelled predictions changed from a
  freshly computed default result. Default parameter distance breaks ties.
  Features and candidate costs are reused within a call; each call starts
  from defaults. Manual nonnegative overrides, including MA, are outside
  both calibration targets and the change penalty; negative missing-score
  sentinels remain unlabelled. Manual overlay is unchanged.
- Exact cached 33-Wake-second example [9582, 9615) now chooses threshold 0.75
  and minimum Wake duration 15 s: raw counts 975 Wake / 8,441 NREM / 884 REM,
  740 changed seconds, 88.97% agreement with the stored local reference
  instead of 17.61%. Six raw misses remain, but all 33 manual labels are
  preserved after overlay. The earlier 521-Wake-label case retains 0.85
  with 86.65% raw reference agreement (previous 86.47%). Reference labels
  are evaluation only, never inputs to candidate selection.
- Added public real-predictor regression coverage for nonmonotonic
  Wake-to-REM promotion and recording-wide collapse. Stage balancing,
  fully labelled corrections, manual override exclusion, missing sentinels,
  one real Wake label, repeatability, and label immutability are covered.
  Two key regressions fail on the old implementation and pass now.
- Updated README, cookbook, citation abstract, and Unreleased changelog for
  the new behavior. Version/date metadata remain v0.17.5; this is local
  feature work, not a published release. Defaults remain an imperfect prior;
  broader recording and interactive validation are still needed. The
  two-pass coordinate search can miss changes requiring joint knob moves.
- Delivery: changes committed locally on `fix/calibration-overfit` for
  maintainer testing. No new PR, push, merge, or release requested.
- Verification: activated `sleep_scoring_dash3.0` through `conda run`;
  `python -m pytest --basetemp .pytest_tmp\calibration-objective-full
  -p no:cacheprovider -q`: 250 passed. Pinned `pre-commit run black
  --all-files` passed using `.pre-commit-codex/atomic-save` after formatting
  the test file. `python run_desktop_app.py --smoke` passed. Private
  `python user_test_files/verify_calibration_objective.py` checked old/new
  regressions, cached and earlier label sets, and exact manual overlay;
  results are in `user_test_files/wake33_diagnostics/implemented_results.json`.
  `git diff --check` passed. Live app cache and recording files were not written.

### v0.17.5 partial release published (Codex, GPT-6; effort/token budget not reported)

- Maintainer requested a partial release so the manuscript describes shipped
  behavior. Candidate includes the per-stage calibration tolerance from PR #17
  and annotation-save protection from local dev commit `a6932cc`.
- Reviewed PR #18: it copies only the publication title/abstract into
  `CITATION.cff`; all seven GitHub checks passed. Maintainer explicitly
  authorized the administrator override after the required-review policy
  blocked normal merging. PR merged as `715fabe` and its official history is
  included in this final candidate; abstract wording matches calibration.
- Aligned app/setup/CFF version 0.17.5 and release date 2026-10-13. Runtime
  changes are confined to compatible `app_src/` code; no frozen dependency,
  launcher, model, or packaging change. Supported bases are v0.17.1–v0.17.4.
- Save verification covers complete sibling staging and retained recovery
  exports on destination failure. Native shared-drive semantics still need
  maintainer testing; filesystem durability and the user's original truncation
  timing are not established by the failure-injection regressions.
- Rotated the prior five-date log chunk to its archive; existing local source
  checkout log edits and manuscript/latency artifacts remain preserved.
- Verification:
  - `Get-Date -Format yyyy-MM-dd`: 2026-10-13. PR #18 diff changes title and
    abstract only; all seven checks succeeded on `ccfda9d`.
  - Initial candidate `b554dd5` passed the full lightweight gate: 244 Python
    tests, 51 JavaScript tests, Black, compilation, source smoke, schema-2
    validation (20 payload files), and a fresh v0.17.1 frozen update/smoke
    preserving customized settings. Its ZIP SHA-256 was
    `204DA779C648E978552F16A3AF38C2FC6897F0210FDADFA4CBD02C126BE559C4`.
  - Final payload `9113f6532cf901beaa103dc61fef77634f1bfceb` passed the
    complete lightweight gate after the official PR merge: 244 Python tests,
    51 JavaScript tests, Black, compilation, source smoke, schema-2 validation
    (20 files), and a fresh frozen v0.17.1 update/smoke preserving settings.
    Final ZIP SHA-256:
    `31644A007CF638B113F0E4CE0A8751C0140E50FAA0612DD77EF95393662F73D1`.
  - GitHub CI run `38448938238` and CodeQL run `38448937619` both succeeded
    on that exact payload. Annotated tag v0.17.5 is pushed and its peeled
    commit is `9113f65`; `main` includes the documentation verification
    follow-up. Synchronize `dev` with this documentation after publication.
  - Published GitHub release v0.17.5 at 2026-10-13T18:48:36Z. GitHub reports
    the ZIP SHA-256 above and checksum-file SHA-256
    `F5F66229D08976AA8404637B48C6488CD4364C19493AD5EF44BFDD600935F45E`,
    both matching local files. Only the update ZIP/checksum were attached.
  - Zenodo record `23349241` confirms version v0.17.5, the tagged GitHub
    source, the new Sleep Scoring App title, version DOI
    `10.5281/zenodo.23349241`, and concept DOI `10.5281/zenodo.21748494`.
    The required `q=sleep_scoring` search omitted the renamed record; querying
    `q=conceptrecid:21748494&all_versions=true` found it. Use that stable
    concept query for future verification. The released webhook returned 202;
    the published event's 409 said the release had already been received.
    No recreation or redelivery was needed.

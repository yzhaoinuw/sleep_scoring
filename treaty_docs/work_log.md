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

## 2026-10-20

### SPINDLE mentioned once in State of the field (Claude Code, Claude Opus 5.5; effort/token budget not reported)

- Maintainer judged three SPINDLE mentions too prominent. The setup sentence
  now says every open tool above requires either setup in a programming
  environment or uploading recordings to a web service; "open" excludes the
  commercial SleepScore named just before it. The first mention adds "now as
  a web service" so readers can tell which tool uploads. Not "paid": rodent
  scoring is free up to a weekly quota (see the State-of-the-field comment).
- Research impact statement: the repository "links to" the Zenodo archive
  and example dataset rather than "provides" them, since neither lives in
  the repository; the README's Example Data section is the link.
- Replaced the review PDF with the `cbf893e` JOSS draft artifact (7 pages,
  checked). Net body word count is about unchanged.

### Example dataset cited in the manuscript (Claude Code, Claude Opus 5.5; effort/token budget not reported)

- Fast-forwarded `publication` to the dataset-link commit on `dev`.
- Cited the dataset in the Research impact statement through its concept
  DOI `10.5281/zenodo.23489641` (`@dataset` entry, matching how the software
  archive is cited by concept DOI). Offset the added words in the same
  paragraph, so the body count is unchanged.
- Maintainer confirmed all four "especially thank" users shared recordings
  for testing; added that to their sentence. The dataset provider is among
  them, so she is not named a second time.
- Replaced the review PDF with the `567929d` JOSS draft artifact (7 pages;
  citation and reference render checked).
- Verification: local pandoc citeproc render, JOSS draft workflow passed,
  `git diff --check`.
- Maintainer confirmed the files downloaded from Zenodo work in the app,
  closing the desktop-walkthrough gap noted below. Independent event
  alignment remains unverified.

### Public example dataset verified and linked (Codex, GPT-6; effort/token budget not reported)

- Maintainer published Zenodo dataset v1.0.0, record 23489642, version DOI
  `10.5281/zenodo.23489642`, concept DOI `10.5281/zenodo.23489641`.
  The dataset is public under CC BY 4.0, credits Yue Zhao as creator and
  Klaudia Tokarska (University of Copenhagen, ORCID 0000-0002-6397-377X)
  as DataCollector, and links the software concept DOI as a supplement.
  The description retains the requested reference-label wording.
- Verified all six published file sizes and API MD5 checksums against the
  staged copies, including the cleaned README and refreshed checksum list.
  DOI resolution follows through to the public record with HTTP 200.
  Zenodo was inaccessible through the web reader; direct curl API retrieval
  succeeded. The API snapshot is retained inside the ignored staging folder.
- Added the example-data entry point and MP4 instructions to the user README;
  moved the paper's public-example item into What is done. No manuscript-body,
  app-version, package, tag, or software-archive metadata changes are needed.
  Full desktop walkthrough and independent event alignment remain unverified.
- Verification: public Zenodo records API, local MD5 comparison for all six
  files, DOI HTTP HEAD/redirect check, documentation diff review, and
  `git diff --check`. Dataset publication was performed by the maintainer.

### Zenodo example selection and AVI playback diagnosis (Codex, GPT-6; effort/token budget not reported)

- Maintainer selected `user_test_files/35_app13.mat` and accompanying
  `35_ymaze_ymaze_Cam2.avi`, with `35_app13_groundtruth.mat` as an optional
  illustrative reference. Credit Klaudia Anna Tokarska for providing the
  recording/video. The scorer is not identified; follow the maintainer's
  requested note that labels follow the manuscript rules and may differ from
  another expert's interpretation; do not describe them as potentially
  erroneous. CC BY 4.0 approved.
- EEG/EMG: 6,286,400 samples at 610.3515625 Hz, 10,299.63776 s. NE:
  104,774 samples at 10.172526245117188 Hz, 10,299.70309 s. Both MATs have
  identical EEG/EMG/NE arrays; the main file is unscored, while the reference
  has 10,300 labels (Wake 1,805, NREM 7,730, REM 765). Saved start/video
  offsets are zero; duration agreement does not prove event synchronization.
- AVI reports 10,296.50 s, about 3.14 s shorter than EEG/EMG. FFmpeg decoded
  all 102,965 frames without warnings/errors. Windows Media Foundation reads
  its native `avc1` subtype but cannot negotiate NV12 decoding (0xC00D5212).
  An MP4 remux maps to H264 and decodes successfully through Windows.
  The app plays it because make_mp4_clip uses FFmpeg to encode MP4 snippets.
- Prepared separate copies under ignored `user_test_files/zenodo_example_staging/`:
  both MATs, a no-reencoding MP4, draft guide, CC BY license, checksum list,
  and draft deposit metadata. Original data/video remain unchanged; nothing
  has been uploaded or published. The app's picker already accepts MP4.
- All 102,965 ordered decoded-frame checksums match between AVI and MP4.
  Compressed-stream hashes differed across containers; use decoded content
  to assess the remux. Initial SHA-256 frame hashing exceeded 300 s; the
  complete Murmur3-128 per-frame comparison passed with matching SHA-256
  aggregates. FFmpeg normalizes the MP4's 0.1 s start unless copyts is set:
  both stored first-frame timestamps are 0.1 s. Do not silently retime data
  based on the normalized probe difference. MP4 reports 10,296.40 s.
- Rotated the previous five unique dates into the 2026-10-14_to_2026-10-19
  archive when starting this date, per the work-log convention.
- Verification: SciPy loadmat/whosmat plus array equality/nonfinite checks;
  original AVI full FFmpeg decode; all-frame content comparison;
  Windows Media Foundation source-reader/decoder probes of AVI and MP4;
  source/copy SHA-256 checks; app video/dialog code inspection;
  `git diff --check`. No desktop walkthrough or independent behavioral
  alignment check was performed. Publication and final metadata remain open.

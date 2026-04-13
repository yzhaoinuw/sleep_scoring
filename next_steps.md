# Next Steps

Use this checklist alongside [codex_work_log.md](C:\Users\yzhao\python_projects\sleep_scoring\codex_work_log.md).

Quick workflow:

1. Read the newest entry at the top of `codex_work_log.md`.
2. Check this file for the next incomplete item.
3. When work is finished, update both files so the current state stays easy to recover.

## Beta Direction

- Keep the existing app review flow.
- Do not build extra beta review UI for now.
- Only write ChatGPT scores where the model is confident enough.
- Leave very uncertain intervals as `Unscored` so the current app workflow can handle manual review.
- Make the confidence threshold user-configurable.

## Current State

- `app_src/chatgpt_tools.py` helper functions are implemented.
- `tests/test_chatgpt_tools.py` covers the helper contracts.
- `app_src/chatgpt_scoring_guidance.md` now contains the active tightened scoring guidance used by the ChatGPT backend.
- `app_src/chatgpt_inference.py` now performs a first real coarse ChatGPT overview pass.
- `app_src/chatgpt_inference.py` now also performs targeted local refinement with zoom snapshots and interval features for uncertain, low-confidence, and transition-heavy intervals.
- The ChatGPT backend now falls back safely when the SDK, API key, snapshot export, or structured output path is unavailable.
- Confidence thresholding is supported through `app_src/config.py` and the backend inference path.
- The app UI now treats ChatGPT as a real backend and reports readiness when the local SDK or API key is missing.
- A live ChatGPT run still requires the local `openai` package to be installed, but the app now auto-loads `OPENAI_API_KEY` from a repo-local `.env` file.
- A first live ChatGPT scoring run completed successfully through the app, but the sleep-stage quality is not yet good enough for beta use.
- The guidance prompt in `app_src/chatgpt_scoring_guidance.md` has been tightened so the model now receives explicit EEG spectrogram first, NE second, and EMG last guidance, along with concise transition and uncertainty instructions and no extra dev notes.
- Optional ChatGPT trace logging is now available through `CHATGPT_SHOW_THOUGHTS` in `app_src/config.py`, writing a `.txt` trace beside the snapshot images for each enabled run.
- The ChatGPT trace file now focuses on model-visible summaries and labeled block outputs instead of dumping the full prompt payload.
- ChatGPT snapshot and trace filenames are now deterministic and human-readable instead of UUID-heavy.
- ChatGPT overview and zoom snapshot titles now use the source `.mat` stem plus the exported interval bounds.
- The overview figure now requests denser major x-axis labels and has temporary hour-mark minor ticks disabled to make the snapshot easier for the vision model to read.
- ChatGPT refinement is now configurable through `CHATGPT_REFINEMENT_MODE`, and the current default is overview-only so the model is forced to prioritize coarse global structure first.
- A four-section broad-refinement mode is now implemented and ready to test next by switching `CHATGPT_REFINEMENT_MODE` to `fixed_sections`.

## Beta Checklist

- [x] Implement the real ChatGPT orchestration in `app_src/chatgpt_inference.py`.
- [x] Load the guidance prompt from `app_src/chatgpt_scoring_guidance.md`.
- [x] Create and attach a deterministic overview snapshot for coarse scoring.
- [x] Define a strict structured response format for contiguous bout output.
- [x] Parse model output into validated `start_s` / `end_s` / `state` blocks.
- [x] Convert validated blocks into per-second sleep scores.
- [x] Support leaving uncertain intervals unscored instead of forcing a label.
- [x] Add confidence handling that maps model certainty into per-second confidence values.
- [x] Add a user-configurable confidence threshold for score writeback.
- [x] Skip writeback below the threshold and preserve those intervals as unchanged or `Unscored`.
- [x] Add targeted zoom-snapshot and interval-feature follow-up calls for uncertain or transition-heavy intervals.
- [x] Add API configuration plumbing for model name and credentials.
- [ ] Add timeout, retry, and failure handling for API calls.
- [x] Add safe fallback behavior when the ChatGPT request fails or returns invalid output.
- [x] Replace placeholder UI messages once real inference is wired.
- [x] Add end-to-end tests for ChatGPT inference with mocked API responses.
- [ ] Add fixture cases covering confident scoring, uncertain intervals, malformed responses, and API failures.
- [ ] Build a small evaluation set of representative `.mat` files for beta validation.
- [ ] Measure agreement with expected labels and review the main failure modes before beta.
- [x] Tighten the prompt with clearer sleep-stage heuristics and uncertainty instructions.
- [ ] Investigate why the live ChatGPT scores are poor and identify the highest-leverage improvements.
- [x] Add optional prediction trace logging so the model can write its visible reasoning summaries, observations, and actions to a `.txt` file during sleep-score generation for debugging.

## Suggested Build Order

### Milestone 1: Working Alpha

- [x] Wire `chatgpt_inference.py` to produce real coarse scores from the overview snapshot.
- [x] Validate and write back structured bout output.
- [x] Leave uncertain regions unscored.
- [x] Add mocked end-to-end tests for the basic flow.

### Milestone 2: Beta Hardening

- [x] Add configurable confidence thresholding.
- [x] Add targeted refinement with zoom snapshots and interval features.
- [ ] Improve retry/failure handling and fallback behavior.
- [x] Replace placeholder UI text.
- [ ] Run evaluation on a small representative dataset.
- [x] Add debug logging for live ChatGPT prediction traces.
- [ ] Use the new traces to diagnose scoring mistakes.
- [ ] Add a compact top-of-trace summary of why each interval was selected for refinement.
- [ ] Evaluate the new overview-only default against the previous adaptive-refinement behavior on a small representative set.
- [ ] If overview-only helps global structure but still misses REM, test `fixed_sections` refinement next.
- [ ] If REM remains weak after the refinement change, add a small curated set of REM-vs-quiet-Wake ground-truth exemplar images.

## Notes

- Keep transition rules primarily in the model guidance unless a clear deterministic post-rule becomes necessary.
- Prefer biologically plausible bout structure over local mechanical rewrites.
- Favor a robust, testable structured-output path before adding more prompt complexity.

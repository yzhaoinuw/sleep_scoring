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
- Keep the backend support for unscored / unchanged intervals available, but note that the current prompt experiment now defaults mixed evidence to `NREM` instead of relying on uncertain intervals.
- Make the confidence threshold user-configurable.

## Immediate Next Experiments

- Speed-quality tradeoff experiments for the improved-but-slower ChatGPT path:
  - test the current zero-shot fixed-zoom-section setup with no overview image and no reference examples
  - compare the current `high`-effort reference-pack setup against `medium` effort
  - test sending fewer reference images, such as overview + the strongest 2 zoom examples instead of all 5 images
  - tighten the guidance prompt further so fewer examples may be needed to teach the same visual cues
  - consider a two-tier strategy: keep examples on the coarse pass only, but reduce or skip refinement windows when the coarse output already looks confident
- Revisit whether sending all overview + zoom images in a single request is still worthwhile now that the curated example pack already improves recall, or whether it will just add latency and token cost without enough quality gain.

## Current State

- `app_src/chatgpt_tools.py` helper functions are implemented.
- `tests/test_chatgpt_tools.py` covers the helper contracts.
- `app_src/chatgpt_scoring_guidance.md` now contains the active concise zero-shot scoring guidance used by the ChatGPT backend.
- `app_src/chatgpt_inference.py` supports both the older coarse ChatGPT overview pass and the current fixed zoom-section-only experiment.
- The current default skips the full-recording overview image and scores fixed broad zoomed sections directly.
- The ChatGPT backend now falls back safely when the SDK, API key, snapshot export, or structured output path is unavailable.
- Confidence thresholding is supported through `app_src/config.py` and the backend inference path.
- The app UI now treats ChatGPT as a real backend and reports readiness when the local SDK or API key is missing.
- A live ChatGPT run still requires the local `openai` package to be installed, but the app now auto-loads `OPENAI_API_KEY` from a repo-local `.env` file.
- A first live ChatGPT scoring run completed successfully through the app, but the sleep-stage quality is not yet good enough for beta use.
- The guidance prompt in `app_src/chatgpt_scoring_guidance.md` has been revised around the current `NREM -> Wake -> REM` workflow for the focused two-panel image.
- The current prompt experiment now defaults weak or mixed evidence to `NREM` instead of explicitly asking the model to produce uncertain intervals.
- Optional ChatGPT trace logging is now available through `CHATGPT_SHOW_THOUGHTS` in `app_src/config.py`, writing a `.txt` trace beside the snapshot images for each enabled run.
- The ChatGPT trace file now focuses on model-visible summaries and labeled block outputs instead of dumping the full prompt payload.
- ChatGPT snapshot and trace filenames are now deterministic and human-readable instead of UUID-heavy.
- ChatGPT overview and zoom snapshot titles now use the source `.mat` stem plus the exported interval bounds.
- The model-facing ChatGPT figure is now isolated in `app_src/make_figure_chatgpt.py` and uses a focused two-panel layout with EEG spectrogram on top and NE on the bottom, while the app UI figure stays unchanged.
- The user-facing spectrogram frequency axis now spans `0-30 Hz`, while the ChatGPT model-facing export uses a tighter `0-15 Hz` range.
- ChatGPT refinement is now configurable through `CHATGPT_REFINEMENT_MODE`, and the current default is `fixed_sections`.
- Non-image helper metadata has been disabled for refinement, so the model now scores from the attached images only.
- A backend-only `vision_figure_mode` comparison hook now exists so `focused` and `full` model-facing image layouts can be compared without changing the app UI.
- The bundled ground-truth reference example pack still exists under `app_src/assets/chatgpt_reference_examples`, but reference examples are currently disabled by default for zero-shot testing.
- ChatGPT reasoning effort is now explicitly configurable, and the current default is `high`.
- The thoughts trace file now includes per-call token usage and estimated cost when the Responses API returns usage data.

## Beta Checklist

- [x] Implement the real ChatGPT orchestration in `app_src/chatgpt_inference.py`.
- [x] Load the guidance prompt from `app_src/chatgpt_scoring_guidance.md`.
- [x] Create and attach a deterministic overview snapshot for coarse scoring.
- [x] Define a strict structured response format for contiguous bout output.
- [x] Parse model output into validated `start_s` / `end_s` / `state` blocks.
- [x] Convert validated blocks into per-second sleep scores.
- [x] Support leaving uncertain intervals unscored instead of forcing a label when the prompt/backend choose to use that path.
- [x] Add confidence handling that maps model certainty into per-second confidence values.
- [x] Add a user-configurable confidence threshold for score writeback.
- [x] Skip writeback below the threshold and preserve those intervals as unchanged or `Unscored`.
- [x] Add targeted zoom-snapshot follow-up calls for uncertain or transition-heavy intervals.
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
- [x] Add curated ground-truth exemplar images and attach them to the coarse pass.
- [ ] Reduce latency and token cost without losing the current quality gains from the example pack.
- [ ] Decide whether the current coarse-pass example pack should be trimmed, reordered, or partially removed.
- [ ] Re-test a single-request multi-image strategy only if the leaner prompt/example experiments still miss obvious bouts.

## Suggested Build Order

### Milestone 1: Working Alpha

- [x] Wire `chatgpt_inference.py` to produce real coarse scores from the overview snapshot.
- [x] Validate and write back structured bout output.
- [x] Leave uncertain regions unscored.
- [x] Add mocked end-to-end tests for the basic flow.

### Milestone 2: Beta Hardening

- [x] Add configurable confidence thresholding.
- [x] Add targeted refinement with zoom snapshots.
- [ ] Improve retry/failure handling and fallback behavior.
- [x] Replace placeholder UI text.
- [ ] Run evaluation on a small representative dataset.
- [x] Add debug logging for live ChatGPT prediction traces.
- [ ] Use the new traces to diagnose scoring mistakes.
- [ ] Add a compact top-of-trace summary of why each interval was selected for refinement.
- [ ] Evaluate the new overview-only default against the previous adaptive-refinement behavior on a small representative set.
- [ ] If overview-only helps global structure but still misses REM, test `fixed_sections` refinement next.
- [ ] Add a small curated set of REM-vs-quiet-Wake ground-truth exemplar images.
- [ ] Test whether a smaller curated example pack can preserve the new quality gains with lower latency.
- [ ] Compare `high` versus `medium` reasoning effort on the same representative recordings.
- [ ] If speed is still unacceptable, test whether a single-request overview-plus-zooms path helps enough to justify the extra token load.

## Notes

- Keep transition rules primarily in the model guidance unless a clear deterministic post-rule becomes necessary.
- Prefer biologically plausible bout structure over local mechanical rewrites.
- Favor a robust, testable structured-output path before adding more prompt complexity.

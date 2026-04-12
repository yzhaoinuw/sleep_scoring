# Codex Work Log

Prepend new session notes to the top of this file.

## 2026-04-11

### Done Today

- Replaced the placeholder ChatGPT backend in `app_src/chatgpt_inference.py` with a first real coarse-scoring orchestration path.
- Added backend support to:
  - load the guidance prompt from `app_src/chatgpt_scoring_guidance.md`
  - render and attach an overview snapshot
  - request structured contiguous bout output from the Responses API
  - validate non-overlapping bout intervals and allowed stage labels
  - write back only high-confidence intervals
  - preserve uncertain regions as unchanged / unscored
- Added safe fallback behavior so the app keeps working when the OpenAI SDK, API key, snapshot export, or structured model output path is unavailable.
- Added backend-level confidence-threshold support through:
  - the `infer(..., confidence_threshold=...)` argument
  - the `SLEEP_SCORING_CHATGPT_CONFIDENCE_THRESHOLD` environment variable
- Moved the active confidence-threshold setting into `app_src/config.py` as `CHATGPT_CONFIDENCE_THRESHOLD`.
- Added `CHATGPT_MODEL` to `app_src/config.py` so the active ChatGPT model is configured in one place.
- Updated `app_src/inference.py` to use the shared config threshold for the ChatGPT backend.
- Extended `app_src/chatgpt_inference.py` with a bounded local refinement pass that:
  - identifies uncertain, low-confidence, and transition-heavy intervals from the coarse pass
  - exports zoomed interval snapshots
  - attaches numeric interval features and current local scores
  - requests local structured refinement without allowing edits outside the requested window
  - overlays confident local updates while preserving unresolved regions
- Expanded `tests/test_inference_scaffold.py` to cover the new refinement path, including a mocked second-pass call that refines an uncertain middle interval with zoom-image plus helper-context input.
- Added an `AI Model Cost Estimate` section to `README.md` under `Developer Notes`, documenting the current overview-image token estimate and rough per-image cost for `gpt-5-mini`.
- Added `ChatGPT Backend Setup` notes to `README.md`, documenting the `openai` package and `OPENAI_API_KEY` requirements.
- Replaced the remaining placeholder app text in `app_src/app_dev.py` and `app_src/components_dev.py` so the UI presents ChatGPT as a real backend.
- Added a ChatGPT-backend readiness helper in `app_src/chatgpt_inference.py` so the app can report missing SDK / API-key setup before the user starts prediction.
- Added `openai>=1.0.0` to `requirements.txt`.
- Added lightweight local `.env` loading in `app_src/env_loader.py` and wired it into `app_src/__init__.py` so the app can auto-load `OPENAI_API_KEY` on startup.
- Added `.env.example` at the repo root and updated `README.md` so users can configure ChatGPT by copying that file to `.env`.
- Expanded `tests/test_inference_scaffold.py` into mocked end-to-end coverage for:
  - confident bout writeback with uncertain intervals left unscored
  - backend readiness status checks
  - fallback when no OpenAI client is available
  - fallback when the model returns invalid overlapping bouts
- Added `tests/test_env_loader.py` to cover local `.env` parsing and non-overriding behavior.
- Verified that a first live ChatGPT scoring run now works end to end through the app.
- Observed that live sleep-stage quality is still weak, so the next work should focus on diagnosing why the model is making poor scoring decisions and improving the prompt / refinement behavior.
- Updated `next_steps.md` with a new debugging direction:
  - investigate the live scoring failure modes
  - add optional `.txt` trace logging of model reasoning / actions during prediction to make debugging easier
- Updated `next_steps.md` to reflect the completed alpha/backend items and the remaining beta work.

### Verification

- Ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m pytest tests/test_inference_scaffold.py tests/test_chatgpt_tools.py`
- Result: 17 tests passed
- Ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m pytest tests/test_inference_scaffold.py tests/test_smoke.py`
- Result: 13 tests passed
- Ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m pytest tests/test_env_loader.py tests/test_inference_scaffold.py tests/test_smoke.py`
- Result: 17 tests passed
- Note: pytest still emitted the existing repo-local cache permission warning, but the test run completed successfully.

## 2026-04-10

### Done Today

- Drafted the first ChatGPT sleep-scoring guidance prompt in `app_src/chatgpt_scoring_guidance.md`.
- Referenced the draft guidance prompt from `app_src/chatgpt_inference.py` and `project_overview.md` so future wiring work can pick it up quickly.
- Implemented `get_current_scores()` in `app_src/chatgpt_tools.py`.
- Added per-second score export for a requested interval with:
  - clamped floor/ceil interval handling
  - explicit `Wake` / `NREM` / `REM` / `Unscored` labels
  - contiguous block summaries for prompt-friendly interval refinement
  - score counts and dominant-state summary
- Added targeted tests for:
  - mixed-state interval slicing
  - clamped intervals with missing labels
  - invalid interval validation
- Implemented the remaining `chatgpt_tools.py` helpers:
  - `set_scores_block()` for half-open contiguous score edits
  - `apply_transition_rules()` as a non-destructive pass-through so transition rules stay in the model guidance prompt
  - `mark_uncertain_interval()` with whole-second normalization, duration, and validation
- Expanded `tests/test_chatgpt_tools.py` to cover the new helper contracts.

### Verification

- Ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m pytest tests/test_chatgpt_tools.py`
- Result: 13 tests passed

## 2026-04-09

### Done Today

- Drafted `get_interval_features()` in `app_src/chatgpt_tools.py`.
- Updated the helper to accept an optional `fig` so spectral summaries can come from the rendered Plotly figure instead of recomputing by default.
- Added interval summaries for:
  - spectrogram delta/theta power and theta-delta ratio
  - EMG amplitude and burst counts
  - current score counts / dominant state
  - NE mean/std/slope/drop when NE is present
- Kept a fallback path that recomputes spectrogram/theta-delta traces if no figure is supplied.
- Added targeted tests in `tests/test_chatgpt_tools.py`.

### Verification

- Ran `C:\Users\yzhao\miniconda3\envs\sleep_scoring_dash3.0\python.exe -m pytest tests/test_chatgpt_tools.py`
- Result: 2 tests passed

## 2026-04-08

### Planned Today

- Add a `ChatGPT` prediction path alongside `SDreamer` without redesigning the current app UI.
- Keep the first integration lightweight by scaffolding backend hooks and placeholder helper APIs.
- Use visualization snapshots plus app-exposed helper functions for coarse scoring and targeted refinement.
- Keep ChatGPT-generated images in an app temp workspace instead of under Dash `assets/`.
- Commit and push the in-progress scaffold on a dedicated feature branch.

### Done Today

- Created branch `codex/chatgpt`.
- Added a prediction-backend dropdown to the active dev UI so users can choose `SDreamer` or `ChatGPT (placeholder)`.
- Updated the active prediction flow to route by backend and allow the app to load even when optional SDreamer ML dependencies are missing.
- Added `app_src/chatgpt_inference.py` as the placeholder ChatGPT backend.
- Added `app_src/chatgpt_tools.py` with typed helper contracts for:
  - overview snapshot export
  - zoom snapshot export
  - interval feature lookup
  - current-score lookup
  - block score writeback
  - transition-rule cleanup
  - uncertain-interval marking
- Implemented `capture_overview_snapshot()` with fixed-size PNG export and parent-directory creation.
- Implemented `capture_zoom_snapshot()` as a hidden export path that deep-copies the `FigureResampler`, applies a resampled zoomed interval, and writes a PNG without changing the live app figure.
- Verified `capture_zoom_snapshot()` on `user_test_files/115_gs.mat`; the exported image worked and the source figure range stayed unchanged.
- Moved the default ChatGPT snapshot location to the app temp workspace:
  - `%TEMP%\\sleep_scoring_app_data\\chatgpt_snapshots`
- Added smoke/scaffold tests for the new inference modules.

### Current State

- The ChatGPT path is scaffolded but still returns placeholder predictions.
- Snapshot export helpers are in place, but the OpenAI API call path is not wired yet.
- The next implementation step is to connect `chatgpt_inference.py` to:
  - overview snapshot creation
  - targeted zoom snapshot requests
  - interval feature helpers
  - score writeback helpers

### Notes

- Hidden zoom export should remain separate from any user-visible graph zoom behavior.
- In this sandbox, Kaleido image export needed elevated permissions because of temp-file restrictions; the app itself should use a writable temp location during normal runtime.

# Codex Work Log

Prepend new session notes to the top of this file.

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

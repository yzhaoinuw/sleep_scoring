# Interactive Visualization App Cookbook

A recipe book for building Dash + Plotly desktop apps that let a user **navigate, inspect,
and annotate long time series** fluidly — zoom, pan, drag-select, auto-pan, keypress
labeling, undo, and video/side-panel integration.

The **Sleep Scoring App** in this repo is the reference implementation. Every recipe points
at the exact file and the exact idea, then tells you how to generalize it to a different
domain (ECG, audio, seismology, financial ticks, sensor logs, anything with a long x-axis
and one or more channels).

## Who this is for

- **Agents** building or modifying a similar app. Read the [Recipe Index](#recipe-index)
  first, then pull only the recipes the task needs. Each recipe lists its dependencies and
  its source-of-truth files so you can jump straight to code.
- **Human app designers** who want the design rationale, not just the code. Each recipe has a
  *Why it's built this way* note explaining the tradeoff, because most of these choices exist
  to fight one enemy: **latency on large signals**.

## How to read a recipe

Every recipe follows the same shape:

| Field | Meaning |
| --- | --- |
| **Goal** | The user-visible behavior it delivers. |
| **Depends on** | Other recipes it needs. Skip a recipe only if you also skip its dependents. |
| **Source** | The reference-app files that implement it. |
| **Mechanism** | How it actually works. |
| **Adapt** | What to change for a different domain. |
| **Gotchas** | The non-obvious traps. |

---

## The one big idea

> **Browser-authoritative interaction, server-authoritative data.**

Everything that must feel *instant* (mode switch, selection box, keypress label, panning the
view) happens **in the browser** — either as a Dash *clientside callback* (JavaScript in
`app_src/assets/clientsideCallbacks.js`, registered from Python via `ClientsideFunction`) or
as a standalone *asset script* under `app_src/assets/`. The Python
server is only invoked when real data work is unavoidable: loading a file, running a model,
resampling a hi-res signal to the pixels currently on screen, saving.

This split is the reason the app stays responsive on multi-hour recordings with millions of
samples. Keep it. If you find yourself round-tripping to the server for a hover, a keypress,
or a drag frame, you are on the slow path.

The three layers, top to bottom:

```
┌─────────────────────────────────────────────────────────────────┐
│  pywebview native window  (run_desktop_app.py)                    │  desktop shell
│   └─ embeds a local URL, owns native OS file dialogs              │
├─────────────────────────────────────────────────────────────────┤
│  Dash app + Flask server  (app_src/server.py)                     │  server
│   ├─ layout & components   (components.py)                        │
│   ├─ figure builder        (make_figure.py, get_fft_plots.py)     │
│   ├─ server-side cache     (flask_caching filesystem)             │
│   ├─ server callbacks      (callbacks/: load, predict, save, ...)  │
│   └─ raw Flask routes      (routes.py: /resample, /profile-log)   │
├─────────────────────────────────────────────────────────────────┤
│  Browser interaction layer                                        │  browser
│   ├─ clientside callbacks  (assets/clientsideCallbacks.js)        │
│   ├─ asset scripts         (app_src/assets/*.js)                  │
│   ├─ hidden dcc.Store state + EventListener bridges               │
│   └─ Plotly figure (FigureResampler-backed)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Recipe Index

**Skeleton — you almost always want all of these:**

1. [Desktop shell (pywebview + threaded Dash)](#recipe-1--desktop-shell)
2. [Layout & component model](#recipe-2--layout--component-model)
3. [Server-side cache as the state store](#recipe-3--server-side-cache-as-the-state-store)
4. [File loading, native dialogs & validation](#recipe-4--file-loading-native-dialogs--validation)
5. [Building the resampler figure](#recipe-5--building-the-resampler-figure)

**The bridge — the glue pattern the whole interaction layer rides on:**

6. [The EventListener bridge (custom DOM event → Dash callback)](#recipe-6--the-eventlistener-bridge)

**Navigation — zoom & pan on large signals:**

7. [The relayout coalescer (the navigation backbone)](#recipe-7--the-relayout-coalescer)
8. [Resampler patch pipeline & direct restyle (fast figure updates)](#recipe-8--resampler-patch-pipeline--direct-restyle)
9. [Keyboard panning](#recipe-9--keyboard-panning)
10. [Custom pointer pan (x+y drag)](#recipe-10--custom-pointer-pan)

**Annotation — selecting and labeling regions:**

11. [Mode switching (navigate ↔ annotate)](#recipe-11--mode-switching)
12. [Selection: box, click, and context-menu (whole-bout)](#recipe-12--selection-box-click-context-menu)
13. [Drag-to-select with auto-pan + live trace refresh](#recipe-13--drag-to-select-with-auto-pan)
14. [Keypress annotation & heatmap overlays](#recipe-14--keypress-annotation--heatmap-overlays)
15. [Undo & crash recovery](#recipe-15--undo--crash-recovery)

**Extras:**

16. [Saving & export](#recipe-16--saving--export)
17. [Side-panel media (video clips)](#recipe-17--side-panel-media)
18. [Performance instrumentation](#recipe-18--performance-instrumentation)

**Reference:**

- [Cross-cutting patterns & conventions](#cross-cutting-patterns)
- [Adaptation checklist (new domain)](#adaptation-checklist)
- [Gotcha catalog](#gotcha-catalog)

---

# Skeleton

## Recipe 1 — Desktop shell

**Goal.** Ship the app as a native desktop window, not a browser tab, with access to native
Open/Save dialogs.

**Depends on.** Nothing (this is the outermost layer). Optional — you can run browser-only.

**Source.** `run_desktop_app.py`, `app_src/config.py` (`WINDOW_CONFIG`, `PORT`),
`app_src/assets/closeWindow.js`.

**Mechanism.**
- The Dash/Flask server runs on `127.0.0.1:PORT` in a **daemon thread**
  (`run_desktop_app.py::run_dash`), so it dies with the process.
- `webview.create_window(...)` opens a native window pointed at that local URL. `WINDOW_CONFIG`
  sets size/min-size/resizable.
- On Windows the app forces the EdgeChromium renderer (`webview.start(gui="edgechromium")`);
  elsewhere it auto-selects. This matters — Plotly + custom pointer events behave differently
  across embedded webviews.
- `webview.windows[0]` is the handle the server later uses to raise native file dialogs
  (Recipe 4). This is the *only* reason the app needs pywebview rather than a plain browser.
- `closeWindow.js` installs a `window.onbeforeunload` guard so an accidental close prompts
  "Do you want to leave?" — cheap protection for unsaved annotations.

**Why it's built this way.** A desktop shell buys you native file dialogs (users pick files
by real OS paths, not browser uploads) and a single-window feel, while still letting you build
the entire UI with web tech. The server-in-a-thread pattern keeps startup to one process and
one entrypoint.

**Adapt.**
- Browser-only variant: drop pywebview, run `app.run(...)` directly, and replace native
  dialogs (Recipe 4) with `dcc.Upload`. Everything else in this cookbook is unchanged.
- Change window chrome in `WINDOW_CONFIG`.

**Gotchas.**
- The daemon thread means you must not rely on the server for clean shutdown work; do
  persistence eagerly (the cache + temp files already do).
- Startup auto-update logic lives *before* `app_src` is imported (`run_startup_update_if_enabled`);
  keep any pre-import bootstrapping there, not inside the Dash app.

---

## Recipe 2 — Layout & component model

**Goal.** A two-phase UI: a minimal **home** screen (just "pick a file"), which is replaced by
the full **visualization** screen after a file loads — plus the hidden plumbing every
interaction needs.

**Depends on.** None structurally; everything else plugs into it.

**Source.** `app_src/components.py`, consumed in `app_src/server.py` (`app.layout =
components.home_div`) and `app_src/callbacks/loading.py` (`create_visualization` swaps in
`components.visualization_div`).

**Mechanism.** `components.py` defines three things:

1. **`home_div`** — the initial layout: the "select a file" button, a message area, and
   `backend_div`.
2. **`backend_div`** — the invisible engine room. It holds:
   - A pile of **`dcc.Store`** components — browser-side state slots
     (`box-select-store`, `updated-sleep-scores-store`, `mat-metadata-store`,
     `graph-direct-restyle-payload-store`, etc.). Stores are how clientside and serverside
     callbacks hand data to each other without a visible widget.
   - Several **`EventListener`** components (Recipe 6) that turn custom DOM events into
     callback inputs (`graph-relayout-coalesced`, `graph-annotation-select`,
     `graph-contextmenu`, `keyboard`, ...).
   - A `dcc.Interval` used as a one-shot timer to clear status messages.
3. **`visualization_div`** (built by `make_visualization_div`) — the real UI: a utility bar
   (sampling-level dropdown, video button, predict button), the `dcc.Graph`, the annotation
   message line, save/undo buttons, and the modals.

The `Components` class just bundles these so `server.py` and the callback modules can
reference `components.graph`,
`components.visualization_div`, etc. The graph itself is created once
(`graph = dcc.Graph(id="graph", config={"scrollZoom": True})`) and its `.figure` is assigned
later by the load callback.

**Why it's built this way.** Splitting *initial* from *dynamic* layout keeps the first paint
tiny and defers building the heavy figure until there's data. Centralizing all hidden state in
one `backend_div` means every callback can find its wiring in one file. `suppress_callback_
exceptions=True` on the Dash app is required because callbacks reference components (in
`visualization_div`) that aren't in the initial layout.

**Adapt.**
- Keep the `home_div` / `visualization_div` split. Rename the domain-specific buttons.
- Add a `dcc.Store` for any new piece of browser-side state you need to pass between
  callbacks. Give it an `id` and wire it as `Input`/`State`/`Output`.
- The set of `EventListener`s you keep depends on which interaction recipes you adopt.

**Gotchas.**
- Dynamic components (those that appear as a *result* of a callback, like the video upload
  button) fire their callbacks on creation; `prevent_initial_call` does **not** protect them.
  Guard inside the callback with `if n_clicks is None or n_clicks == 0: raise PreventUpdate`.
  This pattern is everywhere in `app_src/callbacks/` for a reason.
- Store ids are a global namespace shared by clientside JS and Python — keep them stable.

---

## Recipe 3 — Server-side cache as the state store

**Goal.** A place to hold per-session, per-file server state (the loaded file path, annotation
history, the resampler figure) that survives across callbacks and even across an app restart.

**Depends on.** None.

**Source.** `app_src/server.py` (the `Cache(...)` setup), `app_src/session.py`
(`initialize_cache`), and `app_src/resampling.py` (the module global `FIG_RESAMPLER` +
`store_fig_resampler` / `get_fig_resampler` / `clear_fig_resamplers`).

**Mechanism.** Two distinct stores, chosen by whether the value is serializable:

1. **`flask_caching` filesystem cache** for JSON-ish state:
   ```python
   cache = Cache(app.server, config={
       "CACHE_TYPE": "filesystem",
       "CACHE_DIR": TEMP_PATH,                 # a temp dir under the OS temp root
       "CACHE_THRESHOLD": 30,
       "CACHE_DEFAULT_TIMEOUT": 20*24*3600,    # ~20 days, so state persists between runs
   })
   ```
   Keys used: `filepath`, `filename`, `sleep_scores_history` (a `deque(maxlen=2)`; Recipe 15),
   `recent_files_with_video`, `file_video_record`. `initialize_cache` resets these when a new
   file is opened and clears stale temp `.mat`/`.xlsx` files.

2. **A module-global `FIG_RESAMPLER`** for the one thing that *can't* go through the cache: the
   live `FigureResampler` object. It holds the full-resolution signal in memory and is
   accessed on every zoom/pan (Recipe 8) and every auto-pan fetch (Recipe 13). Serializing it
   to disk per interaction would defeat the purpose, so it lives as a process global, created
   in `create_fig` and read via `get_fig_resampler()`.

**Why it's built this way.** The long cache timeout is deliberate: if the app crashes mid-
annotation, reopening the same file **salvages** the last annotation state from
`sleep_scores_history` (Recipe 15). The filesystem backend (not in-memory) is what makes that
survive a restart. The resampler is global precisely because it's big, hot, and singular —
this app shows one file at a time.

**Adapt.**
- Keep the two-tier split: serializable session state → `flask_caching`; big hot in-memory
  objects → module global (or a small registry keyed by session if you go multi-file).
- If you want multi-file / multi-tab, replace the single global with a dict keyed by a session
  id and add eviction. The current app intentionally shows one recording at a time.

**Gotchas.**
- **`np.nan` becomes `None` when read back from the filesystem cache.** The code accounts for
  this repeatedly (`np.place(..., sleep_scores == None, ...)`, `equal_nan=True` comparisons).
  Any NaN-bearing array you round-trip through the cache needs the same care.
- The global resampler means concurrency is single-user by design. Don't add parallel figure
  interactions without rethinking this.

---

## Recipe 4 — File loading, native dialogs & validation

**Goal.** User clicks a button → native OS Open dialog → file path comes back → load, validate,
initialize state, and render the visualization. Same idea for Save.

**Depends on.** Recipe 1 (needs `webview.windows[0]`), Recipe 2 (buttons/stores), Recipe 3
(cache).

**Source.** `app_src/dialogs.py`: `open_file_dialog`, `save_file_dialog`;
`app_src/callbacks/loading.py`: `choose_mat`, `create_visualization`; `app_src/session.py`:
`write_metadata`, `initialize_cache`.

**Mechanism.** The load is a **two-callback handoff** through a store:

1. `choose_mat` (fires on the upload button) opens the native dialog, calls
   `initialize_cache`, writes a "Creating visualizations..." message, and sets
   `visualization-ready-store = "vis"`.
2. `create_visualization` (fires on that store) loads the `.mat`, **validates required
   fields**, builds metadata, salvages/initializes annotation history, builds the figure
   (`create_fig`), assigns it to `components.graph.figure`, and returns `visualization_div` to
   swap the whole screen.

The split exists so the *first* callback can paint a "loading..." message before the *second*
does the slow work. That's a general Dash idiom: **acknowledge in callback A, do the work in
callback B, chained by a store.**

Native dialogs (`open_file_dialog` / `save_file_dialog`) wrap
`webview.windows[0].create_file_dialog(...)` with file-type filters and normalize the return.

**Why it's built this way.** Native dialogs give real filesystem paths (needed to `loadmat`
directly and to locate matching video files), avoiding browser upload size limits and letting
the app write results next to the source. The metadata store (`mat-metadata-store`) carries
just the few numbers the browser needs (`start_time`, `end_time`, `video_start_time`) so
clientside selection math doesn't need the whole file.

**Adapt.**
- Swap `.mat`/`loadmat` for your format (Parquet, EDF, WAV, CSV...). Keep the shape: dialog →
  `initialize_cache` → ack message → second callback validates + builds figure.
- Change the **validation block** in `create_visualization` to your required fields; return a
  helpful message and bail if missing.
- `mat-metadata-store` should carry the minimal numbers your clientside code needs for
  coordinate math (typically the x-axis start/end).

**Gotchas.**
- **Cross-platform dialog return types differ.** On Windows `create_file_dialog` returns a
  tuple; on macOS the SAVE dialog returns an `objc.pyobjc_unicode` (a *string-like* object).
  `result[0]` on the macOS case grabs the first **character**, not the first path. Both
  helpers handle this explicitly — copy that normalization, don't simplify it.
- Guard the button callbacks against the initial `n_clicks is None` fire.

---

## Recipe 5 — Building the resampler figure

**Goal.** One Plotly figure that shows several stacked channels sharing a common x-axis, stays
fast on millions of samples, and carries overlay(s) you can annotate.

**Depends on.** Recipe 3 (stores the built figure as the global resampler).

**Source.** `app_src/make_figure.py` (`make_figure`, `get_padded_sleep_scores`),
`app_src/get_fft_plots.py`, `app_src/config.py`.

**Mechanism.**
- The figure is a `plotly_resampler.FigureResampler` wrapping a 4-row `make_subplots` with
  `shared_xaxes=True`. High-frequency traces are added with `hf_x`/`hf_y` and a
  `default_n_shown_samples` budget; the resampler swaps in a decimated view for the current
  zoom and refreshes on navigation (Recipe 8). Downsampler is `MinMaxLTTB(parallel=True)`,
  which preserves visual extremes (important for spiky physiological signals).
- **All traces are forced onto one shared x-axis**: `fig.update_traces(xaxis="x4")`. This is
  what gives a single crosshair and synchronized pan/zoom across every row
  (`hovermode="x unified"`). Every clientside script keys off `xaxis4`.
- The **annotation layer is a heatmap**. `sleep_scores` (one integer class per second) is
  rendered as a `go.Heatmap` and added as the **last three traces** (once per signal row).
  Because it's added last, JS can always find it at `figure.data.length - 1/-2/-3`
  (Recipe 14). Class→color mapping is a discrete `colorscale`.
- The spectrogram + theta/delta ratio (a dual-y-axis top row) come from `get_fft_plots.py`
  (`scipy` STFT, Gaussian-smoothed, clipped to 0–30 Hz). This is domain-specific; treat it as
  "an example of a derived analytic panel."
- `dragmode="pan"` is the default (navigate); `"select"` is annotate (Recipe 11).
  `modebar_remove=["lasso2d","zoom","autoScale"]` strips modes that would conflict with the
  custom interactions.
- `create_fig` also stashes the recording's x-bounds into `fig.layout.meta.sleepScoringXBounds`,
  which the pan/auto-pan code reads to clamp the view to the data (Recipes 10, 13).

**Why it's built this way.** `FigureResampler` is the single most important dependency for
performance: it means the browser only ever holds ~a few thousand points per trace regardless
of recording length, and the server recomputes the visible slice on demand. The shared x-axis
+ heatmap-overlay design makes annotation a matter of editing one array (`z`) and patching
three traces, which is cheap.

**Adapt.**
- Change the number/labels of rows to your channels. Keep `shared_xaxes=True` and
  `update_traces(xaxis="x<last>")` so navigation stays synchronized; note which axis id ends
  up shared and update the JS references if it isn't `x4`.
- Replace the heatmap overlay's class set/colors for your labels (2 classes? 6 classes?). Keep
  it as the **last N traces** so the index math stays trivial, or centralize the index lookup.
- Drop the spectrogram row if your domain has no analog; add your own derived panel the same
  way.
- Set `default_n_shown_samples` for your latency/detail tradeoff; expose it via a dropdown
  (the app's "Sampling Level", Recipe 8).

**Gotchas.**
- **Heatmaps need 2-D `z`.** A `(N,)` array renders nothing; it must be `(1, N)`. The code
  `np.expand_dims`es it and notes this at the top of `make_figure.py`.
- Sleep scores must be **padded to the exact duration** (`get_padded_sleep_scores`) or the
  overlay won't line up with the signal.
- `-1` (unscored sentinel) is converted to `np.nan` for display so it renders transparent, and
  back to `-1` on save.

---

# The bridge

## Recipe 6 — The EventListener bridge

**Goal.** Let a plain browser DOM event (dispatched by your own JS) trigger a Dash callback —
serverside *or* clientside — carrying a payload. This is the seam that connects the custom
interaction layer to the Dash callback graph.

**Depends on.** Recipe 2 (the listeners live in `backend_div`).

**Source.** `app_src/components.py` (the `EventListener(...)` components) + the asset scripts
that `document.dispatchEvent(new CustomEvent(...))`, consumed by callbacks in
`app_src/callbacks/`.

**Mechanism.** `dash_extensions.EventListener` attaches a DOM event listener and exposes it to
Dash as a component with two relevant props:
- `n_events` — increments each time the event fires (use as `Input`).
- `event` — the captured event, with the specific `props` you declared (use as `State`).

You declare *which* event and *which* fields to capture:
```python
EventListener(
    id="graph-annotation-select",
    events=[{
        "event": "sleepannotationselect",
        "props": ["detail.x0", "detail.x1", "detail.xref", "detail.yref",
                  "detail.y0", "detail.y1", "detail.kind", "detail.timeStamp"],
    }],
)
```
An asset script then does:
```js
document.dispatchEvent(new CustomEvent("sleepannotationselect", {
    detail: { x0, x1, xref, yref, y0, y1, kind, timeStamp },
}));
```
and a Dash callback listens with `Input("graph-annotation-select", "n_events")` +
`State("graph-annotation-select", "event")`.

The app uses this bridge for: coalesced relayout (`sleepgraphrelayout`), annotation drag/click
(`sleepannotationselect`), context-menu bout select (`sleepboutcontextmenu`), navigation
profiling (`sleepgraphprofile`), and raw keydown (`keyboard`).

**Why it's built this way.** Dash's built-in graph props (`relayoutData`, `selectedData`,
`clickData`) are coarse and fire too often or with the wrong granularity for large-signal
interaction. Emitting your *own* semantic events ("the user finished a drag-select from x0 to
x1") lets the JS do the debouncing/coalescing/geometry and hand Python a clean, minimal
payload. It also lets one event feed both a clientside callback (instant UI) and a serverside
callback (data work).

**Adapt.**
- For any custom interaction, define: (1) a `CustomEvent` name, (2) the `detail` fields, (3) a
  matching `EventListener`, (4) the callback. Keep names namespaced (`sleep...`) to avoid
  collisions.
- Prefer sending *derived, minimal* data in `detail` (times, indices, a mode string) rather
  than raw pixel coordinates — do the geometry in JS where you have `_fullLayout`.

**Gotchas.**
- `n_events` is the trigger; the payload is in `event` as a flat dict keyed by the dotted prop
  names you declared (`event["detail.x0"]`). Declare every field you need up front.
- Events dispatched on `document` are global; use `event.target.closest("#graph")` in the JS
  to scope them to the graph (see `graphContextMenu.js`).

---

# Navigation

## Recipe 7 — The relayout coalescer

**Goal.** Turn Plotly's firehose of `plotly_relayouting` events (one per animation frame while
dragging/zooming) into **one clean, debounced signal** that distinguishes "still moving" from
"settled", so the server does the expensive resample once instead of 60×/second.

**Depends on.** Recipe 5 (the figure), Recipe 6 (dispatches `sleepgraphrelayout`).

**Source.** `app_src/assets/graphRelayoutCoalescer.js`; consumed by `update_fig_resampler` in
`app_src/callbacks/navigation.py` via the `graph-relayout-coalesced` EventListener.

**Mechanism.** The script attaches to the Plotly div and listens to both `plotly_relayouting`
(fires continuously during a gesture) and `plotly_relayout` (fires once at the end). It:
- Extracts the new x-range (`xaxis4.range` or `xaxis.range`).
- Debounces to a **final** dispatch after an idle period (`FINAL_IDLE_MS = 450`, shorter
  `KEYBOARD_FINAL_IDLE_MS = 120` for keyboard-driven moves).
- Suppresses near-duplicate dispatches within a time/tolerance window so identical ranges don't
  re-fire.
- Tags each dispatch with a monotonic `profileId`, a `mode` (`fast`/`final`), and a `source`
  (`plotly-moving`, `keyboard`, `custom-drag`).
- **Critically, it suppresses itself while the custom pointer pan or the annotation auto-pan is
  active** (`shouldSuppressPlotlyRelayout`). Those recipes drive their own updates and would
  otherwise double-fire.

It exposes `window.sleepScoringGraphRelayout.{request, requestFinalOnly, suppressPlotlyRelayoutFor}`
so other scripts (keyboard pan, custom pan) can feed ranges into the same coalescing pipeline
instead of inventing their own.

**Why it's built this way.** Without coalescing, every pan would fire dozens of server
callbacks, each reconstructing a resampler patch — the app would grind. The `profileId`
lets the server drop **stale** updates (Recipe 8): if the user is still moving, an in-flight
resample for an old range can be discarded. This is the single most important perf mechanism
after `FigureResampler` itself.

**Adapt.**
- Reusable almost verbatim. The knobs to tune: `FINAL_IDLE_MS` (responsiveness vs. churn),
  the tolerance constants (how "equal" two ranges must be to skip), and the axis id
  (`xaxis4`).
- If you add a new gesture source, route it through `window.sleepScoringGraphRelayout.request`
  so it inherits coalescing + profiling for free.

**Gotchas.**
- It attaches via a `MutationObserver` because the Plotly div is created/replaced dynamically
  (figure swaps). Keep that — a one-time attach will miss re-renders.
- The self-suppression during custom pan/auto-pan is load-bearing; if you add a new custom
  drag, extend `shouldSuppressPlotlyRelayout` or you'll get competing relayouts.

---

## Recipe 8 — Resampler patch pipeline & direct restyle

**Goal.** When the visible x-range settles, refresh each trace to show the right detail for
that zoom — **fast**, moving as few bytes as possible.

**Depends on.** Recipe 5 (resampler figure), Recipe 7 (coalesced signal).

**Source.** `app_src/callbacks/navigation.py`: `update_fig_resampler`,
`RESAMPLER_CALLBACK_OUTPUT`; `app_src/resampling.py`: `compact_resampler_patch`,
`build_direct_restyle_payload`; `app_src/routes.py`: the `/_sleep_scoring/resample` Flask
route; `app_src/assets/graphDirectRestyle.js`; config flag `ENABLE_DIRECT_PLOTLY_RESTYLE`.

**Mechanism.** There are **two delivery paths** for the same computation
(`fig.construct_update_data_patch({...x range...})`, the resampler's method that returns the
decimated data for the new view):

1. **Dash-figure-patch path** (default off): the callback returns a Dash `Patch` targeting
   `graph.figure`. Simple, but Dash diffs and re-applies through its figure machinery.
2. **Direct-restyle path** (`ENABLE_DIRECT_PLOTLY_RESTYLE = True`, the shipped default): the
   callback puts the patch operations into `graph-direct-restyle-payload-store`; a clientside
   callback hands them to `graphDirectRestyle.js`, which calls `Plotly.restyle(plot, update,
   traceIndices)` **directly** — bypassing Dash's figure reconciliation. This is measurably
   faster for big updates.

`RESAMPLER_CALLBACK_OUTPUT` is chosen at import time based on the flag, so the same callback
body serves both paths.

Two more optimizations:
- **`compact_resampler_patch`** trims float precision before serialization
  (`RESAMPLER_PATCH_X_DECIMALS = 3`, `RESAMPLER_PATCH_Y_DECIMALS = 7`), shrinking payloads with
  no visible loss.
- **Stale-drop via `profileId`** (`mark_navigation_profile_seen`, `is_stale_navigation_profile`):
  if a newer navigation has already been seen, an older in-flight patch is dropped rather than
  applied late.

There's also a **raw Flask GET** `/_sleep_scoring/resample?x0=..&x1=..` that returns the patch
JSON directly. This is used by the auto-pan script (Recipe 13), which fetches during a live
drag *without* going through the Dash callback graph at all — the lowest-latency path.

**Why it's built this way.** Every layer here removes overhead from the hot path: coalescing
removes redundant calls (Recipe 7), direct restyle removes Dash's figure diff, precision
trimming removes bytes, stale-drop removes wasted work, and the raw Flask route removes Dash
entirely for the most latency-sensitive case. On a multi-hour recording these compound into
the difference between "smooth" and "unusable."

**Adapt.**
- Keep `update_fig_resampler` structurally. The `xaxis4.range` keys must match your shared
  axis.
- Start with the direct-restyle path off (simpler) and turn it on once the interaction works;
  it's a pure perf swap gated by one flag.
- The raw Flask route is only needed if you implement auto-pan (Recipe 13). Otherwise skip it.

**Gotchas.**
- The direct-restyle JS only forwards a whitelist of trace props (`x`, `y`, `name`, `marker`).
  Widen `DATA_PROPS` if your patches touch others.
- `construct_update_data_patch` mutates/reads the global resampler — it must be the same object
  you built in `create_fig` (Recipe 3).

---

## Recipe 9 — Keyboard panning

**Goal.** Arrow keys nudge the view left/right by a fixed fraction, instantly.

**Depends on.** Recipe 5, Recipe 7 (feeds ranges into the coalescer), the `keyboard`
EventListener.

**Source.** the `pan_figure` clientside callback (JS in
`app_src/assets/clientsideCallbacks.js`, registered in `app_src/callbacks/clientside.py`).

**Mechanism.** A clientside callback on `keyboard.n_events` reads the current `xaxis4.range`,
computes a new range shifted by ±30% on ArrowRight/ArrowLeft, and:
- Applies it immediately via a `dash_clientside.Patch` on `graph.figure` (instant view move).
- Updates `graph.relayoutData` so downstream logic sees the new range.
- Calls `window.sleepScoringGraphRelayout.request(newRelayoutData, "keyboard")` so the
  resampler refresh (Recipe 8) is scheduled through the same coalescing pipeline, with the
  faster keyboard idle timeout.

**Why it's built this way.** Doing the view shift clientside makes it feel instant; routing the
data refresh through the coalescer means the (slower) resample happens once after the user
stops pressing, not on every keydown.

**Adapt.**
- Change the step fraction (0.3) or bind different keys.
- The pattern — *move the view clientside now, schedule the data refresh through the coalescer*
  — is the template for any custom navigation gesture.

**Gotchas.**
- Return `dash_clientside.no_update` for unrelated keys or you'll clobber other outputs.
- Build a **new** `relayoutData` object (spread), don't mutate the existing one — mutating
  React-owned state causes subtle staleness.

---

## Recipe 10 — Custom pointer pan

**Goal.** Click-drag to pan in both x **and** y, smoothly, replacing Plotly's built-in pan.

**Depends on.** Recipe 5, Recipe 7 (final-range hand-off + suppression).

**Source.** `app_src/assets/graphCustomPointerPan.js`.

**Mechanism.** In `dragmode === "pan"`, the script intercepts `pointerdown` on the graph
(when no modifier keys, left button, not on chrome like the modebar/legend), captures the
starting x-range and the y-range of whichever signal row the pointer is over, then on each
`pointermove`:
- Converts pixel delta → data delta using axis length and current range.
- Schedules the actual `Plotly.relayout` on the next animation frame
  (`requestAnimationFrame`), coalescing multiple moves into one paint.
- Tells the coalescer to suppress its own relayout handling for 250 ms
  (`suppressPlotlyRelayoutFor`) so the two don't fight.

On `pointerup` it applies the final frame and calls
`window.sleepScoringGraphRelayout.requestFinalOnly(...)` to trigger the resample once.

**Why it's built this way.** Plotly's native pan doesn't give per-row y-panning and doesn't
integrate with the coalescer's suppression/profiling. Owning the gesture lets the app pan x and
the hovered channel's y together, throttle to the frame rate, and keep the resample on the
settled path only.

**Adapt.**
- Toggle with `ENABLE_CUSTOM_POINTER_PAN`. If you don't need per-axis y-pan, Plotly's native
  pan may be enough — but you lose coalescer integration.
- The list of pan-eligible y-axes (`["yaxis3", "yaxis4"]`) is app-specific (only EMG/NE rows
  y-pan); set yours.

**Gotchas.**
- `isInteractiveChrome` prevents hijacking drags on the modebar/legend/rangeslider — keep an
  equivalent guard.
- rAF throttling + suppression window are what keep it smooth and non-conflicting; don't remove
  them.

---

# Annotation

## Recipe 11 — Mode switching

**Goal.** One key (`m`) toggles between **navigate** (`dragmode="pan"`) and **annotate**
(`dragmode="select"`), and shows/hides the mode-appropriate controls.

**Depends on.** Recipe 5 (the figure holds `dragmode`), the `keyboard` EventListener.

**Source.** the `switch_mode` clientside callback (JS in
`app_src/assets/clientsideCallbacks.js`, registered in `app_src/callbacks/clientside.py`).

**Mechanism.** A clientside callback on the `m` key patches `figure.layout.dragmode` between
`"pan"` and `"select"`, clears any leftover selections/shapes when leaving select mode, and
flips the visibility of the video/predict buttons accordingly.

**Why it's built this way.** Navigation gestures (pan/scroll-zoom) and annotation gestures
(drag-select) both want the mouse, so they can't coexist in one mode. A single toggle keyed to
muscle memory is faster than clicking a modebar. Doing it clientside keeps it instant. Nearly
every other interaction script gates on `dragmode` (`isSelectMode`, `getDragMode`), so this one
flag is the master switch for the whole interaction layer.

**Adapt.**
- Keep the pattern; rebind the key if `m` collides. Add per-mode UI affordances here.

**Gotchas.**
- When leaving select mode, clear `selections` and `shapes` or a stale selection box lingers.

---

## Recipe 12 — Selection: box, click, context-menu

**Goal.** Three ways to select a time region to label: **drag a box**, **click a point**
(selects a small neighborhood), or **right-click a bout** (selects the whole contiguous
same-label run under the cursor).

**Depends on.** Recipe 5, Recipe 11 (only active in select mode), Recipe 6 (the context-menu
event), `mat-metadata-store`.

**Source.** clientside callbacks in `app_src/assets/clientsideCallbacks.js` (registered in
`app_src/callbacks/clientside.py`): `read_box_select` (Plotly `selectedData`),
`read_click_select` (Plotly `clickData`), `read_bout_context_select`
(`graph-contextmenu` event); `app_src/assets/graphContextMenu.js`.

**Mechanism.** Each selection callback ends the same way: it computes an integer
`[start, end]` index range (in seconds relative to `start_time`), stores it in
`box-select-store`, draws a rectangle `shape` on the figure for visual feedback, and writes a
status message ("You selected [a, b] (n s). Press 1 for Wake...").
- **Box**: reads `figure.layout.selections`, keeps only the last box, rounds to integer
  seconds, clamps to recording bounds.
- **Click**: builds a small neighborhood (0.5% of the visible width) around the clicked x,
  respecting which subplot/axis was clicked.
- **Context-menu**: `graphContextMenu.js` converts the right-click pixel to a data x (via the
  axis `p2l`), dispatches `sleepboutcontextmenu`; the callback walks the heatmap `z` array
  left/right from the clicked index while the label stays equal, selecting the entire bout.

`box-select-store` is the single source of truth that the annotation keypress (Recipe 14)
reads. All three selection methods converge on it.

**Why it's built this way.** Different edits want different granularities — a precise drag, a
quick single-second fix, or "relabel this whole misclassified bout." Funneling all three into
one `[start, end]` store means the labeling step doesn't care how the region was chosen.

**Adapt.**
- Keep the convergence on one selection store. Add/remove selection methods freely.
- The bout-walk logic assumes the label array is the last heatmap trace's `z[0]`; keep that
  invariant or centralize the lookup.

**Gotchas.**
- Rounding to integer seconds has edge cases when `start_round === end_round`; the code expands
  to a full second deliberately — preserve that or you get zero-width selections.
- All three are gated on `dragmode === "select"` (or clear on pan) so they don't fire during
  navigation.

---

## Recipe 13 — Drag-to-select with auto-pan

**Goal.** Drag to select a region; when the pointer nears the edge of the view, the plot
**auto-pans** in that direction so you can select a region longer than the screen — and the
newly revealed signal is **fetched and drawn live** during the pan.

**Depends on.** Recipe 5, Recipe 6 (`sleepannotationselect`), Recipe 8 (the raw
`/_sleep_scoring/resample` Flask route), Recipe 11 (select mode).

**Source.** `app_src/assets/annotationAutoPan.js` (the biggest asset, ~800 lines); consumed by
`read_annotation_auto_pan_select` in `app_src/callbacks/clientside.py`.

**Mechanism.** This is the most elaborate interaction; it owns the whole pointer gesture in
select mode:
- **Begin** (`beginDrag`): on `pointerdown` in select mode, record the anchor time and the
  hovered row's y-range; set `window.sleepScoringAnnotationAutoPanActive = true` (which makes
  the coalescer stand down, Recipe 7); capture the pointer.
- **Continue** (`continueDrag`): update the selection rectangle each frame
  (`scheduleDraw` → `Plotly.relayout` with a `shapes` rect on the next animation frame).
- **Auto-pan** (`edgePressure` + `autoPanStep`): compute an edge "pressure" (0 in the middle,
  ramping to ±1 within `EDGE_PX = 72` of an edge). While pressure ≠ 0, an rAF loop shifts
  `xaxis4.range` proportionally to pressure² × view-width × dt, clamped to recording bounds.
- **Live trace refresh** (`requestTraceRefresh` → `startTraceRequest`): during the pan it
  `fetch`es `/_sleep_scoring/resample` for the lead-edge range, then **merges** the incoming
  decimated points into the existing trace data (`mergeTraceArrays`, throttled to
  `TRACE_REFRESH_MS = 180`, single-flight with a pending slot) and applies via `Plotly.restyle`
  — so the signal appears under the growing selection without waiting for the drag to end.
- **End** (`endDrag`): stop the pan, do one final `replace`-mode refresh, and dispatch
  `sleepannotationselect` with the final `[x0, x1]` and whether it was a `drag` or a `click`.
  The `read_annotation_auto_pan_select` clientside callback then normalizes to integer indices,
  draws the final selection box, and writes `box-select-store` — same convergence point as
  Recipe 12.

**Why it's built this way.** Selecting a region wider than the viewport is otherwise impossible
without zooming out (and losing detail). Auto-pan-while-dragging solves that, but a naive
implementation would show blank space where data hasn't loaded — hence the live fetch+merge
against the raw resample endpoint. It deliberately bypasses Dash (raw `fetch`) because this is
the single most latency-sensitive loop in the app: it runs every frame of a drag.

**Adapt.**
- This is the hardest recipe to port; adopt it only if cross-viewport selection matters.
  Tunables: `EDGE_PX`, `PAN_VIEW_WIDTH_PER_SECOND`, `AUTO_PAN_FRAME_MS`, `TRACE_REFRESH_MS`,
  and the buffer fractions controlling how much lead data to fetch/merge.
- Requires the raw Flask resample route (Recipe 8) and the x-bounds meta (Recipe 5) for
  clamping.

**Gotchas.**
- The `AnnotationAutoPanActive` flag must toggle the coalescer suppression on/off around the
  gesture (with a small release delay) or you get dueling relayouts.
- Single-flight + pending-request bookkeeping prevents request pileup; the stale-guard
  (`requestId < latestAppliedTraceRequestId`) prevents an old fetch from overwriting a newer
  one. Keep both.
- `click` vs `drag` is decided by a `CLICK_PX = 4` movement threshold so a stationary press
  still selects a neighborhood (matching Recipe 12's click behavior).

---

## Recipe 14 — Keypress annotation & heatmap overlays

**Goal.** With a region selected, press `1/2/3/4` to label it; the overlay updates instantly
and the change is pushed to undo history.

**Depends on.** Recipe 5 (heatmap overlay), Recipe 12/13 (`box-select-store`), the `keyboard`
EventListener, Recipe 15 (history).

**Source.** clientside `make_annotation` and `update_sleep_scores` (JS in
`app_src/assets/clientsideCallbacks.js`, registered in `app_src/callbacks/clientside.py`);
serverside `update_sleep_scores_history` in `app_src/callbacks/saving.py`.

**Mechanism.** Two-step, all clientside for the visual part:
1. `make_annotation` fires on a `1–4` keypress (only in select mode, only with a selection):
   it copies the current label array (`figure.data[last].z[0]`), writes `label` into
   `[start, end)`, and puts the new array in `updated-sleep-scores-store`. Then it clears the
   selection.
2. `update_sleep_scores` fires on that store: it patches the `z` of **all three** heatmap
   traces (`num_traces - 1/-2/-3`) to the new array and clears selection shapes.

Separately, the serverside `update_sleep_scores_history` also fires on
`updated-sleep-scores-store`, appends the new array to `sleep_scores_history` if it actually
changed (`np.array_equal(..., equal_nan=True)`), and reveals the Undo button.

**Why it's built this way.** The label array *is* the annotation state; rendering it as a
heatmap means "apply a label" is just "edit an array and repaint three traces" — no server
round-trip for the visual. Keeping three synchronized overlays (one per signal row) is what
lets the user read the score against any channel. Pushing to history on the same store keeps
undo automatic.

**Adapt.**
- Map your keys → class integers. Keep the label array as the heatmap `z`.
- If you have one overlay row instead of three, patch one trace; if N, patch N. Consider a
  helper that returns the overlay trace indices instead of hardcoding offsets.

**Gotchas.**
- `[start, end)` is half-open; the loop writes `i` from `start` to `end-1`. Match your
  selection math to this or you'll be off by one second.
- The overlay index math (`length - 1/-2/-3`) depends on the heatmaps being the last traces
  (Recipe 5). If you add traces after them, this breaks silently.

---

## Recipe 15 — Undo & crash recovery

**Goal.** One level of undo for annotations, and automatic salvage of unsaved work if the app
restarts on the same file.

**Depends on.** Recipe 3 (cache), Recipe 14 (writes history).

**Source.** `app_src/session.py`: `initialize_cache` (creates `deque(maxlen=2)`);
`app_src/callbacks/saving.py`: `update_sleep_scores_history`, `undo_annotation`; the salvage
branch in `create_visualization` (`app_src/callbacks/loading.py`).

**Mechanism.** `sleep_scores_history` is a `collections.deque(maxlen=2)` in the filesystem
cache — it holds at most the previous and current label arrays. Every real change appends
(dropping the oldest). `undo_annotation` restores `history[0]` and pops, then repaints via
`updated-sleep-scores-store`. Because the cache is filesystem-backed with a ~20-day timeout,
reopening the **same** file finds the last history entry and loads it instead of the file's
on-disk scores (the salvage branch in `create_visualization`); opening a **different** file
resets the deque.

**Why it's built this way.** `maxlen=2` is a deliberate scope choice: one-step undo covers the
"oops, wrong label" case cheaply without unbounded memory or a full history stack. Filesystem
persistence turns the same structure into crash recovery for free.

**Adapt.**
- Want multi-level undo? Raise `maxlen` and make `undo` walk back one step at a time. Weigh
  memory (each entry is a full label array).
- The salvage-on-reopen behavior is keyed on filename equality; keep that check if you adopt
  it.

**Gotchas.**
- NaN handling again: history arrays round-trip through the cache, so comparisons use
  `equal_nan=True` and `== None` checks (Recipe 3 gotcha).

---

# Extras

## Recipe 16 — Saving & export

**Goal.** Save annotations back to the source format via a native Save dialog, and export a
derived summary (here: a sleep-bout spreadsheet) when the data is complete.

**Depends on.** Recipe 1 (native dialog), Recipe 3 (state), Recipe 15 (final labels).

**Source.** `app_src/callbacks/saving.py::save_annotations`; `app_src/postprocessing.py`
(`get_sleep_segments`, `get_pred_label_stats`, `get_first_unscored_segment`, `standardize`).

**Mechanism.** On Save: reload the source, replace its label field with the latest history
array (converting `NaN`/`None`→sentinel `-1`), write to a temp file, then a native Save dialog
copies it to the user's chosen path. If scoring is complete (no unscored segment), it also
builds a bout table + stats and offers a second Save dialog for an `.xlsx`. A one-shot
`dcc.Interval` clears the status message after a few seconds.

**Adapt.** Swap the format writer and the derived-export logic. Keep the "temp file then copy
to dialog path" pattern (it decouples computation from the user's save location and survives a
cancelled dialog).

**Gotchas.** Convert display sentinels (`NaN`) back to the on-disk sentinel (`-1`) before
saving. Guard the button's initial fire.

---

## Recipe 17 — Side-panel media

**Goal.** Play a media clip (video) corresponding to the currently selected time window.

**Depends on.** Recipe 12/13 (`box-select-store`), Recipe 4 (native dialog to locate media),
Recipe 3 (remember media per file).

**Source.** `app_src/callbacks/video.py`: `prepare_video`, `choose_video`, `make_clip`,
`show_clip`;
`app_src/make_mp4.py`; a `dbc.Modal` with `dash_player`.

**Mechanism.** The selection range + a `video_start_time` offset define a clip; `make_mp4.py`
cuts it with the bundled ffmpeg into `assets/videos/`, and `dash_player.DashPlayer` plays it in
a modal. The app remembers recently used video paths per file in the cache
(`recent_files_with_video`, `file_video_record`) so it doesn't re-ask.

**Adapt.** Any "selected region → derived artifact in a side panel" (a zoom-in figure, an audio
snippet, a detail table) follows this shape: read `box-select-store`, produce the artifact into
`assets/`, show it in a modal/panel.

**Gotchas.** Serve derived media from the Dash `assets/` folder (it's auto-served at
`/assets/...`). Clean up old clips to avoid unbounded disk use (the app unlinks prior `.mp4`s).

---

## Recipe 18 — Performance instrumentation

**Goal.** Measure where navigation latency goes (server construct time, payload size, browser
apply time, frame gaps) without shipping the noise to end users.

**Depends on.** Recipes 7, 8 (the things being measured).

**Source.** `app_src/config.py` (the `*_PERF_LOG` flags + `SLEEP_SCORING_*` env vars),
`app_src/resampling.py` (`summarize_resampler_patch`), `app_src/callbacks/navigation.py`
(the `[resampler]`/`[browser-nav]` prints), `app_src/routes.py` (the
`/_sleep_scoring/profile-log` route), `app_src/assets/graphNavigationProfiler.js`.

**Mechanism.** Each navigation carries a `profileId` end to end. The browser profiler
(`graphNavigationProfiler.js`) times `dispatch → afterplot`, computes coalesce/apply/frame-gap
ms, and posts them to `/_sleep_scoring/profile-log`, which mirrors them into the server
terminal alongside the server-side `[resampler]` line for the same id. Everything is gated by
env vars / flags and off by default in shipped builds.

**Adapt.** Cheap to keep, invaluable when tuning. The `profileId`-threaded-through-everything
pattern is the reusable idea: one id lets you line up a browser event with the server work it
triggered.

**Gotchas.** Keep it off by default (it's chatty). Use `navigator.sendBeacon` for the log post
so it doesn't block the interaction.

---

# Cross-cutting patterns

These show up across many recipes; internalize them.

- **Clientside for feel, serverside for data.** If an interaction can be done from data already
  in the browser (figure, stores), do it in a clientside callback. Only hit the server for
  loading, model inference, resampling, and saving.
- **Stores are the wiring.** Hidden `dcc.Store`s are how callbacks (client and server) hand
  each other data. Name them stably; treat the set of stores as your app's state schema.
- **Converge many inputs on one store.** Box/click/context/drag all write `box-select-store`;
  the labeling step reads only that. Fan-in keeps downstream logic simple.
- **Ack-then-work handoff.** A slow action is two callbacks: the first paints a "working..."
  message and sets a trigger store; the second does the work. Users see immediate feedback.
- **Emit your own semantic DOM events.** Don't fight Plotly's coarse built-in events; have your
  JS compute the meaningful thing (a settled range, a finished selection) and dispatch a custom
  event the `EventListener` bridge turns into a callback (Recipe 6).
- **One shared x-axis.** Force every trace onto it (`update_traces(xaxis="x4")`). All
  navigation and selection code keys off that single axis.
- **Coalesce, then compute.** Never let a per-frame gesture drive per-frame server work. Route
  it through the coalescer (Recipe 7) and drop stale work by `profileId` (Recipe 8).
- **Thread a `profileId` through everything** for both stale-dropping and profiling.
- **Guard dynamic-component callbacks** with `if n_clicks is None or n_clicks == 0: raise
  PreventUpdate` — `prevent_initial_call` isn't enough for components created by callbacks.
- **`allow_duplicate=True`** is required whenever multiple callbacks target the same output
  (many target `graph.figure` and `annotation-message`). Expect to use it a lot.
- **Assets auto-load.** Any `.js` in `app_src/assets/` is served and executed automatically, in
  filename order. Scripts self-guard with `if (window.sleepScoringX) return;` and re-attach via
  `MutationObserver` because the graph div is recreated on figure swaps.

---

# Adaptation checklist

Building a new app from this template? Work down this list.

1. **Shell** (Recipe 1): keep pywebview, or drop it for browser-only + `dcc.Upload`.
2. **Data contract**: define your file format and the **required vs optional** fields. Put the
   validation in the load callback (Recipe 4).
3. **Figure** (Recipe 5): set your rows/channels, the shared x-axis, `default_n_shown_samples`,
   and — if you annotate — the heatmap overlay's class set and colors. Stash x-bounds in
   `layout.meta` if you'll pan/auto-pan.
4. **State** (Recipe 3): list your session state → filesystem cache; your one big hot object →
   module global.
5. **Stores** (Recipe 2): one `dcc.Store` per piece of browser-side state you pass between
   callbacks.
6. **Navigation**: adopt the coalescer (Recipe 7) + resampler pipeline (Recipe 8) — these are
   near-verbatim reusable. Add keyboard pan (9) and/or custom pointer pan (10) as desired.
7. **Annotation** (if applicable): mode switch (11) → selection methods you want (12, and 13 if
   you need cross-viewport selection) → keypress labeling (14) → undo (15). All converge on one
   selection store and one label array.
8. **Save/export** (16) and **side panels** (17) as needed.
9. **Instrument** (18) while tuning; ship it off.

**Minimum viable interactive viewer** (no annotation): Recipes 1–8. That already gives you a
fast, zoomable, pannable multi-channel viewer on huge signals.

**Add annotation**: Recipes 11, 12, 14, 15 (+13 for long selections).

---

# Gotcha catalog

A quick-reference of the traps, collected:

- **macOS vs Windows file dialogs** return different types (`objc.pyobjc_unicode` vs tuple);
  `result[0]` on macOS grabs a character. Normalize (Recipe 4).
- **`np.nan` → `None`** when read back from the filesystem cache. Use `equal_nan=True` and
  `== None` handling (Recipe 3).
- **Heatmaps need 2-D `z`** (`(1, N)`, not `(N,)`) (Recipe 5).
- **Label array must be padded to the exact duration** or the overlay misaligns (Recipe 5).
- **Overlay index math** (`figure.data.length - 1/-2/-3`) assumes heatmaps are the last traces
  (Recipes 5, 14). Don't append traces after them.
- **Half-open `[start, end)`** annotation range — off-by-one if your selection math disagrees
  (Recipe 14).
- **Coalescer self-suppression** must wrap every custom drag/auto-pan, or relayouts fight
  (Recipes 7, 10, 13).
- **Dynamic-component callbacks** need the manual `n_clicks` guard (Recipe 2).
- **`allow_duplicate=True`** on shared outputs, and give each duplicate a distinct
  `prevent_initial_call`.
- **Re-attach via `MutationObserver`** — the Plotly div is replaced on figure swaps; a one-time
  `addEventListener` misses re-renders (Recipes 7, 10).
- **Clear `selections` and `shapes`** when leaving select mode or after labeling, or ghost
  boxes linger (Recipes 11, 14).
- **Sentinel round-trip**: unscored is `-1` on disk, `NaN` in display. Convert both ways
  (Recipes 5, 16).

---

## Source-file map (reference app)

| Concern | File |
| --- | --- |
| Desktop shell / entrypoint | `run_desktop_app.py` |
| Config (window, port, flags, colors) | `app_src/config.py` |
| App aggregator (importing it registers everything) | `app_src/app.py` |
| Dash instance, cache, components, runtime paths | `app_src/server.py` |
| Raw Flask routes (`/resample`, `/profile-log`) | `app_src/routes.py` |
| Native OS file dialogs | `app_src/dialogs.py` |
| Resampler figure store & patch helpers | `app_src/resampling.py` |
| Per-recording setup (cache init, metadata) | `app_src/session.py` |
| Dash callbacks, one module per concern | `app_src/callbacks/` |
| Clientside callback JS implementations | `app_src/assets/clientsideCallbacks.js` |
| Layout, stores, EventListeners, modals | `app_src/components.py` |
| Figure builder (resampler, overlays) | `app_src/make_figure.py` |
| Spectrogram / derived panel | `app_src/get_fft_plots.py` |
| Relayout coalescing | `app_src/assets/graphRelayoutCoalescer.js` |
| Direct restyle (fast patch apply) | `app_src/assets/graphDirectRestyle.js` |
| Custom pointer pan | `app_src/assets/graphCustomPointerPan.js` |
| Drag-select + auto-pan | `app_src/assets/annotationAutoPan.js` |
| Context-menu bout select | `app_src/assets/graphContextMenu.js` |
| Navigation profiler | `app_src/assets/graphNavigationProfiler.js` |
| Unsaved-work exit guard | `app_src/assets/closeWindow.js` |
| Model inference (domain) | `app_src/inference.py`, `run_inference_*.py` |
| Postprocessing / export | `app_src/postprocessing.py` |
| Video clips | `app_src/make_mp4.py` |

For the codebase orientation itself (active vs legacy, data contract, test map), see
`project_overview.md`. For collaboration/runtime conventions, see `AGENTS.md`.

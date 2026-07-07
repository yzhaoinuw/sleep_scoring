# Clientside Callback Tests (jest)

Unit tests for the app's clientside callbacks — the JavaScript in
[`app_src/assets/clientsideCallbacks.js`](../../app_src/assets/clientsideCallbacks.js)
that handles mode switching, keyboard panning, selection, keypress annotation,
and message cleanup in the browser.

If you know pytest but not the JavaScript ecosystem, start here.

## What jest is

[Jest](https://jestjs.io/) is to JavaScript what pytest is to Python: a test
runner. Test files contain `test("...", () => { expect(f(x)).toEqual(y); })`
blocks; jest runs them and reports pass/fail. It executes in **Node.js**, a
JavaScript interpreter you run from the terminal exactly like the `python`
command. No browser, no app window, no rendering is involved.

## How these tests can cover "visual" features without opening the app

The clientside callbacks never draw anything themselves. Each one is a
decision function: plain data in, plain data out. For example, the
click-select callback receives the figure state (a JSON object), a click at
`x = 50`, and the recording metadata, and returns: the selection `[49, 51]`,
an instruction to draw a rectangle from 49 to 51, a status message, and a
video-button visibility style. *Plotly* is what later turns the rectangle
instruction into pixels.

These tests exercise that decision layer — the rounding, clamping,
tie-breaking, and bout-boundary math — by calling the functions with
hand-built inputs and asserting on the returned data. Testing that a chess
engine returns the right move does not require rendering the board.

## How it works

- [`setup.js`](setup.js) provides the two globals the callbacks expect and
  then loads the real asset file:
  - `window` — pointed at Node's global object.
  - `dash_clientside` — a stub with `no_update` (a sentinel the tests compare
    against by identity) and `Patch`.
- `PatchStub` is the key trick. In the browser, callbacks build a
  `dash_clientside.Patch`: a list of "set this figure property to this value"
  instructions that Dash applies to the live figure afterward. The stub just
  records those instructions into an `ops` array, so a test can assert
  "pressing `m` in pan mode set `layout.dragmode` to `select`" without any
  figure, Dash, or browser existing.
- [`clientsideCallbacks.test.js`](clientsideCallbacks.test.js) holds the
  tests, one `describe` block per callback, mirroring the section order of
  `app_src/callbacks/clientside.py`.

## What is and is not covered

Covered: the logic. If someone breaks the selection rounding, introduces an
off-by-one in the annotation range (it is half-open: `[start, end)`), or
inverts the video-button visibility rule, this suite goes red in seconds.

Not covered: the wiring and the rendering. Whether Dash delivers events in
the assumed shape, whether Plotly applies the patches correctly, asset load
order, and how the interactions feel all still require a manual pass in the
running app (or, someday, browser-automation tests such as Playwright or
`dash[testing]`). Keep doing the manual check for changes that touch the
browser wiring.

## Running

Requires Node.js (`brew install node` on macOS; any recent version works).

```bash
cd tests/js
npm ci     # install jest exactly as pinned in package-lock.json
npm test
```

`npm ci` creates a local `node_modules/` folder (gitignored). CI runs the
same two commands in the `js-test` job of
[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml).

## Adding a test

Grab the callback from the namespace and call it like any function:

```js
const fns = window.dash_clientside.sleep_scoring;

test("pressing 2 labels the selected range as NREM", () => {
  const figure = { layout: { dragmode: "select" }, data: [{ z: [[0, 0, 0, 0, 0]] }] };
  const [, scores] = fns.make_annotation(1, { key: "2" }, [1, 3], figure);
  expect(scores).toEqual([0, 1, 1, 0, 0]);
});
```

Build the smallest `figure`/`metadata` objects that satisfy the callback's
guards; the existing tests are a catalog of the shapes each callback reads.

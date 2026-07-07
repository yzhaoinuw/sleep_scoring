// Unit tests for the clientside callbacks in
// app_src/assets/clientsideCallbacks.js (loaded by setup.js).
//
// Each callback returns either dash_clientside.no_update sentinels or an
// array of outputs matching its registration in
// app_src/callbacks/clientside.py. Figure mutations are asserted through the
// PatchStub's recorded ops.

const fns = window.dash_clientside.sleep_scoring;
const NO = dash_clientside.no_update;

const ANNOTATION_HINT = "Press 1 for Wake, 2 for NREM, 3 for REM, or 4 for MA.";

function lastOp(patch, ...path) {
  const key = path.join(".");
  const hits = patch.ops.filter((op) => op.path.join(".") === key);
  return hits[hits.length - 1];
}

function expectAllNoUpdate(result, n) {
  expect(result).toHaveLength(n);
  result.forEach((value) => expect(value).toBe(NO));
}

afterEach(() => {
  delete window.sleepScoringGraphRelayout;
  delete window.sleepScoringDirectRestyle;
});

// ---- mode switching and navigation ----

describe("switch_mode", () => {
  test("m in pan mode switches to select and shows the predict button", () => {
    const figure = { layout: { dragmode: "pan" } };
    const [patch, message, videoStyle, predStyle] = fns.switch_mode(1, { key: "m" }, figure);

    expect(lastOp(patch, "layout", "dragmode").value).toBe("select");
    expect(message).toBe("");
    expect(videoStyle).toEqual({ visibility: "hidden" });
    expect(predStyle).toEqual({ visibility: "visible" });
  });

  test("M in select mode switches to pan and clears selections and shapes", () => {
    const figure = { layout: { dragmode: "select" } };
    const [patch, , , predStyle] = fns.switch_mode(1, { key: "M" }, figure);

    expect(lastOp(patch, "layout", "dragmode").value).toBe("pan");
    expect(lastOp(patch, "layout", "selections").value).toBeNull();
    expect(lastOp(patch, "layout", "shapes").value).toBeNull();
    expect(predStyle).toEqual({ visibility: "hidden" });
  });

  test("other keys change nothing", () => {
    const figure = { layout: { dragmode: "pan" } };
    expectAllNoUpdate(fns.switch_mode(1, { key: "x" }, figure), 4);
  });

  test("missing event or figure changes nothing", () => {
    expectAllNoUpdate(fns.switch_mode(0, null, { layout: {} }), 4);
    expectAllNoUpdate(fns.switch_mode(1, { key: "m" }, null), 4);
  });
});

describe("pan_figure", () => {
  const figure = { layout: { xaxis4: { range: [0, 100] } } };

  test("ArrowRight shifts the view right by 30% of the current width", () => {
    const [patch, relayout] = fns.pan_figure(1, { key: "ArrowRight" }, null, figure);

    expect(lastOp(patch, "layout", "xaxis4", "range").value).toEqual([30, 130]);
    expect(relayout).toEqual({ "xaxis4.range[0]": 30, "xaxis4.range[1]": 130 });
  });

  test("ArrowLeft uses the range from relayoutData when present", () => {
    const relayoutdata = { "xaxis4.range[0]": 100, "xaxis4.range[1]": 200 };
    const [patch, relayout] = fns.pan_figure(1, { key: "ArrowLeft" }, relayoutdata, figure);

    expect(lastOp(patch, "layout", "xaxis4", "range").value).toEqual([70, 170]);
    expect(relayout["xaxis4.range[0]"]).toBe(70);
    expect(relayout["xaxis4.range[1]"]).toBe(170);
  });

  test("schedules a coalesced resample when the relayout bridge is loaded", () => {
    window.sleepScoringGraphRelayout = { request: jest.fn() };
    const [, relayout] = fns.pan_figure(1, { key: "ArrowRight" }, null, figure);

    expect(window.sleepScoringGraphRelayout.request).toHaveBeenCalledWith(relayout, "keyboard");
  });

  test("other keys change nothing", () => {
    expectAllNoUpdate(fns.pan_figure(1, { key: "a" }, null, figure), 2);
  });
});

describe("apply_direct_restyle", () => {
  const payload = { profileMarker: { profileId: 7 }, operations: [] };

  test("empty payload changes nothing", () => {
    expect(fns.apply_direct_restyle(null)).toBe(NO);
  });

  test("reports an error status when the restyle asset is not loaded", () => {
    const status = fns.apply_direct_restyle(payload);

    expect(status.ok).toBe(false);
    expect(status.error).toBe("sleepScoringDirectRestyle asset is not loaded");
    expect(status.profileId).toBe(7);
    expect(typeof status.timeStamp).toBe("number");
  });

  test("returns the apply result with a timestamp on success", () => {
    window.sleepScoringDirectRestyle = { apply: jest.fn(() => ({ ok: true, applied: 3 })) };
    const status = fns.apply_direct_restyle(payload);

    expect(window.sleepScoringDirectRestyle.apply).toHaveBeenCalledWith(payload);
    expect(status.ok).toBe(true);
    expect(status.applied).toBe(3);
    expect(typeof status.timeStamp).toBe("number");
  });

  test("captures thrown errors as a failure status", () => {
    window.sleepScoringDirectRestyle = {
      apply: () => {
        throw new Error("boom");
      },
    };
    const status = fns.apply_direct_restyle(payload);

    expect(status.ok).toBe(false);
    expect(status.error).toBe("boom");
    expect(status.profileId).toBe(7);
  });
});

// ---- selection ----

describe("read_box_select", () => {
  const metadata = { start_time: 0, end_time: 100 };
  const figureWith = (selections) => ({ layout: { selections } });

  test("rounds the dragged box to whole seconds", () => {
    const figure = figureWith([{ x0: 10.4, x1: 20.6 }]);
    const [range, patch, message, videoStyle] = fns.read_box_select(null, figure, null, metadata);

    expect(range).toEqual([10, 21]);
    expect(lastOp(patch, "layout", "shapes").value).toBeNull();
    expect(message).toBe(`You selected [10, 21] (11 s). ${ANNOTATION_HINT}`);
    expect(videoStyle).toEqual({ visibility: "visible" });
  });

  test("a sub-second box still selects one whole second", () => {
    const figure = figureWith([{ x0: 10.3, x1: 10.2 }]);
    const [range, , , videoStyle] = fns.read_box_select(null, figure, null, metadata);

    expect(range).toEqual([10, 11]);
    expect(videoStyle).toEqual({ visibility: "visible" });
  });

  test("reports selections relative to a nonzero recording start_time", () => {
    const offset = { start_time: 3600, end_time: 3700 };
    const figure = figureWith([{ x0: 3610.2, x1: 3620.8 }]);
    const [range] = fns.read_box_select(null, figure, null, offset);

    expect(range).toEqual([10, 21]);
  });

  test("keeps only the most recent box when several exist", () => {
    const figure = figureWith([
      { x0: 1, x1: 2 },
      { x0: 5.2, x1: 8.9 },
    ]);
    const [range, patch] = fns.read_box_select(null, figure, null, metadata);

    expect(range).toEqual([5, 9]);
    expect(lastOp(patch, "layout", "selections").value).toEqual([{ x0: 5.2, x1: 8.9 }]);
  });

  test("rejects a box entirely outside the recording", () => {
    const figure = figureWith([{ x0: 150, x1: 160 }]);
    const [range, , message, videoStyle] = fns.read_box_select(null, figure, null, metadata);

    expect(range).toEqual([]);
    expect(message).toBe("Out of range. Please select from 0 to 100.");
    expect(videoStyle).toEqual({ visibility: "hidden" });
  });

  test("hides the video button for selections longer than 300 s", () => {
    const figure = figureWith([{ x0: 0, x1: 400 }]);
    const [range, , , videoStyle] = fns.read_box_select(null, figure, null, {
      start_time: 0,
      end_time: 500,
    });

    expect(range).toEqual([0, 400]);
    expect(videoStyle).toEqual({ visibility: "hidden" });
  });

  test("no selections changes nothing", () => {
    expectAllNoUpdate(fns.read_box_select(null, figureWith(undefined), null, metadata), 4);
    expectAllNoUpdate(fns.read_box_select(null, figureWith([]), null, metadata), 4);
  });
});

describe("read_click_select", () => {
  const metadata = { start_time: 0, end_time: 100 };
  const figure = {
    layout: { dragmode: "select", xaxis4: { range: [0, 100] } },
    data: [{}, { xaxis: "x4", yaxis: "y2" }],
  };

  test("selects a small neighborhood around the clicked point", () => {
    const clickData = { points: [{ x: 50, curveNumber: 0 }] };
    const [range, patch, message, videoStyle] = fns.read_click_select(clickData, figure, metadata);

    // 0.5% of the 100 s view is 0.5 s; floor/ceil widen it to whole seconds
    expect(range).toEqual([49, 51]);
    const box = lastOp(patch, "layout", "shapes").value[0];
    expect(box).toMatchObject({ x0: 49, x1: 51, xref: "x4", yref: "y5" });
    expect(message).toBe(`You selected [49, 51] (2 s). ${ANNOTATION_HINT}`);
    expect(videoStyle).toEqual({ visibility: "visible" });
  });

  test("remaps the spectrogram dual axis y2 to y1 so the box avoids the ratio curve", () => {
    const clickData = { points: [{ x: 50, curveNumber: 1 }] };
    const [, patch] = fns.read_click_select(clickData, figure, metadata);

    expect(lastOp(patch, "layout", "shapes").value[0].yref).toBe("y1");
  });

  test("clears the selection in pan mode", () => {
    const panFigure = { ...figure, layout: { ...figure.layout, dragmode: "pan" } };
    const clickData = { points: [{ x: 50, curveNumber: 0 }] };
    const [range, patch, message, videoStyle] = fns.read_click_select(
      clickData,
      panFigure,
      metadata
    );

    expect(range).toEqual([]);
    expect(lastOp(patch, "layout", "shapes").value).toBeNull();
    expect(message).toBe("");
    expect(videoStyle).toEqual({ visibility: "hidden" });
  });

  test("missing figure or metadata changes nothing", () => {
    expectAllNoUpdate(fns.read_click_select(null, null, metadata), 4);
    expectAllNoUpdate(fns.read_click_select(null, figure, null), 4);
  });
});

describe("read_bout_context_select", () => {
  const metadata = { start_time: 0, end_time: 6 };
  const figureWith = (scores) => ({
    layout: { dragmode: "select" },
    data: [{}, { z: [scores] }],
  });

  test("right-click selects the whole contiguous same-label bout", () => {
    const figure = figureWith([0, 0, 1, 1, 1, 2]);
    const [range, patch, message, videoStyle] = fns.read_bout_context_select(
      1,
      { "detail.x": 3 },
      figure,
      metadata
    );

    expect(range).toEqual([2, 5]);
    const box = lastOp(patch, "layout", "shapes").value[0];
    expect(box).toMatchObject({ x0: 2, x1: 5, y0: -30, y1: 30, xref: "x4", yref: "y5" });
    expect(message).toBe(`You selected bout [2, 5] (3 s). ${ANNOTATION_HINT}`);
    expect(videoStyle).toEqual({ visibility: "visible" });
  });

  test("treats null and NaN scores as one contiguous unscored bout", () => {
    // The filesystem cache round-trip turns NaN into null (see app_src/session.py
    // notes), so both must count as the same "unscored" label.
    const figure = figureWith([0, null, NaN, 1]);
    const [range] = fns.read_bout_context_select(
      1,
      { "detail.x": 1 },
      figure,
      { start_time: 0, end_time: 4 }
    );

    expect(range).toEqual([1, 3]);
  });

  test("does nothing outside select mode", () => {
    const figure = { layout: { dragmode: "pan" }, data: [{ z: [[0, 1]] }] };
    expectAllNoUpdate(fns.read_bout_context_select(1, { "detail.x": 1 }, figure, metadata), 4);
  });

  test("does nothing without a click position", () => {
    const figure = figureWith([0, 1]);
    expectAllNoUpdate(fns.read_bout_context_select(1, {}, figure, metadata), 4);
    expectAllNoUpdate(
      fns.read_bout_context_select(1, { "detail.x": NaN }, figure, metadata),
      4
    );
  });
});

describe("read_annotation_auto_pan_select", () => {
  const metadata = { start_time: 0, end_time: 100 };
  const figure = { layout: { dragmode: "select" } };

  test("rounds the auto-pan drag selection and draws the select box", () => {
    const event = { "detail.x0": 10.4, "detail.x1": 20.6 };
    const [range, patch, message, videoStyle] = fns.read_annotation_auto_pan_select(
      1,
      event,
      figure,
      metadata
    );

    expect(range).toEqual([10, 21]);
    expect(lastOp(patch, "layout", "selections").value).toBeNull();
    const box = lastOp(patch, "layout", "shapes").value[0];
    expect(box).toMatchObject({ x0: 10, x1: 21, y0: -30, y1: 30, xref: "x4", yref: "y5" });
    expect(message).toBe(`You selected [10, 21] (11 s). ${ANNOTATION_HINT}`);
    expect(videoStyle).toEqual({ visibility: "visible" });
  });

  test("a sub-second drag at the recording edge still selects one second", () => {
    const event = { "detail.x0": 99.7, "detail.x1": 99.9 };
    const [range, , , videoStyle] = fns.read_annotation_auto_pan_select(
      1,
      event,
      figure,
      metadata
    );

    expect(range).toEqual([99, 100]);
    expect(videoStyle).toEqual({ visibility: "visible" });
  });

  test("rejects a drag entirely outside the recording", () => {
    const event = { "detail.x0": 150, "detail.x1": 160 };
    const [range, , message] = fns.read_annotation_auto_pan_select(1, event, figure, metadata);

    expect(range).toEqual([]);
    expect(message).toBe("Out of range. Please select from 0 to 100.");
  });

  test("does nothing for a zero-width drag or outside select mode", () => {
    const event = { "detail.x0": 10, "detail.x1": 10 };
    expectAllNoUpdate(fns.read_annotation_auto_pan_select(1, event, figure, metadata), 4);

    const panFigure = { layout: { dragmode: "pan" } };
    const realEvent = { "detail.x0": 10, "detail.x1": 20 };
    expectAllNoUpdate(
      fns.read_annotation_auto_pan_select(1, realEvent, panFigure, metadata),
      4
    );
  });
});

// ---- annotation ----

describe("make_annotation", () => {
  const figureWith = (scores) => ({
    layout: { dragmode: "select" },
    data: [{ z: [scores] }],
  });

  test("pressing 2 labels the selected half-open range as NREM", () => {
    const [videoStyle, scores, selection] = fns.make_annotation(
      1,
      { key: "2" },
      [1, 3],
      figureWith([0, 0, 0, 0, 0])
    );

    expect(scores).toEqual([0, 1, 1, 0, 0]); // [start, end): index 3 untouched
    expect(selection).toEqual([]);
    expect(videoStyle).toEqual({ visibility: "hidden" });
  });

  test("does not mutate the figure's own score array", () => {
    const figure = figureWith([0, 0, 0]);
    fns.make_annotation(1, { key: "4" }, [0, 3], figure);

    expect(figure.data[0].z[0]).toEqual([0, 0, 0]);
  });

  test("ignores keys other than 1-4", () => {
    expectAllNoUpdate(fns.make_annotation(1, { key: "5" }, [1, 3], figureWith([0, 0, 0])), 3);
    expectAllNoUpdate(fns.make_annotation(1, { key: "a" }, [1, 3], figureWith([0, 0, 0])), 3);
  });

  test("requires select mode and a selection", () => {
    const panFigure = { layout: { dragmode: "pan" }, data: [{ z: [[0]] }] };
    expectAllNoUpdate(fns.make_annotation(1, { key: "1" }, [0, 1], panFigure), 3);
    expectAllNoUpdate(fns.make_annotation(1, { key: "1" }, [], figureWith([0])), 3);
  });
});

describe("update_sleep_scores", () => {
  test("repaints the z of the last three (heatmap) traces and clears selections", () => {
    const figure = { data: [{}, {}, {}, {}, {}, {}, {}] };
    const scores = [1, 2, 0];
    const [patch, message] = fns.update_sleep_scores(scores, figure);

    [4, 5, 6].forEach((idx) => {
      expect(lastOp(patch, "data", idx, "z").value).toEqual([scores]);
    });
    expect(lastOp(patch, "layout", "selections").value).toBeNull();
    expect(lastOp(patch, "layout", "shapes").value).toBeNull();
    expect(message).toBe("");
  });

  test("non-array scores change nothing", () => {
    expectAllNoUpdate(fns.update_sleep_scores(null, { data: [] }), 2);
    expectAllNoUpdate(fns.update_sleep_scores("nope", { data: [] }), 2);
  });
});

// ---- message cleanup ----

describe("clear_display", () => {
  test("clears the message only on the fifth interval tick", () => {
    expect(fns.clear_display(5)).toBe("");
    expect(fns.clear_display(4)).toBe(NO);
    expect(fns.clear_display(6)).toBe(NO);
  });
});

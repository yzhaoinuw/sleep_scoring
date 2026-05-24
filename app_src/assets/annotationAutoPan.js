(function () {
    "use strict";

    if (window.sleepScoringAnnotationAutoPan) {
        return;
    }

    const EVENT_NAME = "sleepannotationselect";
    const GRAPH_ID = "graph";
    const EDGE_PX = 72;
    const CLICK_PX = 4;
    const MAX_PAN_DT_SECONDS = 0.05;
    const PAN_VIEW_WIDTH_PER_SECOND = 0.45;
    const AUTO_PAN_FRAME_MS = 33;
    const TRACE_REFRESH_MS = 180;
    const LEAD_OVERLAP_FRACTION = 0.2;
    const TRAIL_BUFFER_FRACTION = 0.2;
    const LEAD_BUFFER_FRACTION = 0.7;
    const BUFFER_SAMPLE_MULTIPLIER = 1.6;
    const RESAMPLE_ENDPOINT = "/_sleep_scoring/resample";
    const PROFILE_LOG_ENDPOINT = "/_sleep_scoring/profile-log";

    let dragState = null;
    let panFrame = null;
    let drawFrame = null;
    let traceTimer = null;
    let traceRequestInFlight = false;
    let pendingTraceRequest = null;
    let lastTraceRequestAt = 0;
    let latestTraceRequestId = 0;
    let latestAppliedTraceRequestId = 0;
    let autoPanReleaseTimer = null;
    let lastPanFrameAt = 0;

    function findPlot() {
        const graphRoot = document.getElementById(GRAPH_ID);
        if (!graphRoot) {
            return null;
        }
        return graphRoot.querySelector(".js-plotly-plot");
    }

    function postProfileLog(payload) {
        if (typeof window.sleepScoringProfileLog === "function") {
            window.sleepScoringProfileLog(payload);
            return;
        }

        const body = JSON.stringify(payload);
        if (navigator.sendBeacon) {
            const blob = new Blob([body], { type: "application/json" });
            if (navigator.sendBeacon(PROFILE_LOG_ENDPOINT, blob)) {
                return;
            }
        }

        window
            .fetch(PROFILE_LOG_ENDPOINT, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body,
                keepalive: true,
            })
            .catch(function () {});
    }

    function formatMs(value) {
        return Number.isFinite(value) ? value.toFixed(1) + " ms" : null;
    }

    function formatKb(value) {
        return Number.isFinite(value) ? value.toFixed(1) + " KB" : null;
    }

    function getSharedXAxis(fullLayout) {
        return fullLayout ? fullLayout.xaxis4 || fullLayout.xaxis : null;
    }

    function getAxisRange(axis) {
        if (axis && Array.isArray(axis.range) && axis.range.length === 2) {
            const x0 = Number(axis.range[0]);
            const x1 = Number(axis.range[1]);
            if (Number.isFinite(x0) && Number.isFinite(x1) && x0 !== x1) {
                return [Math.min(x0, x1), Math.max(x0, x1)];
            }
        }
        return null;
    }

    function getRecordingBounds(plot) {
        const meta =
            (plot && plot.layout && plot.layout.meta) ||
            (plot && plot._fullLayout && plot._fullLayout.meta);
        const bounds = meta && meta.sleepScoringXBounds;
        if (!Array.isArray(bounds) || bounds.length !== 2) {
            return null;
        }

        const x0 = Number(bounds[0]);
        const x1 = Number(bounds[1]);
        if (!Number.isFinite(x0) || !Number.isFinite(x1) || x0 === x1) {
            return null;
        }
        return [Math.min(x0, x1), Math.max(x0, x1)];
    }

    function clampRangeToBounds(range, bounds) {
        if (!bounds) {
            return range;
        }

        const width = range[1] - range[0];
        const boundsWidth = bounds[1] - bounds[0];
        if (!Number.isFinite(width) || width <= 0 || width >= boundsWidth) {
            return bounds;
        }

        const x0 = Math.min(Math.max(range[0], bounds[0]), bounds[1] - width);
        return [x0, x0 + width];
    }

    function getPrimaryYAxis(fullLayout, clientY, plotRect) {
        const plotY = clientY - plotRect.top;

        const axes = Object.keys(fullLayout || {})
            .filter((key) => /^yaxis[0-9]*$/.test(key))
            .map((key) => fullLayout[key])
            .filter((axis) => {
                return (
                    axis &&
                    axis._id &&
                    !axis.overlaying &&
                    Number.isFinite(axis._offset) &&
                    Number.isFinite(axis._length)
                );
            });

        for (const axis of axes) {
            const top = axis._offset;
            const bottom = axis._offset + axis._length;
            if (plotY >= top && plotY <= bottom) {
                return axis;
            }
        }

        return null;
    }

    function pointerToTime(plot, clientX) {
        const fullLayout = plot && plot._fullLayout;
        const xaxis = getSharedXAxis(fullLayout);
        if (
            !xaxis ||
            typeof xaxis.p2l !== "function" ||
            !Number.isFinite(xaxis._offset) ||
            !Number.isFinite(xaxis._length) ||
            xaxis._length <= 0
        ) {
            return null;
        }

        const plotRect = plot.getBoundingClientRect();
        const plotX = clientX - plotRect.left - xaxis._offset;
        const clippedPlotX = Math.max(0, Math.min(xaxis._length, plotX));
        const time = Number(xaxis.p2l(clippedPlotX));
        if (!Number.isFinite(time)) {
            return null;
        }

        return {
            time,
            plotX,
            xaxis,
            plotRect,
        };
    }

    function getSelectionShape() {
        if (!dragState) {
            return null;
        }

        return {
            type: "rect",
            xref: dragState.xref,
            yref: dragState.yref,
            x0: dragState.anchorTime,
            x1: dragState.currentTime,
            y0: dragState.y0,
            y1: dragState.y1,
            fillcolor: "rgba(99, 110, 250, 0.14)",
            line: { color: "rgba(99, 110, 250, 0.95)", width: 1, dash: "dot" },
            layer: "above",
        };
    }

    function scheduleDraw() {
        if (!dragState || drawFrame) {
            return;
        }

        drawFrame = window.requestAnimationFrame(function () {
            drawFrame = null;
            const shape = getSelectionShape();
            if (!shape || !window.Plotly) {
                return;
            }
            window.Plotly.relayout(dragState.plot, {
                selections: [],
                shapes: [shape],
            });
        });
    }

    function edgePressure(plotX, axisLength) {
        if (plotX < EDGE_PX) {
            return -Math.min(1, (EDGE_PX - plotX) / EDGE_PX);
        }
        if (plotX > axisLength - EDGE_PX) {
            return Math.min(1, (plotX - (axisLength - EDGE_PX)) / EDGE_PX);
        }
        return 0;
    }

    function stopAutoPan() {
        if (panFrame) {
            window.cancelAnimationFrame(panFrame);
            panFrame = null;
        }
        if (dragState) {
            dragState.lastPanAt = null;
        }
    }

    function updateCurrentTimeFromPointer() {
        if (!dragState) {
            return null;
        }

        const info = pointerToTime(dragState.plot, dragState.lastClientX);
        if (!info) {
            return null;
        }
        dragState.currentTime = info.time;
        return info;
    }

    function groupPatchOperations(patch) {
        const operations = patch && Array.isArray(patch.operations) ? patch.operations : [];
        const grouped = new Map();

        for (const operation of operations) {
            const location = operation.location;
            if (
                !Array.isArray(location) ||
                location.length < 3 ||
                location[0] !== "data" ||
                operation.operation !== "Assign"
            ) {
                continue;
            }

            if (!operation.params || !Object.prototype.hasOwnProperty.call(operation.params, "value")) {
                continue;
            }

            const traceIndex = Number(location[1]);
            const traceProperty = location[2];
            if (
                !Number.isInteger(traceIndex) ||
                (traceProperty !== "x" && traceProperty !== "y")
            ) {
                continue;
            }

            if (!grouped.has(traceIndex)) {
                grouped.set(traceIndex, {});
            }
            grouped.get(traceIndex)[traceProperty] = [operation.params.value];
        }

        return grouped;
    }

    function getLeadRange(visibleRange, pressure) {
        const width = visibleRange[1] - visibleRange[0];
        if (!Number.isFinite(width) || width <= 0 || pressure === 0) {
            return visibleRange;
        }

        if (pressure > 0) {
            return [
                visibleRange[1] - width * LEAD_OVERLAP_FRACTION,
                visibleRange[1] + width * (1 - LEAD_OVERLAP_FRACTION),
            ];
        }

        return [
            visibleRange[0] - width * (1 - LEAD_OVERLAP_FRACTION),
            visibleRange[0] + width * LEAD_OVERLAP_FRACTION,
        ];
    }

    function getTrimRange(visibleRange, pressure) {
        const width = visibleRange[1] - visibleRange[0];
        if (!Number.isFinite(width) || width <= 0 || pressure === 0) {
            return visibleRange;
        }

        if (pressure > 0) {
            return [
                visibleRange[0] - width * TRAIL_BUFFER_FRACTION,
                visibleRange[1] + width * LEAD_BUFFER_FRACTION,
            ];
        }

        return [
            visibleRange[0] - width * LEAD_BUFFER_FRACTION,
            visibleRange[1] + width * TRAIL_BUFFER_FRACTION,
        ];
    }

    function toArray(values) {
        if (!values) {
            return [];
        }

        return Array.from(values);
    }

    function decimatePoints(points, maxPoints) {
        if (points.length <= maxPoints || maxPoints < 2) {
            return points;
        }

        const decimated = [];
        const lastIndex = points.length - 1;
        for (let index = 0; index < maxPoints; index += 1) {
            const sourceIndex = Math.round((index * lastIndex) / (maxPoints - 1));
            decimated.push(points[sourceIndex]);
        }
        return decimated;
    }

    function mergeTraceArrays(existingX, existingY, incomingX, incomingY, trimRange) {
        const points = [];
        const incomingXs = toArray(incomingX);
        const incomingYs = toArray(incomingY);
        const maxPoints = Math.max(2, Math.ceil(incomingXs.length * BUFFER_SAMPLE_MULTIPLIER));

        function addPoints(xs, ys) {
            const count = Math.min(xs.length, ys.length);
            for (let index = 0; index < count; index += 1) {
                const x = Number(xs[index]);
                if (!Number.isFinite(x) || x < trimRange[0] || x > trimRange[1]) {
                    continue;
                }
                points.push({ x, y: ys[index] });
            }
        }

        addPoints(toArray(existingX), toArray(existingY));
        addPoints(incomingXs, incomingYs);
        points.sort((left, right) => left.x - right.x);

        const mergedX = [];
        const mergedY = [];
        for (const point of decimatePoints(points, maxPoints)) {
            const lastIndex = mergedX.length - 1;
            if (lastIndex >= 0 && mergedX[lastIndex] === point.x) {
                mergedY[lastIndex] = point.y;
            } else {
                mergedX.push(point.x);
                mergedY.push(point.y);
            }
        }

        return { x: mergedX, y: mergedY };
    }

    function buildTraceUpdates(plot, groupedOperations, requestConfig) {
        const traceIndices = Array.from(groupedOperations.keys())
            .filter((traceIndex) => {
                const update = groupedOperations.get(traceIndex);
                return update.x && update.y;
            })
            .sort((left, right) => left - right);

        const xUpdates = [];
        const yUpdates = [];
        const updatedTraceIndices = [];
        const shouldMerge = requestConfig.mode === "merge";
        const trimRange = getTrimRange(requestConfig.visibleRange, requestConfig.pressure);

        for (const traceIndex of traceIndices) {
            const update = groupedOperations.get(traceIndex);
            let x = update.x[0];
            let y = update.y[0];

            if (shouldMerge && plot.data && plot.data[traceIndex]) {
                const merged = mergeTraceArrays(
                    plot.data[traceIndex].x,
                    plot.data[traceIndex].y,
                    x,
                    y,
                    trimRange
                );
                x = merged.x;
                y = merged.y;
            }

            updatedTraceIndices.push(traceIndex);
            xUpdates.push(x);
            yUpdates.push(y);
        }

        return {
            traceIndices: updatedTraceIndices,
            update: { x: xUpdates, y: yUpdates },
        };
    }

    function applyTracePatch(plot, patch, requestId, requestConfig) {
        if (!plot || !window.Plotly || requestId < latestAppliedTraceRequestId) {
            return { applied: false, traces: 0, points: 0 };
        }

        latestAppliedTraceRequestId = requestId;
        const groupedOperations = groupPatchOperations(patch);
        const tracePatch = buildTraceUpdates(plot, groupedOperations, requestConfig);

        if (tracePatch.traceIndices.length === 0) {
            return { applied: false, traces: 0, points: 0 };
        }

        window.Plotly.restyle(plot, tracePatch.update, tracePatch.traceIndices);
        const points = tracePatch.update.x.reduce((total, traceX) => total + traceX.length, 0);
        return { applied: true, traces: tracePatch.traceIndices.length, points };
    }

    function buildResampleUrl(range) {
        const url = new URL(RESAMPLE_ENDPOINT, window.location.origin);
        url.searchParams.set("x0", range[0]);
        url.searchParams.set("x1", range[1]);
        return url;
    }

    function startTraceRequest(requestConfig) {
        if (traceRequestInFlight) {
            pendingTraceRequest = requestConfig;
            return;
        }

        traceRequestInFlight = true;
        lastTraceRequestAt = Date.now();
        const requestId = ++latestTraceRequestId;
        const plot = requestConfig.plot;
        const requestStartedAt = performance.now();
        const queuedMs = requestStartedAt - requestConfig.queuedAt;
        let fetchMs = null;
        let payloadKb = null;
        let parseMs = null;

        window
            .fetch(buildResampleUrl(requestConfig.range), { cache: "no-store" })
            .then((response) => {
                if (!response.ok) {
                    throw new Error(`Resample request failed: ${response.status}`);
                }
                const fetchFinishedAt = performance.now();
                fetchMs = fetchFinishedAt - requestStartedAt;
                return response.text();
            })
            .then((text) => {
                payloadKb = new Blob([text]).size / 1024;
                const parseStartedAt = performance.now();
                const patch = JSON.parse(text);
                parseMs = performance.now() - parseStartedAt;
                return patch;
            })
            .then((patch) => {
                const applyStartedAt = performance.now();
                const applyStats = applyTracePatch(plot, patch, requestId, requestConfig);
                const applyMs = performance.now() - applyStartedAt;
                postProfileLog({
                    event: "browser-autopan",
                    id: requestId,
                    mode: requestConfig.mode,
                    queued: formatMs(queuedMs),
                    fetch: formatMs(fetchMs),
                    parse: formatMs(parseMs),
                    apply: formatMs(applyMs),
                    browser_total: formatMs(performance.now() - requestConfig.queuedAt),
                    payload: formatKb(payloadKb),
                    x_width: Math.abs(requestConfig.range[1] - requestConfig.range[0]).toFixed(1) + " s",
                    visible_width:
                        Math.abs(requestConfig.visibleRange[1] - requestConfig.visibleRange[0]).toFixed(1) +
                        " s",
                    traces: applyStats.traces,
                    points: applyStats.points,
                    applied: applyStats.applied,
                });
            })
            .catch(() => {
                // Keep dragging responsive even if one live refresh misses.
            })
            .finally(() => {
                traceRequestInFlight = false;
                if (pendingTraceRequest) {
                    const nextRequest = pendingTraceRequest;
                    pendingTraceRequest = null;
                    requestTraceRefresh(
                        nextRequest.visibleRange,
                        nextRequest.pressure,
                        true,
                        nextRequest.plot,
                        nextRequest.mode
                    );
                }
            });
    }

    function requestTraceRefresh(visibleRange, pressure, immediate, plot, mode) {
        if (!visibleRange || !plot) {
            return;
        }

        const requestMode = mode || "replace";
        const bounds = getRecordingBounds(plot);
        const rawRequestRange =
            requestMode === "merge" ? getLeadRange(visibleRange, pressure) : visibleRange;
        const requestRange = clampRangeToBounds(rawRequestRange, bounds);
        const requestConfig = {
            visibleRange,
            range: requestRange,
            pressure,
            plot,
            mode: requestMode,
            queuedAt: performance.now(),
        };
        pendingTraceRequest = requestConfig;

        window.clearTimeout(traceTimer);
        const elapsed = Date.now() - lastTraceRequestAt;
        if (immediate || elapsed >= TRACE_REFRESH_MS) {
            pendingTraceRequest = null;
            startTraceRequest(requestConfig);
        } else {
            traceTimer = window.setTimeout(function () {
                const nextRequest = pendingTraceRequest;
                pendingTraceRequest = null;
                startTraceRequest(nextRequest);
            }, TRACE_REFRESH_MS - elapsed);
        }
    }

    function finishTraceRefresh() {
        window.clearTimeout(traceTimer);
        traceTimer = null;
        pendingTraceRequest = null;
    }

    function setAutoPanActive(active) {
        window.clearTimeout(autoPanReleaseTimer);
        if (active) {
            window.sleepScoringAnnotationAutoPanActive = true;
            return;
        }

        autoPanReleaseTimer = window.setTimeout(function () {
            window.sleepScoringAnnotationAutoPanActive = false;
        }, 250);
    }

    function autoPanStep(now) {
        panFrame = null;
        if (!dragState || !window.Plotly) {
            return;
        }

        if (lastPanFrameAt && now - lastPanFrameAt < AUTO_PAN_FRAME_MS) {
            panFrame = window.requestAnimationFrame(autoPanStep);
            return;
        }

        const info = pointerToTime(dragState.plot, dragState.lastClientX);
        if (!info) {
            stopAutoPan();
            return;
        }

        const pressure = edgePressure(info.plotX, info.xaxis._length);
        if (pressure === 0) {
            stopAutoPan();
            return;
        }

        const range = getAxisRange(info.xaxis);
        if (!range) {
            stopAutoPan();
            return;
        }

        const previousPanAt = dragState.lastPanAt || now;
        const dtSeconds = Math.min((now - previousPanAt) / 1000, MAX_PAN_DT_SECONDS);
        dragState.lastPanAt = now;

        const width = range[1] - range[0];
        const delta = pressure * Math.abs(pressure) * width * PAN_VIEW_WIDTH_PER_SECOND * dtSeconds;
        const nextRange = [range[0] + delta, range[1] + delta];
        lastPanFrameAt = now;

        const clippedPlotX = Math.max(0, Math.min(info.xaxis._length, info.plotX));
        dragState.currentTime = nextRange[0] + (clippedPlotX / info.xaxis._length) * width;
        const shape = getSelectionShape();

        window.Plotly.relayout(dragState.plot, {
            "xaxis4.range": nextRange,
            selections: [],
            shapes: shape ? [shape] : [],
        });
        requestTraceRefresh(nextRange, pressure, false, dragState.plot, "merge");

        panFrame = window.requestAnimationFrame(autoPanStep);
    }

    function updateAutoPan() {
        if (!dragState) {
            return;
        }

        const info = pointerToTime(dragState.plot, dragState.lastClientX);
        if (!info) {
            stopAutoPan();
            return;
        }

        const pressure = edgePressure(info.plotX, info.xaxis._length);
        if (pressure === 0) {
            stopAutoPan();
            return;
        }

        if (!panFrame) {
            dragState.lastPanAt = null;
            lastPanFrameAt = 0;
            panFrame = window.requestAnimationFrame(autoPanStep);
        }
    }

    function stopEvent(event) {
        event.preventDefault();
        event.stopPropagation();
        if (typeof event.stopImmediatePropagation === "function") {
            event.stopImmediatePropagation();
        }
    }

    function isSelectMode(plot) {
        const fullLayout = plot && plot._fullLayout;
        return fullLayout && fullLayout.dragmode === "select";
    }

    function beginDrag(event) {
        if (event.button !== 0 || dragState) {
            return;
        }

        const graphRoot = event.target.closest ? event.target.closest(`#${GRAPH_ID}`) : null;
        if (!graphRoot || event.target.closest(".modebar")) {
            return;
        }

        const plot = findPlot();
        if (!plot || !plot._fullLayout || !isSelectMode(plot)) {
            return;
        }

        const pointerInfo = pointerToTime(plot, event.clientX);
        if (!pointerInfo || pointerInfo.plotX < 0 || pointerInfo.plotX > pointerInfo.xaxis._length) {
            return;
        }

        const yaxis = getPrimaryYAxis(plot._fullLayout, event.clientY, pointerInfo.plotRect);
        if (!yaxis) {
            return;
        }

        const yRange = Array.isArray(yaxis.range) ? yaxis.range : [-30, 30];
        dragState = {
            plot,
            graphRoot,
            pointerId: event.pointerId,
            anchorTime: pointerInfo.time,
            currentTime: pointerInfo.time,
            startClientX: event.clientX,
            startClientY: event.clientY,
            lastClientX: event.clientX,
            lastClientY: event.clientY,
            lastPanAt: null,
            xref: pointerInfo.xaxis._id || "x4",
            yref: yaxis._id || "y5",
            y0: Number.isFinite(Number(yRange[0])) ? Number(yRange[0]) : -30,
            y1: Number.isFinite(Number(yRange[1])) ? Number(yRange[1]) : 30,
            didDrag: false,
        };

        setAutoPanActive(true);
        stopEvent(event);
        if (graphRoot.setPointerCapture && event.pointerId !== undefined) {
            graphRoot.setPointerCapture(event.pointerId);
        }
    }

    function continueDrag(event) {
        if (!dragState || event.pointerId !== dragState.pointerId) {
            return;
        }

        dragState.lastClientX = event.clientX;
        dragState.lastClientY = event.clientY;

        const dx = event.clientX - dragState.startClientX;
        const dy = event.clientY - dragState.startClientY;
        if (Math.hypot(dx, dy) >= CLICK_PX) {
            dragState.didDrag = true;
        }

        updateCurrentTimeFromPointer();
        scheduleDraw();
        updateAutoPan();
        stopEvent(event);
    }

    function dispatchSelection(event) {
        if (!dragState) {
            return;
        }

        updateCurrentTimeFromPointer();

        let x0 = dragState.anchorTime;
        let x1 = dragState.currentTime;
        const kind = dragState.didDrag ? "drag" : "click";

        if (kind === "click") {
            const info = pointerToTime(dragState.plot, dragState.lastClientX);
            const range = info ? getAxisRange(info.xaxis) : null;
            const width = range ? range[1] - range[0] : 200;
            const delta = width * 0.005;
            x0 = Math.floor(dragState.anchorTime - delta / 2);
            x1 = Math.ceil(dragState.anchorTime + delta / 2);
        }

        document.dispatchEvent(
            new CustomEvent(EVENT_NAME, {
                detail: {
                    x0: Math.min(x0, x1),
                    x1: Math.max(x0, x1),
                    xref: dragState.xref,
                    yref: dragState.yref,
                    y0: dragState.y0,
                    y1: dragState.y1,
                    kind,
                    timeStamp: event.timeStamp,
                },
            })
        );
    }

    function endDrag(event) {
        if (!dragState || event.pointerId !== dragState.pointerId) {
            return;
        }

        stopEvent(event);
        stopAutoPan();
        const finalInfo = updateCurrentTimeFromPointer();
        const finalRange = finalInfo ? getAxisRange(finalInfo.xaxis) : null;
        if (finalRange) {
            requestTraceRefresh(finalRange, 0, true, dragState.plot, "replace");
        } else {
            finishTraceRefresh();
        }
        dispatchSelection(event);

        if (dragState.graphRoot.releasePointerCapture && event.pointerId !== undefined) {
            try {
                dragState.graphRoot.releasePointerCapture(event.pointerId);
            } catch (error) {
                // Pointer capture may already be released by the browser.
            }
        }

        dragState = null;
        setAutoPanActive(false);
    }

    document.addEventListener("pointerdown", beginDrag, true);
    document.addEventListener("pointermove", continueDrag, true);
    document.addEventListener("pointerup", endDrag, true);
    document.addEventListener("pointercancel", endDrag, true);

    window.sleepScoringAnnotationAutoPan = true;
})();

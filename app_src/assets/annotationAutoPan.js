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
    const RESAMPLE_ENDPOINT = "/_sleep_scoring/resample";

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

    function applyTracePatch(plot, patch, requestId) {
        if (!plot || !window.Plotly || requestId < latestAppliedTraceRequestId) {
            return;
        }

        latestAppliedTraceRequestId = requestId;
        const groupedOperations = groupPatchOperations(patch);
        const traceIndices = Array.from(groupedOperations.keys())
            .filter((traceIndex) => {
                const update = groupedOperations.get(traceIndex);
                return update.x && update.y;
            })
            .sort((left, right) => left - right);

        if (traceIndices.length === 0) {
            return;
        }

        window.Plotly.restyle(
            plot,
            {
                x: traceIndices.map((traceIndex) => groupedOperations.get(traceIndex).x[0]),
                y: traceIndices.map((traceIndex) => groupedOperations.get(traceIndex).y[0]),
            },
            traceIndices
        );
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

        window
            .fetch(buildResampleUrl(requestConfig.range), { cache: "no-store" })
            .then((response) => {
                if (!response.ok) {
                    throw new Error(`Resample request failed: ${response.status}`);
                }
                return response.json();
            })
            .then((patch) => applyTracePatch(plot, patch, requestId))
            .catch(() => {
                // Keep dragging responsive even if one live refresh misses.
            })
            .finally(() => {
                traceRequestInFlight = false;
                if (pendingTraceRequest) {
                    const nextRequest = pendingTraceRequest;
                    pendingTraceRequest = null;
                    requestTraceRefresh(nextRequest.rawRange, nextRequest.pressure, true, nextRequest.plot);
                }
            });
    }

    function requestTraceRefresh(rawRange, pressure, immediate, plot) {
        if (!rawRange || !plot) {
            return;
        }

        const requestConfig = { rawRange, range: rawRange, pressure, plot };
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
        requestTraceRefresh(nextRange, pressure, false, dragState.plot);

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
            requestTraceRefresh(finalRange, 0, true, dragState.plot);
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

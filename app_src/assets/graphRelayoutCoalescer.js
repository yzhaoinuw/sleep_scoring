(function () {
    "use strict";

    if (window.sleepScoringGraphRelayout) {
        return;
    }

    const EVENT_NAME = "sleepgraphrelayout";
    const GRAPH_ID = "graph";
    const DEBOUNCE_MS = 90;
    const MAX_WAIT_MS = 250;
    const FINAL_IDLE_MS = 450;

    let pendingRange = null;
    let debounceTimer = null;
    let finalTimer = null;
    let firstPendingAt = null;
    let lastDispatch = null;
    let attachedPlot = null;

    function getValue(data, dottedKey, bracketIndex) {
        if (!data) {
            return undefined;
        }

        const bracketKey = `${dottedKey}[${bracketIndex}]`;
        if (data[bracketKey] !== undefined) {
            return data[bracketKey];
        }

        const range = data[dottedKey];
        if (Array.isArray(range)) {
            return range[bracketIndex];
        }

        return undefined;
    }

    function extractXRange(relayoutData) {
        let x0 = getValue(relayoutData, "xaxis4.range", 0);
        let x1 = getValue(relayoutData, "xaxis4.range", 1);

        if (x0 === undefined || x1 === undefined) {
            x0 = getValue(relayoutData, "xaxis.range", 0);
            x1 = getValue(relayoutData, "xaxis.range", 1);
        }

        x0 = Number(x0);
        x1 = Number(x1);
        if (!Number.isFinite(x0) || !Number.isFinite(x1) || x0 === x1) {
            return null;
        }

        return [Math.min(x0, x1), Math.max(x0, x1)];
    }

    function dispatchRange(range, source, mode) {
        const [x0, x1] = range;
        const now = Date.now();
        if (
            lastDispatch &&
            lastDispatch.mode === mode &&
            Math.abs(lastDispatch.x0 - x0) < 1e-9 &&
            Math.abs(lastDispatch.x1 - x1) < 1e-9 &&
            now - lastDispatch.timeStamp < MAX_WAIT_MS
        ) {
            return;
        }

        lastDispatch = { x0, x1, mode, timeStamp: now };
        const event = new CustomEvent(EVENT_NAME, {
            detail: {
                x0,
                x1,
                source,
                mode,
                timeStamp: now,
            },
        });
        document.dispatchEvent(event);
    }

    function dispatchPending() {
        if (!pendingRange) {
            return;
        }

        dispatchRange(pendingRange.range, pendingRange.source, "fast");
        pendingRange = null;
        firstPendingAt = null;
    }

    function requestFast(range, source) {
        const now = Date.now();
        pendingRange = { range, source: source || "plotly" };
        if (firstPendingAt === null) {
            firstPendingAt = now;
        }

        window.clearTimeout(debounceTimer);
        if (now - firstPendingAt >= MAX_WAIT_MS) {
            dispatchPending();
        } else {
            debounceTimer = window.setTimeout(dispatchPending, DEBOUNCE_MS);
        }
    }

    function requestFinal(range, source, delay) {
        window.clearTimeout(finalTimer);
        finalTimer = window.setTimeout(function () {
            dispatchRange(range, source || "plotly", "final");
        }, delay);
    }

    function request(relayoutData, source) {
        const range = extractXRange(relayoutData);
        if (!range) {
            return;
        }

        requestFast(range, source);
        requestFinal(range, source, FINAL_IDLE_MS);
    }

    function findPlot() {
        const graphRoot = document.getElementById(GRAPH_ID);
        if (!graphRoot) {
            return null;
        }
        return graphRoot.querySelector(".js-plotly-plot");
    }

    function attachPlotlyListener() {
        const plot = findPlot();
        if (!plot || plot === attachedPlot || typeof plot.on !== "function") {
            return;
        }

        attachedPlot = plot;
        plot.on("plotly_relayouting", function (relayoutData) {
            request(relayoutData, "plotly-moving");
        });
        plot.on("plotly_relayout", function (relayoutData) {
            request(relayoutData, "plotly");
        });
    }

    window.sleepScoringGraphRelayout = {
        request,
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", attachPlotlyListener);
    } else {
        attachPlotlyListener();
    }

    const observer = new MutationObserver(attachPlotlyListener);
    observer.observe(document.documentElement, { childList: true, subtree: true });
})();

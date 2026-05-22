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
    const KEYBOARD_FINAL_IDLE_MS = 120;
    const RELEASE_FINAL_DELAY_MS = 25;
    const NAVIGATION_OPTIONS_ID = "navigation-options";
    const RANGE_EQUAL_REL_TOLERANCE = 0.00005;
    const RANGE_EQUAL_ABS_TOLERANCE = 0.05;
    const FINAL_RANGE_EQUAL_REL_TOLERANCE = 0.001;
    const FINAL_RANGE_EQUAL_ABS_TOLERANCE = 0.25;

    let pendingRange = null;
    let debounceTimer = null;
    let finalTimer = null;
    let firstPendingAt = null;
    let lastDispatch = null;
    let lastFinalDispatch = null;
    let attachedPlot = null;
    let profileId = 0;
    let suppressPlotlyRelayoutUntil = 0;

    function nowPerformance() {
        if (window.performance && typeof window.performance.now === "function") {
            return window.performance.now();
        }
        return Date.now();
    }

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

    function rangesNearlyEqual(a, b, relTolerance, absTolerance) {
        if (!a || !b) {
            return false;
        }

        const width = Math.max(Math.abs(a[1] - a[0]), Math.abs(b[1] - b[0]), 1);
        const tolerance = Math.max(absTolerance, width * relTolerance);
        return Math.abs(a[0] - b[0]) <= tolerance && Math.abs(a[1] - b[1]) <= tolerance;
    }

    function dispatchRange(range, source, mode, inputPerformanceTime) {
        const [x0, x1] = range;
        const now = Date.now();
        if (
            lastDispatch &&
            lastDispatch.mode === mode &&
            rangesNearlyEqual(
                [lastDispatch.x0, lastDispatch.x1],
                range,
                RANGE_EQUAL_REL_TOLERANCE,
                RANGE_EQUAL_ABS_TOLERANCE
            ) &&
            (mode === "fast" || now - lastDispatch.timeStamp < MAX_WAIT_MS)
        ) {
            return;
        }
        if (
            mode === "final" &&
            lastFinalDispatch &&
            rangesNearlyEqual(
                [lastFinalDispatch.x0, lastFinalDispatch.x1],
                range,
                FINAL_RANGE_EQUAL_REL_TOLERANCE,
                FINAL_RANGE_EQUAL_ABS_TOLERANCE
            )
        ) {
            return;
        }

        const dispatchPerformanceTime = nowPerformance();
        const currentProfileId = ++profileId;
        lastDispatch = { x0, x1, mode, timeStamp: now };
        if (mode === "final") {
            lastFinalDispatch = { x0, x1, timeStamp: now };
        }
        if (window.sleepScoringNavigationProfiler) {
            window.sleepScoringNavigationProfiler.markDispatched({
                profileId: currentProfileId,
                mode,
                source,
                x0,
                x1,
                inputPerformanceTime,
                dispatchPerformanceTime,
            });
        }
        const event = new CustomEvent(EVENT_NAME, {
            detail: {
                x0,
                x1,
                source,
                mode,
                timeStamp: now,
                profileId: currentProfileId,
                inputPerformanceTime,
                dispatchPerformanceTime,
            },
        });
        document.dispatchEvent(event);
    }

    function dispatchPending() {
        if (!pendingRange) {
            return;
        }

        dispatchRange(
            pendingRange.range,
            pendingRange.source,
            "fast",
            pendingRange.inputPerformanceTime
        );
        pendingRange = null;
        firstPendingAt = null;
    }

    function shouldSendFastTraceEvents() {
        const options = document.getElementById(NAVIGATION_OPTIONS_ID);
        return Boolean(
            options &&
                options.dataset &&
                options.dataset.sendFastTraceEvents === "true"
        );
    }

    function requestFast(range, source) {
        if (!shouldSendFastTraceEvents()) {
            return;
        }

        const now = Date.now();
        pendingRange = {
            range,
            source: source || "plotly",
            inputPerformanceTime: nowPerformance(),
        };
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
        const inputPerformanceTime = nowPerformance();
        window.clearTimeout(finalTimer);
        finalTimer = window.setTimeout(function () {
            dispatchRange(range, source || "plotly", "final", inputPerformanceTime);
        }, delay);
    }

    function finalDelayForSource(source) {
        if (source === "keyboard") {
            return KEYBOARD_FINAL_IDLE_MS;
        }
        return FINAL_IDLE_MS;
    }

    function request(relayoutData, source) {
        const range = extractXRange(relayoutData);
        if (!range) {
            return;
        }

        requestFast(range, source);
        requestFinal(range, source, finalDelayForSource(source));
    }

    function requestFinalOnly(relayoutData, source, delay) {
        const range = extractXRange(relayoutData);
        if (!range) {
            return;
        }

        window.clearTimeout(debounceTimer);
        pendingRange = null;
        firstPendingAt = null;
        requestFinal(range, source, delay === undefined ? 0 : delay);
    }

    function isCustomPointerPanActive() {
        return Boolean(
            window.sleepScoringCustomPointerPan &&
                window.sleepScoringCustomPointerPan.isActive === true
        ) || Date.now() < suppressPlotlyRelayoutUntil;
    }

    function suppressPlotlyRelayoutFor(durationMs) {
        suppressPlotlyRelayoutUntil = Math.max(
            suppressPlotlyRelayoutUntil,
            Date.now() + durationMs
        );
    }

    function resetDispatchState() {
        pendingRange = null;
        firstPendingAt = null;
        lastDispatch = null;
        lastFinalDispatch = null;
        window.clearTimeout(debounceTimer);
        window.clearTimeout(finalTimer);
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

        resetDispatchState();
        attachedPlot = plot;
        plot.on("plotly_relayouting", function (relayoutData) {
            if (isCustomPointerPanActive()) {
                return;
            }
            request(relayoutData, "plotly-moving");
        });
        plot.on("plotly_relayout", function (relayoutData) {
            if (isCustomPointerPanActive()) {
                return;
            }
            requestFinalOnly(relayoutData, "plotly", RELEASE_FINAL_DELAY_MS);
        });
    }

    window.sleepScoringGraphRelayout = {
        request,
        requestFinalOnly,
        suppressPlotlyRelayoutFor,
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", attachPlotlyListener);
    } else {
        attachPlotlyListener();
    }

    const observer = new MutationObserver(attachPlotlyListener);
    observer.observe(document.documentElement, { childList: true, subtree: true });
})();

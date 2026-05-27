(function () {
    "use strict";

    if (window.sleepScoringCustomPointerPan) {
        return;
    }

    const GRAPH_ID = "graph";
    const ENABLE_CUSTOM_POINTER_PAN = true;

    let attachedPlot = null;
    let dragState = null;
    let pendingClientX = null;
    let pendingClientY = null;
    let rafId = null;

    function findPlot() {
        const graphRoot = document.getElementById(GRAPH_ID);
        if (!graphRoot) {
            return null;
        }
        return graphRoot.querySelector(".js-plotly-plot");
    }

    function isInteractiveChrome(target) {
        return Boolean(
            target &&
                target.closest &&
                target.closest(".modebar, .legend, .rangeslider-container, .select-outline")
        );
    }

    function getXAxis(plot) {
        return plot && plot._fullLayout && plot._fullLayout.xaxis4;
    }

    function getPointerYAxis(plot, clientY) {
        if (!plot || !plot._fullLayout) {
            return null;
        }

        const rect = plot.getBoundingClientRect();
        const plotY = clientY - rect.top;
        const candidates = ["yaxis3", "yaxis4"];
        for (const axisName of candidates) {
            const axis = plot._fullLayout[axisName];
            const offset = axis && Number(axis._offset);
            const length = axis && Number(axis._length);
            if (
                axis &&
                !axis.fixedrange &&
                Number.isFinite(offset) &&
                Number.isFinite(length) &&
                plotY >= offset &&
                plotY <= offset + length
            ) {
                return { axis, axisName };
            }
        }

        return null;
    }

    function getDragMode(plot) {
        const layout = (plot && plot.layout) || {};
        const fullLayout = (plot && plot._fullLayout) || {};
        return layout.dragmode || fullLayout.dragmode;
    }

    function canPan(plot, event) {
        if (!ENABLE_CUSTOM_POINTER_PAN || !plot || event.button !== 0) {
            return false;
        }
        if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) {
            return false;
        }
        if (isInteractiveChrome(event.target)) {
            return false;
        }
        return getDragMode(plot) === "pan";
    }

    function stopNativeDrag(event) {
        event.preventDefault();
        event.stopPropagation();
        if (typeof event.stopImmediatePropagation === "function") {
            event.stopImmediatePropagation();
        }
    }

    function currentRange(axis) {
        if (!axis || !Array.isArray(axis.range) || axis.range.length !== 2) {
            return null;
        }
        const x0 = Number(axis.range[0]);
        const x1 = Number(axis.range[1]);
        if (!Number.isFinite(x0) || !Number.isFinite(x1) || x0 === x1) {
            return null;
        }
        return [x0, x1];
    }

    function applyPendingPan() {
        rafId = null;
        if (
            !dragState ||
            pendingClientX === null ||
            pendingClientY === null ||
            !window.Plotly
        ) {
            return;
        }

        const dx = pendingClientX - dragState.startClientX;
        const width = dragState.startXRange[1] - dragState.startXRange[0];
        const secondsPerPixel = width / dragState.xAxisLength;
        const shift = -dx * secondsPerPixel;
        const nextXRange = [
            dragState.startXRange[0] + shift,
            dragState.startXRange[1] + shift,
        ];
        const relayout = {
            "xaxis4.range[0]": nextXRange[0],
            "xaxis4.range[1]": nextXRange[1],
        };

        if (dragState.startYRange && dragState.yAxisName && dragState.yAxisLength) {
            const dy = pendingClientY - dragState.startClientY;
            const yWidth = dragState.startYRange[1] - dragState.startYRange[0];
            const unitsPerPixel = yWidth / dragState.yAxisLength;
            const yShift = dy * unitsPerPixel;
            const nextYRange = [
                dragState.startYRange[0] + yShift,
                dragState.startYRange[1] + yShift,
            ];
            relayout[`${dragState.yAxisName}.range[0]`] = nextYRange[0];
            relayout[`${dragState.yAxisName}.range[1]`] = nextYRange[1];
        }

        dragState.latestRange = nextXRange;
        if (
            window.sleepScoringGraphRelayout &&
            typeof window.sleepScoringGraphRelayout.suppressPlotlyRelayoutFor === "function"
        ) {
            window.sleepScoringGraphRelayout.suppressPlotlyRelayoutFor(250);
        }
        window.Plotly.relayout(dragState.plot, relayout);
    }

    function schedulePan(clientX, clientY) {
        pendingClientX = clientX;
        pendingClientY = clientY;
        if (rafId === null) {
            rafId = window.requestAnimationFrame(applyPendingPan);
        }
    }

    function onPointerMove(event) {
        if (!dragState) {
            return;
        }
        stopNativeDrag(event);
        schedulePan(event.clientX, event.clientY);
    }

    function requestFinalRange() {
        if (!dragState || !dragState.latestRange || !window.sleepScoringGraphRelayout) {
            return;
        }
        if (typeof window.sleepScoringGraphRelayout.suppressPlotlyRelayoutFor === "function") {
            window.sleepScoringGraphRelayout.suppressPlotlyRelayoutFor(500);
        }
        const [x0, x1] = dragState.latestRange;
        window.sleepScoringGraphRelayout.requestFinalOnly(
            {
                "xaxis4.range[0]": x0,
                "xaxis4.range[1]": x1,
            },
            "custom-drag",
            0
        );
    }

    function clearDragState() {
        if (rafId !== null) {
            window.cancelAnimationFrame(rafId);
            rafId = null;
        }
        pendingClientX = null;
        pendingClientY = null;
        dragState = null;
        window.sleepScoringCustomPointerPan.isActive = false;
    }

    function onPointerUp(event) {
        if (!dragState) {
            return;
        }
        stopNativeDrag(event);
        schedulePan(event.clientX, event.clientY);
        if (rafId !== null) {
            window.cancelAnimationFrame(rafId);
            applyPendingPan();
        }
        requestFinalRange();
        clearDragState();
        window.removeEventListener("pointermove", onPointerMove, true);
        window.removeEventListener("pointerup", onPointerUp, true);
        window.removeEventListener("pointercancel", onPointerUp, true);
    }

    function onPointerDown(event) {
        const plot = attachedPlot;
        if (!canPan(plot, event)) {
            return;
        }

        const axis = getXAxis(plot);
        const range = currentRange(axis);
        const axisLength = axis && Number(axis._length);
        if (!range || !Number.isFinite(axisLength) || axisLength <= 0) {
            return;
        }
        const yAxisMatch = getPointerYAxis(plot, event.clientY);
        const yRange = yAxisMatch ? currentRange(yAxisMatch.axis) : null;
        const yAxisLength = yAxisMatch && Number(yAxisMatch.axis._length);

        stopNativeDrag(event);
        window.sleepScoringCustomPointerPan.isActive = true;
        dragState = {
            plot,
            startClientX: event.clientX,
            startClientY: event.clientY,
            startXRange: range,
            startYRange:
                yRange && Number.isFinite(yAxisLength) && yAxisLength > 0 ? yRange : null,
            xAxisLength: axisLength,
            yAxisLength: Number.isFinite(yAxisLength) && yAxisLength > 0 ? yAxisLength : null,
            yAxisName: yAxisMatch ? yAxisMatch.axisName : null,
            latestRange: range,
        };
        pendingClientX = event.clientX;
        pendingClientY = event.clientY;

        window.addEventListener("pointermove", onPointerMove, true);
        window.addEventListener("pointerup", onPointerUp, true);
        window.addEventListener("pointercancel", onPointerUp, true);
    }

    function attachPlotlyListener() {
        const plot = findPlot();
        if (!plot || plot === attachedPlot) {
            return;
        }

        attachedPlot = plot;
        plot.addEventListener("pointerdown", onPointerDown, true);
    }

    window.sleepScoringCustomPointerPan = {
        enabled: ENABLE_CUSTOM_POINTER_PAN,
        isActive: false,
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", attachPlotlyListener);
    } else {
        attachPlotlyListener();
    }

    const observer = new MutationObserver(attachPlotlyListener);
    observer.observe(document.documentElement, { childList: true, subtree: true });
})();

(function () {
    "use strict";

    if (window.sleepScoringNavigationProfiler) {
        return;
    }

    const GRAPH_ID = "graph";
    const EVENT_NAME = "sleepgraphprofile";
    const MAX_PENDING = 50;

    const pendingProfiles = new Map();
    const emittedProfiles = new Set();
    let attachedPlot = null;
    let lastAfterplotTime = null;

    function nowPerformance() {
        if (window.performance && typeof window.performance.now === "function") {
            return window.performance.now();
        }
        return Date.now();
    }

    function trimPending() {
        while (pendingProfiles.size > MAX_PENDING) {
            const oldestKey = pendingProfiles.keys().next().value;
            pendingProfiles.delete(oldestKey);
        }
    }

    function markDispatched(profile) {
        if (!profile || profile.profileId === undefined) {
            return;
        }

        pendingProfiles.set(Number(profile.profileId), profile);
        trimPending();
    }

    function readProfileMarker(plot) {
        const meta = plot && plot.layout && plot.layout.meta;
        const marker = meta && meta.sleepScoringNavigationProfile;
        if (!marker || marker.profileId === undefined) {
            return null;
        }
        return marker;
    }

    function dispatchProfile(profile, marker, afterplotTime) {
        const profileId = Number(profile.profileId);
        if (emittedProfiles.has(profileId)) {
            return;
        }

        emittedProfiles.add(profileId);
        pendingProfiles.delete(profileId);

        const frameGapMs = lastAfterplotTime === null ? null : afterplotTime - lastAfterplotTime;
        const inputTime = Number(profile.inputPerformanceTime);
        const dispatchTime = Number(profile.dispatchPerformanceTime);
        const event = new CustomEvent(EVENT_NAME, {
            detail: {
                profileId,
                mode: marker.mode || profile.mode || "",
                source: profile.source || "",
                x0: profile.x0,
                x1: profile.x1,
                coalesceMs: dispatchTime - inputTime,
                dashApplyMs: afterplotTime - dispatchTime,
                browserTotalMs: afterplotTime - inputTime,
                frameGapMs,
            },
        });
        document.dispatchEvent(event);
    }

    function dispatchProfileForMarker(marker, afterplotTime) {
        if (!marker) {
            return false;
        }

        const profileId = Number(marker.profileId);
        const profile = pendingProfiles.get(profileId);
        if (profile) {
            dispatchProfile(profile, marker, afterplotTime);
            return true;
        }
        return false;
    }

    function emitProfileForMarker(marker, afterplotTime) {
        const profileTime = afterplotTime === undefined ? nowPerformance() : afterplotTime;
        const didDispatch = dispatchProfileForMarker(marker, profileTime);
        if (didDispatch) {
            lastAfterplotTime = profileTime;
        }
        return didDispatch;
    }

    function onAfterplot() {
        const afterplotTime = nowPerformance();
        const marker = readProfileMarker(attachedPlot);
        dispatchProfileForMarker(marker, afterplotTime);
        lastAfterplotTime = afterplotTime;
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
        plot.on("plotly_afterplot", onAfterplot);
    }

    window.sleepScoringNavigationProfiler = {
        markDispatched,
        emitProfileForMarker,
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", attachPlotlyListener);
    } else {
        attachPlotlyListener();
    }

    const observer = new MutationObserver(attachPlotlyListener);
    observer.observe(document.documentElement, { childList: true, subtree: true });
})();

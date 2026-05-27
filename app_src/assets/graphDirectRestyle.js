(function () {
    "use strict";

    if (window.sleepScoringDirectRestyle) {
        return;
    }

    const GRAPH_ID = "graph";
    const DATA_PROPS = new Set(["x", "y", "name", "marker"]);

    function findPlot() {
        const graphRoot = document.getElementById(GRAPH_ID);
        if (!graphRoot) {
            return null;
        }
        return graphRoot.querySelector(".js-plotly-plot");
    }

    function operationValue(operation) {
        return operation && operation.params ? operation.params.value : undefined;
    }

    function markerFromOperations(operations) {
        for (const operation of operations) {
            const location = operation.location || [];
            if (
                location.length === 3 &&
                location[0] === "layout" &&
                location[1] === "meta" &&
                location[2] === "sleepScoringNavigationProfile"
            ) {
                return operationValue(operation);
            }
        }
        return null;
    }

    function setProfileMarker(plot, marker) {
        if (!plot || !marker) {
            return;
        }

        plot.layout = plot.layout || {};
        plot.layout.meta = {
            ...(plot.layout.meta || {}),
            sleepScoringNavigationProfile: marker,
        };
    }

    function buildTraceUpdates(operations) {
        const traceUpdates = new Map();
        for (const operation of operations) {
            const location = operation.location || [];
            if (
                location.length < 3 ||
                location[0] !== "data" ||
                !DATA_PROPS.has(location[2])
            ) {
                continue;
            }

            const traceIndex = Number(location[1]);
            if (!Number.isInteger(traceIndex)) {
                continue;
            }

            const traceUpdate = traceUpdates.get(traceIndex) || {};
            traceUpdate[location[2]] = operationValue(operation);
            traceUpdates.set(traceIndex, traceUpdate);
        }
        return traceUpdates;
    }

    function buildRestyleArgs(traceUpdates) {
        const traceIndices = Array.from(traceUpdates.keys()).sort((a, b) => a - b);
        const propNames = new Set();
        for (const traceUpdate of traceUpdates.values()) {
            for (const propName of Object.keys(traceUpdate)) {
                propNames.add(propName);
            }
        }

        const restyleUpdate = {};
        for (const propName of propNames) {
            restyleUpdate[propName] = traceIndices.map((traceIndex) => {
                const traceUpdate = traceUpdates.get(traceIndex);
                return traceUpdate ? traceUpdate[propName] : undefined;
            });
        }

        return { restyleUpdate, traceIndices };
    }

    function emitProfile(marker) {
        if (
            marker &&
            window.sleepScoringNavigationProfiler &&
            typeof window.sleepScoringNavigationProfiler.emitProfileForMarker === "function"
        ) {
            window.sleepScoringNavigationProfiler.emitProfileForMarker(marker);
        }
    }

    function apply(payload) {
        const operations = (payload && payload.operations) || [];
        const plot = findPlot();
        if (!plot || !window.Plotly) {
            return {
                ok: false,
                error: "Plotly graph is not ready",
                profileId: payload && payload.profileMarker && payload.profileMarker.profileId,
            };
        }

        const profileMarker =
            (payload && payload.profileMarker) || markerFromOperations(operations);
        const traceUpdates = buildTraceUpdates(operations);
        const { restyleUpdate, traceIndices } = buildRestyleArgs(traceUpdates);

        if (!traceIndices.length) {
            emitProfile(profileMarker);
            return {
                ok: true,
                profileId: profileMarker && profileMarker.profileId,
                traceCount: 0,
                operationCount: operations.length,
            };
        }

        setProfileMarker(plot, profileMarker);
        Promise.resolve(window.Plotly.restyle(plot, restyleUpdate, traceIndices))
            .then(function () {
                emitProfile(profileMarker);
            })
            .catch(function (error) {
                console.error("sleepScoringDirectRestyle failed", error);
            });

        return {
            ok: true,
            profileId: profileMarker && profileMarker.profileId,
            traceCount: traceIndices.length,
            operationCount: operations.length,
        };
    }

    window.sleepScoringDirectRestyle = {
        apply,
    };
})();

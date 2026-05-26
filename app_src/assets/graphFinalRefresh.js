(function () {
    "use strict";

    if (window.sleepScoringFinalRefresh) {
        return;
    }

    const RESAMPLE_ENDPOINT = "/_sleep_scoring/resample";
    const GRAPH_ID = "graph";

    let latestRequestId = 0;
    let lastAppliedRequestId = 0;

    function isEnabled() {
        return Boolean(
            window.sleepScoringConfig &&
                window.sleepScoringConfig.directRestyleFinal &&
                window.sleepScoringDirectRestyle &&
                typeof window.sleepScoringDirectRestyle.apply === "function"
        );
    }

    function findPlot() {
        const graphRoot = document.getElementById(GRAPH_ID);
        if (!graphRoot) {
            return null;
        }
        return graphRoot.querySelector(".js-plotly-plot");
    }

    function buildUrl(x0, x1, marker) {
        const url = new URL(RESAMPLE_ENDPOINT, window.location.origin);
        url.searchParams.set("x0", x0);
        url.searchParams.set("x1", x1);
        if (marker) {
            if (marker.profileId !== undefined && marker.profileId !== null) {
                url.searchParams.set("profile_id", marker.profileId);
            }
            if (marker.mode) {
                url.searchParams.set("mode", marker.mode);
            }
            if (marker.source) {
                url.searchParams.set("source", marker.source);
            }
        }
        return url;
    }

    function tryDirectFetch(detail) {
        if (!isEnabled() || !detail || detail.mode !== "final") {
            return false;
        }
        if (detail.x0 === undefined || detail.x1 === undefined) {
            return false;
        }
        const plot = findPlot();
        if (!plot) {
            return false;
        }

        const requestId = ++latestRequestId;
        const profileMarker = {
            profileId: detail.profileId,
            mode: detail.mode,
            source: detail.source,
        };

        window
            .fetch(buildUrl(detail.x0, detail.x1, profileMarker), { cache: "no-store" })
            .then(function (response) {
                if (!response.ok) {
                    throw new Error("resample request failed: " + response.status);
                }
                return response.json();
            })
            .then(function (patch) {
                if (requestId < lastAppliedRequestId) {
                    return;
                }
                lastAppliedRequestId = requestId;
                const payload = {
                    operations: (patch && patch.operations) || [],
                    profileMarker: (patch && patch.profileMarker) || profileMarker,
                };
                window.sleepScoringDirectRestyle.apply(payload);
            })
            .catch(function (error) {
                console.error("sleepScoringFinalRefresh failed", error);
            });

        return true;
    }

    window.sleepScoringFinalRefresh = {
        isEnabled,
        tryDirectFetch,
    };
})();

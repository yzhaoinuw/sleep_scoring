(function () {
    const EVENT_NAME = "sleepboutcontextmenu";

    function getPrimaryYAxis(fullLayout, clientY, plotRect) {
        const plotY = clientY - plotRect.top;
        let nearestAxis = null;
        let nearestDistance = Infinity;

        Object.keys(fullLayout)
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
            })
            .forEach((axis) => {
                const top = axis._offset;
                const bottom = axis._offset + axis._length;

                if (plotY >= top && plotY <= bottom) {
                    nearestAxis = axis;
                    nearestDistance = 0;
                    return;
                }

                const distance = Math.min(Math.abs(plotY - top), Math.abs(plotY - bottom));
                if (distance < nearestDistance) {
                    nearestAxis = axis;
                    nearestDistance = distance;
                }
            });

        return nearestAxis ? nearestAxis._id : "y5";
    }

    // Keep Plotly out of the right-click path. Plotly starts its drag/click
    // machinery from `mousedown` on the drag layer, and its `clickFn` runs for
    // the right button too (the right-click flag only suppresses the synthetic
    // `click` re-dispatch). Left alone, the press lands a `plotly_click` in
    // `clickData` on mouseup, and read_click_select overwrites the bout box
    // selected below with its 0.5%-of-view strip — intermittently, because
    // Plotly only emits the click when it has live hover data. Swallowing the
    // press in the capture phase stops that at the source, the same way
    // annotationAutoPan swallows the left-button press.
    //
    // stopPropagation only: `preventDefault` on mousedown cancels `contextmenu`
    // in WebKit (the pywebview runtime on macOS), which is the event we need.
    // ctrlKey is here because ctrl+left-click is the macOS secondary click.
    document.addEventListener(
        "mousedown",
        function (event) {
            if (event.button !== 2 && !event.ctrlKey) {
                return;
            }

            const graphRoot = event.target.closest ? event.target.closest("#graph") : null;
            if (graphRoot) {
                event.stopPropagation();
            }
        },
        true
    );

    document.addEventListener(
        "contextmenu",
        function (event) {
            const graphRoot = event.target.closest ? event.target.closest("#graph") : null;
            if (!graphRoot) {
                return;
            }

            event.preventDefault();
            event.stopPropagation();

            const plot = graphRoot.querySelector(".js-plotly-plot");
            if (!plot || !plot._fullLayout) {
                return;
            }

            const fullLayout = plot._fullLayout;
            const xaxis = fullLayout.xaxis4 || fullLayout.xaxis;
            if (!xaxis || typeof xaxis.p2l !== "function") {
                return;
            }

            const plotRect = plot.getBoundingClientRect();
            const plotX = event.clientX - plotRect.left - xaxis._offset;
            if (plotX < 0 || plotX > xaxis._length) {
                return;
            }

            document.dispatchEvent(
                new CustomEvent(EVENT_NAME, {
                    detail: {
                        x: xaxis.p2l(plotX),
                        xref: xaxis._id || "x4",
                        yref: getPrimaryYAxis(fullLayout, event.clientY, plotRect),
                        timeStamp: event.timeStamp,
                    },
                })
            );
        },
        true
    );
})();

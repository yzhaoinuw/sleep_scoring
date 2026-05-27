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

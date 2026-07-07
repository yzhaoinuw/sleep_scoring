// Clientside callback implementations for the sleep scoring app.
//
// Each function is registered in app_src/callbacks/clientside.py via
// ClientsideFunction(namespace="sleep_scoring", function_name=...); the
// sections and names mirror that module.

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    sleep_scoring: {
        // ---- mode switching and navigation ----

        // switch_mode by pressing "m"
        switch_mode: function(keyboard_nevents, keyboard_event, figure) {
            if (!keyboard_event || !figure) {
                return [dash_clientside.no_update, dash_clientside.no_update, dash_clientside.no_update, dash_clientside.no_update];
            }

            var key = keyboard_event.key;

            if (key === "m" || key === "M") {
                var patched_figure = new dash_clientside.Patch;
                var predVisibility;

                if (figure.layout.dragmode === "pan") {
                    // Switch to select mode
                    patched_figure.assign(['layout', 'dragmode'], "select");
                    predVisibility = {"visibility": "visible"};
                } else if (figure.layout.dragmode === "select") {
                    // Switch to pan mode and clear selections
                    patched_figure.assign(['layout', 'selections'], null);
                    patched_figure.assign(['layout', 'shapes'], null);
                    patched_figure.assign(['layout', 'dragmode'], "pan");
                    predVisibility = {"visibility": "hidden"};
                }

                return [patched_figure.build(), "", {"visibility": "hidden"}, predVisibility];
            }

            return [dash_clientside.no_update, dash_clientside.no_update, dash_clientside.no_update, dash_clientside.no_update];
        },

        // pan_figure
        pan_figure: function(keyboard_nevents, keyboard_event, relayoutdata, figure) {
            if (!keyboard_event || !figure) {
                return [dash_clientside.no_update, dash_clientside.no_update];
            }

            var key = keyboard_event.key;
            var xaxisRange = figure.layout.xaxis4.range;
            if (
                relayoutdata &&
                relayoutdata["xaxis4.range[0]"] !== undefined &&
                relayoutdata["xaxis4.range[1]"] !== undefined
            ) {
                xaxisRange = [
                    relayoutdata["xaxis4.range[0]"],
                    relayoutdata["xaxis4.range[1]"]
                ];
            }
            var x0 = xaxisRange[0];
            var x1 = xaxisRange[1];
            var newRange;

            if (key === "ArrowRight") {
                newRange = [x0 + (x1 - x0) * 0.3, x1 + (x1 - x0) * 0.3];
            } else if (key === "ArrowLeft") {
                newRange = [x0 - (x1 - x0) * 0.3, x1 - (x1 - x0) * 0.3];
            }

            if (newRange) {
                // Use Patch for efficient partial update
                var patched_figure = new dash_clientside.Patch;
                patched_figure.assign(['layout', 'xaxis4', 'range'], newRange);

                // Create NEW object instead of mutating
                var newRelayoutData = {
                    ...(relayoutdata || {}),  // Spread existing properties
                    'xaxis4.range[0]': newRange[0],
                    'xaxis4.range[1]': newRange[1]
                };

                if (window.sleepScoringGraphRelayout) {
                    window.sleepScoringGraphRelayout.request(newRelayoutData, "keyboard");
                }

                return [patched_figure.build(), newRelayoutData];
            }

            return [dash_clientside.no_update, dash_clientside.no_update];
        },

        // apply_direct_restyle
        apply_direct_restyle: function(payload) {
            if (!payload) {
                return dash_clientside.no_update;
            }

            if (!window.sleepScoringDirectRestyle) {
                return {
                    ok: false,
                    error: "sleepScoringDirectRestyle asset is not loaded",
                    profileId: payload.profileMarker && payload.profileMarker.profileId,
                    timeStamp: Date.now()
                };
            }

            try {
                const result = window.sleepScoringDirectRestyle.apply(payload);
                return {
                    ...result,
                    timeStamp: Date.now()
                };
            } catch (error) {
                return {
                    ok: false,
                    error: error && error.message ? error.message : String(error),
                    profileId: payload.profileMarker && payload.profileMarker.profileId,
                    timeStamp: Date.now()
                };
            }
        },

        // ---- selection ----

        // read_box_select
        read_box_select: function(box_select, figure, clickData, metadata) {
            // Return no_update for all outputs if conditions not met
            const no_update = dash_clientside.no_update;

            if (!figure || !metadata) {
                return [no_update, no_update, no_update, no_update];
            }

            const video_button_style = {"visibility": "hidden"};
            const selections = figure.layout.selections;

            // When selections is None/undefined, prevent update
            if (!selections || selections.length === 0) {
                return [no_update, no_update, no_update, no_update];
            }

            // Clone figure to avoid mutating state
            var patched_figure = new dash_clientside.Patch;

            // Allow only at most one select box in all subplots
            if (selections.length > 1) {
                patched_figure.assign(['layout', 'selections'], [selections[selections.length - 1]]);
            }

            // Remove existing click select box if any
            patched_figure.assign(['layout', 'shapes'], null);

            const selection = selections[selections.length - 1];

            // Take the min as start and max as end
            let start = Math.min(selection.x0, selection.x1);
            let end = Math.max(selection.x0, selection.x1);

            const eeg_start_time = metadata.start_time;
            const eeg_end_time = metadata.end_time;

            // Check if out of range
            if (end < eeg_start_time || start > eeg_end_time) {
                return [
                    [],
                    patched_figure.build(),
                    `Out of range. Please select from ${eeg_start_time} to ${eeg_end_time}.`,
                    video_button_style
                ];
            }

            // Round start and end
            let start_round = Math.round(start);
            let end_round = Math.round(end);

            start_round = Math.max(start_round, eeg_start_time);
            end_round = Math.min(end_round, eeg_end_time);

            // Handle case where start_round equals end_round
            if (start_round === end_round) {
                if (start_round - start > end - end_round) {
                    // Spanning over two consecutive seconds
                    end_round = Math.ceil(start);
                    start_round = Math.floor(start);
                } else {
                    end_round = Math.ceil(end);
                    start_round = Math.floor(end);
                }
            }

            // Adjust relative to eeg_start_time
            const final_start = start_round - eeg_start_time;
            const final_end = end_round - eeg_start_time;

            // Show video button if valid range
            let final_video_button_style = {"visibility": "hidden"};
            if (final_end - final_start >= 1 && final_end - final_start <= 300) {
                final_video_button_style = {"visibility": "visible"};
            }

            const duration = final_end - final_start;

            return [
                [final_start, final_end],
                patched_figure.build(),
                `You selected [${final_start}, ${final_end}] (${duration} s). Press 1 for Wake, 2 for NREM, 3 for REM, or 4 for MA.`,
                final_video_button_style
            ];
        },

        // read_click_select
        read_click_select: function(clickData, figure, metadata) {
            const no_update = dash_clientside.no_update;

            if (!figure || !metadata) {
                return [no_update, no_update, no_update, no_update];
            }

            // Clone figure to avoid mutating state
            var patched_figure = new dash_clientside.Patch;

            // Remove existing select box if any
            patched_figure.assign(['layout', 'shapes'], null);

            const video_button_style = {"visibility": "hidden"};
            const dragmode = figure.layout.dragmode;

            // If no click data or in pan mode, return defaults
            if (!clickData || dragmode === "pan") {
                return [[], patched_figure.build(), "", video_button_style];
            }

            // Remove the box selection if present
            patched_figure.assign(['layout', 'selections'], null);

            // Grab clicked x value
            const x_click = clickData.points[0].x;

            // Determine current x-axis visible range
            const x_min = figure.layout.xaxis4.range[0];
            const x_max = figure.layout.xaxis4.range[1];
            const total_range = x_max - x_min;

            // Decide neighborhood size: 0.5% of current view range
            const fraction = 0.005;
            const delta = total_range * fraction;

            const eeg_start_time = metadata.start_time;
            const eeg_end_time = metadata.end_time;

            const x0 = Math.floor(x_click - delta / 2);
            const x1 = Math.ceil(x_click + delta / 2);

            // Get curve information
            const curve_index = clickData.points[0].curveNumber;
            const trace = figure.data[curve_index];
            const xref = trace.xaxis || "x4";  // x4 is the shared x-axis
            let yref = trace.yaxis || "y5";    // spectrogram has dual y-axis

            // Use the left y-axis to avoid interfering with theta/delta curve
            if (yref === "y2") {
                yref = "y1";
            }

            // Create select box
            const select_box = {
                "type": "rect",
                "xref": xref,
                "yref": yref,
                "x0": x0,
                "x1": x1,
                "y0": -30,
                "y1": 30,
                "line": {"width": 1, "dash": "dot"}
            };

            patched_figure.assign(['layout', 'shapes'], [select_box]);

            // Convert absolute plot times to sleep-score indices relative to start_time
            const start = Math.max(x0, eeg_start_time) - eeg_start_time;
            const end = Math.min(x1, eeg_end_time) - eeg_start_time;

            // Show video button if valid range
            let final_video_button_style = {"visibility": "hidden"};
            if (end - start >= 1 && end - start <= 300) {
                final_video_button_style = {"visibility": "visible"};
            }

            const duration = end - start;

            return [
                [start, end],
                patched_figure.build(),
                `You selected [${start}, ${end}] (${duration} s). Press 1 for Wake, 2 for NREM, 3 for REM, or 4 for MA.`,
                final_video_button_style
            ];
        },

        // read_bout_context_select
        read_bout_context_select: function(context_n_events, context_event, figure, metadata) {
            const no_update = dash_clientside.no_update;

            if (!context_event || !figure || !metadata) {
                return [no_update, no_update, no_update, no_update];
            }

            if (figure.layout.dragmode !== "select") {
                return [no_update, no_update, no_update, no_update];
            }

            const x_click = context_event["detail.x"];
            if (x_click === undefined || x_click === null || Number.isNaN(x_click)) {
                return [no_update, no_update, no_update, no_update];
            }

            const last_trace = figure.data[figure.data.length - 1];
            const current_sleep_scores = last_trace.z && last_trace.z[0];
            if (!Array.isArray(current_sleep_scores) || current_sleep_scores.length === 0) {
                return [no_update, no_update, no_update, no_update];
            }

            const eeg_start_time = metadata.start_time;
            const eeg_end_time = metadata.end_time;
            const clicked_index = Math.max(
                0,
                Math.min(current_sleep_scores.length - 1, Math.floor(x_click - eeg_start_time))
            );

            function scoreKey(value) {
                if (value === null || value === undefined || Number.isNaN(value)) {
                    return "unscored";
                }
                return String(value);
            }

            const clicked_key = scoreKey(current_sleep_scores[clicked_index]);
            let start = clicked_index;
            let end = clicked_index + 1;

            while (start > 0 && scoreKey(current_sleep_scores[start - 1]) === clicked_key) {
                start -= 1;
            }
            while (
                end < current_sleep_scores.length &&
                scoreKey(current_sleep_scores[end]) === clicked_key
            ) {
                end += 1;
            }

            const absolute_start = Math.max(start + eeg_start_time, eeg_start_time);
            const absolute_end = Math.min(end + eeg_start_time, eeg_end_time);
            const final_start = absolute_start - eeg_start_time;
            const final_end = absolute_end - eeg_start_time;

            const yref = context_event["detail.yref"] || "y5";
            function yAxisLayoutKey(axisRef) {
                return axisRef === "y" ? "yaxis" : "yaxis" + axisRef.slice(1);
            }

            const yaxis = figure.layout[yAxisLayoutKey(yref)] || {};
            const y_range = Array.isArray(yaxis.range) ? yaxis.range : [-30, 30];

            const select_box = {
                "type": "rect",
                "xref": context_event["detail.xref"] || "x4",
                "yref": yref,
                "x0": absolute_start,
                "x1": absolute_end,
                "y0": y_range[0],
                "y1": y_range[1],
                "line": {"width": 2, "dash": "solid"}
            };

            var patched_figure = new dash_clientside.Patch;
            patched_figure.assign(['layout', 'selections'], null);
            patched_figure.assign(['layout', 'shapes'], [select_box]);

            let final_video_button_style = {"visibility": "hidden"};
            if (final_end - final_start >= 1 && final_end - final_start <= 300) {
                final_video_button_style = {"visibility": "visible"};
            }

            const duration = final_end - final_start;

            return [
                [final_start, final_end],
                patched_figure.build(),
                `You selected bout [${final_start}, ${final_end}] (${duration} s). Press 1 for Wake, 2 for NREM, 3 for REM, or 4 for MA.`,
                final_video_button_style
            ];
        },

        // read_annotation_auto_pan_select
        read_annotation_auto_pan_select: function(annotation_n_events, annotation_event, figure, metadata) {
            const no_update = dash_clientside.no_update;

            if (!annotation_n_events || !annotation_event || !figure || !metadata) {
                return [no_update, no_update, no_update, no_update];
            }

            if (figure.layout.dragmode !== "select") {
                return [no_update, no_update, no_update, no_update];
            }

            const raw_x0 = Number(annotation_event["detail.x0"]);
            const raw_x1 = Number(annotation_event["detail.x1"]);
            if (!Number.isFinite(raw_x0) || !Number.isFinite(raw_x1) || raw_x0 === raw_x1) {
                return [no_update, no_update, no_update, no_update];
            }

            const eeg_start_time = Number(metadata.start_time);
            const eeg_end_time = Number(metadata.end_time);
            if (!Number.isFinite(eeg_start_time) || !Number.isFinite(eeg_end_time)) {
                return [no_update, no_update, no_update, no_update];
            }

            const video_button_style = {"visibility": "hidden"};
            const start = Math.min(raw_x0, raw_x1);
            const end = Math.max(raw_x0, raw_x1);

            var patched_figure = new dash_clientside.Patch;
            patched_figure.assign(['layout', 'selections'], null);

            if (end < eeg_start_time || start > eeg_end_time) {
                patched_figure.assign(['layout', 'shapes'], null);
                return [
                    [],
                    patched_figure.build(),
                    `Out of range. Please select from ${eeg_start_time} to ${eeg_end_time}.`,
                    video_button_style
                ];
            }

            let start_round = Math.round(start);
            let end_round = Math.round(end);

            if (start_round === end_round) {
                if (start_round - start > end - end_round) {
                    end_round = Math.ceil(start);
                    start_round = Math.floor(start);
                } else {
                    end_round = Math.ceil(end);
                    start_round = Math.floor(end);
                }
            }

            start_round = Math.max(start_round, eeg_start_time);
            end_round = Math.min(end_round, eeg_end_time);

            if (end_round <= start_round) {
                if (start_round < eeg_end_time) {
                    end_round = start_round + 1;
                } else if (end_round > eeg_start_time) {
                    start_round = end_round - 1;
                }
            }

            start_round = Math.max(start_round, eeg_start_time);
            end_round = Math.min(end_round, eeg_end_time);
            if (end_round <= start_round) {
                patched_figure.assign(['layout', 'shapes'], null);
                return [
                    [],
                    patched_figure.build(),
                    `Out of range. Please select from ${eeg_start_time} to ${eeg_end_time}.`,
                    video_button_style
                ];
            }

            const final_start = start_round - eeg_start_time;
            const final_end = end_round - eeg_start_time;
            const duration = final_end - final_start;

            let y0 = Number(annotation_event["detail.y0"]);
            let y1 = Number(annotation_event["detail.y1"]);
            if (!Number.isFinite(y0) || !Number.isFinite(y1)) {
                y0 = -30;
                y1 = 30;
            }

            const select_box = {
                "type": "rect",
                "xref": annotation_event["detail.xref"] || "x4",
                "yref": annotation_event["detail.yref"] || "y5",
                "x0": start_round,
                "x1": end_round,
                "y0": y0,
                "y1": y1,
                "fillcolor": "rgba(99, 110, 250, 0.14)",
                "line": {"color": "rgba(99, 110, 250, 0.95)", "width": 1, "dash": "dot"},
                "layer": "above"
            };

            patched_figure.assign(['layout', 'shapes'], [select_box]);

            let final_video_button_style = {"visibility": "hidden"};
            if (duration >= 1 && duration <= 300) {
                final_video_button_style = {"visibility": "visible"};
            }

            return [
                [final_start, final_end],
                patched_figure.build(),
                `You selected [${final_start}, ${final_end}] (${duration} s). Press 1 for Wake, 2 for NREM, 3 for REM, or 4 for MA.`,
                final_video_button_style
            ];
        },

        // ---- annotation ----

        // make_annotation
        make_annotation: function(keyboard_press, keyboard_event, box_select_range, figure) {
            const no_update = dash_clientside.no_update;

            // Only proceed if we have all required data
            if (!keyboard_event || !box_select_range || box_select_range.length === 0 || !figure) {
                return [no_update, no_update, no_update];
            }

            // Check if in select mode
            if (figure.layout.dragmode !== "select") {
                return [no_update, no_update, no_update];
            }

            const label = keyboard_event.key;
            if (!["1", "2", "3", "4"].includes(label)) {
                return [no_update, no_update, no_update];
            }

            const label_int = parseInt(label) - 1;
            const [start, end] = box_select_range;

            // Get current sleep scores from last trace
            const last_trace = figure.data[figure.data.length - 1];
            const current_sleep_scores = last_trace.z[0];

            // Create a copy using spread operator
            const sleep_scores = [...current_sleep_scores];

            // Update the range
            for (let i = start; i < end; i++) {
                sleep_scores[i] = label_int;
            }

            return [
                {"visibility": "hidden"},
                sleep_scores,
                []  // Clear box selection after annotation
            ];
        },

        // update_sleep_scores
        update_sleep_scores: function(sleep_scores, figure) {
            const no_update = dash_clientside.no_update;

            if (!sleep_scores || !Array.isArray(sleep_scores) || !figure) {
                return [no_update, no_update];
            }

            // Use Patch for efficient update
            var patched_figure = new dash_clientside.Patch;

            // Wrap in array for heatmap z-data format
            const sleep_scores_wrapped = [sleep_scores];

            // Calculate actual indices (last 3 traces)
            const num_traces = figure.data.length;
            const indices = [num_traces - 3, num_traces - 2, num_traces - 1];

            // Update all 3 heatmaps
            for (const idx of indices) {
                patched_figure.assign(['data', idx, 'z'], sleep_scores_wrapped);
            }

            // Clear selections
            patched_figure.assign(['layout', 'selections'], null);
            patched_figure.assign(['layout', 'shapes'], null);

            return [patched_figure.build(), ""];
        },

        // ---- message cleanup ----

        // clear_display
        clear_display: function(n_intervals) {
            return n_intervals === 5 ? "" : dash_clientside.no_update;
        }
    }
});

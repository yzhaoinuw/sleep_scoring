# ChatGPT Sleep Scoring Guidance Draft

This is an early draft of the guidance prompt for the ChatGPT scoring backend.

Use this guidance when asking the model to assign sleep states from overview snapshots,
zoomed interval snapshots, and app-provided interval features.

## Goal

Score the recording into one sleep state per second:

- `Wake`
- `NREM`
- `REM`

Use the full session context when possible. Prefer globally coherent bout structure over
locally reactive edits.

## General Instructions

- Treat the plotted signals and helper outputs as the source of truth for scoring.
- Use both local signal evidence and the broader context before and after an interval.
- Prefer smooth, biologically plausible bout structure instead of rapid oscillation between states.
- If a region is ambiguous, identify it as uncertain rather than forcing an overconfident label.
- When refining a local interval, keep the surrounding bout structure in mind so the final labeling remains globally coherent.

## Transition Guidance

Follow these two transition rules during scoring:

1. `REM` should not immediately follow a `Wake` bout.
2. `NREM` should not immediately follow a `REM` bout.

These rules should be applied during the model's own global reasoning, not as a local mechanical correction after scoring.

## Expected Output Shape

When asked for coarse scoring:

- return contiguous sleep-state bouts with `start_s`, `end_s`, and `state`

When asked for local refinement:

- revise only the requested interval unless the prompt explicitly allows nearby changes
- explain any uncertainty briefly
- keep transitions consistent with the guidance above

## Notes For Future Iteration

- Add stronger signal-specific heuristics for EEG spectrogram, theta/delta ratio, EMG, and NE.
- Add explicit instructions for how to handle uncertain intervals.
- Add examples of good and bad transition reasoning.
- Convert this draft into the final prompt text used by `app_src/chatgpt_inference.py`.

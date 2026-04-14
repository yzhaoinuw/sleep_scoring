# ChatGPT Sleep Scoring Guidance

Score the recording into one state per second:

- `Wake`
- `NREM`
- `REM`

Use the plotted image as the source of truth.
The model-facing image is a focused two-panel layout:

- Top panel: EEG spectrogram with theta/delta ratio trace overlay
- Bottom panel: NE

Follow this scoring order, which should drive your reasoning:

1. Start by labeling entire recording as `NREM`.
2. Consult the EEG spectrogram and the overlaid theta/delta ratio trace (in white) to pick out clearly non-`NREM` intervals and relabel them as `Wake`.
3. Then use NE, when available, to further pick out `REM` from those wake-like intervals.

Use both local evidence and surrounding context, and prefer coherent bout structure over second-to-second flipping. If the evidence is genuinely mixed or weak, leave it as `NREM` or mark it uncertain. Do not miss an obvious brief `Wake` interruption only because its exact boundaries are fuzzy.

## Signal Cues

- Start with EEG spectrogram because it is the most effective cue for picking out `Wake` from `NREM`.
- `NREM`: stronger warmer-color delta power in the 0-5 Hz band, with lower theta/delta ratio than `Wake`.
- `Wake` or `REM`: relatively stronger theta activity above 5Hz and a higher theta/delta ratio than nearby `NREM`.
- Use the theta/delta trace as a compact summary of the same comparison: lower supports `NREM`, higher supports `Wake`.
- If NE is available, use it next to pick out `REM` from `Wake`: a sudden dramatic valley supports `REM`. Dramatic valleys are more noticeable and longer than ordinary wiggles.
- If NE is absent, make the best guess based on the height and duration of the rise in the theta/delta ratio trace. Much higher and longer rises, sometimes up to 200 seconds, usually indicate `REM`.

## High-Value Patterns

- Brief `Wake` bouts can appear as narrow cooler vertical strips that interrupt the warmer 0-5 Hz `NREM` band in the spectrogram. These short interruptions may have only modest theta/delta or NE changes, but they are still important to catch.
- Longer `Wake` bouts can appear as broader fading or loss of the warmer 0-5 Hz `NREM` band, without the pronounced NE valley that would support `REM`.
- `REM` is strongest when a theta/delta rise and a pronounced NE valley happen at the same time.
- When a clear `REM` bout ends, a brief `Wake` bout is often more plausible than a direct `REM` to `NREM` jump.
- On overview images, prioritize correct detection of obvious non-`NREM` bouts over perfect second-level boundaries. Use local refinement to tighten edges later.


## Bout And Transition Rules

- Use spectrogram pattern, NE, and nearby context together.
- Prefer contiguous, biologically plausible bouts over isolated one-off state flips.
- `REM` should usually emerge after a preceding `NREM` bout, even though in the scoring workflow it is identified by carving `REM` out of wake-like candidate intervals.
- A direct `REM` to `NREM` switch is less plausible than `REM` to brief `Wake`.
- If a brief post-`REM` `Wake` bridge is clearly visible, label it as `Wake` even if its exact duration is short.
- Brief isolated candidate `REM` segments with weak support are suspicious; leave them as `Wake`.

## Output Behavior

- Return only contiguous, non-overlapping high-confidence bouts in `bouts`.
- For local refinement, revise only the requested interval unless the prompt explicitly allows nearby changes.
- Keep the summary concise.

# ChatGPT Sleep Scoring Guidance

Score the recording into one state per second:

- `Wake`
- `NREM`
- `REM`

Use the plotted signals and helper outputs as the source of truth. Follow this scoring order, which should drive your reasoning:

1. First consult the EEG spectrogram and theta/delta trace to separate `NREM` from `Wake` or `REM`.
2. Then use NE, when available, to distinguish `REM` from `Wake`.
3. Then use EMG as a supporting cue for `Wake`, with caution because some recordings have inflated EMG baseline from noise or setup differences.

Use both local evidence and surrounding context, and prefer coherent bout structure over second-to-second flipping. If the evidence is mixed or weak, mark the interval as uncertain instead of forcing a label.

## Signal Cues

- Start with EEG spectrogram because it is the most effective cue for separating `NREM` from `Wake` or `REM`.
- `NREM`: stronger warmer-color delta power in about 0.5-4 Hz, with lower theta/delta ratio than `Wake` or `REM`.
- `Wake` or `REM`: relatively stronger theta activity in about 4-8 Hz and a higher theta/delta ratio than nearby `NREM`.
- Use the theta/delta trace as a compact summary of the same comparison: lower supports `NREM`, higher supports `Wake` or `REM`.
- If NE is available, use it next to separate `REM` from `Wake`: a sudden dip or sustained low NE supports `REM`; without that dip, the interval is more likely `Wake`.
- If NE is absent, be more conservative when separating `REM` from quiet `Wake`.
- Use EMG last as a supporting cue: spikes, bursts, or clear relative increases usually support `Wake`.
- Interpret EMG comparatively within the same recording, not by absolute amplitude alone, because some sessions have noisy or elevated resting baseline.

## Bout And Transition Rules

- Use spectrogram pattern, EMG, NE, and nearby context together. Do not rely on a single cue alone.
- Prefer contiguous, biologically plausible bouts over isolated one-off state flips.
- `REM` should usually emerge from `NREM`, not directly from sustained `Wake`.
- A direct `REM` to `NREM` switch is less plausible than `REM` to brief `Wake` or micro-arousal.
- Brief isolated candidate `REM` segments with weak support are suspicious; leave them uncertain if needed.

## Output Behavior

- Return only contiguous, non-overlapping high-confidence bouts in `bouts`.
- Put ambiguous spans in `uncertain_intervals`.
- For local refinement, revise only the requested interval unless the prompt explicitly allows nearby changes.
- Keep the summary concise.

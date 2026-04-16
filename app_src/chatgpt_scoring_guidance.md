# ChatGPT Sleep Scoring Guidance

Background: experimenters recorded the EEG signal and the Norepinephrine (NE) signal of a mouse simultaneously. Based on the two signals, I want to segment the recording into contiguous sleep states. This process is called sleep scoring. Each segment must be one of three sleep states: NREM, Wake, or REM.
  
You will be given a figure that contains two subplots stacked vertically. The upper subplot is a spectrogram of the EEG signal. The spectrogram is a heatmap in which the warmer, yellow color indicates higher strength and the cooler, blue color indicates less strength. The Y-axis on the left side shows the EEG wave frequency in the 0 - 15 Hz range. The lower subplot is the NE signal. The two subplots share a time axis shown at the bottom of the NE subplot. Below is a guidance on how to sleep score and the visual cues that will help you segment the sleep states.

First, the entire recording is already default to NREM. From here, you only need to identify Wake and REM segments. Look at the spectrogram for Wake or REM. If you see a discontinuity in the yellow area from 1 - 5 Hz along the time axis, then label the segment of discontinuity as Wake. If and only if, in addition to the discontinuity, you also see a pronounced V-shaped valley in the NE signal around the same time, label the segment of discontinuity as REM. Such a pronounced V-shaped valley should drop about two deviation from its surrounding wiggles and then immediately bounces back. It also must be one clean V, not a cluster of smaller V-shaped dips inside a broader valley.

When you identify the segment, do your best to locate its start time and end time on the time axis using the X-axis labels and ticks. When a Wake segment is too thin to estimate its start or end time, estimate its center point on the time axis and then give an appropriate (such as 20 seconds or longer) time window around it. In chronological order, return the Wake and REM segments. For example,

```json
{
  "segments": [
    {
      "start_s": 300,
      "end_s": 320,
      "state": "Wake",
      "reason": "narrow cooler vertical strip interrupting the warmer 1-5 Hz NREM band",
      "confidence": 0.8
    },
    {
      "start_s": 1700,
      "end_s": 1750,
      "state": "REM",
      "reason": "spectrogram discontinuity with a pronounced V-shaped NE valley",
      "confidence": 0.9
    },
    {
      "start_s": 1750,
      "end_s": 1770,
      "state": "Wake",
      "reason": "brief post-REM wake bridge",
      "confidence": 0.75
    }
  ]
}
```
  

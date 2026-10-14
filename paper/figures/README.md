# Paper figure source snapshots

These five replacement screenshots were supplied by the maintainer on
2026-10-13. All show
the same time range from the same MATLAB recording. The agreed two-figure
layout and proposed captions are in
[`manuscript_layout.md`](../manuscript_layout.md#recommended-figures).

| File | Captured state | Planned panel |
| --- | --- | --- |
| [`app_figure.png`](app_figure.png) | Prediction before the correction shown in the next snapshot. | Figure 1A |
| [`app_figure_correction.png`](app_figure_correction.png) | Expert NREM label at 9752–9802 s (50 s). | Figure 1B |
| [`app_figure_after_correction.png`](app_figure_after_correction.png) | Predictions regenerated from 228 labelled seconds: 178 Wake and 50 NREM. | Figure 1C |
| [`app_figure_right_click_select.png`](app_figure_right_click_select.png) | Right-click selection of the 9176–9272 s bout. | Figure 2A |
| [`app_figure_video.png`](app_figure_video.png) | Behavior clip for that selected 96-second interval. | Figure 2B |

Keep these files unchanged as source assets. The composed figures used in
`paper.md` are generated from them:

```
python paper/figures/build_figures.py
```

| Output | Contents |
| --- | --- |
| [`figure1_adaptation.png`](figure1_adaptation.png) | Three full views of the same viewport stacked on one time axis: prediction (A), expert label (B), and after calibration (C). All include the spectrogram; tick labels appear only under C. C outlines the labeled interval; readers compare stages down the shared axis. |
| [`figure2_video_review.png`](figure2_video_review.png) | The right-click view from the controls row (Check Video outlined) through the time axis, cut at 9675 s, with the selected bout outlined (A), beside the video dialog (B) scaled with its aspect ratio kept to equal height. |

The 1980-by-1202-pixel sources share the same horizontal axis; crops align
to the 9200/9400 s tick centres (about 1.56 px/s). The post-calibration image
is two pixels higher and the selection image one pixel lower. Figure 1
keeps the recording name and colour bar on A only, drops Plotly's hover
toolbar, and moves the axis title below C's ticks, which it overlapped.
Figure 2A keeps the original tick values and signal pixels, recenters the
subplot titles on the cropped plot area, and adds a time-axis title.
Both figures omit the app's status messages (Figure 1C's calibration
summary and Figure 2A's selection message); the captions state what they say. The cached label layer confirms
the 228-second total, including the earlier Wake interval [2873, 3051) and
the NREM correction [9752, 9802); the earlier Wake interval is outside the view.

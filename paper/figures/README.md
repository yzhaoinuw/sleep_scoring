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
| [`figure1_adaptation.png`](figure1_adaptation.png) | Overview (A) plus matching 9345–9902 s crops before (B) and after (C) calibration; B and C together span exactly A's plot width, so both rows share plot edges. C outlines the labeled interval; readers compare stages with B across the shared time axis. |
| [`figure2_video_review.png`](figure2_video_review.png) | A 9070–9675 s crop with the right-clicked bout outlined (A) and the video dialog (B), scaled with its aspect ratio kept so both panels have equal height. |

The 1980-by-1202-pixel sources share the same horizontal axis; crops align
to the 9200/9400 s tick centres (about 1.56 px/s). The post-calibration image
is two pixels higher and the selection image one pixel lower. Zoomed panels
retain the original tick values and signal pixels, recenter subplot titles,
and add a common time-axis title; panel A keeps the full stage legend, drops
Plotly's hover toolbar, and moves its axis title below the ticks it overlapped.
Screenshot messages are wrapped at word gaps. The cached label layer confirms
the 228-second total, including the earlier Wake interval [2873, 3051) and
the NREM correction [9752, 9802); the earlier Wake interval is outside the view.

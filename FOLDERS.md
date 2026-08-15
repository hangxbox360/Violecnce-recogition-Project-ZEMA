# ZEMA-Net Demo GIFs — Folder Guide

Optimized GIFs (620px wide, 10 fps) of the ZEMA-Net demo clips, grouped by
scenario. Each GIF is a side-by-side panel: input clip with the `+ZEMA (ours)`
prediction on the left, and a `P(violence)`-vs-time plot (ZEMA vs. six baselines)
on the right. `+ZEMA (ours)` = the full ZEMA + GhostNetV2 + ConvLSTM pipeline.

| Folder | Scenario | Clips |
|:--|:--|:--:|
| `01_Camera-Motion-Spectrum` | Stationary → non-stationary camera range | 12 |
| `02_Best-Cases` | ZEMA at its strongest (confident, correct) | 3 |
| `03_Interesting-Cases` | Challenging / edge cases where ZEMA struggles | 3 |
| `04_Robustness-Cases` | Crowd / clutter where ZEMA stays stable | 6 |
| `05_Bad-Cases` | Clear failures (false negative + false positive) | 2 |

Delivered as two ZIPs (each under GitHub's 25 MB web-upload limit); extract both
into the same location to reassemble the full tree.

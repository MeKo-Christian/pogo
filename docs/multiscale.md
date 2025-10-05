# Multi-Scale Detection and Adaptive Pyramid

This document explains how POGO’s multi-scale detection improves small-text sensitivity, how the adaptive pyramid works, and how to tune it via CLI flags.

## Why Multi-Scale?

Text of different sizes benefits from evaluating an image at multiple scales. A higher resolution (scale 1.0) catches large and medium text. Additional, smaller scales can help reveal small text that would otherwise be missed at coarser feature resolutions.

POGO runs detection at multiple scales and merges results using IoU-based strategies (hard NMS, Soft‑NMS, adaptive, or size‑aware NMS).

## How It Works

1. Scale selection

- Fixed scales: `--det-scales 1.0,0.75,0.5` (or configured via file).
- Adaptive scales: `--det-ms-adaptive` auto-generates additional scales based on image size and limits.

2. Per-scale detection

- Each scale is resized to multiples of 32.
- Optional morphology and adaptive thresholds applied on probability maps.
- Regions are computed and re-mapped to original image coordinates.

3. Fusion

- Results across scales are merged using the configured strategy:
  - Hard NMS (default), Soft‑NMS (linear/gaussian), adaptive, or size‑aware NMS.
- Cross-scale merging threshold is controlled by `--det-merge-iou`.

4. Memory efficiency (incremental merge)

- With `--det-ms-incremental-merge` (default: true), results are merged after every scale to keep working sets small.

## Visual Intuition (ASCII)

Original image (1.0):

```
+----------------------------+
|   LARGE TEXT        MEDIUM |
|                            |
|         small small        |
+----------------------------+
```

Downscaled (0.75):

```
+----------------------------+
|  larg txt        medium    |
|                            |
|     small  small           |
+----------------------------+
```

Downscaled (0.5):

```
+----------------------------+
| lrg           med          |
|                            |
|   small small small        |
+----------------------------+
```

Merging boxes across scales recovers both large and small text while suppressing duplicates.

## Recommended Settings

- Start with:
  - `--det-multiscale --det-scales 1.0,0.75,0.5 --det-merge-iou 0.3`
  - Or: `--det-multiscale --det-ms-adaptive` for auto scales on large images
- For very large inputs (e.g., > 3000 px min side):
  - `--det-ms-adaptive --det-ms-max-levels 4 --det-ms-min-side 320`
- Keep `--det-ms-incremental-merge` enabled to reduce memory footprint.

## CLI Flags Summary

- `--det-multiscale` Enable multi-scale detection
- `--det-scales` Fixed scale list (e.g., `1.0,0.75,0.5`)
- `--det-merge-iou` IoU threshold for cross-scale deduplication
- `--det-ms-adaptive` Adaptive pyramid generation
- `--det-ms-max-levels` Max levels when adaptive (including 1.0)
- `--det-ms-min-side` Stop when min(image side × scale) ≤ this value
- `--det-ms-incremental-merge` Merge after each scale (default: true)

## Server & PDF

All flags above are available for `serve` and `pdf` commands. For server, flags are applied at startup and affect all requests until restart.

## Notes

- Scales always include 1.0.
- All detections are mapped back to original coordinates before merging.
- Choose Soft‑NMS (`--nms-method linear|gaussian`) if you want gentler deduplication across scales.

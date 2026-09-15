# Evaluation Methodology

All numbers shown anywhere in the app are computed live from the loaded
images and current parameter settings — nothing is hardcoded or simulated
after the fact.

## Metrics computed

| Metric | Computed as |
|---|---|
| Keypoints (ref/src) | `len(cv2.SIFT.detectAndCompute(...)[0])` per image |
| Candidate matches | Number of KNN (k=2) pairs returned by BFMatcher |
| Good matches | Candidates surviving Lowe's ratio test (`m.distance < ratio * n.distance`) |
| Inliers / Outliers | RANSAC mask from `cv2.findHomography` / `cv2.estimateAffinePartial2D` |
| Inlier ratio | `inliers / good_matches` |
| Spatial coverage % | `occupied_cells / total_cells * 100` over an N×N grid of inlier points |
| Distribution score | Normalized Shannon entropy of per-cell inlier counts (0 = all one cell, 1 = perfectly uniform) |
| Registration error (px) | Mean corner-displacement between the estimated transform and the **known** ground-truth transform — only available for the synthetic demo pair |
| Refinement error (px) | Mean sub-pixel displacement magnitude reported by phase-correlation refinement |
| Runtime (s) | Wall-clock time for detection + matching + RANSAC + spatial analysis, measured with `time.perf_counter()` |
| Status | SUCCESS if RANSAC verification passed and met the minimum-inlier threshold; FAILED otherwise, with a stated reason |

## Experiment comparison table

The **Robustness Testing** page reruns the entire baseline pipeline (not just
the geometric fit) against:

- **Scale experiment**: source image resized to 0.5×, 0.75×, 1.0×, 1.25×, 1.5×
  and pasted back onto the original canvas size before rerunning detection.
- **Illumination experiment**: source image brightness/contrast/gamma varied
  per preset (see `src/evaluation/robustness.py::ILLUMINATION_PRESETS`) before
  rerunning detection.

Each row of the comparison table on the **📈 Evaluation** page is a real,
independently computed `PipelineMetrics` instance.

## What is NOT claimed

- No claim of SIH-level or production-grade sub-pixel accuracy.
- No claim that simulated illumination changes are equivalent to real
  Sun-angle-varying observations.
- No claim that the classical SIFT baseline solves cross-modal
  (OHRC↔IIRS / TMC-2↔IIRS) correspondence — those results are explicitly
  flagged as an advanced/research-mode preview.
- No registration error number is shown for image pairs without a known
  ground-truth transform; the field displays `N/A` instead.

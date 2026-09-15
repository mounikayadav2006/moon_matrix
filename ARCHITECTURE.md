# Architecture

## Pipeline flow

```text
Reference image ──┐
                   ├─▶ Preprocessing ──▶ SIFT Feature Detection ──┐
Source image ──────┘                                              │
                                                                    ▼
                                                          Feature Matching
                                                     (BFMatcher + KNN + Lowe ratio)
                                                                    │
                                                                    ▼
                                                     RANSAC Geometric Verification
                                                       (homography / affine, inliers)
                                                                    │
                                        ┌───────────────────────────┼──────────────────────────┐
                                        ▼                           ▼                          ▼
                          Spatial Distribution Analysis    Image Registration        Sub-pixel Refinement
                             (N×N grid, coverage %)      (warp, overlay, diff)     (phase correlation, prototype)
                                        │                           │                          │
                                        └───────────────┬───────────┴──────────────────────────┘
                                                         ▼
                                                Evaluation Dashboard
                                          (real metrics, experiment comparison,
                                           hackathon one-screen summary)
```

Robustness experiments (`src/evaluation/robustness.py`) rerun this same
pipeline against systematically perturbed source images (scaled or
illumination-varied) to produce the experiment comparison table.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `src/preprocessing/preprocess.py` | Grayscale/normalize/CLAHE/contrast/denoise/resize, each independently toggleable |
| `src/features/sift.py` | SIFT keypoint + descriptor extraction, visualization |
| `src/matching/matcher.py` | BFMatcher KNN matching + Lowe's ratio test, visualization |
| `src/matching/cross_modal.py` | Sensor-pair difficulty classification; `AdvancedMatcher` interface for future SuperPoint/LightGlue/LoFTR integration |
| `src/geometry/ransac.py` | RANSAC-based homography/affine estimation with explicit rejection on failure |
| `src/spatial/grid.py` | N×N grid occupancy, coverage %, entropy-based distribution score |
| `src/registration/register.py` | Warping, overlay, difference map, ground-truth error computation |
| `src/subpixel/refine.py` | Local phase-correlation sub-pixel correspondence refinement |
| `src/geospatial/coords.py` | Metadata-gated pixel↔lat/lon conversion; never fabricates values |
| `src/mapping/lunar_map.py` | Footprint plotting, only invoked when metadata is available |
| `src/evaluation/metrics.py` | `PipelineMetrics` dataclass shared by the live run and all experiments |
| `src/evaluation/robustness.py` | Scale and illumination robustness experiment runners |
| `src/utils/sample_data.py` | Synthetic demo image + ground-truth-transform generator |
| `src/utils/image_io.py` | Upload/path loading, validation, format normalization |
| `app.py` | Streamlit UI — one page per pipeline stage, backed by `st.session_state` |

## Design choices

- **State machine via `st.session_state`**: each pipeline stage writes its
  result object into session state so later pages (Spatial Analysis,
  Registration, Evaluation) can consume upstream results without recomputation,
  while still allowing the user to re-run any stage with new parameters.
- **Fail-loud, not fail-silent**: `InsufficientFeaturesError` and
  `RansacResult.success=False` propagate real failure reasons to the UI instead
  of the app silently producing a degraded or fabricated result.
- **Ground truth only where it genuinely exists**: the synthetic demo pair
  carries a known transform (`demo_ground_truth_H.npy`) specifically so
  registration error can be reported as a real number; user uploads never get
  a numeric error invented for them.
- **Cross-modal honesty boundary**: `src/matching/cross_modal.py` is the single
  place that decides whether a sensor pair is "baseline" or
  "advanced/research mode," so this judgment isn't duplicated (and potentially
  inconsistently applied) across UI pages.

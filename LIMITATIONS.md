# Limitations

This document is the single source of truth for what MoonMatrix AI's internal
hackathon prototype does and does not do. Keep it up to date as the project
evolves — a hackathon judge asking "does this actually do X?" should be
answerable by pointing here.

## Data

- Bundled sample images are **synthetic, procedurally generated** grayscale
  textures with painted craters — not Chandrayaan-2 imagery. Labeled as such
  everywhere in the UI ("Demo Dataset — Synthetic Test Data").
- No PDS3/PDS4 label parser is implemented; geospatial metadata must be
  supplied manually as a small JSON sidecar (see DATA_GUIDE.md), and the app
  explicitly says "Geospatial metadata unavailable for this dataset" when
  none is supplied — coordinates are never fabricated.
- Only PNG/JPG/TIFF uploads are supported; converting raw `.img`/`.qub`
  Chandrayaan-2 products to those formats is a manual pre-processing step
  outside this prototype (candidate for a GDAL-based converter post-selection).

## Algorithms

- **Feature detection/matching** uses classical SIFT + BFMatcher + Lowe ratio
  test only. This works well for same-sensor or visually-similar sensor pairs,
  and noticeably degrades for large illumination differences or genuinely
  cross-modal pairs (especially anything involving IIRS, which is
  hyperspectral).
- **Cross-modal matching**: no SuperPoint, LightGlue, or LoFTR model is bundled
  or executed. `src/matching/cross_modal.py` provides a clean `AdvancedMatcher`
  interface and a sensor-pair classifier that flags hard pairs, but the actual
  matching for those pairs still runs the classical baseline and is labeled as
  an "advanced/research mode preview," not a validated result.
- **Sub-pixel refinement** is a local, patch-based phase-correlation estimate
  around each inlier correspondence. It has not been validated against a
  known sub-pixel ground truth beyond the synthetic demo pair, and patch-based
  phase correlation can be sensitive to local texture quality, patch size, and
  correspondence density. Labeled "Sub-pixel refinement prototype" throughout.
- **Illumination robustness** experiments apply synthetic brightness/contrast/
  gamma transforms to a single source image. This is a reasonable first-order
  proxy for illumination sensitivity but is **not** a substitute for real
  Sun-angle-varying observation pairs, which have more complex effects
  (shadow direction/length changes, not just global brightness/contrast).
- **Registration error** is only numerically reportable for the synthetic
  demo pair, which carries a known ground-truth transform. There is currently
  no independent, external ground truth for real Chandrayaan-2 image pairs in
  this prototype.
- **Lunar map** uses simple bilinear interpolation of four corner lat/lon
  values across the image footprint — not a rigorous selenographic map
  projection (e.g. no accounting for lens distortion, terrain relief, or
  orbital geometry).

## Engineering

- No GPU acceleration; all processing is CPU-bound OpenCV, fine for hackathon
  demo image sizes but not benchmarked at operational Chandrayaan-2 scene
  resolutions.
- No persistence layer — results reset when the Streamlit session ends
  (by design, to keep the prototype simple and dependency-light).
- Minimal automated test coverage (`tests/`) — covers preprocessing, feature
  detection, matching, RANSAC, and spatial analysis at the unit level. No
  end-to-end UI tests.

## Roadmap (post-selection)

1. Integrate a learned cross-modal-robust matcher (SuperPoint + LightGlue,
   or LoFTR) behind the existing `AdvancedMatcher` interface.
2. Implement real Chandrayaan-2 PDS3/PDS4 label parsing for accurate
   geospatial metadata (replacing the manual JSON sidecar).
3. Add DEM-aware / ortho-rectified registration for terrain-corrected
   correspondence.
4. Build a benchmark suite against real, labeled Chandrayaan-2 image pairs
   (including genuine multi-date, multi-Sun-angle pairs).
5. Validate and improve sub-pixel correspondence accuracy against a
   controlled ground truth beyond the synthetic demo pair.
